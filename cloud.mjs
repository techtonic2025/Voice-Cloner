// Vast AI orchestration: rent a GPU, run one Qwen job, download, always destroy.
import { createWriteStream } from 'node:fs';
import { mkdir, readFile, writeFile, rename } from 'node:fs/promises';
import { join } from 'node:path';
import { randomBytes, randomUUID } from 'node:crypto';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';

const VAST = process.env.CLOVA_VAST_API || 'https://console.vast.ai/api/v0';
const LABEL_PREFIX = 'clova-voice-';
const IMAGE = process.env.CLOVA_WORKER_IMAGE || 'ghcr.io/techtonic2025/clova-voice-qwen:latest';
const IMAGE_LOGIN = process.env.CLOVA_IMAGE_LOGIN || '';
const BOOT_TIMEOUT_MS = 20 * 60_000;
const LOAD_TIMEOUT_MS = 10 * 60_000;

export function createCloud({ root, voicesDir }) {
  const outputsDir = join(root, 'outputs');
  const activeFile = join(root, 'data', 'active-instances.json');
  const jobs = new Map();
  let running = null;

  async function apiKey() {
    if (process.env.VAST_API_KEY) return process.env.VAST_API_KEY.trim();
    try {
      const key = (await readFile(join(root, 'api.txt'), 'utf8')).trim().split(/\s+/)[0];
      if (key) return key;
    } catch { /* Fall through to the error below. */ }
    throw new Error('Chiave Vast mancante: mettila in api.txt o nella variabile VAST_API_KEY.');
  }

  async function vast(path, { method = 'GET', body, signal, base = VAST } = {}) {
    const response = await fetch(`${base}${path}`, {
      method, signal,
      headers: { authorization: `Bearer ${await apiKey()}`, 'content-type': 'application/json', accept: 'application/json' },
      body: body ? JSON.stringify(body) : undefined
    });
    const text = await response.text();
    let data = {};
    try { data = text ? JSON.parse(text) : {}; } catch { data = { msg: text }; }
    if (!response.ok || data.success === false) {
      const error = new Error(`Vast ${method} ${path}: ${data.msg || data.error || data.detail || response.status}`);
      error.status = response.status;
      throw error;
    }
    return data;
  }

  // Instances we created are tracked on disk so a crash or closed window never leaves one billing.
  async function readActive() {
    try { return JSON.parse(await readFile(activeFile, 'utf8')); } catch { return []; }
  }
  async function writeActive(list) {
    await mkdir(join(root, 'data'), { recursive: true });
    await writeFile(`${activeFile}.tmp`, JSON.stringify(list, null, 2));
    await rename(`${activeFile}.tmp`, activeFile);
  }
  async function track(id) { await writeActive([...new Set([...(await readActive()), id])]); }
  async function untrack(id) { await writeActive((await readActive()).filter((item) => item !== id)); }

  // Destroy, then confirm it is gone. If Vast cannot confirm, the id stays tracked
  // and the next server start retries it.
  async function destroy(id) {
    for (let attempt = 1; attempt <= 8; attempt++) {
      try {
        await vast(`/instances/${id}/`, { method: 'DELETE' });
      } catch (error) {
        if (error.status !== 404) console.warn(`Distruzione ${id}, tentativo ${attempt}: ${error.message}`);
      }
      await sleep(Math.min(attempt, 4) * 2500);
      try {
        const data = await vast(`/instances/${id}/`);
        if (!data.instances) { await untrack(id); return true; }
      } catch (error) {
        if (error.status === 404) { await untrack(id); return true; }
      }
    }
    console.error(`ATTENZIONE: non riesco a confermare la distruzione dell'istanza Vast ${id}. Controlla su https://cloud.vast.ai/instances/`);
    return false;
  }

  async function cleanupOrphans() {
    const ids = new Set(await readActive());
    try {
      const data = await vast('/instances/', { base: VAST.replace('/v0', '/v1') });
      for (const instance of data.instances || []) if (String(instance.label || '').startsWith(LABEL_PREFIX)) ids.add(instance.id);
    } catch (error) { if (!ids.size) return []; console.warn(`Controllo istanze non riuscito: ${error.message}`); }
    for (const id of ids) await destroy(id);
    return [...ids];
  }

  // compute_cap 800-900 = Ampere/Ada: Blackwell (1200) is not supported by torch 2.6 + flash-attn 2.7.
  async function searchOffers(maxRate, characters) {
    const data = await vast('/bundles/', {
      method: 'POST',
      body: {
        limit: 200, type: 'ondemand',
        verified: { eq: true }, rentable: { eq: true }, rented: { eq: false },
        num_gpus: { eq: 1 }, gpu_ram: { gte: 16000 }, compute_cap: { gte: 800, lte: 900 },
        cuda_max_good: { gte: 12.6 }, reliability: { gte: 0.98 },
        inet_down: { gte: 1000 }, disk_bw: { gte: 1000 }, disk_space: { gte: 40 }, direct_port_count: { gte: 1 },
        dph_total: { lte: maxRate }, order: [['dlperf', 'desc']]
      }
    });
    const offers = Array.isArray(data.offers) ? data.offers : data.offers ? [data.offers] : [];
    return rankOffers(offers, characters);
  }

  function workerAddress(instance) {
    const mapping = instance.ports?.['8000/tcp'];
    const port = Array.isArray(mapping) ? mapping.find((item) => item.HostPort)?.HostPort : null;
    return instance.public_ipaddr && port ? `http://${instance.public_ipaddr.trim()}:${port}` : null;
  }

  function startJob({ voice, text, language, maxRate, budgetCap, pause }) {
    if (running) throw new Error('C’è già una generazione in corso. Attendi o annullala.');
    const job = {
      id: randomUUID(), status: 'searching', percent: 2, detail: 'Cerco una GPU adatta…',
      startedAt: Date.now(), cost: 0, files: null, error: null
    };
    const controller = new AbortController();
    // Hard ceiling: the budget cap turns into a maximum rental time.
    const maxMs = Math.max(5, (budgetCap / maxRate) * 60) * 60_000;
    const watchdog = setTimeout(() => controller.abort(new Error('Tempo massimo raggiunto per il budget scelto.')), maxMs);
    job.cancel = () => controller.abort(new Error('Generazione annullata.'));
    jobs.set(job.id, job);
    running = job;
    run(job, controller.signal, { voice, text, language, maxRate, pause })
      .catch((error) => Object.assign(job, { status: 'error', error: controller.signal.reason?.message || error.message }))
      .finally(() => { clearTimeout(watchdog); running = null; });
    return job;
  }

  async function run(job, signal, { voice, text, language, maxRate, pause }) {
    const set = (status, percent, detail) => Object.assign(job, { status, percent, detail });
    let instanceId = null;
    let rentedAt = 0;
    try {
      const offers = await searchOffers(maxRate, text.length);
      if (!offers.length) throw new Error(`Nessuna GPU veloce disponibile sotto $${maxRate}/ora. Alza la tariffa massima e riprova.`);

      const token = randomBytes(24).toString('hex');
      for (const offer of offers.slice(0, 4)) {
        signal.throwIfAborted();
        set('creating', 6, `Noleggio ${offer.gpu_name} · rete ${(offer.inet_down / 1000).toFixed(1)} Gbit/s · $${offer.dph_total.toFixed(3)}/ora · tempo stimato ~${Math.ceil(offer.estimatedSeconds / 60)} min…`);
        try {
          // 24 GB+ cards fit twice as many chunks per batch.
          const batchSize = offer.gpu_ram >= 23000 ? 8 : 4;
          const created = await vast(`/asks/${offer.id}/`, {
            method: 'PUT', signal,
            body: {
              client_id: 'me', image: IMAGE, label: `${LABEL_PREFIX}${job.id.slice(0, 8)}`,
              disk: 40, runtype: 'args', env: `-p 8000:8000 -e CLOVA_JOB_TOKEN=${token} -e CLOVA_BATCH_SIZE=${batchSize}`,
              ...(IMAGE_LOGIN ? { image_login: IMAGE_LOGIN } : {})
            }
          });
          instanceId = created.new_contract;
          rentedAt = Date.now();
          await track(instanceId);
          Object.assign(job, { instanceId, gpu: offer.gpu_name, rate: offer.dph_total });
          break;
        } catch (error) {
          if (signal.aborted) throw error;
          console.warn(`Offerta ${offer.id} non disponibile: ${error.message}`);
        }
      }
      if (!instanceId) throw new Error('Nessuna delle GPU trovate ha accettato il noleggio. Riprova tra poco.');
      const updateCost = () => { job.cost = +((Date.now() - rentedAt) / 3_600_000 * job.rate).toFixed(4); };

      // 1. Wait for the container (image pull is the slow part).
      let address = null;
      const bootDeadline = Date.now() + BOOT_TIMEOUT_MS;
      while (!address) {
        if (Date.now() > bootDeadline) throw new Error('La GPU non si è avviata in tempo.');
        await sleep(5000, signal);
        updateCost();
        const { instances: instance } = await vast(`/instances/${instanceId}/`, { signal });
        const elapsed = Math.round((Date.now() - rentedAt) / 1000);
        set('booting', Math.min(38, 10 + elapsed / 15), `Avvio GPU e download del worker (${elapsed}s)… ${instance?.status_msg ? `· ${String(instance.status_msg).slice(0, 80)}` : ''}`);
        if (instance?.actual_status === 'running') address = workerAddress(instance);
      }

      const worker = (path, options = {}) => fetch(`${address}${path}`, {
        ...options, signal, headers: { authorization: `Bearer ${token}`, ...(options.headers || {}) }
      });

      // 2. Wait for Qwen and Whisper to load into VRAM.
      const loadDeadline = Date.now() + LOAD_TIMEOUT_MS;
      for (;;) {
        if (Date.now() > loadDeadline) throw new Error('Il modello non si è caricato in tempo.');
        updateCost();
        set('loading', 42, 'Caricamento di Qwen in memoria GPU…');
        try {
          const health = await (await worker('/health')).json();
          if (health.error) throw new Error(`Worker: ${health.error}`);
          if (health.ready) { job.attention = health.attention; break; }
        } catch (error) {
          if (signal.aborted || error.message.startsWith('Worker:')) throw error;
        }
        await sleep(3000, signal);
      }

      // 3. Submit the job. A cached short clip + transcript skips Whisper entirely.
      set('uploading', 48, 'Invio della voce e del testo…');
      const useClip = Boolean(voice.clipFilename && voice.referenceText);
      const referenceFile = join(voicesDir, voice.id, useClip ? voice.clipFilename : voice.filename);
      const form = new FormData();
      form.append('reference', new Blob([await readFile(referenceFile)], { type: useClip ? 'audio/wav' : voice.contentType }), useClip ? 'reference.wav' : voice.filename);
      form.append('reference_text', useClip ? voice.referenceText : '');
      form.append('text', text);
      form.append('language', language);
      form.append('pause', String(pause));
      const submitted = await worker('/jobs', { method: 'POST', body: form });
      const submittedData = await submitted.json();
      if (!submitted.ok) throw new Error(submittedData.detail || 'Il worker ha rifiutato il lavoro.');

      // 4. Follow progress.
      let remote;
      for (;;) {
        await sleep(2500, signal);
        updateCost();
        remote = await (await worker(`/jobs/${submittedData.id}`)).json();
        if (remote.status === 'error') throw new Error(remote.error);
        if (remote.status === 'done') break;
        if (remote.status === 'generating' && remote.total) {
          set('generating', 50 + Math.round((remote.done / remote.total) * 40), `Sintesi con la tua voce: blocco ${remote.done} di ${remote.total}`);
        } else {
          set('generating', 50, remote.status === 'preparing' ? 'Scelgo il pezzo migliore del campione e lo trascrivo…' : 'In coda…');
        }
      }

      // 5. Download before destroying anything.
      set('downloading', 92, 'Download dei file audio…');
      const folder = join(outputsDir, job.id);
      await mkdir(folder, { recursive: true });
      const download = async (name, destination) => {
        const response = await worker(`/jobs/${submittedData.id}/files/${name}`);
        if (!response.ok) throw new Error(`Download di ${name} non riuscito.`);
        await pipeline(Readable.fromWeb(response.body), createWriteStream(destination));
      };
      await download('full.wav', join(folder, 'clova-voice.wav'));
      await download('parts.zip', join(folder, 'clova-voice-blocchi.zip'));
      if (remote.prepared_reference) {
        await download('reference-clip.wav', join(voicesDir, voice.id, 'reference-clip.wav'));
        const profilePath = join(voicesDir, voice.id, 'profile.json');
        const profile = JSON.parse(await readFile(profilePath, 'utf8'));
        await writeFile(profilePath, JSON.stringify({ ...profile, clipFilename: 'reference-clip.wav', referenceText: remote.reference_text }, null, 2));
      }
      updateCost();
      Object.assign(job, { files: { full: `/api/jobs/${job.id}/files/clova-voice.wav`, parts: `/api/jobs/${job.id}/files/clova-voice-blocchi.zip` }, chunks: remote.total, audioSeconds: remote.seconds });
    } finally {
      if (instanceId) {
        set(job.files ? 'destroying' : job.status, job.files ? 97 : job.percent, 'Distruzione della GPU e del suo disco…');
        job.destroyed = await destroy(instanceId);
        job.cost = +((Date.now() - rentedAt) / 3_600_000 * job.rate).toFixed(4);
        if (!job.destroyed) job.detail = `Non riesco a confermare la distruzione dell’istanza ${instanceId}: controlla su cloud.vast.ai.`;
      }
    }
    set('done', 100, job.destroyed
      ? `Fatto: ${job.chunks} blocchi, ${job.audioSeconds}s di audio. GPU distrutta, costo stimato $${job.cost}.`
      : `Audio pronto, ma ${job.detail}`);
  }

  function publicJob(job) {
    if (!job) return null;
    const { cancel, ...rest } = job;
    return rest;
  }

  return {
    outputsDir, cleanupOrphans, startJob,
    getJob: (id) => publicJob(jobs.get(id)),
    cancelJob: (id) => { jobs.get(id)?.cancel?.(); return Boolean(jobs.get(id)); },
    activeInstanceIds: readActive,
    destroyAll: async () => { running?.cancel?.(); for (const id of await readActive()) await destroy(id); }
  };
}

// Rough end-to-end time per offer, used only to rank them: pulling and unpacking
// the ~9 GB image dominates short jobs, GPU speed (dlperf) dominates long ones.
const IMAGE_MB = 9000;
const EXCLUDED_GPUS = /^CMP /; // Mining cards: high dlperf score, crippled bf16.

export function rankOffers(offers, characters) {
  const audioSeconds = Math.max(10, characters / 15);
  return offers
    .filter((offer) => !EXCLUDED_GPUS.test(offer.gpu_name || '') && offer.inet_down > 0 && offer.dlperf > 0)
    .map((offer) => {
      const diskMBs = Math.max(200, offer.disk_bw || 200);
      const pull = (IMAGE_MB * 8) / offer.inet_down + (IMAGE_MB * 2.2) / diskMBs + 45;
      const load = 4500 / diskMBs + 25;
      // About 0.5 s of compute per second of audio on a dlperf-100 card.
      const generate = audioSeconds * 0.5 * (100 / offer.dlperf);
      return { ...offer, estimatedSeconds: Math.round(pull + load + generate) };
    })
    // Faster first; one minute of waiting is worth about $1/h of price difference.
    .sort((a, b) => (a.estimatedSeconds + a.dph_total * 60) - (b.estimatedSeconds + b.dph_total * 60));
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason);
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => { clearTimeout(timer); reject(signal.reason); }, { once: true });
  });
}
