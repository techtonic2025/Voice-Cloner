// Vast AI orchestration: rent a GPU, run Qwen jobs, download, always destroy.
import { createWriteStream } from 'node:fs';
import { mkdir, readFile, writeFile, rename } from 'node:fs/promises';
import { join } from 'node:path';
import { randomBytes, randomUUID } from 'node:crypto';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';

const VAST = process.env.CLOVA_VAST_API || 'https://console.vast.ai/api/v0';
const LABEL_PREFIX = 'clova-voice-';
const IMAGE = process.env.CLOVA_WORKER_IMAGE || 'ghcr.io/techtonic2025/clova-voice-qwen:latest';
const IMAGE_FLUX = process.env.CLOVA_IMAGE_WORKER || 'ghcr.io/techtonic2025/clova-image-flux:latest';
const IMAGE_LOGIN = process.env.CLOVA_IMAGE_LOGIN || '';
// LoRAs loaded into FLUX.1-dev at startup: its training data was filtered, so on its own it
// draws adult nudity poorly. They apply only to requests with "Contenuti per adulti" ticked;
// set IMAGE_LORAS_ENABLED to false to not even load them.
// Format "repo:file:scale"; the minors filter in server.mjs applies either way.
const IMAGE_LORAS_ENABLED = true;
const IMAGE_LORAS = [
  'lustlyai/Flux_Lustly.ai_Uncensored_nsfw_v1:flux_lustly-ai_v1.safetensors:0.9',
  'Keltezaa/NSFW_MASTER_FLUX:NSFW_master_Flux.safetensors:0.5',
  // Realistic gore; strongest when the prompt contains its trigger word "ccRealGore".
  'CosmicCrafter/real-gore-by-cosmic:real-gore-by-cosmic.safetensors:0.8'
];
// Edits (FLUX Kontext) keep the subject's body only with lighter LoRAs, and no gore.
const IMAGE_EDIT_LORAS = [
  'lustlyai/Flux_Lustly.ai_Uncensored_nsfw_v1:flux_lustly-ai_v1.safetensors:0.75',
  'Keltezaa/NSFW_MASTER_FLUX:NSFW_master_Flux.safetensors:0.4'
];
// What each kind of worker needs from a GPU. imageMB is the Docker image, extraMB what
// the worker downloads at startup (FLUX weights), downloadGB what the host bills as traffic
// (measured on Vast invoices: some hosts charge up to $0.04/GB, more than the GPU itself).
const WORKERS = {
  voice: { image: IMAGE, minGpuRam: 16_000, disk: 40, imageMB: 10_500, extraMB: 0, loadMB: 4_500, downloadGB: 10, loadTimeoutMs: 10 * 60_000 },
  images: { image: IMAGE_FLUX, minGpuRam: 45_000, minCpuRam: 64_000, disk: 110, imageMB: 7_000, extraMB: 34_000, loadMB: 34_000, downloadGB: 30, loadTimeoutMs: 30 * 60_000 }
};
const BOOT_TIMEOUT_MS = 20 * 60_000;
const KEEP_ALIVE_MS = Number(process.env.CLOVA_KEEP_ALIVE_MS) || 5 * 60_000;
// Bump when the worker changes how it prepares the reference, so old clips are redone.
const CLIP_VERSION = 2;
// Syllables per second of natural Italian narration (measured on the TechTonic recordings: 7.04).
const NATURAL_RATE = 7.0;
// Worker errors that mean the host itself cannot reach Hugging Face.
const HOST_NETWORK_ERROR = /CERTIFICATE_VERIFY_FAILED|SSLError|MaxRetryError|ConnectionPool|Connection(Error| refused| reset)|Name or service not known|Temporary failure in name resolution/i;
// Sampling presets sent to the worker; "natural" keeps the model defaults (temperature 0.9, top_p 1).
const STABILITY = { stable: { temperature: 0.6, topP: 0.9 }, very: { temperature: 0.4, topP: 0.8 } };

export function createCloud({ root, voicesDir, getHfToken = async () => '' }) {
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
  // Hosts that loaded the image but never started it are skipped in later searches.
  const badHostsFile = join(root, 'data', 'bad-hosts.json');
  async function readBadHosts() {
    try { return JSON.parse(await readFile(badHostsFile, 'utf8')); } catch { return []; }
  }
  async function markBadHost(machineId) {
    if (!machineId) return;
    const hosts = [...new Set([...(await readBadHosts()), machineId])];
    await mkdir(join(root, 'data'), { recursive: true });
    await writeFile(badHostsFile, JSON.stringify(hosts, null, 2));
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
  async function searchOffers(maxRate, characters, minGpuRam, profile) {
    const data = await vast('/bundles/', {
      method: 'POST',
      body: {
        limit: 200, type: 'ondemand',
        verified: { eq: true }, rentable: { eq: true }, rented: { eq: false },
        num_gpus: { eq: 1 }, gpu_ram: { gte: minGpuRam }, compute_cap: { gte: 800, lte: 900 },
        cuda_max_good: { gte: 12.6 }, reliability: { gte: 0.98 },
        inet_down: { gte: 1000 }, disk_bw: { gte: 1000 }, disk_space: { gte: profile.disk }, cpu_ram: { gte: profile.minCpuRam || 0 }, direct_port_count: { gte: 1 },
        dph_total: { lte: maxRate }, order: [['dlperf', 'desc']]
      }
    });
    const offers = Array.isArray(data.offers) ? data.offers : data.offers ? [data.offers] : [];
    const badHosts = new Set(await readBadHosts());
    return rankOffers(offers.filter((offer) => !badHosts.has(offer.machine_id)), characters, profile);
  }

  function workerAddress(instance) {
    const mapping = instance.ports?.['8000/tcp'];
    const port = Array.isArray(mapping) ? mapping.find((item) => item.HostPort)?.HostPort : null;
    return instance.public_ipaddr && port ? `http://${instance.public_ipaddr.trim()}:${port}` : null;
  }

  // The rented GPU. With "keep alive" it survives KEEP_ALIVE_MS after a job so the
  // next generation skips boot and model load. It is destroyed on that timer, by the
  // "Distruggi istanza" button, on any error, and on app shutdown.
  let session = null;
  let lastDestroy = null;

  // The host's download fee is paid once per rental, so it counts only from the rental start.
  function costSince(current, from) {
    const traffic = from === current.rentedAt ? current.downloadCost || 0 : 0;
    return +((Date.now() - from) / 3_600_000 * current.rate + traffic).toFixed(4);
  }

  async function endSession() {
    const current = session;
    if (!current) return true;
    session = null;
    clearTimeout(current.idleTimer);
    const destroyed = await destroy(current.instanceId);
    lastDestroy = { instanceId: current.instanceId, destroyed, at: Date.now(), cost: costSince(current, current.rentedAt) };
    return destroyed;
  }

  function keepWarm(current) {
    current.keepUntil = Date.now() + KEEP_ALIVE_MS;
    current.idleTimer = setTimeout(() => { if (session === current && !running) endSession(); }, KEEP_ALIVE_MS);
  }

  function workerFetch(current, path, options = {}) {
    return fetch(`${current.address}${path}`, {
      ...options, headers: { authorization: `Bearer ${current.token}`, ...(options.headers || {}) }
    });
  }

  // Reuse the kept-alive GPU if it still answers; otherwise destroy it.
  async function warmSession(kind = 'voice') {
    const current = session;
    if (current && current.kind !== kind) {
      // A kept-alive GPU of the other kind (voice vs images) runs a different worker.
      await endSession();
      return null;
    }
    if (!current?.address) return null;
    clearTimeout(current.idleTimer);
    current.keepUntil = null;
    try {
      const health = await (await workerFetch(current, '/health', { signal: AbortSignal.timeout(8000) })).json();
      if (health.ready) return current;
    } catch { /* Unreachable: replaced below. */ }
    await endSession();
    return null;
  }

  function startJob({ voice, text, language, maxRate, budgetCap, pause, keepAlive, stability, takes, pronunciations, speed, refSeconds, chunkChars }) {
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
    const options = { voice, text, language, maxRate, pause, keepAlive, stability, takes, pronunciations, speed, refSeconds, chunkChars };
    run(job, controller.signal, options)
      .catch(async (error) => {
        // Some hosts cannot reach Hugging Face over HTTPS (broken certificates or a proxy):
        // that host is already marked bad and destroyed, so one more try lands elsewhere.
        if (!error.hostNetwork || controller.signal.aborted) throw error;
        const previousCost = job.cost || 0;
        Object.assign(job, { status: 'searching', percent: 2, detail: 'La GPU non raggiungeva Hugging Face: è stata distrutta, ne cerco un’altra…', reused: false });
        await run(job, controller.signal, options);
        job.cost = +(job.cost + previousCost).toFixed(4);
      })
      .catch((error) => Object.assign(job, { status: 'error', error: controller.signal.reason?.message || error.message }))
      .finally(() => { clearTimeout(watchdog); running = null; });
    return job;
  }

  async function rent(job, signal, { maxRate, characters, minGpuRam, kind = 'voice', disk }) {
    const set = (status, percent, detail) => Object.assign(job, { status, percent, detail });
    const profile = { ...WORKERS[kind], ...(disk ? { disk } : {}) };
    const offers = await searchOffers(maxRate, characters, minGpuRam || profile.minGpuRam, profile);
    // Needed on the GPU to download private fine-tuned models and to upload new ones.
    const hfToken = await getHfToken();
    if (!offers.length) throw new Error(`Nessuna GPU veloce disponibile sotto $${maxRate}/ora. Alza la tariffa massima e riprova.`);

    const token = randomBytes(24).toString('hex');
    const bootDeadline = Date.now() + BOOT_TIMEOUT_MS;
    let current = null;
    let failedHosts = 0;
    for (const offer of offers.slice(0, 4)) {
      signal.throwIfAborted();
      if (Date.now() > bootDeadline) break;
      const retry = failedHosts ? ` (macchina precedente bloccata, provo la successiva)` : '';
      set('creating', 6, `Noleggio ${offer.gpu_name} · rete ${(offer.inet_down / 1000).toFixed(1)} Gbit/s · $${offer.dph_total.toFixed(3)}/ora${offer.downloadCost >= 0.01 ? ` + download ~$${offer.downloadCost.toFixed(2)}` : ''} · tempo stimato ~${Math.ceil(offer.estimatedSeconds / 60)} min${retry}…`);
      try {
        // 24 GB+ cards fit twice as many chunks per batch.
        const batchSize = offer.gpu_ram >= 23000 ? 8 : 4;
        const created = await vast(`/asks/${offer.id}/`, {
          method: 'PUT', signal,
          body: {
            client_id: 'me', image: profile.image, label: `${LABEL_PREFIX}${job.id.slice(0, 8)}`,
            disk: profile.disk, runtype: 'args', env: `-p 8000:8000 -e CLOVA_JOB_TOKEN=${token} -e CLOVA_BATCH_SIZE=${batchSize}${kind === 'images' && IMAGE_LORAS_ENABLED ? ` -e FLUX_LORAS=${IMAGE_LORAS.join(',')} -e FLUX_EDIT_LORAS=${IMAGE_EDIT_LORAS.join(',')}` : ''}${hfToken ? ` -e HF_TOKEN=${hfToken}` : ''}`,
            ...(IMAGE_LOGIN ? { image_login: IMAGE_LOGIN } : {})
          }
        });
        // Registered immediately so any later failure still destroys it.
        session = { instanceId: created.new_contract, machineId: offer.machine_id, kind, token, gpu: offer.gpu_name, rate: offer.dph_total, downloadCost: offer.downloadCost, rentedAt: Date.now(), address: null, keepUntil: null };
        await track(session.instanceId);
      } catch (error) {
        if (signal.aborted) throw error;
        console.warn(`Offerta ${offer.id} non disponibile: ${error.message}`);
        continue;
      }
      current = session;
      Object.assign(job, { instanceId: current.instanceId, gpu: current.gpu, rate: current.rate });

      // 1. Wait for the container (image pull is the slow part). Some hosts load the
      // image and then park the contract as "stopped" without ever starting it:
      // destroy those at once and move on to the next offer.
      let stoppedPolls = 0;
      while (!current.address && Date.now() <= bootDeadline) {
        await sleep(5000, signal);
        job.cost = costSince(current, current.rentedAt);
        const { instances: instance } = await vast(`/instances/${current.instanceId}/`, { signal });
        const elapsed = Math.round((Date.now() - current.rentedAt) / 1000);
        set('booting', Math.min(38, 10 + elapsed / 15), `Avvio GPU e download del worker (${elapsed}s)… ${instance?.status_msg ? `· ${String(instance.status_msg).slice(0, 80)}` : ''}`);
        if (instance?.actual_status === 'running') current.address = workerAddress(instance);
        const stopped = instance && (instance.intended_status === 'stopped' || instance.cur_state === 'stopped' || ['exited', 'offline'].includes(instance.actual_status));
        stoppedPolls = stopped ? stoppedPolls + 1 : 0;
        if (stoppedPolls >= 12) break; // about one minute: brief stopped states can be transient
      }
      if (current.address) break;
      if (Date.now() > bootDeadline) throw new Error('La GPU non si è avviata in tempo.');
      console.warn(`Istanza ${current.instanceId} ferma dopo il caricamento dell'immagine: la distruggo e provo un'altra macchina.`);
      failedHosts++;
      await markBadHost(offer.machine_id);
      set('creating', 6, `La ${current.gpu} si è bloccata all’avvio: la distruggo e provo un’altra macchina…`);
      await endSession();
      current = null;
    }
    if (!current) {
      throw new Error(failedHosts
        ? 'Le GPU noleggiate si sono bloccate all’avvio e sono state distrutte. Riprova tra poco.'
        : 'Nessuna delle GPU trovate ha accettato il noleggio. Riprova tra poco.');
    }

    // 2. Wait for the models to load (the image worker first downloads FLUX).
    const loadDeadline = Date.now() + profile.loadTimeoutMs;
    for (;;) {
      if (Date.now() > loadDeadline) throw new Error('Il modello non si è caricato in tempo.');
      job.cost = costSince(current, current.rentedAt);
      try {
        const health = await (await workerFetch(current, '/health', { signal })).json();
        if (health.error) throw new Error(`Worker: ${health.error}`);
        if (health.ready) {
          job.attention = health.attention;
          if (health.lora_error) console.warn(`LoRA non caricato, FLUX funziona senza: ${health.lora_error}`);
          if (health.flash_attn_error) console.warn(`FlashAttention non attivo (torch ${health.torch}): ${health.flash_attn_error}`);
          break;
        }
        if (health.phase === 'downloading' && health.download != null) {
          set('loading', 38 + Math.round(health.download * 8), `Download di FLUX da Hugging Face: ${Math.round(health.download * 100)}%…`);
        } else {
          set('loading', 46, kind === 'images' ? 'Caricamento di FLUX in memoria GPU…' : 'Caricamento di Qwen in memoria GPU…');
        }
      } catch (error) {
        if (signal.aborted || error.message.startsWith('Worker:')) throw error;
      }
      await sleep(3000, signal);
    }
    return current;
  }

  async function run(job, signal, { voice, text, language, maxRate, pause, keepAlive, stability, takes, pronunciations, speed, refSeconds, chunkChars }) {
    const set = (status, percent, detail) => Object.assign(job, { status, percent, detail });
    let current = null;
    let success = false;
    let costFrom = Date.now();
    try {
      current = await warmSession('voice');
      if (current) {
        Object.assign(job, { instanceId: current.instanceId, gpu: current.gpu, rate: current.rate, reused: true });
      } else {
        try {
          current = await rent(job, signal, { maxRate, characters: text.length });
        } catch (error) {
          current = session;
          throw error;
        }
        costFrom = current.rentedAt;
      }
      const worker = (path, options = {}) => workerFetch(current, path, { ...options, signal });
      const updateCost = () => { job.cost = costSince(current, costFrom); };

      // 3. Submit the job. A cached full-sample clip + transcript skips Whisper entirely.
      set('uploading', 48, job.reused ? 'GPU già accesa: invio subito voce e testo…' : 'Invio della voce e del testo…');
      const form = new FormData();
      if (voice.type === 'trained') {
        // Fine-tuned voice: the worker downloads the model from the private HF repo.
        form.append('trained_repo', voice.repo);
        form.append('speaker', voice.speaker);
        // Holding takes to the speaker's pace stretches the audio (WSOLA), which drags the
        // vowels and sounds coarse, so it only runs for voices that opt in.
        if (voice.autoPace) form.append('target_rate', String(voice.naturalRate || NATURAL_RATE));
      } else {
        const useClip = Boolean(voice.clipFilename && voice.referenceText && voice.clipVersion === CLIP_VERSION);
        const referenceFile = join(voicesDir, voice.id, useClip ? voice.clipFilename : voice.filename);
        form.append('reference', new Blob([await readFile(referenceFile)], { type: useClip ? 'audio/wav' : voice.contentType }), useClip ? 'reference.wav' : voice.filename);
        form.append('reference_text', useClip ? voice.referenceText : '');
      }
      form.append('text', text);
      form.append('language', language);
      form.append('pause', String(pause));
      // Lower sampling temperature/top_p: fewer slips like doubled consonants, flatter intonation.
      const sampling = STABILITY[stability];
      if (sampling) { form.append('temperature', String(sampling.temperature)); form.append('top_p', String(sampling.topP)); }
      form.append('takes', String(takes));
      form.append('pronunciations', JSON.stringify(pronunciations || []));
      form.append('speed', String(speed || 1));
      form.append('ref_seconds', String(refSeconds || 120));
      if (chunkChars) form.append('chunk_chars', String(chunkChars));
      const submitted = await worker('/jobs', { method: 'POST', body: form });
      const submittedData = await submitted.json();
      if (!submitted.ok) throw new Error(submittedData.detail || 'Il worker ha rifiutato il lavoro.');

      // 4. Follow progress.
      let remote;
      for (;;) {
        await sleep(2500, signal);
        updateCost();
        remote = await (await worker(`/jobs/${submittedData.id}`)).json();
        if (remote.status === 'error' && HOST_NETWORK_ERROR.test(remote.error)) {
          await markBadHost(current.machineId);
          throw Object.assign(new Error(`La GPU ${current.gpu} non riesce a collegarsi a Hugging Face per scaricare il modello addestrato.`), { hostNetwork: true });
        }
        if (remote.status === 'error') throw new Error(remote.error);
        if (remote.status === 'done') break;
        if (remote.status === 'generating' && remote.total) {
          set('generating', 50 + Math.round((remote.done / remote.total) * 40), `Sintesi con la tua voce: blocco ${remote.done} di ${remote.total}`);
        } else if (remote.status === 'checking') {
          set('generating', 91, `Whisper ascolta i ${remote.takes} tentativi di ogni blocco e sceglie i migliori…`);
        } else {
          set('generating', 50, {
            preparing: 'Preparo il campione e lo trascrivo…',
            loading_trained: 'Scarico e carico il tuo modello addestrato (solo la prima volta per ogni GPU)…'
          }[remote.status] || 'In coda…');
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
        await writeFile(profilePath, JSON.stringify({ ...profile, clipFilename: 'reference-clip.wav', referenceText: remote.reference_text, clipVersion: CLIP_VERSION }, null, 2));
      }
      updateCost();
      Object.assign(job, { files: { full: `/api/jobs/${job.id}/files/clova-voice.wav`, parts: `/api/jobs/${job.id}/files/clova-voice-blocchi.zip` }, chunks: remote.total, audioSeconds: remote.seconds, quality: remote.quality, weakest: remote.weakest });
      // Saved next to the audio so the history survives app restarts.
      await writeFile(join(folder, 'meta.json'), JSON.stringify({
        id: job.id, createdAt: new Date(job.startedAt).toISOString(), voiceId: voice.id, voiceName: voice.name,
        language, text, stability, takes, speed: speed || 1, refSeconds: refSeconds || 120, chunks: job.chunks, audioSeconds: job.audioSeconds,
        quality: job.quality ?? null, weakest: job.weakest ?? [], gpu: job.gpu, cost: job.cost
      }, null, 2));
      success = true;
    } finally {
      if (current && session === current) {
        if (success && keepAlive) {
          keepWarm(current);
          job.keptUntil = current.keepUntil;
        } else {
          set(job.files ? 'destroying' : job.status, job.files ? 97 : job.percent, 'Distruzione della GPU e del suo disco…');
          job.destroyed = await endSession();
          if (!job.destroyed) job.detail = `Non riesco a confermare la distruzione dell’istanza ${current.instanceId}: controlla su cloud.vast.ai.`;
        }
        job.cost = costSince(current, costFrom);
      } else if (current) {
        // Destroyed meanwhile by the button.
        job.destroyed = lastDestroy?.instanceId === current.instanceId ? lastDestroy.destroyed : true;
      }
    }
    const summary = `Fatto: ${job.chunks} blocchi, ${job.audioSeconds}s di audio, costo stimato $${job.cost}.`;
    set('done', 100, job.keptUntil
      ? `${summary} GPU ancora accesa per 5 minuti: puoi generare di nuovo senza attendere l’avvio.`
      : job.destroyed ? `${summary} GPU distrutta.` : `Audio pronto, ma ${job.detail}`);
  }

  function publicJob(job) {
    if (!job) return null;
    const { cancel, ...rest } = job;
    return rest;
  }

  // Fine-tuning: always on a fresh GPU with 40 GB+ (full fine-tune of the 1.7B model),
  // always destroyed at the end; the result lives in the user's private HF repo.
  function startTraining({ files, repo, speaker, epochs, maxRate, maxHours = 3, onTrained }) {
    if (running) throw new Error('C’è già un lavoro in corso. Attendi o annullalo.');
    const job = {
      id: randomUUID(), kind: 'training', status: 'searching', percent: 2,
      detail: 'Cerco una GPU da almeno 40 GB…', startedAt: Date.now(), cost: 0, error: null
    };
    const controller = new AbortController();
    const watchdog = setTimeout(() => controller.abort(new Error(`Tempo massimo di addestramento (${maxHours} ore) raggiunto.`)), maxHours * 3_600_000);
    job.cancel = () => controller.abort(new Error('Addestramento annullato.'));
    jobs.set(job.id, job);
    running = job;
    train(job, controller.signal, { files, repo, speaker, epochs, maxRate, onTrained })
      .catch((error) => Object.assign(job, { status: 'error', error: controller.signal.reason?.message || error.message }))
      .finally(() => { clearTimeout(watchdog); running = null; });
    return job;
  }

  async function train(job, signal, { files, repo, speaker, epochs, maxRate, onTrained }) {
    const set = (status, percent, detail) => Object.assign(job, { status, percent, detail });
    let current = null;
    try {
      await endSession(); // A kept-alive voice GPU is usually too small for training.
      try {
        // characters: 0 ranks offers on boot time only (training time is not text-based).
        // Every epoch saves a full ~4.5 GB checkpoint next to the ~12 GB image: size the disk to fit.
        current = await rent(job, signal, { maxRate, characters: 0, minGpuRam: 40_000, disk: 40 + epochs * 6 + 20 });
      } catch (error) {
        current = session;
        throw error;
      }
      const worker = (path, options = {}) => workerFetch(current, path, { ...options, signal });

      const form = new FormData();
      let bytes = 0;
      for (const file of files) {
        const data = await readFile(file.path);
        bytes += data.length;
        form.append('files', new Blob([data]), file.name);
      }
      form.append('repo', repo);
      form.append('speaker', speaker);
      form.append('epochs', String(epochs));
      form.append('language', 'Italian');
      set('uploading', 45, `Invio di ${files.length} registrazioni (${Math.round(bytes / 1_048_576)} MB)…`);
      const submitted = await worker('/train', { method: 'POST', body: form });
      const submittedData = await submitted.json();
      if (!submitted.ok) throw new Error(submittedData.detail || 'Il worker ha rifiutato l’addestramento.');

      const base = { queued: 46, segmenting: 50, codes: 58, checking: 95, uploading: 97 };
      let remote;
      for (;;) {
        await sleep(5000, signal);
        job.cost = costSince(current, current.rentedAt);
        remote = await (await worker(`/train/${submittedData.id}`)).json();
        if (remote.status === 'error') throw new Error(remote.error);
        Object.assign(job, {
          clips: remote.clips, minutes: remote.minutes, dropped: remote.dropped, loss: remote.loss, epoch: remote.epoch, epochs: remote.epochs,
          epochScores: remote.epoch_scores, baseline: remote.baseline, bestEpoch: remote.best_epoch, fidelity: remote.fidelity, similarity: remote.similarity
        });
        if (remote.status === 'done') break;
        const percent = remote.phase === 'training' ? 60 + Math.round(35 * (remote.epoch || 0) / (remote.epochs || epochs)) : base[remote.phase] ?? 46;
        set('training', percent, remote.detail || 'In corso…');
      }
      Object.assign(job, { repo, speaker });
      // Keep the test sentences of every epoch next to the real voice, for listening.
      if (remote.samples?.length) {
        set('training', 97, 'Download dei campioni di confronto…');
        const folder = join(outputsDir, job.id);
        await mkdir(folder, { recursive: true });
        for (const name of remote.samples) {
          const response = await worker(`/train/${submittedData.id}/samples/${name}`);
          if (response.ok) await pipeline(Readable.fromWeb(response.body), createWriteStream(join(folder, name)));
        }
        await writeFile(join(folder, 'meta.json'), JSON.stringify({
          id: job.id, kind: 'training', createdAt: new Date(job.startedAt).toISOString(), repo, speaker,
          clips: job.clips, minutes: job.minutes, epochs, bestEpoch: job.bestEpoch, baseline: job.baseline,
          epochScores: job.epochScores, samples: remote.samples
        }, null, 2));
      }
      await onTrained?.(job);
      job.trained = true;
    } finally {
      if (current && session === current) {
        set(job.trained ? 'destroying' : job.status, job.trained ? 98 : job.percent, 'Distruzione della GPU e del suo disco…');
        job.destroyed = await endSession();
        job.cost = costSince(current, current.rentedAt);
      }
    }
    set('done', 100, `Addestramento completato: ${job.clips} frasi (${job.minutes} min) · `
      + (job.bestEpoch ? `salvata l'epoca ${job.bestEpoch} (fedeltà ${Math.round(job.fidelity * 100)}%, somiglianza ${Math.round((job.similarity ?? 0) * 100)}%) · ` : '')
      + `modello in ${repo} (privato) · `
      + (job.destroyed ? `GPU distrutta, costo $${job.cost}.` : `controlla su cloud.vast.ai che l’istanza sia distrutta.`));
  }

  // Images: FLUX on a 45 GB+ GPU, same keep-alive and destruction rules as voice jobs.
  // With `source` (a PNG from the history) the job is an edit made with FLUX Kontext.
  function startImageJob({ prompt, fullPrompt, style, format, width, height, count, seed, steps, maxRate, budgetCap, keepAlive, source, adult }) {
    if (running) throw new Error('C’è già un lavoro in corso. Attendi o annullalo.');
    const job = {
      id: randomUUID(), kind: 'images', status: 'searching', percent: 2, detail: 'Cerco una GPU da almeno 45 GB…',
      startedAt: Date.now(), cost: 0, images: null, error: null
    };
    const controller = new AbortController();
    const maxMs = Math.max(10, (budgetCap / maxRate) * 60) * 60_000;
    const watchdog = setTimeout(() => controller.abort(new Error('Tempo massimo raggiunto per il budget scelto.')), maxMs);
    job.cancel = () => controller.abort(new Error('Generazione annullata.'));
    jobs.set(job.id, job);
    running = job;
    runImages(job, controller.signal, { prompt, fullPrompt, style, format, width, height, count, seed, steps, maxRate, keepAlive, source, adult })
      .catch((error) => Object.assign(job, { status: 'error', error: controller.signal.reason?.message || error.message }))
      .finally(() => { clearTimeout(watchdog); running = null; });
    return job;
  }

  async function runImages(job, signal, { prompt, fullPrompt, style, format, width, height, count, seed, steps, maxRate, keepAlive, source, adult }) {
    const set = (status, percent, detail) => Object.assign(job, { status, percent, detail });
    let current = null;
    let success = false;
    let costFrom = Date.now();
    try {
      current = await warmSession('images');
      if (current) {
        Object.assign(job, { instanceId: current.instanceId, gpu: current.gpu, rate: current.rate, reused: true });
      } else {
        try {
          current = await rent(job, signal, { maxRate, characters: 0, kind: 'images' });
        } catch (error) {
          current = session;
          throw error;
        }
        costFrom = current.rentedAt;
      }
      const worker = (path, options = {}) => workerFetch(current, path, { ...options, signal });

      set('uploading', 50, job.reused ? 'GPU già accesa: invio subito la richiesta…' : 'Invio della richiesta…');
      const form = new FormData();
      form.append('prompt', fullPrompt);
      form.append('width', String(width));
      form.append('height', String(height));
      form.append('count', String(count));
      form.append('seed', String(seed ?? -1));
      form.append('steps', String(steps));
      form.append('loras', adult && IMAGE_LORAS_ENABLED ? 'true' : 'false');
      if (source) form.append('image', new Blob([await readFile(source.path)], { type: 'image/png' }), 'source.png');
      const submitted = await worker('/images', { method: 'POST', body: form });
      const submittedData = await submitted.json();
      if (!submitted.ok) throw new Error(submittedData.detail || 'Il worker ha rifiutato la richiesta.');

      let remote;
      for (;;) {
        await sleep(2500, signal);
        job.cost = costSince(current, costFrom);
        remote = await (await worker(`/images/${submittedData.id}`)).json();
        if (remote.status === 'error') throw new Error(remote.error);
        if (remote.status === 'done') break;
        if (remote.status === 'preparing') {
          set('generating', 52, `Prima modifica su questa GPU: download di FLUX Kontext ${Math.round((remote.download || 0) * 100)}%…`);
          continue;
        }
        set('generating', 55 + Math.round(((remote.done || 0) / (remote.total || count)) * 35),
          `${source ? 'Modifica con FLUX Kontext' : 'Generazione con FLUX'}: immagine ${Math.min((remote.done || 0) + 1, remote.total || count)} di ${remote.total || count}…`);
      }

      set('downloading', 92, 'Download delle immagini…');
      const folder = join(outputsDir, job.id);
      await mkdir(folder, { recursive: true });
      const names = [...Array.from({ length: remote.total }, (_, i) => `img-${i + 1}.png`), 'immagini.zip'];
      for (const name of names) {
        const response = await worker(`/images/${submittedData.id}/files/${name}`);
        if (!response.ok) throw new Error(`Download di ${name} non riuscito.`);
        await pipeline(Readable.fromWeb(response.body), createWriteStream(join(folder, name)));
      }
      job.cost = costSince(current, costFrom);
      Object.assign(job, { images: names.filter((name) => name.endsWith('.png')), seeds: remote.seeds, width: remote.width, height: remote.height });
      // Saved next to the files so the history survives app restarts.
      await writeFile(join(folder, 'meta.json'), JSON.stringify({
        id: job.id, kind: 'images', createdAt: new Date(job.startedAt).toISOString(), prompt, style, format,
        width: remote.width, height: remote.height, steps, seeds: remote.seeds, images: job.images, gpu: job.gpu, cost: job.cost, adult: Boolean(adult),
        ...(source ? { editedFrom: { id: source.id, name: source.name } } : {})
      }, null, 2));
      success = true;
    } finally {
      if (current && session === current) {
        if (success && keepAlive) {
          keepWarm(current);
          job.keptUntil = current.keepUntil;
        } else {
          set(success ? 'destroying' : job.status, success ? 97 : job.percent, 'Distruzione della GPU e del suo disco…');
          job.destroyed = await endSession();
        }
        job.cost = costSince(current, costFrom);
      } else if (current) {
        job.destroyed = lastDestroy?.instanceId === current.instanceId ? lastDestroy.destroyed : true;
      }
    }
    const summary = `Fatto: ${job.images.length} ${job.images.length === 1 ? 'immagine' : 'immagini'}, costo stimato $${job.cost}.`;
    set('done', 100, job.keptUntil
      ? `${summary} GPU ancora accesa per 5 minuti: la prossima immagine parte subito.`
      : job.destroyed ? `${summary} GPU distrutta.` : `${summary} Controlla su cloud.vast.ai che l’istanza sia distrutta.`);
  }

  function instanceStatus() {
    if (!session) return { active: false, lastDestroy };
    return {
      active: true, instanceId: session.instanceId, gpu: session.gpu, rate: session.rate,
      ready: Boolean(session.address), busy: Boolean(running), keepUntil: session.keepUntil,
      cost: costSince(session, session.rentedAt)
    };
  }

  async function destroyInstance() {
    running?.cancel?.();
    return endSession();
  }

  return {
    outputsDir, cleanupOrphans, startJob, startTraining, startImageJob, instanceStatus, destroyInstance,
    getJob: (id) => publicJob(jobs.get(id)),
    cancelJob: (id) => { jobs.get(id)?.cancel?.(); return Boolean(jobs.get(id)); },
    activeInstanceIds: readActive,
    destroyAll: async () => { await destroyInstance(); for (const id of await readActive()) await destroy(id); }
  };
}

// Rough end-to-end time per offer, used only to rank them: pulling and unpacking the
// image (plus any startup download) dominates short jobs, GPU speed (dlperf) long ones.
const EXCLUDED_GPUS = /^CMP /; // Mining cards: high dlperf score, crippled bf16.

export function rankOffers(offers, characters, profile = { imageMB: 10_500, extraMB: 0, loadMB: 4_500, downloadGB: 10 }) {
  const audioSeconds = Math.max(10, characters / 15);
  return offers
    .filter((offer) => !EXCLUDED_GPUS.test(offer.gpu_name || '') && offer.inet_down > 0 && offer.dlperf > 0)
    .map((offer) => {
      const diskMBs = Math.max(200, offer.disk_bw || 200);
      const pull = (profile.imageMB * 8) / offer.inet_down + (profile.imageMB * 2.2) / diskMBs + 45;
      // Hugging Face rarely serves faster than ~5 Gbit/s, whatever the host's link.
      const fetchExtra = profile.extraMB ? (profile.extraMB * 8) / Math.min(offer.inet_down, 5000) : 0;
      const load = profile.loadMB / diskMBs + 25;
      // About 0.5 s of compute per second of audio on a dlperf-100 card.
      const generate = audioSeconds * 0.5 * (100 / offer.dlperf);
      const downloadCost = (profile.downloadGB || 0) * Math.max(0, offer.inet_down_cost || 0);
      return { ...offer, downloadCost, estimatedSeconds: Math.round(pull + fetchExtra + load + generate) };
    })
    // Faster first; one minute of waiting is worth about $1/h of price difference,
    // and each dollar of download fee about 20 minutes, so traffic-billing hosts sink.
    .sort((a, b) => (a.estimatedSeconds + a.dph_total * 60 + a.downloadCost * 1200) - (b.estimatedSeconds + b.dph_total * 60 + b.downloadCost * 1200));
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason);
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => { clearTimeout(timer); reject(signal.reason); }, { once: true });
  });
}
