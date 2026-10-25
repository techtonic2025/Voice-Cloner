import { createServer } from 'node:http';
import { mkdir, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { extname, join, normalize, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';
import { createCloud } from './cloud.mjs';

const PORT = Number(process.env.PORT || 4173);
const ROOT = dirname(fileURLToPath(import.meta.url));
const PUBLIC = join(ROOT, 'public');
const VOICES = join(ROOT, 'data', 'voices');
const MAX_VOICES = 3;
const MAX_AUDIO_BYTES = 75 * 1024 * 1024;
const MAX_TEXT_CHARS = 60_000;
const cloud = createCloud({ root: ROOT, voicesDir: VOICES });

const mime = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml', '.json': 'application/json; charset=utf-8',
  '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.m4a': 'audio/mp4', '.ogg': 'audio/ogg'
};
const extensions = { 'audio/mpeg': '.mp3', 'audio/wav': '.wav', 'audio/x-wav': '.wav', 'audio/mp4': '.m4a', 'audio/ogg': '.ogg' };

function send(res, status, payload) {
  res.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' });
  res.end(JSON.stringify(payload));
}

async function body(req) {
  let value = '';
  for await (const chunk of req) {
    value += chunk;
    if (value.length > 400_000) throw new Error('Richiesta troppo grande.');
  }
  return value ? JSON.parse(value) : {};
}

async function binaryBody(req) {
  const chunks = [];
  let total = 0;
  for await (const chunk of req) {
    total += chunk.length;
    if (total > MAX_AUDIO_BYTES) throw new Error('Il file supera 75 MB.');
    chunks.push(chunk);
  }
  if (!total) throw new Error('File audio vuoto.');
  return Buffer.concat(chunks);
}

function safeName(value) {
  return String(value || '').trim().replace(/\s+/g, ' ').slice(0, 60);
}

function safeId(value) {
  return /^[a-f0-9-]{36}$/i.test(value || '') ? value : null;
}

async function listVoices() {
  await mkdir(VOICES, { recursive: true });
  const entries = await readdir(VOICES, { withFileTypes: true });
  const voices = [];
  for (const entry of entries) {
    if (!entry.isDirectory() || !safeId(entry.name)) continue;
    try {
      const meta = JSON.parse(await readFile(join(VOICES, entry.name, 'profile.json'), 'utf8'));
      voices.push({ id: entry.name, name: meta.name, createdAt: meta.createdAt, size: meta.size, prepared: Boolean(meta.referenceText) });
    } catch { /* Ignore incomplete profiles. */ }
  }
  return voices.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

function estimate({ characters = 5000, hourlyRate = 0.20, minutes = 15, safetyMinutes = 10 }) {
  const chars = Math.max(1, Number(characters) || 1);
  const rate = Math.max(0.01, Number(hourlyRate) || 0.20);
  const totalMinutes = Math.max(1, Number(minutes) || 15) + Math.max(0, Number(safetyMinutes) || 0);
  return {
    characters: chars, assumedAudioMinutes: +(chars / 900).toFixed(1), totalMinutes,
    estimatedGpuCost: +((rate * totalMinutes) / 60).toFixed(3),
    recommendedBudget: +((rate * totalMinutes) / 60 * 1.35).toFixed(2)
  };
}

// Pronunciation dictionary: Qwen reads every word with Italian rules, so foreign
// words are respelled the way they sound ("file" -> "fàil"). The worker applies it,
// so it can still check each take against the original spelling.
// One "parola = pronuncia" per line, editable from the page; lines starting with # are comments.
const PRONUNCIATIONS = join(ROOT, 'data', 'pronunce.txt');
const DEFAULT_PRONUNCIATIONS = `# Una riga per parola: parola = come va pronunciata (vale solo per i testi in italiano).
# Le maiuscole non contano. Gli accenti indicano dove cade l'accento.
file = fàil
like = làik
download = daunlòd
upload = aplòd
online = onlàin
offline = offlàin
browser = brauser
computer = compiùter
software = sòftuer
hardware = hàrduer
mouse = maus
email = imèil
e-mail = imèil
team = tìm
feedback = fìdbek
smartphone = smàrtfon
YouTube = iutùb
workflow = uòrkflou
password = pàssuord
backup = bècap
update = apdèit
setup = setàp
display = displèi
privacy = pràivasi
weekend = uìkend
streaming = strìming
`;

async function readPronunciations() {
  try { return await readFile(PRONUNCIATIONS, 'utf8'); } catch { return DEFAULT_PRONUNCIATIONS; }
}

function parsePronunciations(source) {
  return source.split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith('#') && line.includes('='))
    .map((line) => line.split('=').map((part) => part.trim()))
    .filter(([word, spoken]) => word && spoken)
    // Longest first, so "e-mail" wins over "mail".
    .sort((a, b) => b[0].length - a[0].length);
}

async function staticFile(req, res) {
  const requested = req.url === '/' ? '/index.html' : req.url.split('?')[0];
  const file = normalize(join(PUBLIC, requested));
  if (!file.startsWith(PUBLIC)) return send(res, 403, { error: 'Percorso non valido.' });
  try {
    const contents = await readFile(file);
    res.writeHead(200, { 'content-type': mime[extname(file)] || 'application/octet-stream' });
    res.end(contents);
  } catch { send(res, 404, { error: 'File non trovato.' }); }
}

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://${req.headers.host}`);
    if (req.method === 'GET' && url.pathname === '/api/health') return send(res, 200, { ok: true, provider: 'Vast AI one-shot', persistentStorage: false });
    // The custom header forces a CORS preflight, so other websites cannot trigger this.
    if (req.method === 'POST' && url.pathname === '/api/shutdown' && req.headers['x-clova-takeover'] === '1') {
      send(res, 200, { ok: true });
      setImmediate(shutdown);
      return;
    }
    if (req.method === 'GET' && url.pathname === '/api/voices') return send(res, 200, { voices: await listVoices(), limit: MAX_VOICES });
    if (req.method === 'GET' && url.pathname === '/api/pronunciations') return send(res, 200, { text: await readPronunciations() });
    if (req.method === 'PUT' && url.pathname === '/api/pronunciations') {
      const { text = '' } = await body(req);
      if (String(text).length > 50_000) return send(res, 400, { error: 'Dizionario troppo lungo.' });
      await mkdir(dirname(PRONUNCIATIONS), { recursive: true });
      await writeFile(PRONUNCIATIONS, String(text));
      return send(res, 200, { ok: true, entries: parsePronunciations(String(text)).length });
    }

    if (req.method === 'POST' && url.pathname === '/api/voices') {
      const name = safeName(decodeURIComponent(req.headers['x-voice-name'] || ''));
      const contentType = String(req.headers['content-type'] || '').split(';')[0].toLowerCase();
      if (!name) return send(res, 400, { error: 'Dai un nome alla voce.' });
      if (!extensions[contentType]) return send(res, 400, { error: 'Formato non supportato. Usa MP3, WAV, M4A o OGG.' });
      const voices = await listVoices();
      if (voices.length >= MAX_VOICES) return send(res, 400, { error: `Puoi salvare al massimo ${MAX_VOICES} voci. Elimina una voce prima di aggiungerne un'altra.` });
      const audio = await binaryBody(req);
      const id = randomUUID();
      const folder = join(VOICES, id);
      const filename = `reference${extensions[contentType]}`;
      const profile = { id, name, createdAt: new Date().toISOString(), size: audio.length, filename, contentType };
      await mkdir(folder, { recursive: true });
      await writeFile(join(folder, filename), audio);
      await writeFile(join(folder, 'profile.json'), JSON.stringify(profile, null, 2));
      return send(res, 201, { voice: { id, name, createdAt: profile.createdAt, size: profile.size } });
    }

    const deleteMatch = url.pathname.match(/^\/api\/voices\/([a-f0-9-]{36})$/i);
    if (req.method === 'DELETE' && deleteMatch) {
      await rm(join(VOICES, deleteMatch[1]), { recursive: true, force: true });
      return send(res, 200, { ok: true });
    }

    if (req.method === 'POST' && url.pathname === '/api/estimate') return send(res, 200, estimate(await body(req)));
    if (req.method === 'POST' && url.pathname === '/api/jobs') {
      const input = await body(req);
      if (!input.confirmDestroy) return send(res, 400, { error: 'Per proseguire devi confermare la distruzione automatica dell’istanza.' });
      const selectedVoice = safeId(input.voiceId);
      if (!selectedVoice || !(await listVoices()).some((voice) => voice.id === selectedVoice)) return send(res, 400, { error: 'Scegli una voce salvata.' });
      const text = String(input.text || '').trim();
      if (!text) return send(res, 400, { error: 'Incolla il testo da trasformare in voce.' });
      if (text.length > MAX_TEXT_CHARS) return send(res, 400, { error: `Testo troppo lungo: massimo ${MAX_TEXT_CHARS.toLocaleString('it-IT')} caratteri.` });
      const maxRate = Math.min(5, Math.max(0.05, Number(input.hourlyRate) || 0.8));
      const budgetCap = Math.max(0.10, Number(input.budgetCap) || 1);
      const cost = estimate({ ...input, characters: text.length, hourlyRate: maxRate });
      if (cost.recommendedBudget > budgetCap) return send(res, 400, { error: `La stima prudente ($${cost.recommendedBudget}) supera il tuo tetto di $${budgetCap}.` });
      const voice = JSON.parse(await readFile(join(VOICES, selectedVoice, 'profile.json'), 'utf8'));
      const language = ['Italian', 'English', 'Spanish', 'French'].includes(input.language) ? input.language : 'Italian';
      const pause = Math.min(2, Math.max(0, Number(input.pause) || 0.35));
      try {
        const job = cloud.startJob({ voice: { ...voice, id: selectedVoice }, text, language, maxRate, budgetCap, pause, keepAlive: input.keepAlive === true,
          stability: ['natural', 'stable', 'very'].includes(input.stability) ? input.stability : 'stable',
          takes: input.quality === 'high' ? 3 : 1,
          pronunciations: language === 'Italian' ? parsePronunciations(await readPronunciations()) : [] });
        return send(res, 202, { id: job.id, cost });
      } catch (error) { return send(res, 409, { error: error.message }); }
    }

    const jobMatch = url.pathname.match(/^\/api\/jobs\/([a-f0-9-]{36})(\/cancel|\/files\/(clova-voice\.wav|clova-voice-blocchi\.zip))?$/i);
    if (jobMatch) {
      const job = cloud.getJob(jobMatch[1]);
      if (!job) return send(res, 404, { error: 'Generazione non trovata.' });
      if (req.method === 'GET' && !jobMatch[2]) return send(res, 200, { job });
      if (req.method === 'POST' && jobMatch[2] === '/cancel') { cloud.cancelJob(job.id); return send(res, 200, { ok: true }); }
      if (req.method === 'GET' && jobMatch[3]) {
        const file = join(cloud.outputsDir, job.id, jobMatch[3]);
        const info = await stat(file).catch(() => null);
        if (!info) return send(res, 404, { error: 'File non disponibile.' });
        res.writeHead(200, {
          'content-type': jobMatch[3].endsWith('.zip') ? 'application/zip' : 'audio/wav', 'content-length': info.size,
          ...(url.searchParams.has('download') ? { 'content-disposition': `attachment; filename="${jobMatch[3]}"` } : {})
        });
        return createReadStream(file).pipe(res);
      }
    }

    if (req.method === 'GET' && url.pathname === '/api/instance') return send(res, 200, cloud.instanceStatus());
    if (req.method === 'POST' && url.pathname === '/api/instance/destroy') return send(res, 200, { destroyed: await cloud.destroyInstance() });
    if (req.method === 'GET' && url.pathname === '/api/instances') return send(res, 200, { active: await cloud.activeInstanceIds() });
    if (req.method === 'POST' && url.pathname === '/api/cleanup') return send(res, 200, { destroyed: await cloud.cleanupOrphans() });
    return staticFile(req, res);
  } catch (error) { return send(res, 500, { error: error.message || 'Errore inatteso.' }); }
});

// Reopening START.bat while an older copy still runs (e.g. a minimized window)
// would otherwise keep serving stale code: ask it to shut down, then take over.
let tookOver = false;
server.on('error', async (error) => {
  if (error.code !== 'EADDRINUSE') throw error;
  if (tookOver) { console.log(`La porta ${PORT} e' ancora occupata. Chiudi la vecchia finestra di Clova Voice e riapri START.bat.`); process.exitCode = 1; return; }
  tookOver = true;
  console.log('Trovata una copia di Clova Voice gia\' aperta: la chiudo (distruggendo eventuali GPU) e la sostituisco…');
  try {
    const response = await fetch(`http://127.0.0.1:${PORT}/api/shutdown`, { method: 'POST', headers: { 'x-clova-takeover': '1' }, signal: AbortSignal.timeout(5000) });
    if (!response.ok) throw new Error(`risposta ${response.status}`);
  } catch (reason) {
    console.log(`La vecchia copia non risponde alla richiesta di chiusura (${reason.message}). Chiudi la sua finestra e riapri START.bat.`);
    process.exitCode = 1;
    return;
  }
  // Destroying a GPU can take a minute: wait for the port to be released.
  for (let attempt = 0; attempt < 90; attempt++) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    const free = await fetch(`http://127.0.0.1:${PORT}/api/health`, { signal: AbortSignal.timeout(1000) }).then(() => false, () => true);
    if (free) return server.listen(PORT, '127.0.0.1');
  }
  console.log('La vecchia copia non si e\' chiusa in tempo. Chiudi la sua finestra e riapri START.bat.');
  process.exitCode = 1;
});
server.on('listening', () => {
  console.log(`Clova Voice: http://127.0.0.1:${PORT}`);
  cloud.cleanupOrphans()
    .then((ids) => { if (ids.length) console.log(`Distrutte istanze Vast rimaste attive: ${ids.join(', ')}`); })
    .catch((error) => console.warn(`Pulizia istanze Vast saltata: ${error.message}`));
  if (process.argv.includes('--open')) spawn('cmd.exe', ['/c', 'start', '', `http://127.0.0.1:${PORT}`], { detached: true, stdio: 'ignore' }).unref();
});
server.listen(PORT, '127.0.0.1');

// Closing the window or Ctrl+C must not leave a GPU billing.
let shuttingDown = false;
async function shutdown() {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log('Chiusura: distruggo eventuali GPU attive…');
  await cloud.destroyAll().catch((error) => console.error(error.message));
  process.exit(0);
}
for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP', 'SIGBREAK']) process.on(signal, shutdown);
