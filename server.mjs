import { createServer } from 'node:http';
import { mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { extname, join, normalize, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';

const PORT = Number(process.env.PORT || 4173);
const ROOT = dirname(fileURLToPath(import.meta.url));
const PUBLIC = join(ROOT, 'public');
const VOICES = join(ROOT, 'data', 'voices');
const MAX_VOICES = 3;
const MAX_AUDIO_BYTES = 75 * 1024 * 1024;

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
    if (value.length > 100_000) throw new Error('Richiesta troppo grande.');
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
      voices.push({ id: entry.name, name: meta.name, createdAt: meta.createdAt, size: meta.size });
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
    if (req.method === 'GET' && url.pathname === '/api/voices') return send(res, 200, { voices: await listVoices(), limit: MAX_VOICES });

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
    if (req.method === 'POST' && url.pathname === '/api/job-plan') {
      const input = await body(req);
      if (!input.confirmDestroy) return send(res, 400, { error: 'Per proseguire devi confermare la distruzione automatica dell’istanza.' });
      const selectedVoice = safeId(input.voiceId);
      const voices = await listVoices();
      if (!selectedVoice || !voices.some((voice) => voice.id === selectedVoice)) return send(res, 400, { error: 'Scegli una voce salvata.' });
      if (!String(input.text || '').trim()) return send(res, 400, { error: 'Incolla il testo da trasformare in voce.' });
      const cost = estimate(input);
      const cap = Math.max(0.10, Number(input.budgetCap) || 1);
      if (cost.recommendedBudget > cap) return send(res, 400, { error: `La stima prudente ($${cost.recommendedBudget}) supera il tuo tetto di $${cap}.` });
      return send(res, 200, { ok: true, cost, plan: [
        'Usa il profilo vocale scelto, salvato solo su questo PC.',
        'Crea una GPU temporanea senza volume persistente.',
        'Invia automaticamente il riferimento al worker Qwen, genera e scarica il WAV.',
        'Distruggi istanza e disco al termine, anche in caso di errore.'
      ], warning: 'La generazione cloud automatica verrà collegata nel prossimo modulo: questo controllo non crea costi.' });
    }
    return staticFile(req, res);
  } catch (error) { return send(res, 500, { error: error.message || 'Errore inatteso.' }); }
});

server.on('error', (error) => {
  if (error.code === 'EADDRINUSE') { console.log(`Clova Voice e' gia' aperta su http://127.0.0.1:${PORT}`); process.exitCode = 0; return; }
  throw error;
});
server.listen(PORT, '127.0.0.1', () => {
  console.log(`Clova Voice: http://127.0.0.1:${PORT}`);
  if (process.argv.includes('--open')) spawn('cmd.exe', ['/c', 'start', '', `http://127.0.0.1:${PORT}`], { detached: true, stdio: 'ignore' }).unref();
});
