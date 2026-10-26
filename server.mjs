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
const MAX_VOICES = 5;
const MAX_AUDIO_BYTES = 75 * 1024 * 1024;
const MAX_TEXT_CHARS = 60_000;
const HF_SETTINGS = join(ROOT, 'data', 'hf.json');
const TRAINING = join(ROOT, 'data', 'training');
const cloud = createCloud({ root: ROOT, voicesDir: VOICES, getHfToken: async () => (await readHf()).token || '' });

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
      voices.push({ id: entry.name, name: meta.name, createdAt: meta.createdAt, size: meta.size, type: meta.type || 'clone', prepared: Boolean(meta.referenceText) });
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
face = fèis
Hugging Face = àgghin fèis
Facebook = fèisbuk
interface = ìnterfeis
feature = fìciur
cloud = clàud
tool = tùl
free = frì
store = stòr
home = hòum
design = disàin
game = ghèim
gaming = ghèiming
web = uèb
voice = vòis
image = ìmmig
prompt = pròmpt
open source = òpen sors
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

// Minors in a sexual context are refused before any GPU is rented.
const MINOR_WORDS = /\b(bambin[aeio]|ragazzin[aeio]|minorenn[ei]|scolar[aeio]|neonat[aeio]|child(ren)?|kids?|teen(ager)?s?|underage|loli|preteens?|schoolgirls?|schoolboys?|(1[0-7]|[1-9])\s*(anni|years?\s*old|yo))\b/i;
const SEXUAL_WORDS = /\b(nud[aeio]|nudes?|naked|sex\w*|sess\w*|erotic\w*|porn\w*|lingerie|intimo|seducent[ei]|sensual\w*|topless|nsfw)\b/i;
function forbiddenImagePrompt(prompt) {
  return MINOR_WORDS.test(prompt) && SEXUAL_WORDS.test(prompt);
}

// Image presets: the style is appended to the user's prompt (FLUX follows English best);
// formats are FLUX-friendly sizes around one megapixel, multiples of 16.
const IMAGE_STYLES = {
  photo: ', photorealistic photograph, natural lighting, sharp focus, highly detailed, 35mm',
  illustration: ', detailed digital illustration, clean lines, rich colors, professional artwork',
  free: ''
};
const IMAGE_FORMATS = { wide: [1344, 768], square: [1024, 1024], tall: [768, 1344], portrait: [896, 1120] };

// History: every finished generation lives in outputs/<id>/ (audio, zip, meta.json).
// Folders from before meta.json existed are listed with what can be read from disk.
async function historyEntry(id) {
  const folder = join(cloud.outputsDir, id);
  let meta = {};
  try { meta = JSON.parse(await readFile(join(folder, 'meta.json'), 'utf8')); } catch { /* Older generation. */ }
  if (meta.kind === 'training') {
    return {
      id, kind: 'training', createdAt: meta.createdAt, repo: meta.repo, clips: meta.clips, minutes: meta.minutes,
      epochs: meta.epochs, bestEpoch: meta.bestEpoch, baseline: meta.baseline, epochScores: meta.epochScores || {}, samples: meta.samples || []
    };
  }
  if (meta.kind === 'images') {
    const sizes = await Promise.all([...(meta.images || []), 'immagini.zip'].map((name) => stat(join(folder, name)).then((s) => s.size, () => 0)));
    return {
      id, kind: 'images', createdAt: meta.createdAt, prompt: meta.prompt, style: meta.style, format: meta.format,
      width: meta.width, height: meta.height, seeds: meta.seeds || [], images: meta.images || [], cost: meta.cost ?? null,
      editedFrom: meta.editedFrom || null, bytes: sizes.reduce((sum, size) => sum + size, 0)
    };
  }
  const audio = await stat(join(folder, 'clova-voice.wav')).catch(() => null);
  if (!audio) return null;
  const parts = await stat(join(folder, 'clova-voice-blocchi.zip')).catch(() => null);
  return {
    id, kind: 'voice', createdAt: meta.createdAt || new Date(audio.mtimeMs).toISOString(),
    voiceName: meta.voiceName || null, text: meta.text || null, language: meta.language || null,
    stability: meta.stability || null, takes: meta.takes || null, speed: meta.speed || null, refSeconds: meta.refSeconds || null, chunks: meta.chunks ?? null,
    // 16-bit mono 24 kHz WAV: 48,000 bytes per second after the 44-byte header.
    audioSeconds: meta.audioSeconds ?? +((audio.size - 44) / 48_000).toFixed(1),
    quality: meta.quality ?? null, weakest: meta.weakest || [], cost: meta.cost ?? null,
    bytes: audio.size + (parts?.size || 0), hasParts: Boolean(parts)
  };
}

async function listHistory() {
  const entries = await readdir(cloud.outputsDir, { withFileTypes: true }).catch(() => []);
  const items = await Promise.all(entries.filter((entry) => entry.isDirectory() && safeId(entry.name)).map((entry) => historyEntry(entry.name)));
  return items.filter(Boolean).sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

// Hugging Face token: stored only in data/ (git-ignored), never sent back to the page.
async function readHf() {
  // Tolerates a BOM, e.g. if the file was edited with Notepad.
  try { return JSON.parse((await readFile(HF_SETTINGS, 'utf8')).replace(/^﻿/, '')); } catch { return {}; }
}

async function checkHfToken(token) {
  const response = await fetch('https://huggingface.co/api/whoami-v2', { headers: { authorization: `Bearer ${token}` }, signal: AbortSignal.timeout(15_000) });
  if (!response.ok) throw new Error('Token Hugging Face non valido.');
  const data = await response.json();
  const access = data.auth?.accessToken || {};
  const scoped = (access.fineGrained?.scoped || []).flatMap((entry) => entry.permissions || []);
  // Classic tokens have a role; fine-grained ones list scoped permissions instead.
  const canWrite = ['write', 'admin'].includes(access.role) || scoped.includes('repo.write');
  return { user: data.name, role: access.role || 'unknown', canWrite };
}

async function listTrainingFiles() {
  await mkdir(TRAINING, { recursive: true });
  const entries = await readdir(TRAINING, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    if (!entry.isFile() || !/\.(mp3|wav|m4a|ogg|flac)$/i.test(entry.name)) continue;
    files.push({ name: entry.name, size: (await stat(join(TRAINING, entry.name))).size });
  }
  return files.sort((a, b) => a.name.localeCompare(b.name));
}

function safeFileName(value) {
  const name = String(value || '').replace(/[^\p{L}\p{N} ._-]/gu, '_').trim().slice(0, 120);
  return name && !name.startsWith('.') ? name : null;
}

async function staticFile(req, res) {
  const path = req.url.split('?')[0];
  const requested = path === '/' ? '/index.html' : path;
  const file = normalize(join(PUBLIC, requested));
  if (!file.startsWith(PUBLIC)) return send(res, 403, { error: 'Percorso non valido.' });
  try {
    const contents = await readFile(file);
    // no-cache: after an update the browser must re-check, or it keeps showing the old page.
    res.writeHead(200, { 'content-type': mime[extname(file)] || 'application/octet-stream', 'cache-control': 'no-cache' });
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

    if (req.method === 'GET' && url.pathname === '/api/hf') {
      const hf = await readHf();
      return send(res, 200, { configured: Boolean(hf.token), user: hf.user || null, canWrite: Boolean(hf.canWrite) });
    }
    if (req.method === 'PUT' && url.pathname === '/api/hf') {
      const token = String((await body(req)).token || '').trim();
      if (!/^hf_[A-Za-z0-9]{20,}$/.test(token)) return send(res, 400, { error: 'Il token Hugging Face inizia con “hf_”.' });
      let info;
      try { info = await checkHfToken(token); } catch (error) { return send(res, 400, { error: error.message }); }
      await mkdir(dirname(HF_SETTINGS), { recursive: true });
      await writeFile(HF_SETTINGS, JSON.stringify({ token, ...info }, null, 2));
      return send(res, 200, { configured: true, user: info.user, canWrite: info.canWrite });
    }

    if (req.method === 'GET' && url.pathname === '/api/training/files') return send(res, 200, { files: await listTrainingFiles() });
    if (req.method === 'POST' && url.pathname === '/api/training/files') {
      const name = safeFileName(decodeURIComponent(req.headers['x-file-name'] || ''));
      if (!name || !/\.(mp3|wav|m4a|ogg|flac)$/i.test(name)) return send(res, 400, { error: 'Usa file MP3, WAV, M4A, OGG o FLAC.' });
      const chunks = [];
      let total = 0;
      for await (const chunk of req) {
        total += chunk.length;
        if (total > 500 * 1024 * 1024) return send(res, 400, { error: 'File troppo grande (massimo 500 MB).' });
        chunks.push(chunk);
      }
      await mkdir(TRAINING, { recursive: true });
      await writeFile(join(TRAINING, name), Buffer.concat(chunks));
      return send(res, 201, { files: await listTrainingFiles() });
    }
    const trainingFileMatch = url.pathname.match(/^\/api\/training\/files\/(.+)$/);
    if (req.method === 'DELETE' && trainingFileMatch) {
      const name = safeFileName(decodeURIComponent(trainingFileMatch[1]));
      if (name) await rm(join(TRAINING, name), { force: true });
      return send(res, 200, { files: await listTrainingFiles() });
    }
    if (req.method === 'POST' && url.pathname === '/api/training/start') {
      const input = await body(req);
      const hf = await readHf();
      if (!hf.token) return send(res, 400, { error: 'Prima salva il token Hugging Face.' });
      if (!hf.canWrite) return send(res, 400, { error: 'Il token Hugging Face deve avere il permesso di scrittura (Write).' });
      const files = (await listTrainingFiles()).map((file) => ({ ...file, path: join(TRAINING, file.name) }));
      if (!files.length) return send(res, 400, { error: 'Aggiungi almeno una registrazione della tua voce.' });
      const voiceName = safeName(input.voiceName) || 'Voce addestrata';
      const speaker = voiceName.toLowerCase().normalize('NFD').replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 32) || 'voce';
      const repo = `${hf.user}/clova-voice-${speaker.replace(/_/g, '-')}`;
      // Retraining under the same name overwrites the HF repo: reuse that voice instead of adding one.
      let existingId = null;
      for (const voice of await listVoices()) {
        const profile = JSON.parse(await readFile(join(VOICES, voice.id, 'profile.json'), 'utf8'));
        if (profile.type === 'trained' && profile.repo === repo) existingId = voice.id;
      }
      if (!existingId && (await listVoices()).length >= MAX_VOICES) return send(res, 400, { error: `Hai già ${MAX_VOICES} voci: eliminane una per fare spazio a quella addestrata.` });
      try {
        const job = cloud.startTraining({
          files, repo, speaker, epochs: Math.min(10, Math.max(1, Number(input.epochs) || 5)),
          maxRate: Math.min(5, Math.max(0.2, Number(input.hourlyRate) || 1)),
          onTrained: async (result) => {
            const id = existingId || randomUUID();
            await mkdir(join(VOICES, id), { recursive: true });
            await writeFile(join(VOICES, id, 'profile.json'), JSON.stringify({
              id, name: voiceName, type: 'trained', repo: result.repo, speaker: result.speaker,
              createdAt: new Date().toISOString(), size: 0, clips: result.clips, minutes: result.minutes,
              bestEpoch: result.bestEpoch ?? null, fidelity: result.fidelity ?? null
            }, null, 2));
          }
        });
        return send(res, 202, { id: job.id, repo });
      } catch (error) { return send(res, 409, { error: error.message }); }
    }
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

    // Transcript of the cached reference clip: Qwen pairs it word by word with the
    // audio, so a Whisper mistake here hurts pronunciation on every generation.
    const transcriptMatch = url.pathname.match(/^\/api\/voices\/([a-f0-9-]{36})\/transcript$/i);
    if (transcriptMatch) {
      const profilePath = join(VOICES, transcriptMatch[1], 'profile.json');
      const profile = JSON.parse(await readFile(profilePath, 'utf8').catch(() => 'null'));
      if (!profile) return send(res, 404, { error: 'Voce non trovata.' });
      if (req.method === 'GET') return send(res, 200, { text: profile.referenceText || '', prepared: Boolean(profile.referenceText) });
      if (req.method === 'PUT') {
        const value = String((await body(req)).text || '').replace(/\s+/g, ' ').trim();
        if (!profile.referenceText) return send(res, 400, { error: 'La trascrizione si crea alla prima generazione con questa voce.' });
        if (value.length < 10) return send(res, 400, { error: 'Trascrizione troppo corta.' });
        await writeFile(profilePath, JSON.stringify({ ...profile, referenceText: value }, null, 2));
        return send(res, 200, { ok: true });
      }
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
      const pause = Math.min(2, Math.max(0, Number.isFinite(Number(input.pause)) && input.pause !== '' && input.pause != null ? Number(input.pause) : 0.35));
      try {
        const job = cloud.startJob({ voice: { ...voice, id: selectedVoice }, text, language, maxRate, budgetCap, pause, keepAlive: input.keepAlive === true,
          stability: ['natural', 'stable', 'very'].includes(input.stability) ? input.stability : 'stable',
          takes: input.quality === 'high' ? 3 : 1,
          speed: Math.min(1.3, Math.max(0.7, Number(input.speed) || 1)),
          refSeconds: { short: 20, medium: 40 }[input.refLength] || 120,
          // Fine-tuned voices learned from 3-12 s clips and speak best one or two sentences at a time.
          chunkChars: Math.min(600, Math.max(80, Number(input.chunkChars) || (voice.type === 'trained' ? 150 : 400))),
          pronunciations: language === 'Italian' ? parsePronunciations(await readPronunciations()) : [] });
        return send(res, 202, { id: job.id, cost });
      } catch (error) { return send(res, 409, { error: error.message }); }
    }

    if (req.method === 'POST' && url.pathname === '/api/images') {
      const input = await body(req);
      if (!input.confirmDestroy) return send(res, 400, { error: 'Per proseguire devi confermare la distruzione automatica dell’istanza.' });
      const prompt = String(input.prompt || '').trim().slice(0, 2000);
      if (!prompt) return send(res, 400, { error: 'Descrivi l’immagine da generare.' });
      if (!(await readHf()).token) return send(res, 400, { error: 'Serve il token Hugging Face (riquadro “Addestra la tua voce”) e la licenza di FLUX.1-dev accettata su Hugging Face.' });
      if (forbiddenImagePrompt(String(input.prompt || ''))) return send(res, 400, { error: 'Richiesta rifiutata: niente contenuti sessuali che coinvolgono minori.' });
      const style = IMAGE_STYLES[input.style] ? input.style : 'photo';
      const format = IMAGE_FORMATS[input.format] ? input.format : 'square';
      const [width, height] = IMAGE_FORMATS[format];
      const seed = Number.isInteger(Number(input.seed)) && String(input.seed).trim() !== '' ? Math.max(0, Number(input.seed)) : -1;
      // Edits start only from images this app generated (history), never from uploaded photos.
      let source = null;
      if (input.source) {
        const id = String(input.source.id || '');
        const name = String(input.source.name || '');
        if (!/^[a-f0-9-]{36}$/i.test(id) || !/^img-[1-4]\.png$/.test(name)) return send(res, 400, { error: 'Immagine da modificare non valida.' });
        const entry = await historyEntry(id);
        if (entry?.kind !== 'images' || !entry.images.includes(name)) return send(res, 400, { error: 'L’immagine da modificare non è più nella cronologia.' });
        source = { id, name, path: join(cloud.outputsDir, id, name) };
      }
      // "No censura" and "Horror" turn on separate LoRA groups; the gore LoRA needs its trigger word.
      const loraGroups = [input.uncensored === true && 'nsfw', input.horror === true && 'gore'].filter(Boolean);
      const trigger = loraGroups.includes('gore') && !/ccRealGore/i.test(prompt) ? 'ccRealGore, ' : '';
      try {
        const job = cloud.startImageJob({
          // An edit is an instruction ("make the sky red"): no style suffix appended.
          prompt, fullPrompt: trigger + (source ? prompt : prompt + IMAGE_STYLES[style]), style, format, width, height, source, loraGroups,
          count: Math.min(4, Math.max(1, Number(input.count) || 1)), seed,
          steps: { fast: 20, normal: 28, max: 40 }[input.quality] || 28,
          maxRate: Math.min(5, Math.max(0.2, Number(input.hourlyRate) || 1)),
          budgetCap: Math.max(0.1, Number(input.budgetCap) || 1), keepAlive: input.keepAlive === true
        });
        return send(res, 202, { id: job.id });
      } catch (error) { return send(res, 409, { error: error.message }); }
    }

    const jobMatch = url.pathname.match(/^\/api\/jobs\/([a-f0-9-]{36})(\/cancel|\/files\/(clova-voice\.wav|clova-voice-blocchi\.zip|img-[1-4]\.png|immagini\.zip|(?:vera|clone|epoca-\d{2})-[12]\.wav))?$/i);
    if (jobMatch) {
      // Files are served straight from outputs/, so history entries keep working after a restart.
      if (req.method === 'GET' && jobMatch[3]) {
        const file = join(cloud.outputsDir, jobMatch[1], jobMatch[3]);
        const info = await stat(file).catch(() => null);
        if (!info) return send(res, 404, { error: 'File non disponibile.' });
        const entry = await historyEntry(jobMatch[1]);
        const when = new Date(entry?.createdAt || info.mtimeMs);
        const pad = (value) => String(value).padStart(2, '0');
        const stamp = `${when.getFullYear()}-${pad(when.getMonth() + 1)}-${pad(when.getDate())}_${pad(when.getHours())}-${pad(when.getMinutes())}`;
        const name = jobMatch[3];
        const filename = name === 'clova-voice.wav' ? `clova-${stamp}.wav` : name === 'clova-voice-blocchi.zip' ? `clova-${stamp}-blocchi.zip`
          : name === 'immagini.zip' ? `clova-${stamp}-immagini.zip` : `clova-${stamp}-${name.replace('img-', 'immagine-')}`;
        const type = name.endsWith('.zip') ? 'application/zip' : name.endsWith('.png') ? 'image/png' : 'audio/wav';
        res.writeHead(200, {
          'content-type': type, 'content-length': info.size,
          ...(url.searchParams.has('download') ? { 'content-disposition': `attachment; filename="${filename}"` } : {})
        });
        return createReadStream(file).pipe(res);
      }
      const job = cloud.getJob(jobMatch[1]);
      if (!job) return send(res, 404, { error: 'Generazione non trovata.' });
      if (req.method === 'GET' && !jobMatch[2]) return send(res, 200, { job });
      if (req.method === 'POST' && jobMatch[2] === '/cancel') { cloud.cancelJob(job.id); return send(res, 200, { ok: true }); }
    }

    if (req.method === 'GET' && url.pathname === '/api/history') return send(res, 200, { entries: await listHistory() });
    const historyMatch = url.pathname.match(/^\/api\/history\/([a-f0-9-]{36})$/i);
    if (req.method === 'DELETE' && historyMatch) {
      if (cloud.getJob(historyMatch[1]) && !(await historyEntry(historyMatch[1]))) return send(res, 409, { error: 'Generazione ancora in corso.' });
      await rm(join(cloud.outputsDir, historyMatch[1]), { recursive: true, force: true });
      return send(res, 200, { ok: true });
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
