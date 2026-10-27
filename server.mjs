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
const BASE_PRONUNCIATIONS = join(ROOT, 'pronunce-base.txt');

async function readPronunciations() {
  try { return await readFile(PRONUNCIATIONS, 'utf8'); } catch { return readFile(BASE_PRONUNCIATIONS, 'utf8'); }
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

// Acronyms written in capitals ("AI", "GPU") only match capitals, so the Italian "ai" stays as it is.
// The worker matches case-insensitively, so these are applied here and the rest is sent along.
const isAcronym = (word) => /\p{Lu}/u.test(word) && word === word.toUpperCase();
const escapeRegExp = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const wordPattern = (word, flags) => new RegExp(`(?<![\\p{L}\\p{N}])${escapeRegExp(word)}(?![\\p{L}\\p{N}])`, flags);

function applyAcronyms(text, entries) {
  return entries.filter(([word]) => isAcronym(word))
    .reduce((result, [word, spoken]) => result.replace(wordPattern(word, 'gu'), spoken), text);
}

// Numbers in words, since Qwen reads "14,7k" as "quattordici sette kappa" and slurs long figures.
const UNITS = ['zero', 'uno', 'due', 'tre', 'quattro', 'cinque', 'sei', 'sette', 'otto', 'nove', 'dieci', 'undici', 'dodici', 'tredici',
  'quattordici', 'quindici', 'sedici', 'diciassette', 'diciotto', 'diciannove'];
const TENS = ['', '', 'venti', 'trenta', 'quaranta', 'cinquanta', 'sessanta', 'settanta', 'ottanta', 'novanta'];

function belowThousand(n) {
  const hundreds = Math.floor(n / 100), rest = n % 100;
  let tail = '';
  if (rest < 20) tail = rest ? UNITS[rest] : '';
  else {
    const unit = rest % 10;
    // "ventuno", "ventotto": the tens drop their vowel before uno and otto; "ventitré" takes the accent.
    tail = (unit === 1 || unit === 8 ? TENS[Math.floor(rest / 10)].slice(0, -1) : TENS[Math.floor(rest / 10)])
      + (unit === 3 ? 'tré' : unit ? UNITS[unit] : '');
  }
  if (!hundreds) return tail;
  const head = hundreds === 1 ? 'cento' : `${UNITS[hundreds]}cento`;
  return /^ott/.test(tail) ? head.slice(0, -1) + tail : head + tail;
}

function numberToWords(n) {
  if (n === 0) return 'zero';
  if (n >= 1e12) return String(n);
  const parts = [];
  for (const [size, one, many] of [[1e9, 'un miliardo', 'miliardi'], [1e6, 'un milione', 'milioni']]) {
    const count = Math.floor(n / size);
    if (count) parts.push(count === 1 ? one : `${numberToWords(count)} ${many}`);
    n %= size;
  }
  const thousands = Math.floor(n / 1000), rest = n % 1000;
  const small = (thousands ? (thousands === 1 ? 'mille' : `${belowThousand(thousands)}mila`) : '') + belowThousand(rest);
  if (small) parts.push(small);
  return parts.join(parts.length > 1 && !/^(mille|\w+mila)?$/.test(small) ? ' e ' : ' ');
}

// "3,25" -> "tre virgola venticinque"; leading zeros after the comma are read one by one.
function decimalToWords(whole, fraction) {
  const zeros = fraction.match(/^0*/)[0].length;
  const digits = fraction.slice(zeros);
  return `${numberToWords(Number(whole))} virgola ${'zero '.repeat(zeros)}${digits ? numberToWords(Number(digits)) : ''}`.trim();
}

function numbersToWords(text) {
  const multipliers = { k: 1e3, K: 1e3, mila: 1e3, M: 1e6, mln: 1e6, mld: 1e9 };
  const amount = String.raw`\d[\d.,]*(?:\s?(?:k|K|M|mln|mld)(?![\p{L}\d]))?`;
  return text
    // "$1,05" and "1,05 $" -> "1,05 dollari"; then the figure is read like any other.
    .replace(new RegExp(String.raw`\$\s?(${amount})`, 'gu'), '$1 dollari').replace(new RegExp(String.raw`€\s?(${amount})`, 'gu'), '$1 euro')
    .replace(/(\d)\s?\$/g, '$1 dollari').replace(/(\d)\s?€/g, '$1 euro')
    // "09:57" -> "nove e cinquantasette", "18:00" -> "diciotto".
    .replace(/(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])/g, (_, h, m) => (m === '00' ? numberToWords(+h) : `${numberToWords(+h)} e ${numberToWords(+m)}`))
    // "RTX 4090", "GTX 1080": model numbers are read in pairs, "quaranta novanta".
    .replace(/(\b\p{Lu}{2,}\s)(\d{2})(\d{2})(?!\d|[.,]\d)/gu, (_, name, a, b) => `${name}${numberToWords(+a)} ${b === '00' ? 'cento' : numberToWords(+b)}`)
    // "14,7k", "1,5M", "3 mln": the amount is worked out and read in full.
    .replace(/(?<![\p{L}\d]|\d[.,])(\d+(?:[.,]\d+)?)\s?(k|K|M|mln|mld)(?![\p{L}\d])/gu, (_, value, unit) => {
      const amount = Math.round(Number(value.replace(',', '.')) * multipliers[unit]);
      return numberToWords(amount);
    })
    // "14.700" and "1.250.000": dots as thousands separators.
    .replace(/(?<![\p{L}\d]|\d[.,])\d{1,3}(?:\.\d{3})+(?!\d|[.,]\d)/gu, (match) => numberToWords(Number(match.replace(/\./g, ''))))
    .replace(/(?<![\p{L}\d]|\d[.,])(\d+)[.,](\d+)(?![\p{L}\d]|[.,]\d)/gu, (_, whole, fraction) => decimalToWords(whole, fraction))
    .replace(/(?<![\p{L}\d]|\d[.,])\d+(?![\p{L}\d]|[.,]\d)/gu, (match) => (match.length > 1 && match.startsWith('0') ? [...match].map((d) => UNITS[+d]).join(' ') : numberToWords(Number(match))))
    .replace(/\s?%/g, ' per cento');
}

// Words a dictionary entry already covers are blanked out, then what is left is checked
// for things the Italian rules read wrong: acronyms, English spellings, final consonants.
const ITALIAN_LETTERS = { A: 'a', B: 'bi', C: 'ci', D: 'di', E: 'e', F: 'effe', G: 'gi', H: 'acca', I: 'i', J: 'i lunga', K: 'cappa', L: 'elle', M: 'emme',
  N: 'enne', O: 'o', P: 'pi', Q: 'cu', R: 'erre', S: 'esse', T: 'ti', U: 'u', V: 'vu', W: 'doppia vu', X: 'ics', Y: 'ipsilon', Z: 'zeta' };
const ITALIAN_ENDINGS_OK = new Set(['quel', 'buon', 'gran', 'esser', 'aver', 'signor', 'dottor', 'professor', 'nessun', 'alcun', 'ciascun', 'qual',
  'tal', 'cuor', 'amor', 'bus', 'autobus', 'camion', 'virus', 'bonus', 'film', 'sport', 'tram', 'gas', 'bar', 'super', 'extra', 'taxi', 'test', 'top', 'stop']);

function suggestSpelling(word) {
  if (isAcronym(word) && /^[A-Z0-9]{2,6}$/.test(word)) return [...word].map((c) => ITALIAN_LETTERS[c] || c).join(' ');
  let s = word.toLowerCase();
  if (/[^aeiou]e$/.test(s) && s.length > 3 && /[aeiouy]/.test(s.slice(0, -1))) s = s.slice(0, -1);
  return s
    .replace(/tion/g, 'scion').replace(/igh/g, 'ai').replace(/ph/g, 'f').replace(/th/g, 't').replace(/ck/g, 'k')
    .replace(/c(?=[ei])/g, 's').replace(/sh(?=[ei])/g, 'sc').replace(/sh/g, 'sci').replace(/ch(?=[ei])/g, 'c').replace(/ch/g, 'ci')
    .replace(/ee|ea/g, 'i').replace(/oo/g, 'u').replace(/ou|ow$/g, 'au').replace(/ay|ey$/g, 'ei')
    .replace(/j(?=[ei])/g, 'g').replace(/j/g, 'gi')
    .replace(/k(?=[ei])/g, 'ch').replace(/k/g, 'c').replace(/w/g, 'u').replace(/y$/g, 'i').replace(/y/g, 'ai');
}

function looksForeign(word) {
  if (/[àèéìòóù]/i.test(word) || /^\d+$/.test(word)) return false;
  if (isAcronym(word)) return /^[A-Z0-9]{2,6}$/.test(word) && /[A-Z]/.test(word) && word.length > 1;
  const lower = word.toLowerCase();
  if (/[kwyjx]|sh|th|ee|ck|ght|ay|ey/.test(lower)) return true;
  return lower.length >= 4 && /[^aeiou]$/.test(lower) && !ITALIAN_ENDINGS_OK.has(lower);
}

function unknownWords(text, entries) {
  let rest = text;
  for (const [word] of entries) rest = rest.replace(wordPattern(word, isAcronym(word) ? 'gu' : 'giu'), ' ');
  const counts = new Map();
  for (const word of rest.match(/[\p{L}\p{N}]+(?:[-'’][\p{L}\p{N}]+)*/gu) || []) {
    const clean = word.replace(/^(?:l|d|un|all|dell|nell|sull|dall|quell)['’]/i, '');
    if (!looksForeign(clean)) continue;
    const key = isAcronym(clean) ? clean : clean.toLowerCase();
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return [...counts].map(([word, count]) => ({ word, count, suggestion: suggestSpelling(word) }))
    .sort((a, b) => b.count - a.count || a.word.localeCompare(b.word));
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

// Image tab settings: the Vast key lives in api.txt (as before), the credit and the
// FLUX license checks next to it in data/settings.json.
const SETTINGS = join(ROOT, 'data', 'settings.json');
const FLUX_MODELS = { flux: 'black-forest-labs/FLUX.1-dev', kontext: 'black-forest-labs/FLUX.1-Kontext-dev' };

async function readSettings() {
  try { return JSON.parse((await readFile(SETTINGS, 'utf8')).replace(/^\uFEFF/, '')); } catch { return {}; }
}

async function vastKeySaved() {
  if (process.env.VAST_API_KEY) return true;
  try { return Boolean((await readFile(join(ROOT, 'api.txt'), 'utf8')).trim()); } catch { return false; }
}

async function checkVastKey(key) {
  const response = await fetch('https://console.vast.ai/api/v0/users/current/', { headers: { authorization: `Bearer ${key}` }, signal: AbortSignal.timeout(15_000) });
  if (!response.ok) throw new Error('Chiave Vast non valida.');
  const data = await response.json();
  return Number(data.credit ?? data.balance ?? 0);
}

// A gated file answers 200 (or a redirect) only once the license is accepted.
async function checkLicenses(token) {
  const licenses = {};
  for (const [name, repo] of Object.entries(FLUX_MODELS)) {
    const file = await fetch(`https://huggingface.co/${repo}/resolve/main/model_index.json`, { headers: { authorization: `Bearer ${token}` }, redirect: 'manual', signal: AbortSignal.timeout(15_000) }).catch(() => null);
    licenses[name] = Boolean(file && file.status < 400);
  }
  return licenses;
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

    if (req.method === 'GET' && url.pathname === '/api/settings') {
      const hf = await readHf();
      return send(res, 200, {
        vast: await vastKeySaved(), credit: (await readSettings()).credit ?? null,
        hf: Boolean(hf.token), hfUser: hf.user || null, canWrite: Boolean(hf.canWrite), licenses: hf.licenses || {}
      });
    }
    if (req.method === 'PUT' && url.pathname === '/api/settings/vast') {
      const key = String((await body(req)).key || '').trim();
      if (!/^[A-Za-z0-9]{32,}$/.test(key)) return send(res, 400, { error: 'La chiave Vast è una lunga sequenza di lettere e numeri: copiala da cloud.vast.ai → Account → API Keys.' });
      let credit;
      try { credit = await checkVastKey(key); } catch (error) { return send(res, 400, { error: error.message }); }
      await writeFile(join(ROOT, 'api.txt'), key);
      await mkdir(dirname(SETTINGS), { recursive: true });
      await writeFile(SETTINGS, JSON.stringify({ ...(await readSettings()), credit }, null, 2));
      return send(res, 200, { vast: true, credit });
    }
    if (req.method === 'PUT' && url.pathname === '/api/settings/hf') {
      // Without a token: re-check the licenses of the saved one (after accepting them on the site).
      const token = String((await body(req)).token || '').trim() || (await readHf()).token || '';
      if (!/^hf_[A-Za-z0-9]{20,}$/.test(token)) return send(res, 400, { error: 'Il token Hugging Face inizia con “hf_”.' });
      let info;
      try { info = { ...(await checkHfToken(token)), licenses: await checkLicenses(token) }; } catch (error) { return send(res, 400, { error: error.message }); }
      await mkdir(dirname(HF_SETTINGS), { recursive: true });
      await writeFile(HF_SETTINGS, JSON.stringify({ token, ...info }, null, 2));
      return send(res, 200, { hf: true, hfUser: info.user, licenses: info.licenses });
    }

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
    if (req.method === 'POST' && url.pathname === '/api/pronunciations/check') {
      const { text = '' } = await body(req);
      return send(res, 200, { words: unknownWords(String(text).slice(0, 200_000), parsePronunciations(await readPronunciations())) });
    }
    // Adds the base list's words that are missing from the user's dictionary, keeping every edit.
    if (req.method === 'POST' && url.pathname === '/api/pronunciations/merge-base') {
      const current = await readPronunciations();
      const known = new Set(parsePronunciations(current).map(([word]) => (isAcronym(word) ? word : word.toLowerCase())));
      const missing = parsePronunciations(await readFile(BASE_PRONUNCIATIONS, 'utf8'))
        .filter(([word]) => !known.has(isAcronym(word) ? word : word.toLowerCase()))
        .map(([word, spoken]) => `${word} = ${spoken}`).sort((a, b) => a.localeCompare(b));
      const merged = missing.length ? `${current.trimEnd()}\n\n# Aggiunte dalla lista base\n${missing.join('\n')}\n` : current;
      return send(res, 200, { text: merged, added: missing.length });
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
        const pronunciations = language === 'Italian' ? parsePronunciations(await readPronunciations()) : [];
        const job = cloud.startJob({ voice: { ...voice, id: selectedVoice }, text: applyAcronyms(language === 'Italian' ? numbersToWords(text) : text, pronunciations), displayText: text, language, maxRate, budgetCap, pause, keepAlive: input.keepAlive === true,
          stability: ['natural', 'stable', 'very'].includes(input.stability) ? input.stability : 'stable',
          takes: ['high', 'max'].includes(input.quality) ? 3 : 1, retry: input.quality === 'max',
          speed: Math.min(1.3, Math.max(0.7, Number(input.speed) || 1)),
          refSeconds: { short: 20, medium: 40 }[input.refLength] || 120,
          // Fine-tuned voices learned from 3-12 s clips and speak best one or two sentences at a time.
          chunkChars: Math.min(600, Math.max(80, Number(input.chunkChars) || (voice.type === 'trained' ? 150 : 400))),
          pronunciations: pronunciations.filter(([word]) => !isAcronym(word)) });
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
