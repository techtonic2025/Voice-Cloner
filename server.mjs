import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const PORT = Number(process.env.PORT || 4173);
const ROOT = dirname(fileURLToPath(import.meta.url));
const PUBLIC = join(ROOT, 'public');

const mime = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml'
};

function send(res, status, payload) {
  res.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' });
  res.end(JSON.stringify(payload));
}

async function body(req) {
  let value = '';
  for await (const chunk of req) value += chunk;
  if (value.length > 100_000) throw new Error('Richiesta troppo grande.');
  return value ? JSON.parse(value) : {};
}

function estimate({ characters = 5000, hourlyRate = 0.20, minutes = 15, safetyMinutes = 10 }) {
  const chars = Math.max(1, Number(characters) || 1);
  const rate = Math.max(0.01, Number(hourlyRate) || 0.20);
  const runMinutes = Math.max(1, Number(minutes) || 15);
  const margin = Math.max(0, Number(safetyMinutes) || 0);
  const totalMinutes = runMinutes + margin;
  return {
    characters: chars,
    assumedAudioMinutes: +(chars / 900).toFixed(1),
    totalMinutes,
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
  } catch {
    send(res, 404, { error: 'File non trovato.' });
  }
}

const server = createServer(async (req, res) => {
  try {
    if (req.method === 'GET' && req.url === '/api/health') return send(res, 200, { ok: true, provider: 'Vast AI one-shot', persistentStorage: false });

    if (req.method === 'POST' && req.url === '/api/estimate') {
      return send(res, 200, estimate(await body(req)));
    }

    if (req.method === 'POST' && req.url === '/api/job-plan') {
      const input = await body(req);
      if (!input.confirmDestroy) return send(res, 400, { error: 'Per proseguire devi confermare la distruzione automatica dell’istanza.' });
      if (!input.referenceReady) return send(res, 400, { error: 'Carica o registra prima un campione vocale pulito.' });
      if (!input.referenceTranscriptReady) return send(res, 400, { error: 'Inserisci la trascrizione esatta del campione: Qwen la usa per clonare la voce.' });
      const cost = estimate(input);
      const cap = Math.max(0.10, Number(input.budgetCap) || 1);
      if (cost.recommendedBudget > cap) return send(res, 400, { error: `La stima prudente ($${cost.recommendedBudget}) supera il tuo tetto di $${cap}.` });
      return send(res, 200, {
        ok: true,
        cost,
        plan: [
          'Cerca una GPU RTX 3090/4090 con almeno 16 GB VRAM e tariffa entro il tetto scelto.',
          'Crea un’istanza effimera senza volume persistente.',
          'Carica campione e testo con URL firmati o tunnel temporaneo.',
          'Avvia Qwen3-TTS Base 1.7B già incluso nel worker, genera i blocchi e scarica il WAV.',
          'Esegui DELETE dell’istanza anche in caso di errore o timeout.'
        ],
        warning: 'Questo prototipo non invia ancora file al cloud: nessuna GPU viene creata e nessun costo viene addebitato.'
      });
    }

    return staticFile(req, res);
  } catch (error) {
    return send(res, 500, { error: error.message || 'Errore inatteso.' });
  }
});

server.on('error', (error) => {
  if (error.code === 'EADDRINUSE') {
    console.log(`Clova Voice e' gia' aperta su http://127.0.0.1:${PORT}`);
    console.log('Puoi chiudere questa finestra.');
    process.exitCode = 0;
    return;
  }
  throw error;
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`Clova Voice: http://127.0.0.1:${PORT}`);
  if (process.argv.includes('--open')) {
    spawn('cmd.exe', ['/c', 'start', '', `http://127.0.0.1:${PORT}`], {
      detached: true,
      stdio: 'ignore'
    }).unref();
  }
});
