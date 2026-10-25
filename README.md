# Clova Voice One-Shot

Clonazione vocale personale con **Qwen3-TTS 12Hz 1.7B Base** e GPU temporanea.

## Come funziona

1. Avvia `START.bat` e apri l'app (http://127.0.0.1:4587).
2. In "Le mie voci" carica fino a tre campioni (30–120 s vanno bene).
3. Scegli la voce, incolla il testo (fino a 60.000 caratteri) e premi "Genera".
4. L'app noleggia la GPU Vast più economica adatta, cioè con almeno 16 GB, architettura
   Ampere/Ada e rete veloce. Poi avvia il worker, genera, scarica i file e
   **distrugge l'istanza**, anche in caso di errore, annullamento o budget esaurito.
5. Scarica il **file unico** (WAV) oppure **più file** (ZIP con un WAV per blocco e
   `testi.txt` con il testo di ogni blocco).

La chiave Vast si legge da `api.txt`, escluso da git, oppure dalla variabile `VAST_API_KEY`.
Non arriva mai al browser.

Variabili opzionali:

- `CLOVA_WORKER_IMAGE`: immagine del worker (predefinita `ghcr.io/techtonic2025/clova-voice-qwen:latest`);
- `CLOVA_IMAGE_LOGIN`: credenziali registro per Vast se l'immagine è privata,
  es. `-u utente -p TOKEN ghcr.io`.

## Velocità

- Alla prima generazione il worker sceglie i ~15 secondi più puliti del campione,
  li trascrive con Whisper e l'app li salva nel profilo (`reference-clip.wav` +
  trascrizione). Dalla volta successiva Whisper viene saltato e Qwen lavora su un
  riferimento corto.
- Il prompt della voce viene calcolato una volta sola per job e riusato.
- Il testo viene diviso in blocchi di circa 280 caratteri sui confini delle frasi e generato in
  batch (`CLOVA_BATCH_SIZE`, predefinito 4).
- FlashAttention 2 se disponibile, altrimenti SDPA (`/health` indica quale).
- Il tempo maggiore resta l'avvio dell'istanza e il download dell'immagine:
  per questo vengono scelte solo macchine con almeno 500 MB/s in download.

## Worker Qwen

- `GET /health`: modello caricato, errori di caricamento, tipo di attenzione;
- `POST /jobs`: riferimento, trascrizione facoltativa, testo, lingua, pausa;
- `GET /jobs/{id}`: avanzamento (`done`/`total`);
- `GET /jobs/{id}/files/{full.wav|parts.zip|reference-clip.wav}`.

Ogni istanza riceve un `CLOVA_JOB_TOKEN` casuale, inviato come `Authorization: Bearer ...`.
Il workflow `.github/workflows/publish-worker.yml` ricostruisce l'immagine a ogni
push su `main` che modifica `worker/`.

## Regola di costo

Le istanze vengono create senza volume persistente e distrutte al termine, perché
fermarle non basta: il disco continuerebbe a essere fatturato. Ogni istanza creata viene
annotata in `data/active-instances.json` finché Vast non ne conferma la distruzione.
Alla chiusura dell'app e a ogni avvio vengono distrutte le istanze rimaste, sia quelle
annotate sia quelle con etichetta `clova-voice-`. Il tetto di budget diventa anche un
tempo massimo di noleggio.
