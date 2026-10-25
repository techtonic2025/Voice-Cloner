# Clova Voice One-Shot

Clonazione vocale personale con **Qwen3-TTS 12Hz 1.7B Base** e GPU temporanea.

## Come funziona

1. Avvia `START.bat` e apri l'app (http://127.0.0.1:4587).
2. In "Le mie voci" carica fino a tre campioni (30–120 s vanno bene).
3. Scegli la voce, incolla il testo (fino a 60.000 caratteri) e premi "Genera".
4. L'app noleggia la GPU Vast **più veloce** entro la tariffa massima (predefinita
   0,80 $/ora). Per ogni offerta stima il tempo totale: download e
   estrazione del worker, caricamento del modello, generazione. Considera rete
   (almeno 1 Gbit/s), disco (almeno 1 GB/s) e potenza GPU (`dlperf`), poi prende la più rapida.
   Restano escluse le schede da mining e le Blackwell, non supportate. Poi avvia il worker, genera, scarica i file e
   **distrugge l'istanza**, anche in caso di errore, annullamento o budget esaurito.
5. Scegli cosa fare a fine lavoro: **distruggi subito** la GPU oppure **tienila accesa
   5 minuti**. In questo caso la generazione successiva riparte subito, senza avvio e
   senza ricaricare il modello, e i 5 minuti ricominciano. Finché la GPU è accesa compare in alto
   un avviso con conto alla rovescia, spesa e tasto **Distruggi istanza**.
6. Scarica il **file unico** (WAV) oppure **più file** (ZIP con un WAV per blocco e
   `testi.txt` con il testo di ogni blocco).

La chiave Vast si legge da `api.txt`, escluso da git, oppure dalla variabile `VAST_API_KEY`.
Non arriva mai al browser.

Variabili opzionali:

- `CLOVA_WORKER_IMAGE`: immagine del worker (predefinita `ghcr.io/techtonic2025/clova-voice-qwen:latest`);
- `CLOVA_IMAGE_LOGIN`: credenziali registro per Vast se l'immagine è privata,
  es. `-u utente -p TOKEN ghcr.io`.

## Voce addestrata (fine-tuning)

Come il "Professional Voice Cloning": da 30–60 minuti di registrazioni **vere**
(niente audio già generati con l'IA) il modello Qwen3-TTS 1.7B impara come parli,
accento compreso.

1. Crea su Hugging Face un token con permesso **Write** e salvalo nel riquadro
   "Addestra la tua voce" (resta in `data/hf.json`, escluso da git).
2. Aggiungi le registrazioni (finiscono in `data/training/`).
3. Premi "Avvia addestramento": su una GPU da almeno 40 GB Whisper taglia le
   registrazioni in frasi da 3–12 s (scarta quelle incerte), gli script ufficiali
   di `QwenLM/Qwen3-TTS/finetuning` calcolano i codici audio e addestrano, e il
   modello viene caricato in un repository **privato** `utente/clova-voice-<nome>`.
   Poi la GPU viene distrutta.
4. Compare la voce addestrata: le GPU la scaricano da Hugging Face e non serve più
   inviare un campione.

Il token Hugging Face viene passato alla GPU noleggiata (variabile `HF_TOKEN`)
per scaricare e caricare il modello privato.

## Velocità

- Alla prima generazione il worker usa tutto il campione (fino a 2 minuti, tolti
  i silenzi ai bordi), lo trascrive con Whisper large-v3-turbo e l'app salva
  campione e trascrizione nel profilo (`reference-clip.wav`). Dalla volta successiva
  Whisper viene saltato. Un campione lungo dà una somiglianza nettamente migliore
  di un pezzo da 15 secondi.
- Il prompt della voce viene calcolato una volta sola per job e riusato.
- Il testo viene diviso in blocchi di circa 400 caratteri sui confini delle frasi e generato in
  batch (`CLOVA_BATCH_SIZE`, predefinito 4).
- FlashAttention 2 se disponibile, altrimenti SDPA (`/health` indica quale).
- Il tempo maggiore resta l'avvio dell'istanza e il download dell'immagine:
  per questo pesa molto nella scelta della macchina. Sulle GPU da 24 GB in su si
  generano 8 blocchi alla volta invece di 4.

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
