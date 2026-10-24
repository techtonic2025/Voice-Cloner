# Clova Voice One-Shot

Clonazione vocale personale con **Qwen3-TTS 12Hz 1.7B Base** e GPU temporanea.

## Stato del progetto

L'interfaccia locale e il pianificatore sono pronti. Il worker usa ora Qwen,
non Chatterbox, ed e' predisposto per contenere gia' i pesi del modello. In
questo modo una GPU appena affittata non deve riscaricare circa 4,5 GB a ogni
generazione.

Per completare l'automazione manca la pubblicazione di `worker/Dockerfile` in
un registro Docker privato (per esempio GitHub Container Registry). Solo allora
l'app potra' creare la GPU Vast, inviare il campione, scaricare il WAV e
distruggere l'istanza senza passaggi manuali.

Non inserire chiavi API nel progetto o nel pacchetto distribuito. Ogni utente
deve collegare il proprio account Vast e il proprio credito.

Il workflow `.github/workflows/publish-worker.yml` costruisce l'immagine a ogni
push su `main` e la pubblica come `ghcr.io/techtonic2025/clova-voice-qwen`.

## Worker Qwen

Il worker espone endpoint temporanei:

- `GET /health`: verifica che il container sia raggiungibile;
- `POST /warm`: carica Qwen in GPU prima del lavoro;
- `POST /generate`: riceve audio di riferimento, trascrizione esatta e testo,
  poi restituisce un WAV.

Usa sempre una variabile casuale `CLOVA_JOB_TOKEN`. Il controllo locale la
invia come `Authorization: Bearer ...`. L'istanza deve restare esposta solo per
la durata del lavoro e va distrutta soltanto dopo il download del WAV.

Qwen richiede la trascrizione del campione: deve corrispondere a cio' che si
sente nell'audio. Per il clone rapido usa 10--30 secondi puliti. I 37 minuti di
materiale sono invece utili per un futuro fine-tuning, dopo il taglio in clip
brevi e la verifica delle trascrizioni.

## Avvio dell'interfaccia locale

1. Per usare il pianificatore senza server, fai doppio clic su `APRI-APP.html`.
2. L'app si apre direttamente nel browser e non usa porte di rete.
3. Il server Node resta disponibile per il futuro modulo cloud: fai doppio clic
   su `START.bat` se necessario.

## Regola di costo

L'app deve creare istanze Vast AI senza volume persistente e distruggerle al
termine. Non basta fermare un'istanza: il disco puo' continuare a essere
fatturato. Nessuna chiave viene salvata dal pianificatore locale.
