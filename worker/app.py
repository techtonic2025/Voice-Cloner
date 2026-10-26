"""Temporary Qwen voice-cloning worker used by Clova Voice.

Flow: the model loads at startup, the orchestrator polls /health until ready,
submits a job to /jobs, polls /jobs/{id} and finally downloads the files.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import secrets
import tempfile
import threading
import time
import unicodedata
import zipfile
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel

import train as trainer

app = FastAPI(title="Clova Voice Qwen worker", docs_url=None, redoc_url=None)
MODEL: Qwen3TTSModel | None = None
TRANSCRIBER: WhisperModel | None = None
LOAD_ERROR: str | None = None
GPU_LOCK = threading.Lock()
MODEL_PATH = os.getenv("QWEN_MODEL_PATH", "/models/qwen3-tts-1.7b-base")
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "/models/faster-whisper-large-v3-turbo")
JOB_TOKEN = os.getenv("CLOVA_JOB_TOKEN", "")
BATCH_SIZE = max(1, int(os.getenv("CLOVA_BATCH_SIZE", "4")))
MAX_TEXT_CHARS = 60_000
CHUNK_CHARS = 400
# A long reference (up to ~2 min) gives better likeness of the timbre, a shorter
# one can keep a more native accent on new text: the length is chosen per job.
REF_MIN_SECONDS, REF_MAX_SECONDS = 6.0, 120.0
WORK_DIR = Path(tempfile.gettempdir()) / "clova-jobs"
JOBS: dict[str, dict] = {}
# Voice prompts by reference hash: a warm instance reuses them across jobs.
PROMPTS: dict[str, object] = {}
# Fine-tuned models by HF repo, downloaded on first use (one kept at a time).
TRAINED: dict[str, Qwen3TTSModel] = {}
HF_TOKEN = os.getenv("HF_TOKEN", "")
LANGUAGE_CODES = {"Italian": "it", "English": "en", "Spanish": "es", "French": "fr"}


def require_token(authorization: str | None) -> None:
    if not JOB_TOKEN:
        return
    expected = f"Bearer {JOB_TOKEN}"
    if not authorization or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Token non valido")


FLASH_ATTN_ERROR: str | None = None


def attention_implementation() -> str:
    global FLASH_ATTN_ERROR
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError as exc:  # Kept for /health: a wrong wheel fails here, not at install.
        FLASH_ATTN_ERROR = str(exc)[:300]
        return "sdpa"


def load_models() -> None:
    global MODEL, TRANSCRIBER, LOAD_ERROR
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU CUDA non disponibile")
        MODEL = Qwen3TTSModel.from_pretrained(
            MODEL_PATH, device_map="cuda:0", dtype=torch.bfloat16,
            attn_implementation=attention_implementation(),
        )
        TRANSCRIBER = WhisperModel(WHISPER_MODEL_PATH, device="cuda", compute_type="float16")
    except Exception as exc:  # Reported through /health so the orchestrator can give up early.
        LOAD_ERROR = str(exc)


def get_trained(repo: str) -> Qwen3TTSModel:
    if repo not in TRAINED:
        TRAINED.clear()
        torch.cuda.empty_cache()
        folder = Path("/models/trained") / re.sub(r"[^A-Za-z0-9_.-]", "_", repo)
        snapshot_download(repo, local_dir=folder, token=HF_TOKEN or None)
        TRAINED[repo] = Qwen3TTSModel.from_pretrained(
            str(folder), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation=attention_implementation(),
        )
    return TRAINED[repo]


@app.on_event("startup")
def startup() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=load_models, daemon=True).start()


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        return audio.mean(axis=1), rate
    except Exception:
        import librosa
        audio, rate = librosa.load(str(path), sr=None, mono=True)
        return audio.astype("float32"), rate


def prepare_reference(source: Path, language: str, folder: Path,
                      target_seconds: float = REF_MAX_SECONDS) -> tuple[Path, str]:
    """Keep the clean stretch of speech closest to target_seconds, cut on Whisper
    segment boundaries (whole phrases), and transcribe it for Qwen."""
    segments, _ = TRANSCRIBER.transcribe(
        str(source), language=LANGUAGE_CODES.get(language), vad_filter=True, beam_size=5
    )
    segments = [s for s in segments if s.text.strip() and s.no_speech_prob < 0.6]
    if not segments:
        raise RuntimeError("Non riesco a capire il campione vocale. Prova con un audio più pulito.")

    best, best_score = (0, 0), float("-inf")
    for i in range(len(segments)):
        for j in range(i, len(segments)):
            duration = segments[j].end - segments[i].start
            if duration > target_seconds * 1.15:
                break
            window = segments[i:j + 1]
            confidence = sum(s.avg_logprob for s in window) / len(window)
            score = -abs(target_seconds - duration) + 6 * confidence
            if duration < REF_MIN_SECONDS:
                score -= 10
            if score > best_score:
                best, best_score = (i, j), score
    window = segments[best[0]:best[1] + 1]

    audio, rate = read_audio(source)
    start = max(0, int((window[0].start - 0.15) * rate))
    end = min(len(audio), int((window[-1].end + 0.25) * rate))
    clip = folder / "reference-clip.wav"
    sf.write(clip, audio[start:end], rate)
    return clip, " ".join(s.text.strip() for s in window).strip()


def clean_text(text: str) -> str:
    """Fix what pasted text (web pages, manuals) usually gets wrong, so pauses land
    where a reader would put them: "Mostra.Per" -> "Mostra. Per", hard-wrapped
    lines joined, list lines turned into sentences, bullets and odd spaces removed."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ").replace("\t", " ")
    text = re.sub(r"(?m)^[ ]*(?:[•·▪◦‣*–—-]|\d{1,2}[.)])[ ]+", "", text)  # bullets and "1." / "1)"
    text = re.sub(r"([.!?…;:])(?=[A-ZÀ-ÖØ-Þ])", r"\1 ", text)  # missing space after a sentence
    paragraphs = []
    for block in re.split(r"\n[ ]*\n", text):
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        joined = ""
        for line in lines:
            if not joined:
                joined = line
            elif joined[-1] in ".!?…;:,":
                joined += " " + line
            elif line[:1].isupper():  # A new list item or heading without its own punctuation.
                joined += ". " + line
            else:  # Hard-wrapped sentence.
                joined += " " + line
        if joined:
            paragraphs.append(re.sub(r" {2,}", " ", joined))
    return "\n\n".join(paragraphs)


def change_speed(wav: np.ndarray, speed: float) -> np.ndarray:
    """Slow down or speed up speech without changing its pitch (WSOLA)."""
    if abs(speed - 1.0) < 0.01:
        return wav
    from audiotsm import wsola
    from audiotsm.io.array import ArrayReader, ArrayWriter
    reader = ArrayReader(wav.reshape(1, -1))
    writer = ArrayWriter(channels=1)
    wsola(channels=1, speed=speed).run(reader, writer)
    return writer.data[0].astype("float32")


def split_text(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Split into chunks of whole sentences. Blank lines are read straight through
    (a paragraph without final punctuation gets a period); overlong sentences fall
    back to commas."""
    chunks: list[str] = []
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text.strip())]
    flowing = " ".join(p if p[-1] in ".!?…;:" else p + "." for p in paragraphs if p)
    current = ""
    for sentence in re.split(r"(?<=[.!?…;:])\s+", flowing):
        if not sentence:
            continue
        pieces = [sentence]
        if len(sentence) > limit:
            pieces = [p.strip() for p in re.split(r"(?<=[,])\s+", sentence) if p.strip()]
        for piece in pieces:
            while len(piece) > limit:
                cut = piece.rfind(" ", 0, limit)
                cut = cut if cut > limit // 2 else limit
                chunks.append(piece[:cut].strip())
                piece = piece[cut:].strip()
            if current and len(current) + 1 + len(piece) > limit:
                chunks.append(current)
                current = piece
            else:
                current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


def voiced_frames(wav: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """Per 10 ms frame: is there speech? (Energy above 3% of the loudest frame.)"""
    frame = int(sample_rate * 0.01)
    energy = np.sqrt(np.mean(wav[: len(wav) // frame * frame].reshape(-1, frame) ** 2, axis=1))
    return energy > max(energy.max(initial=0) * 0.03, 1e-4), frame


def longest_pause(wav: np.ndarray, sample_rate: int) -> float:
    """Longest silence inside the speech, in seconds."""
    voiced, frame = voiced_frames(wav, sample_rate)
    index = np.flatnonzero(voiced)
    return float(np.diff(index).max(initial=1) - 1) * frame / sample_rate if len(index) > 1 else 0.0


def token_budget(texts: list[str]) -> int:
    """Most audio frames a batch may produce. The model makes 12 frames per second and
    reads ~15 characters per second, so a chunk needs ~0.8 frames per character:
    2.5x that (plus a margin for short chunks) stops takes that never end, which
    otherwise hold the whole batch until 2048 frames (almost three minutes of audio)."""
    return min(2048, int(max(len(text) for text in texts) * 2 + 120))


def trim_silence(wav: np.ndarray, sample_rate: int, keep: float = 0.05, max_pause: float = 0.7) -> np.ndarray:
    """Cut the silence the model leaves before and after a chunk, so the gap between
    chunks is only the chosen pause, and shorten the occasional multi-second stop
    the model drops between sentences to max_pause."""
    if len(wav) < sample_rate * 0.1:
        return wav
    voiced, frame = voiced_frames(wav, sample_rate)
    index = np.flatnonzero(voiced)
    if not len(index):
        return wav
    margin = int(sample_rate * keep)
    pieces = []
    start = max(index[0] * frame - margin, 0)
    for before, after in zip(index[:-1], index[1:]):
        if (after - before) * frame > max_pause * sample_rate:
            half = int(max_pause * sample_rate / 2)
            pieces.append(wav[start:(before + 1) * frame + half])
            start = after * frame - half
    pieces.append(wav[start:min((index[-1] + 1) * frame + margin, len(wav))])
    return np.concatenate(pieces)


def syllables(text: str) -> int:
    """Rough Italian syllable count: one per group of vowels (English words come out close enough)."""
    return len(re.findall(r"[aeiouyàáèéìíòóùú]+", text.lower()))


def speech_seconds(wav: np.ndarray, sample_rate: int) -> float:
    """Time spent talking: voiced frames plus gaps under 0.2 s (stops inside words),
    without the pauses between phrases."""
    voiced, frame = voiced_frames(wav, sample_rate)
    index = np.flatnonzero(voiced)
    if not len(index):
        return 0.0
    gaps = np.diff(index)
    frames = 1 + int(np.sum(np.where(gaps <= 20, gaps, 1)))
    return frames * frame / sample_rate


def natural_speed(wav: np.ndarray, sample_rate: int, text: str, target_rate: float) -> float:
    """Speed factor that brings a take to the speaker's own pace (syllables per second).
    Only slows down or speeds up within 0.8-1.15, never touching takes already close."""
    seconds = speech_seconds(wav, sample_rate)
    if target_rate <= 0 or seconds < 1:
        return 1.0
    factor = target_rate / (syllables(text) / seconds)
    return 1.0 if abs(factor - 1) < 0.03 else min(max(factor, 0.8), 1.15)


def apply_pronunciations(text: str, entries: list[list[str]]) -> str:
    """Respell foreign words as they sound ("file" -> "fàil"), whole words only,
    keeping a leading capital. Takes are still checked against the original text,
    because Whisper writes a well-pronounced "browser" as "browser"."""
    for word, spoken in sorted(entries, key=lambda entry: -len(entry[0])):
        def replace(match: re.Match) -> str:
            return spoken[:1].upper() + spoken[1:] if match.group(0)[:1].isupper() else spoken
        text = re.sub(rf"(?<!\w){re.escape(word)}(?!\w)", replace, text, flags=re.IGNORECASE)
    return text


def normalize_words(value: str) -> str:
    """Lowercase, drop accents and punctuation: "Bràuser!" and "brauser" compare equal."""
    value = unicodedata.normalize("NFD", value.lower())
    value = "".join(c for c in value if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def take_score(wav: np.ndarray, sample_rate: int, target: str, language: str) -> float:
    """How faithfully a take says the target text, judged by Whisper (0..1)."""
    length = int(len(wav) * 16_000 / sample_rate)
    audio16 = np.interp(np.linspace(0, len(wav) - 1, length), np.arange(len(wav)), wav).astype("float32")
    segments, _ = TRANSCRIBER.transcribe(
        audio16, language=LANGUAGE_CODES.get(language), beam_size=1,
        vad_filter=False, condition_on_previous_text=False,
    )
    heard = " ".join(segment.text for segment in segments)
    score = SequenceMatcher(None, normalize_words(target), normalize_words(heard)).ratio()
    # Runaway takes (babbling, long silences) run far longer than ~15 characters per second.
    if len(wav) / sample_rate > max(4.0, len(target) / 15 * 2.2):
        score *= 0.8
    # A take that stops for seconds mid-text sounds broken even after the pause is shortened.
    if longest_pause(wav, sample_rate) > 1.5:
        score *= 0.85
    return score


def run_job(job_id: str, source: Path | None, reference_text: str, text: str, language: str, pause: float,
            temperature: float, top_p: float, takes: int, pronunciations: list[list[str]], speed: float,
            ref_seconds: float, trained_repo: str = "", speaker: str = "", target_rate: float = 0.0,
            chunk_chars: int = CHUNK_CHARS) -> None:
    job = JOBS[job_id]
    folder = WORK_DIR / job_id
    sampling: dict[str, object] = {}
    if temperature > 0:
        sampling.update(do_sample=True, temperature=temperature)
    if 0 < top_p < 1:
        sampling["top_p"] = top_p
    try:
        with GPU_LOCK:
            if trained_repo:
                # Fine-tuned voice: no reference needed, the speaker lives in the model.
                job["status"] = "loading_trained"
                trained_model = get_trained(trained_repo)

                def synthesize(texts: list[str]):
                    return trained_model.generate_custom_voice(
                        text=texts, speaker=speaker, language=[language] * len(texts),
                        max_new_tokens=token_budget(texts), **sampling,
                    )
            else:
                job["status"] = "preparing"
                reference = source
                full_length = ref_seconds >= REF_MAX_SECONDS
                if not reference_text or not full_length:
                    # Only the full-length clip is cached by the app; shorter ones are cut
                    # (and transcribed) from whatever was sent, every time.
                    reference, reference_text = prepare_reference(source, language, folder, ref_seconds)
                    job["prepared_reference"] = full_length
                job["reference_text"] = reference_text
                key = hashlib.sha1(reference.read_bytes() + reference_text.encode()).hexdigest()
                prompt = PROMPTS.get(key)
                if prompt is None:
                    prompt = MODEL.create_voice_clone_prompt(
                        ref_audio=str(reference), ref_text=reference_text, x_vector_only_mode=False
                    )
                    PROMPTS.clear()
                    PROMPTS[key] = prompt

                def synthesize(texts: list[str]):
                    return MODEL.generate_voice_clone(
                        text=texts, language=[language] * len(texts), voice_clone_prompt=prompt,
                        max_new_tokens=token_budget(texts), **sampling,
                    )

            chunks = split_text(clean_text(text), chunk_chars)
            spoken = [apply_pronunciations(chunk, pronunciations) for chunk in chunks]
            job.update(status="generating", total=len(chunks), done=0, takes=takes)
            # With several takes per chunk, each copy is sampled independently in the
            # same batch and Whisper keeps the one that best matches the text.
            items = [index for index in range(len(chunks)) for _ in range(takes)]
            candidates: list[list[np.ndarray]] = [[] for _ in chunks]
            sample_rate = 24_000
            for offset in range(0, len(items), BATCH_SIZE):
                batch = items[offset:offset + BATCH_SIZE]
                wavs, sample_rate = synthesize([spoken[i] for i in batch])
                for index, wav in zip(batch, wavs):
                    candidates[index].append(np.asarray(wav, dtype="float32").reshape(-1))
                job["done"] = sum(len(c) == takes for c in candidates)

            parts: list[np.ndarray] = []
            scores: list[float] = []
            for chunk, options in zip(chunks, candidates):
                if takes == 1:
                    parts.append(options[0])
                    continue
                job["status"] = "checking"
                ranked = [(take_score(wav, sample_rate, chunk, language), wav) for wav in options]
                best_score, best = max(ranked, key=lambda pair: pair[0])
                parts.append(best)
                scores.append(round(best_score, 3))
            if scores:
                job["quality"] = round(sum(scores) / len(scores), 3)
                job["weakest"] = [i + 1 for i, value in enumerate(scores) if value < 0.85]

        job["status"] = "packing"
        # Applied after Whisper picked the takes, so scoring always sees the raw audio.
        # With a target pace (the speaker's own syllables per second) each chunk is first
        # brought to it, then the user's speed applies on top.
        paces = [natural_speed(part, sample_rate, chunk, target_rate) for chunk, part in zip(chunks, parts)]
        job["pace"] = [round(pace, 2) for pace in paces]
        parts = [change_speed(trim_silence(part, sample_rate), speed * pace) for part, pace in zip(parts, paces)]
        silence = np.zeros(int(sample_rate * pause), dtype="float32")
        full = []
        for index, part in enumerate(parts):
            full.extend([part, silence] if index < len(parts) - 1 else [part])
        sf.write(folder / "full.wav", np.concatenate(full), sample_rate)

        with zipfile.ZipFile(folder / "parts.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for index, (chunk, part) in enumerate(zip(chunks, parts), start=1):
                buffer = io.BytesIO()
                sf.write(buffer, part, sample_rate, format="WAV")
                archive.writestr(f"{index:03d}.wav", buffer.getvalue())
            archive.writestr("testi.txt", "\n".join(f"{i:03d}\t{c}" for i, c in enumerate(chunks, 1)))
        job.update(status="done", seconds=round(sum(len(p) for p in parts) / sample_rate, 1),
                   elapsed=round(time.time() - job["started"], 1))
    except Exception as exc:
        job.update(status="error", error=f"Generazione non riuscita: {exc}")
    finally:
        torch.cuda.empty_cache()


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "ok": LOAD_ERROR is None,
        "ready": MODEL is not None and TRANSCRIBER is not None,
        "error": LOAD_ERROR,
        "cuda": torch.cuda.is_available(),
        "attention": attention_implementation(),
        "flash_attn_error": FLASH_ATTN_ERROR,
        "torch": torch.__version__,
        "model": "Qwen3-TTS-12Hz-1.7B-Base",
    }


@app.post("/jobs")
async def create_job(
    reference: UploadFile | None = File(None),
    reference_text: str = Form(""),
    trained_repo: str = Form(""),
    speaker: str = Form(""),
    text: str = Form(...),
    language: str = Form("Italian"),
    pause: float = Form(0.35),
    temperature: float = Form(0.0),
    top_p: float = Form(0.0),
    takes: int = Form(1),
    pronunciations: str = Form("[]"),
    speed: float = Form(1.0),
    ref_seconds: float = Form(REF_MAX_SECONDS),
    target_rate: float = Form(0.0),
    chunk_chars: int = Form(CHUNK_CHARS),
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    require_token(authorization)
    try:
        entries = [[str(word), str(said)] for word, said in json.loads(pronunciations) if str(word).strip() and str(said).strip()]
    except (ValueError, TypeError):
        raise HTTPException(status_code=422, detail="Dizionario di pronuncia non valido") from None
    if MODEL is None:
        raise HTTPException(status_code=503, detail=LOAD_ERROR or "Modello non ancora caricato")
    text, reference_text = text.strip(), reference_text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Serve il testo da generare")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=422, detail=f"Testo troppo lungo (massimo {MAX_TEXT_CHARS} caratteri)")
    trained_repo, speaker = trained_repo.strip(), speaker.strip()
    if bool(trained_repo) != bool(speaker) or (not trained_repo and reference is None):
        raise HTTPException(status_code=422, detail="Serve un campione vocale oppure un modello addestrato con il suo speaker")

    job_id = secrets.token_hex(12)
    folder = WORK_DIR / job_id
    folder.mkdir(parents=True)
    source = None
    if reference is not None:
        suffix = Path(reference.filename or "reference.wav").suffix.lower() or ".wav"
        source = folder / f"reference{suffix}"
        with source.open("wb") as destination:
            while chunk := await reference.read(1024 * 1024):
                destination.write(chunk)

    JOBS[job_id] = {"status": "queued", "done": 0, "total": 0, "started": time.time()}
    threading.Thread(
        target=run_job,
        args=(job_id, source, reference_text, text, language, min(max(pause, 0), 2), min(max(temperature, 0), 1.5),
              min(max(top_p, 0), 1), min(max(takes, 1), 3), entries, min(max(speed, 0.7), 1.3),
              min(max(ref_seconds, 10.0), REF_MAX_SECONDS), trained_repo, speaker, min(max(target_rate, 0.0), 12.0),
              min(max(chunk_chars, 80), 600)),
        daemon=True,
    ).start()
    return {"id": job_id}


@app.post("/train")
async def create_training(
    files: list[UploadFile] = File(...),
    repo: str = Form(...),
    speaker: str = Form("techtonic"),
    language: str = Form("Italian"),
    epochs: int = Form(5),
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    require_token(authorization)
    if MODEL is None or TRANSCRIBER is None:
        raise HTTPException(status_code=503, detail=LOAD_ERROR or "Modello non ancora caricato")
    if not HF_TOKEN:
        raise HTTPException(status_code=422, detail="Serve un token Hugging Face con permesso di scrittura")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo.strip()) or not re.fullmatch(r"[a-z0-9_]{2,32}", speaker):
        raise HTTPException(status_code=422, detail="Nome del repository o dello speaker non valido")

    job_id = f"train-{secrets.token_hex(8)}"
    folder = WORK_DIR / job_id
    inputs = folder / "input"
    inputs.mkdir(parents=True)
    paths = []
    for index, upload in enumerate(files, start=1):
        path = inputs / f"{index:02d}{Path(upload.filename or 'audio.mp3').suffix.lower() or '.mp3'}"
        with path.open("wb") as destination:
            while chunk := await upload.read(4 * 1024 * 1024):
                destination.write(chunk)
        paths.append(path)

    JOBS[job_id] = {"status": "running", "phase": "queued", "detail": "In coda…", "started": time.time()}

    def run() -> None:
        job = JOBS[job_id]
        try:
            with GPU_LOCK:
                trainer.train(
                    job, paths, folder, transcriber=TRANSCRIBER, clean_text=clean_text, language=language,
                    language_code=LANGUAGE_CODES.get(language), speaker=speaker, repo=repo.strip(), token=HF_TOKEN,
                    epochs=min(max(epochs, 1), 10), base_model=MODEL_PATH, attention=attention_implementation(),
                    load_model=lambda path: Qwen3TTSModel.from_pretrained(
                        path, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation=attention_implementation(),
                    ),
                    score=take_score, clone_model=MODEL,
                )
            job.update(status="done", elapsed=round(time.time() - job["started"], 1))
        except Exception as exc:
            job.update(status="error", error=f"Addestramento non riuscito: {exc}")
        finally:
            torch.cuda.empty_cache()

    threading.Thread(target=run, daemon=True).start()
    return {"id": job_id}


@app.get("/train/{job_id}")
def training_status(job_id: str, authorization: str | None = Header(default=None)) -> dict:
    require_token(authorization)
    return get_job(job_id)


@app.get("/train/{job_id}/samples/{name}")
def training_sample(job_id: str, name: str, authorization: str | None = Header(default=None)) -> FileResponse:
    """Test sentences read by each epoch (and the real voice), to listen side by side."""
    require_token(authorization)
    if name not in get_job(job_id).get("samples", []):
        raise HTTPException(status_code=404, detail="Campione non disponibile")
    return FileResponse(WORK_DIR / job_id / "samples" / name)


def get_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job non trovato")
    return job


@app.get("/jobs/{job_id}")
def job_status(job_id: str, authorization: str | None = Header(default=None)) -> dict:
    require_token(authorization)
    return get_job(job_id)


@app.get("/jobs/{job_id}/files/{name}")
def job_file(job_id: str, name: str, authorization: str | None = Header(default=None)) -> FileResponse:
    require_token(authorization)
    if get_job(job_id)["status"] != "done" or name not in {"full.wav", "parts.zip", "reference-clip.wav"}:
        raise HTTPException(status_code=404, detail="File non disponibile")
    path = WORK_DIR / job_id / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="File non disponibile")
    return FileResponse(path)
