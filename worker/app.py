"""Temporary Qwen voice-cloning worker used by Clova Voice.

Flow: the model loads at startup, the orchestrator polls /health until ready,
submits a job to /jobs, polls /jobs/{id} and finally downloads the files.
"""

from __future__ import annotations

import io
import os
import re
import secrets
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from qwen_tts import Qwen3TTSModel

app = FastAPI(title="Clova Voice Qwen worker", docs_url=None, redoc_url=None)
MODEL: Qwen3TTSModel | None = None
TRANSCRIBER: WhisperModel | None = None
LOAD_ERROR: str | None = None
GPU_LOCK = threading.Lock()
MODEL_PATH = os.getenv("QWEN_MODEL_PATH", "/models/qwen3-tts-1.7b-base")
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "/models/faster-whisper-small")
JOB_TOKEN = os.getenv("CLOVA_JOB_TOKEN", "")
BATCH_SIZE = max(1, int(os.getenv("CLOVA_BATCH_SIZE", "4")))
MAX_TEXT_CHARS = 60_000
CHUNK_CHARS = 280
# Qwen clones best from a short clean clip: longer references only slow every chunk.
REF_MIN_SECONDS, REF_TARGET_SECONDS, REF_MAX_SECONDS = 6.0, 15.0, 20.0
WORK_DIR = Path(tempfile.gettempdir()) / "clova-jobs"
JOBS: dict[str, dict] = {}
LANGUAGE_CODES = {"Italian": "it", "English": "en", "Spanish": "es", "French": "fr"}


def require_token(authorization: str | None) -> None:
    if not JOB_TOKEN:
        return
    expected = f"Bearer {JOB_TOKEN}"
    if not authorization or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Token non valido")


def attention_implementation() -> str:
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
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


def prepare_reference(source: Path, language: str, folder: Path) -> tuple[Path, str]:
    """Pick the best 6-20 s stretch of speech and transcribe it for Qwen."""
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
            if duration > REF_MAX_SECONDS:
                break
            window = segments[i:j + 1]
            confidence = sum(s.avg_logprob for s in window) / len(window)
            score = -abs(REF_TARGET_SECONDS - duration) + 6 * confidence
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


def split_text(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Split on sentence boundaries, then on commas, so each chunk stays short."""
    sentences = re.split(r"(?<=[.!?…;:])\s+|\n{2,}", re.sub(r"[ \t]+", " ", text.strip()))
    chunks: list[str] = []
    current = ""
    for sentence in (s.strip() for s in sentences):
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


def run_job(job_id: str, source: Path, reference_text: str, text: str, language: str, pause: float) -> None:
    job = JOBS[job_id]
    folder = WORK_DIR / job_id
    try:
        with GPU_LOCK:
            job["status"] = "preparing"
            reference = source
            if not reference_text:
                reference, reference_text = prepare_reference(source, language, folder)
                job["prepared_reference"] = True
            job["reference_text"] = reference_text
            prompt = MODEL.create_voice_clone_prompt(
                ref_audio=str(reference), ref_text=reference_text, x_vector_only_mode=False
            )

            chunks = split_text(text)
            job.update(status="generating", total=len(chunks), done=0)
            parts: list[np.ndarray] = []
            sample_rate = 24_000
            for offset in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[offset:offset + BATCH_SIZE]
                wavs, sample_rate = MODEL.generate_voice_clone(
                    text=batch, language=[language] * len(batch), voice_clone_prompt=prompt,
                    max_new_tokens=2048,
                )
                parts.extend(np.asarray(w, dtype="float32").reshape(-1) for w in wavs)
                job["done"] = len(parts)

        job["status"] = "packing"
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
        "model": "Qwen3-TTS-12Hz-1.7B-Base",
    }


@app.post("/jobs")
async def create_job(
    reference: UploadFile = File(...),
    reference_text: str = Form(""),
    text: str = Form(...),
    language: str = Form("Italian"),
    pause: float = Form(0.35),
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    require_token(authorization)
    if MODEL is None:
        raise HTTPException(status_code=503, detail=LOAD_ERROR or "Modello non ancora caricato")
    text, reference_text = text.strip(), reference_text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Serve il testo da generare")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=422, detail=f"Testo troppo lungo (massimo {MAX_TEXT_CHARS} caratteri)")

    job_id = secrets.token_hex(12)
    folder = WORK_DIR / job_id
    folder.mkdir(parents=True)
    suffix = Path(reference.filename or "reference.wav").suffix.lower() or ".wav"
    source = folder / f"reference{suffix}"
    with source.open("wb") as destination:
        while chunk := await reference.read(1024 * 1024):
            destination.write(chunk)

    JOBS[job_id] = {"status": "queued", "done": 0, "total": 0, "started": time.time()}
    threading.Thread(
        target=run_job, args=(job_id, source, reference_text, text, language, min(max(pause, 0), 2)),
        daemon=True,
    ).start()
    return {"id": job_id}


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
