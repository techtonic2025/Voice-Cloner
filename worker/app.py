"""Temporary Qwen voice-cloning worker used by Clova Voice."""

from __future__ import annotations

import os
import secrets
import tempfile
from pathlib import Path

import soundfile as sf
import torch
from faster_whisper import WhisperModel
from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from qwen_tts import Qwen3TTSModel

app = FastAPI(title="Clova Voice Qwen worker", docs_url=None, redoc_url=None)
MODEL: Qwen3TTSModel | None = None
TRANSCRIBER: WhisperModel | None = None
MODEL_PATH = os.getenv("QWEN_MODEL_PATH", "/models/qwen3-tts-1.7b-base")
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "/models/faster-whisper-small")
JOB_TOKEN = os.getenv("CLOVA_JOB_TOKEN", "")
MAX_TEXT_CHARS = 2_000


def require_token(authorization: str | None) -> None:
    if not JOB_TOKEN:
        return
    expected = f"Bearer {JOB_TOKEN}"
    if not authorization or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Token non valido")


def get_model() -> Qwen3TTSModel:
    global MODEL
    if MODEL is None:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU CUDA non disponibile")
        MODEL = Qwen3TTSModel.from_pretrained(
            MODEL_PATH, device_map="cuda:0", dtype=torch.bfloat16
        )
    return MODEL


def get_transcriber() -> WhisperModel:
    global TRANSCRIBER
    if TRANSCRIBER is None:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU CUDA non disponibile")
        TRANSCRIBER = WhisperModel(
            WHISPER_MODEL_PATH, device="cuda", compute_type="float16"
        )
    return TRANSCRIBER


def transcribe_reference(path: Path, language: str) -> str:
    """Get the words actually spoken in the reference audio for Qwen."""
    language_code = {"Italian": "it", "English": "en", "Spanish": "es", "French": "fr"}.get(language)
    segments, _ = get_transcriber().transcribe(
        str(path), language=language_code, vad_filter=True, beam_size=5
    )
    transcription = " ".join(segment.text.strip() for segment in segments).strip()
    if not transcription:
        raise RuntimeError("Non riesco a capire il campione vocale. Prova con un audio più pulito.")
    return transcription


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "ok": True,
        "ready": MODEL is not None,
        "cuda": torch.cuda.is_available(),
        "model": "Qwen3-TTS-12Hz-1.7B-Base",
    }


@app.post("/warm")
def warm(authorization: str | None = Header(default=None)) -> dict[str, object]:
    require_token(authorization)
    get_model()
    return {"ok": True, "ready": True}


def remove_paths(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


@app.post("/generate")
async def generate(
    background_tasks: BackgroundTasks,
    reference: UploadFile = File(...),
    reference_text: str = Form(""),
    text: str = Form(...),
    language: str = Form("Italian"),
    authorization: str | None = Header(default=None),
) -> FileResponse:
    require_token(authorization)
    text, reference_text = text.strip(), reference_text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Serve il testo da generare")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=422, detail=f"Testo troppo lungo (massimo {MAX_TEXT_CHARS} caratteri)")
    if not (reference.content_type or "").startswith("audio/"):
        raise HTTPException(status_code=422, detail="Il campione deve essere un file audio")

    suffix = Path(reference.filename or "reference.wav").suffix.lower() or ".wav"
    directory = Path(tempfile.gettempdir())
    source = directory / f"clova-reference-{secrets.token_hex(12)}{suffix}"
    output = directory / f"clova-output-{secrets.token_hex(12)}.wav"
    try:
        with source.open("wb") as destination:
            while chunk := await reference.read(1024 * 1024):
                destination.write(chunk)
        if not reference_text:
            reference_text = transcribe_reference(source, language)
        wavs, sample_rate = get_model().generate_voice_clone(
            text=text, language=language, ref_audio=str(source), ref_text=reference_text
        )
        sf.write(output, wavs[0], sample_rate)
    except Exception as exc:
        remove_paths(source, output)
        raise HTTPException(status_code=500, detail=f"Generazione non riuscita: {exc}") from exc

    background_tasks.add_task(remove_paths, source, output)
    return FileResponse(output, media_type="audio/wav", filename="clova-voice.wav", background=background_tasks)
