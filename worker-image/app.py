"""Temporary FLUX.1 [dev] image worker used by Clova Voice.

Flow: at startup the weights are downloaded from Hugging Face (gated repo: the
user accepts the license and the app passes HF_TOKEN) and loaded in bf16; the
orchestrator polls /health until ready, submits /images, polls /images/{id}
and downloads the PNG files.
"""

from __future__ import annotations

import os
import secrets
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import torch
from diffusers import FluxPipeline
from fastapi import FastAPI, Form, Header, HTTPException
from fastapi.responses import FileResponse
from huggingface_hub import snapshot_download

app = FastAPI(title="Clova Voice FLUX worker", docs_url=None, redoc_url=None)
PIPE: FluxPipeline | None = None
PHASE = "starting"
LOAD_ERROR: str | None = None
GPU_LOCK = threading.Lock()
REPO = os.getenv("FLUX_REPO", "black-forest-labs/FLUX.1-dev")
MODEL_DIR = Path(os.getenv("FLUX_DIR", "/models/flux1-dev"))
# The repo also ships a 23.8 GB single-file copy at the root: only the diffusers folders are needed.
ALLOW = ["model_index.json", "scheduler/*", "text_encoder/*", "text_encoder_2/*",
         "tokenizer/*", "tokenizer_2/*", "transformer/*", "vae/*"]
EXPECTED_BYTES = 33.7e9
JOB_TOKEN = os.getenv("CLOVA_JOB_TOKEN", "")
WORK_DIR = Path(tempfile.gettempdir()) / "clova-images"
JOBS: dict[str, dict] = {}


def require_token(authorization: str | None) -> None:
    if not JOB_TOKEN:
        return
    expected = f"Bearer {JOB_TOKEN}"
    if not authorization or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Token non valido")


def load_model() -> None:
    global PIPE, PHASE, LOAD_ERROR
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU CUDA non disponibile")
        PHASE = "downloading"
        snapshot_download(REPO, local_dir=MODEL_DIR, allow_patterns=ALLOW, token=os.getenv("HF_TOKEN") or None)
        PHASE = "loading"
        PIPE = FluxPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16).to("cuda")
        PHASE = "ready"
    except Exception as exc:  # Reported through /health so the orchestrator can give up early.
        message = str(exc)
        if any(code in message for code in ("401", "403")) or "gated" in message.lower():
            message = ("Accesso a FLUX.1-dev negato: accetta la licenza su huggingface.co/black-forest-labs/FLUX.1-dev "
                       "con il tuo account e controlla il token Hugging Face nell'app.")
        LOAD_ERROR = message
        PHASE = "error"


@app.on_event("startup")
def startup() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=load_model, daemon=True).start()


def downloaded_bytes() -> int:
    return sum(path.stat().st_size for path in MODEL_DIR.rglob("*") if path.is_file()) if MODEL_DIR.exists() else 0


@app.get("/health")
def health() -> dict[str, object]:
    progress = round(min(1.0, downloaded_bytes() / EXPECTED_BYTES), 3) if PHASE == "downloading" else None
    return {
        "ok": LOAD_ERROR is None, "ready": PIPE is not None, "error": LOAD_ERROR,
        "phase": PHASE, "download": progress, "cuda": torch.cuda.is_available(), "model": REPO,
    }


def run_job(job_id: str, prompt: str, width: int, height: int, count: int, seed: int,
            steps: int, guidance: float) -> None:
    job = JOBS[job_id]
    folder = WORK_DIR / job_id
    try:
        with GPU_LOCK:
            job.update(status="generating", total=count, done=0)
            seeds = []
            for index in range(count):
                image_seed = seed + index
                image = PIPE(
                    prompt=prompt, width=width, height=height, num_inference_steps=steps,
                    guidance_scale=guidance, max_sequence_length=512,
                    generator=torch.Generator("cuda").manual_seed(image_seed),
                ).images[0]
                image.save(folder / f"img-{index + 1}.png")
                seeds.append(image_seed)
                job["done"] = index + 1

        with zipfile.ZipFile(folder / "immagini.zip", "w", zipfile.ZIP_STORED) as archive:
            for index, image_seed in enumerate(seeds, start=1):
                archive.write(folder / f"img-{index}.png", f"immagine-{index}-seed{image_seed}.png")
            archive.writestr("prompt.txt", prompt)
        job.update(status="done", seeds=seeds, width=width, height=height,
                   elapsed=round(time.time() - job["started"], 1))
    except Exception as exc:
        job.update(status="error", error=f"Generazione non riuscita: {exc}")
    finally:
        torch.cuda.empty_cache()


@app.post("/images")
def create_job(
    prompt: str = Form(...),
    width: int = Form(1024),
    height: int = Form(1024),
    count: int = Form(1),
    seed: int = Form(-1),
    steps: int = Form(28),
    guidance: float = Form(3.5),
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    require_token(authorization)
    if PIPE is None:
        raise HTTPException(status_code=503, detail=LOAD_ERROR or "Modello non ancora caricato")
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Serve una descrizione dell'immagine")
    # FLUX works on multiples of 16 pixels.
    width, height = (min(max(256, value // 16 * 16), 2048) for value in (width, height))
    seed = seed if seed >= 0 else secrets.randbelow(2**31)

    job_id = secrets.token_hex(12)
    (WORK_DIR / job_id).mkdir(parents=True)
    JOBS[job_id] = {"status": "queued", "done": 0, "total": 0, "started": time.time()}
    threading.Thread(
        target=run_job,
        args=(job_id, prompt, width, height, min(max(count, 1), 4), seed,
              min(max(steps, 8), 50), min(max(guidance, 1.0), 10.0)),
        daemon=True,
    ).start()
    return {"id": job_id}


def get_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job non trovato")
    return job


@app.get("/images/{job_id}")
def job_status(job_id: str, authorization: str | None = Header(default=None)) -> dict:
    require_token(authorization)
    return get_job(job_id)


@app.get("/images/{job_id}/files/{name}")
def job_file(job_id: str, name: str, authorization: str | None = Header(default=None)) -> FileResponse:
    require_token(authorization)
    job = get_job(job_id)
    allowed = {"immagini.zip"} | {f"img-{i}.png" for i in range(1, job.get("total", 0) + 1)}
    if job["status"] != "done" or name not in allowed:
        raise HTTPException(status_code=404, detail="File non disponibile")
    return FileResponse(WORK_DIR / job_id / name)
