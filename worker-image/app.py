"""Temporary FLUX.1 [dev] image worker used by Clova Voice.

Flow: at startup the weights are downloaded from Hugging Face (gated repo: the
user accepts the license and the app passes HF_TOKEN) and loaded in bf16; the
orchestrator polls /health until ready, submits /images, polls /images/{id}
and downloads the PNG files.

A request with an image attached is an edit: FLUX.1 Kontext [dev] is downloaded on
the first one (transformer only, the rest is shared with FLUX.1-dev) and the two
transformers take turns on the GPU, the idle one waiting in system RAM.
"""

from __future__ import annotations

import os
import secrets
import tempfile
import threading
import time
import zipfile
from io import BytesIO
from pathlib import Path

import torch
from diffusers import FluxKontextPipeline, FluxPipeline, FluxTransformer2DModel
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image

app = FastAPI(title="Clova Voice FLUX worker", docs_url=None, redoc_url=None)
PIPE: FluxPipeline | None = None
KONTEXT: FluxKontextPipeline | None = None
PHASE = "starting"
LOAD_ERROR: str | None = None
GPU_LOCK = threading.Lock()
REPO = os.getenv("FLUX_REPO", "black-forest-labs/FLUX.1-dev")
MODEL_DIR = Path(os.getenv("FLUX_DIR", "/models/flux1-dev"))
# The repo also ships a 23.8 GB single-file copy at the root: only the diffusers folders are needed.
ALLOW = ["model_index.json", "scheduler/*", "text_encoder/*", "text_encoder_2/*",
         "tokenizer/*", "tokenizer_2/*", "transformer/*", "vae/*"]
EXPECTED_BYTES = 33.7e9
KONTEXT_REPO = os.getenv("KONTEXT_REPO", "black-forest-labs/FLUX.1-Kontext-dev")
KONTEXT_DIR = Path(os.getenv("KONTEXT_DIR", "/models/flux1-kontext-dev"))
KONTEXT_BYTES = 23.8e9
# LoRAs fused at startup, "repo:file:scale" separated by commas (set by the app; empty = plain FLUX).
def lora_list(variable: str) -> list[list[str]]:
    return [entry.split(":") for entry in os.getenv(variable, "").split(",") if entry.count(":") == 2]


LORAS = lora_list("FLUX_LORAS")
# Kontext gets its own, usually lighter, list: strong LoRAs reshape the body being edited.
EDIT_LORAS = lora_list("FLUX_EDIT_LORAS")
LORA_ERROR: str | None = None
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
        pipe = FluxPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16).to("cuda")
        load_loras(pipe, LORAS)
        PIPE = pipe
        PHASE = "ready"
    except Exception as exc:  # Reported through /health so the orchestrator can give up early.
        message = str(exc)
        if any(code in message for code in ("401", "403")) or "gated" in message.lower():
            message = ("Accesso a FLUX.1-dev negato: accetta la licenza su huggingface.co/black-forest-labs/FLUX.1-dev "
                       "con il tuo account e controlla il token Hugging Face nell'app.")
        LOAD_ERROR = message
        PHASE = "error"


def load_loras(pipe: FluxPipeline | FluxKontextPipeline, entries: list[list[str]]) -> None:
    """Fuse the LoRAs into the weights, so generation runs as fast as plain FLUX.
    A LoRA that fails to load is reported in /health and skipped: base FLUX still works."""
    global LORA_ERROR
    names, weights = [], []
    for index, (repo, filename, scale) in enumerate(entries):
        try:
            path = hf_hub_download(repo, filename, token=os.getenv("HF_TOKEN") or None)
            pipe.load_lora_weights(path, adapter_name=f"lora{index}")
            names.append(f"lora{index}")
            weights.append(float(scale))
        except Exception as exc:
            LORA_ERROR = f"{repo}: {exc}"
    if names:
        pipe.set_adapters(names, adapter_weights=weights)
        pipe.fuse_lora(adapter_names=names)
        pipe.unload_lora_weights()


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
        "loras": [repo for repo, _, _ in LORAS], "edit_loras": [repo for repo, _, _ in EDIT_LORAS], "lora_error": LORA_ERROR,
    }


def kontext_downloaded() -> int:
    return sum(path.stat().st_size for path in KONTEXT_DIR.rglob("*") if path.is_file()) if KONTEXT_DIR.exists() else 0


def kontext_pipeline(job: dict) -> FluxKontextPipeline:
    """Built on the first edit, sharing text encoders and VAE with FLUX.1-dev."""
    global KONTEXT
    if KONTEXT is None:
        job.update(status="preparing", download=0.0)
        watcher = threading.Event()

        def progress() -> None:
            while not watcher.wait(3):
                job["download"] = round(min(1.0, kontext_downloaded() / KONTEXT_BYTES), 3)

        threading.Thread(target=progress, daemon=True).start()
        try:
            snapshot_download(KONTEXT_REPO, local_dir=KONTEXT_DIR, allow_patterns=["transformer/*"],
                              token=os.getenv("HF_TOKEN") or None)
        except Exception as exc:
            message = str(exc)
            if any(code in message for code in ("401", "403")) or "gated" in message.lower():
                raise RuntimeError("Accesso a FLUX.1-Kontext-dev negato: accetta la licenza su "
                                   "huggingface.co/black-forest-labs/FLUX.1-Kontext-dev con il tuo account.") from exc
            raise
        finally:
            watcher.set()
        job.update(status="generating", download=None)
        transformer = FluxTransformer2DModel.from_pretrained(KONTEXT_DIR / "transformer", torch_dtype=torch.bfloat16)
        PIPE.transformer.to("cpu")
        torch.cuda.empty_cache()
        KONTEXT = FluxKontextPipeline(
            transformer=transformer.to("cuda"), scheduler=PIPE.scheduler, vae=PIPE.vae,
            text_encoder=PIPE.text_encoder, tokenizer=PIPE.tokenizer,
            text_encoder_2=PIPE.text_encoder_2, tokenizer_2=PIPE.tokenizer_2,
        )
        load_loras(KONTEXT, EDIT_LORAS)
    return KONTEXT


def on_gpu(wanted: FluxTransformer2DModel, idle: FluxTransformer2DModel | None) -> None:
    """Only one 24 GB transformer fits next to the shared encoders: swap through system RAM."""
    if idle is not None and idle.device.type == "cuda":
        idle.to("cpu")
        torch.cuda.empty_cache()
    if wanted.device.type != "cuda":
        wanted.to("cuda")


def run_job(job_id: str, prompt: str, width: int, height: int, count: int, seed: int,
            steps: int, guidance: float, source: Image.Image | None = None) -> None:
    job = JOBS[job_id]
    folder = WORK_DIR / job_id
    try:
        with GPU_LOCK:
            if source is not None:
                pipe = kontext_pipeline(job)
                on_gpu(pipe.transformer, PIPE.transformer)
            else:
                pipe = PIPE
                on_gpu(PIPE.transformer, KONTEXT.transformer if KONTEXT else None)
            job.update(status="generating", total=count, done=0)
            seeds = []
            for index in range(count):
                image_seed = seed + index
                size = {"image": source} if source is not None else {"width": width, "height": height}
                image = pipe(
                    prompt=prompt, num_inference_steps=steps, **size,
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
        width, height = image.size
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
    guidance: float = Form(-1),
    image: UploadFile | None = File(None),
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
    source = Image.open(BytesIO(image.file.read())).convert("RGB") if image is not None else None
    # Kontext follows edit instructions best at a lower guidance than text-to-image.
    guidance = guidance if guidance > 0 else (2.5 if source is not None else 3.5)

    job_id = secrets.token_hex(12)
    (WORK_DIR / job_id).mkdir(parents=True)
    JOBS[job_id] = {"status": "queued", "done": 0, "total": 0, "started": time.time()}
    threading.Thread(
        target=run_job,
        args=(job_id, prompt, width, height, min(max(count, 1), 4), seed,
              min(max(steps, 8), 50), min(max(guidance, 1.0), 10.0), source),
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
