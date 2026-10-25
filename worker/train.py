"""Fine-tune Qwen3-TTS 1.7B Base on one speaker and push it to a private HF repo.

Pipeline: Whisper cuts the uploaded recordings into 3-12 s phrases (uncertain
ones are dropped), clips are written as 24 kHz mono WAV, the official Qwen
scripts in /app/finetuning compute the audio codes and train, and the last
checkpoint is uploaded to Hugging Face so any GPU can download it in seconds.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

FINETUNE_DIR = Path(os.getenv("FINETUNE_DIR", "/app/finetuning"))
SAMPLE_RATE = 24_000  # The official dataset loader asserts 24 kHz reference audio.
CLIP_MIN, CLIP_TARGET_MAX, CLIP_HARD_MAX, MAX_GAP = 3.0, 12.0, 20.0, 1.0
MIN_CONFIDENCE, MAX_NO_SPEECH = -0.6, 0.5


def load_audio(path: Path) -> np.ndarray:
    """Mono 24 kHz, loudness evened out (the recordings sit around -28 LUFS)."""
    import librosa
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    peak = float(np.percentile(np.abs(audio), 99.9)) or 1.0
    return np.clip(audio * min(0.9 / peak, 10.0), -1.0, 1.0).astype("float32")


def find_clips(transcriber, path: Path, language_code: str | None, clean_text: Callable[[str], str]) -> list[dict]:
    """Group Whisper phrases into clips of up to ~12 s, split at pauses. Uncertain
    phrases are dropped one by one and act as a cut, so good neighbours survive."""
    segments, _ = transcriber.transcribe(str(path), language=language_code, vad_filter=True, beam_size=5)
    groups: list[list] = []
    current: list = []
    for segment in segments:
        if not segment.text.strip() or segment.avg_logprob < MIN_CONFIDENCE or segment.no_speech_prob >= MAX_NO_SPEECH:
            if current:
                groups.append(current)
                current = []
            continue
        if current and (segment.end - current[0].start > CLIP_TARGET_MAX or segment.start - current[-1].end > MAX_GAP):
            groups.append(current)
            current = []
        current.append(segment)
    if current:
        groups.append(current)
    return [{
        "start": group[0].start, "end": group[-1].end,
        "text": clean_text(" ".join(s.text.strip() for s in group)),
        "confidence": sum(s.avg_logprob for s in group) / len(group),
        "no_speech": max(s.no_speech_prob for s in group),
    } for group in groups]


def usable(clip: dict) -> bool:
    duration = clip["end"] - clip["start"]
    return (CLIP_MIN <= duration <= CLIP_HARD_MAX and clip["confidence"] >= MIN_CONFIDENCE
            and clip["no_speech"] < MAX_NO_SPEECH and len(clip["text"]) >= 8)


def run_logged(command: list[str], job: dict, log_path: Path, on_line: Callable[[str], None] | None = None) -> None:
    with log_path.open("a", encoding="utf8") as log:
        process = subprocess.Popen(command, cwd=FINETUNE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        for line in process.stdout:
            log.write(line)
            job["log"] = (job.get("log", []) + [line.rstrip()])[-12:]
            if on_line:
                on_line(line)
        if process.wait() != 0:
            raise RuntimeError(f"{Path(command[1]).name} terminato con errore: " + " | ".join(job.get("log", [])[-4:]))


def patched_sft_script(attention: str) -> str:
    """The official script hard-codes FlashAttention 2; fall back to SDPA when it is missing."""
    source = (FINETUNE_DIR / "sft_12hz.py").read_text(encoding="utf8")
    if attention != "flash_attention_2":
        source = source.replace('"flash_attention_2"', f'"{attention}"').replace("'flash_attention_2'", f"'{attention}'")
    target = FINETUNE_DIR / "sft_clova.py"
    target.write_text(source, encoding="utf8")
    return target.name


def train(job: dict, files: list[Path], folder: Path, *, transcriber, clean_text: Callable[[str], str],
          language: str, language_code: str | None, speaker: str, repo: str, token: str, epochs: int,
          base_model: str, attention: str) -> None:
    clips_dir = folder / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    log_path = folder / "train.log"

    # 1. Cut the recordings into phrases.
    records, dropped, seconds = [], 0, 0.0
    for number, path in enumerate(files, start=1):
        job.update(phase="segmenting", detail=f"Taglio in frasi: file {number} di {len(files)}")
        audio = load_audio(path)
        clips = find_clips(transcriber, path, language_code, clean_text)
        for index, clip in enumerate(clips):
            if not usable(clip):
                dropped += 1
                continue
            # Pad a little, never into the neighbouring phrases.
            before = clips[index - 1]["end"] if index else 0.0
            after = clips[index + 1]["start"] if index + 1 < len(clips) else len(audio) / SAMPLE_RATE
            start = max(clip["start"] - 0.1, (before + clip["start"]) / 2)
            end = min(clip["end"] + 0.2, (clip["end"] + after) / 2)
            wav = clips_dir / f"clip-{len(records) + 1:04d}.wav"
            sf.write(wav, audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)], SAMPLE_RATE)
            records.append({"audio": str(wav), "text": clip["text"], "language": language,
                            "confidence": clip["confidence"], "duration": end - start})
            seconds += end - start
    if len(records) < 30:
        raise RuntimeError(f"Solo {len(records)} frasi utilizzabili: servono registrazioni più lunghe o più pulite.")
    job.update(clips=len(records), dropped=dropped, minutes=round(seconds / 60, 1))

    # Same reference for every sample (recommended by Qwen): the clearest 8-12 s clip.
    candidates = [r for r in records if 8 <= r["duration"] <= 12] or records
    reference = folder / "ref.wav"
    shutil.copy(max(candidates, key=lambda r: r["confidence"])["audio"], reference)
    raw = folder / "train_raw.jsonl"
    with raw.open("w", encoding="utf8") as handle:
        for record in records:
            handle.write(json.dumps({"audio": record["audio"], "text": record["text"], "ref_audio": str(reference),
                                     "language": language}, ensure_ascii=False) + "\n")

    # 2. Audio codes with the official 12 Hz tokenizer.
    job.update(phase="codes", detail=f"Calcolo dei codici audio di {len(records)} frasi…")
    codes = folder / "train_with_codes.jsonl"
    run_logged([sys.executable, "prepare_data.py", "--device", "cuda:0",
                "--input_jsonl", str(raw), "--output_jsonl", str(codes)], job, log_path)

    # 3. Training.
    output = folder / "output"
    batch_size = 2
    steps_per_epoch = -(-len(records) // batch_size)
    job.update(phase="training", epoch=0, epochs=epochs, step=0, total_steps=steps_per_epoch * epochs,
               detail=f"Addestramento: epoca 1 di {epochs}")

    def on_line(line: str) -> None:
        loss = re.search(r"loss[^0-9\-]*(-?[0-9]+\.[0-9]+)", line, re.IGNORECASE)
        step = re.search(r"step[^0-9]*([0-9]+)", line, re.IGNORECASE)
        if loss:
            job["loss"] = float(loss.group(1))
        if step:
            job["step"] = int(step.group(1))
        done = len(list(output.glob("checkpoint-epoch-*"))) if output.exists() else 0
        job["epoch"] = done
        job["detail"] = f"Addestramento: epoca {min(done + 1, epochs)} di {epochs}" + (f" · loss {job['loss']:.3f}" if "loss" in job else "")

    run_logged([sys.executable, patched_sft_script(attention), "--init_model_path", base_model,
                "--output_model_path", str(output), "--train_jsonl", str(codes),
                "--batch_size", str(batch_size), "--lr", "2e-5", "--num_epochs", str(epochs),
                "--speaker_name", speaker], job, log_path, on_line)
    checkpoints = sorted(output.glob("checkpoint-epoch-*"), key=lambda p: int(p.name.rsplit("-", 1)[-1]))
    if not checkpoints:
        raise RuntimeError("L'addestramento non ha prodotto nessun checkpoint.")

    # 4. Private Hugging Face repo.
    job.update(phase="uploading", epoch=epochs, detail=f"Caricamento del modello su Hugging Face ({repo}, privato)…")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo, private=True, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=str(checkpoints[-1]), repo_id=repo, repo_type="model",
                      commit_message=f"Clova Voice: {len(records)} frasi, {epochs} epoche")
    job.update(phase="done", repo=repo, speaker=speaker, detail="Modello addestrato e salvato.")
