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


GRADIENT_ACCUMULATION = 8  # With batch 4: 32 phrases per optimizer step, as recommended.

# The official script trains on inputs the model never sees at inference, which makes
# fine-tuned voices output pure noise (QwenLM/Qwen3-TTS issues #39, #179, #371):
SFT_FIXES = [
    # 1. Inference always passes text embeddings through talker.text_projection; training did not
    #    (silent on 1.7B, where text_hidden_size == hidden_size == 2048).
    ("model.talker.model.text_embedding(input_text_ids) * text_embedding_mask",
     "model.talker.text_projection(model.talker.model.text_embedding(input_text_ids)) * text_embedding_mask"),
    # 2. Labels were shifted here and again inside the HF causal-LM loss: frame t+2 was learned.
    ("labels=codec_0_labels[:, 1:],", "labels=codec_0_labels[:, :-1],"),
    # 3. The sub-talker got the hidden state that had already seen the frame it must predict;
    #    generation uses the previous position.
    ("hidden_states[codec_mask[:, :-1]]", "hidden_states[codec_mask[:, 1:]]"),
    # Smaller default step size and a larger effective batch (see train()).
    ("gradient_accumulation_steps=4", f"gradient_accumulation_steps={GRADIENT_ACCUMULATION}"),
]

# 4. The code predictor's finetune loss goes through the HF causal-LM loss, which shifts the
#    15 already-aligned codebook labels by one: plain cross-entropy instead.
SUB_TALKER_FIX = '''
import torch.nn.functional as _clova_F
from qwen_tts.core.models import modeling_qwen3_tts as _clova_modeling
_clova_cls = _clova_modeling.Qwen3TTSTalkerCodePredictorModelForConditionalGeneration
_clova_original_forward_finetune = _clova_cls.forward_finetune
def _clova_forward_finetune(self, *args, labels=None, **kwargs):
    output = _clova_original_forward_finetune(self, *args, labels=None, **kwargs)
    if labels is not None:
        logits = output.logits
        output = type(output)(loss=_clova_F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1)),
                              logits=logits)
    return output
_clova_cls.forward_finetune = _clova_forward_finetune
'''


def patched_sft_script(attention: str) -> str:
    """Write sft_clova.py: the official script with the fixes above, TensorBoard logging off
    (recent Accelerate requires a logging dir and the script never logs anything), and
    SDPA instead of FlashAttention 2 when that is missing. Fails loudly if upstream changed."""
    source = (FINETUNE_DIR / "sft_12hz.py").read_text(encoding="utf8")
    source = re.sub(r"log_with\s*=\s*[\"']tensorboard[\"']", "log_with=None", source)
    for old, new in SFT_FIXES:
        if source.count(old) != 1:
            raise RuntimeError(f"Lo script ufficiale di Qwen è cambiato: non trovo «{old}». Va aggiornata la correzione.")
        source = source.replace(old, new)
    if source.count("\ndef train():") != 1:
        raise RuntimeError("Lo script ufficiale di Qwen è cambiato: non trovo dove inserire la correzione del sub-talker.")
    source = source.replace("\ndef train():", SUB_TALKER_FIX + "\ndef train():", 1)
    if attention != "flash_attention_2":
        source = source.replace('"flash_attention_2"', f'"{attention}"').replace("'flash_attention_2'", f"'{attention}'")
    target = FINETUNE_DIR / "sft_clova.py"
    target.write_text(source, encoding="utf8")
    return target.name


TEST_SENTENCES = [
    "Ciao a tutti e bentornati sul canale: oggi vediamo insieme una novità davvero interessante.",
    "Se il video vi è piaciuto, lasciate un like e iscrivetevi per non perdere i prossimi.",
]


SPEAKER_MODEL = os.getenv("SPEAKER_MODEL_PATH", "/models/wavlm-base-plus-sv")
MIN_FIDELITY = 0.9  # Below this an epoch is only kept if nothing better exists.


class SpeakerMeter:
    """Speaker similarity with WavLM-SV (independent of Qwen): cosine between x-vectors,
    the usual way TTS papers measure how close a clone is to the real speaker."""

    def __init__(self) -> None:
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.torch = torch
        self.extractor = AutoFeatureExtractor.from_pretrained(SPEAKER_MODEL)
        self.model = WavLMForXVector.from_pretrained(SPEAKER_MODEL).to("cuda").eval()

    def embed(self, wavs: list[np.ndarray], sample_rate: int):
        audio16 = [np.interp(np.linspace(0, len(w) - 1, int(len(w) * 16_000 / sample_rate)), np.arange(len(w)), w).astype("float32")
                   for w in wavs]
        inputs = self.extractor(audio16, sampling_rate=16_000, return_tensors="pt", padding=True).to("cuda")
        with self.torch.no_grad():
            embeddings = self.model(**inputs).embeddings
        return self.torch.nn.functional.normalize(embeddings, dim=-1)

    def reference(self, paths: list[str]):
        """Average voiceprint of real recordings."""
        wavs = [sf.read(path, dtype="float32")[0] for path in paths]
        return self.torch.nn.functional.normalize(self.embed(wavs, SAMPLE_RATE).mean(dim=0), dim=-1)

    def similarity(self, wav: np.ndarray, sample_rate: int, reference) -> float:
        return float(self.embed([wav], sample_rate)[0] @ reference)


def evaluate(wavs, sample_rate: int, language: str, score, meter: SpeakerMeter, reference, samples: Path, label: str) -> dict:
    """Fidelity (Whisper) and similarity (WavLM) of the test sentences; keeps the audio for listening."""
    fidelity, similarity = [], []
    for index, (wav, text) in enumerate(zip(wavs, TEST_SENTENCES), start=1):
        wav = np.asarray(wav, dtype="float32").reshape(-1)
        sf.write(samples / f"{label}-{index}.wav", wav, sample_rate)
        fidelity.append(score(wav, sample_rate, text, language))
        similarity.append(meter.similarity(wav, sample_rate, reference))
    return {"fidelity": round(sum(fidelity) / len(fidelity), 3), "similarity": round(sum(similarity) / len(similarity), 3)}


def checkpoint_metrics(checkpoint: Path, speaker: str, language: str, load_model, score, meter, reference, samples, label) -> dict:
    """Let a checkpoint read two sentences it never saw, then measure it."""
    import torch
    model = load_model(str(checkpoint))
    try:
        wavs, sample_rate = model.generate_custom_voice(
            text=TEST_SENTENCES, speaker=speaker, language=[language] * len(TEST_SENTENCES),
            max_new_tokens=600, do_sample=True, temperature=0.6, top_p=0.9,
        )
        return evaluate(wavs, sample_rate, language, score, meter, reference, samples, label)
    finally:
        del model
        torch.cuda.empty_cache()


def train(job: dict, files: list[Path], folder: Path, *, transcriber, clean_text: Callable[[str], str],
          language: str, language_code: str | None, speaker: str, repo: str, token: str, epochs: int,
          base_model: str, attention: str, load_model, score, clone_model) -> None:
    script = patched_sft_script(attention)  # Before any GPU work, so an upstream change fails fast.
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
    reference_record = max(candidates, key=lambda r: r["confidence"])
    shutil.copy(reference_record["audio"], reference)
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
    batch_size = 4
    steps_per_epoch = -(-len(records) // batch_size)
    job.update(phase="training", epoch=0, epochs=epochs, step=0, total_steps=steps_per_epoch * epochs,
               detail=f"Addestramento: epoca 1 di {epochs}")

    def on_line(line: str) -> None:
        # The official script prints "Epoch 0 | Step 120 | Loss: 4.1234" every 10 steps.
        progress = re.search(r"Epoch\s+(\d+)\s*\|\s*Step\s+(\d+)\s*\|\s*Loss:\s*(-?[0-9.]+)", line)
        if not progress:
            return
        epoch, step, loss = int(progress.group(1)), int(progress.group(2)), float(progress.group(3))
        # Fractional epochs, so the progress bar moves within an epoch.
        job.update(epoch=round(epoch + min(step / steps_per_epoch, 1.0), 3), step=step, loss=loss,
                   detail=f"Addestramento: epoca {epoch + 1} di {epochs} · passo {step} di {steps_per_epoch} · loss {loss:.3f}")

    # 2e-6, not the official 2e-5: the higher rate is widely reported to collapse the 1.7B model into noise.
    run_logged([sys.executable, script, "--init_model_path", base_model,
                "--output_model_path", str(output), "--train_jsonl", str(codes),
                "--batch_size", str(batch_size), "--lr", "2e-6", "--num_epochs", str(epochs),
                "--speaker_name", speaker], job, log_path, on_line)
    checkpoints = sorted(output.glob("checkpoint-epoch-*"), key=lambda p: int(p.name.rsplit("-", 1)[-1]))
    if not checkpoints:
        raise RuntimeError("L'addestramento non ha prodotto nessun checkpoint.")

    # 4. Each epoch reads two unseen sentences: Whisper measures fidelity to the text, WavLM
    #    how close the voice is to the real recordings. The quick clone from the reference
    #    clip is measured too, as the baseline the training has to beat.
    samples = folder / "samples"
    samples.mkdir(exist_ok=True)
    job.update(phase="checking", detail="Preparo il confronto con la tua voce vera…")
    meter = SpeakerMeter()
    rng = np.random.default_rng(0)
    real_clips = [records[i]["audio"] for i in rng.choice(len(records), size=min(24, len(records)), replace=False)]
    voiceprint = meter.reference(real_clips)
    for index, path in enumerate(real_clips[:2], start=1):
        shutil.copy(path, samples / f"vera-{index}.wav")  # The real voice, to listen side by side.

    import torch
    prompt = clone_model.create_voice_clone_prompt(ref_audio=str(reference), ref_text=reference_record["text"])
    wavs, sample_rate = clone_model.generate_voice_clone(
        text=TEST_SENTENCES, language=[language] * len(TEST_SENTENCES), voice_clone_prompt=prompt,
        max_new_tokens=600, do_sample=True, temperature=0.6, top_p=0.9,
    )
    baseline = evaluate(wavs, sample_rate, language, score, meter, voiceprint, samples, "clone")
    job["baseline"] = baseline
    torch.cuda.empty_cache()

    metrics: dict[str, dict] = {}
    for number, checkpoint in enumerate(checkpoints, start=1):
        job.update(detail=f"Prova d'ascolto dell'epoca {number} di {len(checkpoints)}: fedeltà al testo e somiglianza alla tua voce…")
        metrics[str(number)] = checkpoint_metrics(checkpoint, speaker, language, load_model, score, meter, voiceprint,
                                                  samples, f"epoca-{number:02d}")
        job["epoch_scores"] = metrics
    del meter
    torch.cuda.empty_cache()
    job["samples"] = sorted(path.name for path in samples.glob("*.wav"))

    # Among the epochs that read correctly, keep the one that sounds most like you;
    # ties go to the latest epoch, which has adapted most.
    readable = [n for n in metrics if metrics[n]["fidelity"] >= MIN_FIDELITY]
    pool = readable or list(metrics)
    best_number = max(pool, key=lambda n: (metrics[n]["similarity"] if readable else metrics[n]["fidelity"], int(n)))
    best = checkpoints[int(best_number) - 1]
    chosen = metrics[best_number]
    job.update(best_epoch=int(best_number), fidelity=chosen["fidelity"], similarity=chosen["similarity"])
    if chosen["fidelity"] < 0.6:
        raise RuntimeError(f"Nessuna epoca legge in modo comprensibile (fedeltà migliore {chosen['fidelity']:.0%}): "
                           "il modello non viene caricato su Hugging Face.")

    # 5. Private Hugging Face repo.
    summary = f"epoca {best_number} di {epochs}, fedeltà {chosen['fidelity']:.0%}, somiglianza {chosen['similarity']:.0%}"
    job.update(phase="uploading", epoch=epochs, detail=f"Caricamento ({summary}) su Hugging Face ({repo}, privato)…")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo, private=True, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=str(best), repo_id=repo, repo_type="model",
                      commit_message=f"Clova Voice: {len(records)} frasi, {summary}")
    job.update(phase="done", repo=repo, speaker=speaker, detail="Modello addestrato, verificato e salvato.")
