FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Prebuilt FlashAttention 2 wheel (Python 3.11, torch 2.6, CUDA 12). If it cannot
# be installed the worker falls back to PyTorch SDPA attention.
RUN pip install --no-cache-dir "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" \
    || echo "flash-attn non installato: uso SDPA"
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base', local_dir='/models/qwen3-tts-1.7b-base')"
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('mobiuslabsgmbh/faster-whisper-large-v3-turbo', local_dir='/models/faster-whisper-large-v3-turbo')"
# Speaker verification model, used by /train to measure how close each epoch is to the real voice.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/wavlm-base-plus-sv', local_dir='/models/wavlm-base-plus-sv')"
# Official Qwen3-TTS fine-tuning scripts, used by /train.
RUN mkdir -p /app/finetuning && python -c "import urllib.request as u; [u.urlretrieve(f'https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/finetuning/{n}', f'/app/finetuning/{n}') for n in ('sft_12hz.py', 'dataset.py', 'prepare_data.py')]"
COPY app.py train.py ./

ENV PYTHONUNBUFFERED=1
ENV CLOVA_JOB_TOKEN=""
ENV CLOVA_BATCH_SIZE="4"
ENV QWEN_MODEL_PATH="/models/qwen3-tts-1.7b-base"
ENV WHISPER_MODEL_PATH="/models/faster-whisper-large-v3-turbo"
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
