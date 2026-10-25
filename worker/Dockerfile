FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Prebuilt FlashAttention 2 wheel (Python 3.11, torch 2.6, CUDA 12). If it cannot
# be installed the worker falls back to PyTorch SDPA attention.
RUN pip install --no-cache-dir "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" \
    || echo "flash-attn non installato: uso SDPA"
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base', local_dir='/models/qwen3-tts-1.7b-base')"
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-small', local_dir='/models/faster-whisper-small')"
COPY app.py .

ENV PYTHONUNBUFFERED=1
ENV CLOVA_JOB_TOKEN=""
ENV CLOVA_BATCH_SIZE="4"
ENV QWEN_MODEL_PATH="/models/qwen3-tts-1.7b-base"
ENV WHISPER_MODEL_PATH="/models/faster-whisper-small"
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
