FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base', local_dir='/models/qwen3-tts-1.7b-base')"
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-small', local_dir='/models/faster-whisper-small')"
COPY app.py .

ENV PYTHONUNBUFFERED=1
ENV CLOVA_JOB_TOKEN=""
ENV QWEN_MODEL_PATH="/models/qwen3-tts-1.7b-base"
ENV WHISPER_MODEL_PATH="/models/faster-whisper-small"
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
