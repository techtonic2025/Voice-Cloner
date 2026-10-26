FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Prebuilt FlashAttention 2 wheel (Python 3.11, torch 2.6, CUDA 12). Which C++ ABI
# matches depends on how torch was built, so both wheels are tried and the one that
# actually imports is kept; if neither does, the worker falls back to SDPA attention.
RUN python -c "import torch, sys; print('torch', torch.__version__, 'cxx11abi', torch.compiled_with_cxx11_abi(), 'python', sys.version)" \
    && for abi in FALSE TRUE; do \
         pip install --no-cache-dir --force-reinstall --no-deps "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abi${abi}-cp311-cp311-linux_x86_64.whl" \
         && python -c "import flash_attn, flash_attn_2_cuda; print('flash-attn OK', flash_attn.__version__)" && break; \
         pip uninstall -y flash-attn; \
       done; \
    python -c "import flash_attn" 2>/dev/null || echo "flash-attn non installato: uso SDPA"
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
