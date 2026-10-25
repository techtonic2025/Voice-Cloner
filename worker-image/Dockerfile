FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

# The FLUX weights (~34 GB) are downloaded at startup with HF_TOKEN: GitHub's
# builders cannot hold an image that size. hf_transfer makes the download fast.
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV CLOVA_JOB_TOKEN=""
ENV HF_TOKEN=""
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
