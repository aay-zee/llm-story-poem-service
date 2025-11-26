FROM python:3.10

# Prevent python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Set Hugging Face cache directory
ENV HF_HOME=/app/model_cache

# Download models during build
COPY download_models.py .
RUN python download_models.py

COPY . .

# Use shell form so the container will respect the runtime $PORT value if provided by the platform.
# Cloud Run provides PORT env; default to 8080 when not set.
ENTRYPOINT ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
