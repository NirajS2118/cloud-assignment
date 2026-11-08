# Service container for Command Parser
# Multi-stage or lightweight slim base to ensure image <= 2 GB
FROM python:3.11-slim

WORKDIR /app

# Ensure curl / certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pinned dependencies (CPU-only PyTorch to minimize size)
COPY service/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (Strictly NO model weights included in image)
COPY service/app.py app.py
COPY service/schema.json schema.json

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
