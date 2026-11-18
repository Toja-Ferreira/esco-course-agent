# ESCO skill assistant (web app + MCP server) for the university VM.
# The VM's own Python is 3.8; the app needs 3.10+, so it runs in this image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf HOME=/tmp

WORKDIR /app

# CPU-only torch first, so sentence-transformers does not pull the CUDA build
# (the 300M embedding model only encodes short queries here).
COPY requirements.txt .
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

EXPOSE 8090
CMD ["python", "-m", "esco_agent.web"]
