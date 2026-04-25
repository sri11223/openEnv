# Single-stage build - avoids pulling the same base image twice (prevents
# manifest-digest cache errors on the validator's Docker daemon).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    ENABLE_WEB_INTERFACE=true \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache

WORKDIR /app

# Install dependencies first (layer cache friendly)
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy application source as a numeric non-root owner. This avoids a fragile
# useradd/chown build layer on Hugging Face Spaces while still avoiding root.
COPY --chown=1000:1000 . .

USER 1000

# HF Spaces requires port 7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; port=os.environ.get('PORT','7860'); urllib.request.urlopen(f'http://localhost:{port}/health').read()"

# Single worker - session state is in-process. server.app reads $PORT.
CMD ["python", "-m", "server.app"]
