# Single-stage build — avoids pulling the same base image twice (prevents
# manifest-digest cache errors on the validator's Docker daemon).
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create a dedicated non-root user for security
RUN useradd -m -u 1000 -s /sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

# HF Spaces requires port 7860
EXPOSE 7860

# Enable visual web testing at /web (bootcamp requirement)
ENV ENABLE_WEB_INTERFACE=true

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health').read()"

# Single worker — session state is in-process
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
