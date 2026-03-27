# ── Stage 1: dependency builder ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
# Install into an isolated prefix so the runtime layer stays minimal
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: lean runtime (non-root) ────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy pre-built packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Create a dedicated non-root user for security
RUN useradd -m -u 1000 -s /sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

# HF Spaces requires port 7860
EXPOSE 7860

# Use urllib.request (stdlib) so we don't need extra deps for the health probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/').read()"

# Single worker — session state is in-process; scale via HF Spaces replicas
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
