# -*- coding: utf-8 -*-
"""SENTINEL Worker Training — Google Colab T4 (FREE)

This script trains an IRT (Incident Response Triage) worker agent that learns
to do investigate → classify → diagnose → remediate. When trained, this worker
is BETTER than the scripted baseline but still makes natural LLM mistakes
(hallucinations, scope violations, skipped investigation).

These natural mistakes are what make the 2×2 evaluation matrix compelling:
the trained SENTINEL must catch REAL LLM errors, not just scripted injections.

Run in Google Colab with T4 GPU (free tier).
"""

# ═══════════════════════════════════════════════════════════════════════════
# CELL 1: Setup & Install
# ═══════════════════════════════════════════════════════════════════════════

import os
import subprocess
import sys

# Clone the repo
if not os.path.exists("/content/openEnv"):
    subprocess.run(["git", "clone", "https://github.com/sri11223/openEnv.git", "/content/openEnv"], check=True)
os.chdir("/content/openEnv")

# Install training dependencies (T4-compatible)
subprocess.run([
    sys.executable, "-m", "pip", "install", "--quiet",
    "torch>=2.1.0",
    "transformers>=4.51.0",
    "peft>=0.15.0",
    "trl>=0.25.0",
    "bitsandbytes>=0.45.0",
    "datasets>=3.4.1",
    "accelerate>=1.0.0",
    "matplotlib>=3.10.0",
    "wandb>=0.19.0",
    "-r", "requirements.txt",
], check=True)

print("✅ Dependencies installed")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 2: Configure Worker Training
# ═══════════════════════════════════════════════════════════════════════════

# Worker model: small enough for T4 16GB
WORKER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # 1.5B fits T4 easily

# Training config
os.environ.update({
    "MODEL_NAME": WORKER_MODEL,
    "USE_SENTINEL": "0",          # IRT mode — train the WORKER, not overseer
    "USE_UNSLOTH": "0",           # Standard HF path
    "TRAIN_STEPS": "100",         # 100 steps is enough for worker
    "WARM_START_STEPS": "10",     # Quick format learning
    "NUM_GENERATIONS": "4",
    "MAX_NEW_TOKENS": "384",
    "LR": "1e-5",                 # Higher LR for small model
    "KL_COEF": "0.05",
    "LORA_R": "8",                # Smaller LoRA for 1.5B model
    "USE_AGENT_MEMORY": "0",      # Workers don't use memory
    "USE_FEEDBACK_MEMORY": "0",
    "WANDB_PROJECT": "sentinel-worker",
    "OUTPUT_DIR": "outputs/worker_checkpoints",
})

print(f"✅ Worker training configured: {WORKER_MODEL}")
print(f"   Mode: IRT (incident response)")
print(f"   Steps: 100")
print(f"   GPU: {os.popen('nvidia-smi --query-gpu=name --format=csv,noheader').read().strip()}")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 3: Run Worker Training
# ═══════════════════════════════════════════════════════════════════════════

os.makedirs("outputs/worker_checkpoints", exist_ok=True)
print("🏃 Starting worker training...")
print("   This takes ~1-2 hours on T4")
print("   Worker learns: investigate → classify → diagnose → remediate")
print("=" * 60)

# Run training
exec(open("train.py").read())

print("=" * 60)
print("✅ Worker training complete!")
print(f"   Model saved: outputs/worker_checkpoints/final")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 4: Export Trained Worker for Evaluation
# ═══════════════════════════════════════════════════════════════════════════

import json
from pathlib import Path

# Save worker training metadata
worker_meta = {
    "model": WORKER_MODEL,
    "steps": 100,
    "mode": "IRT",
    "checkpoint": "outputs/worker_checkpoints/final",
    "description": "Trained IRT worker for 2x2 co-training evaluation",
}
Path("outputs/worker_training_meta.json").write_text(
    json.dumps(worker_meta, indent=2)
)
print("✅ Worker metadata saved")
print("   Next: download outputs/worker_checkpoints/final/ and run 2×2 eval")
