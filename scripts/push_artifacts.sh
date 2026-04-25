#!/usr/bin/env bash
# push_artifacts.sh — collect training proof artifacts and push to GitHub + HF Hub.
# Designed to run AFTER train.py succeeds. Safe to run multiple times.
#
# Required env vars:
#   GITHUB_TOKEN     PAT with `repo` scope
#   GITHUB_REPO_URL  e.g. https://github.com/your-user/openEnv.git
#   GIT_USER_NAME    Display name for commits
#   GIT_USER_EMAIL   Email for commits
#
# Optional env vars:
#   HF_TOKEN         HuggingFace token with write access
#   HF_REPO          e.g. your-user/sentinel-qwen3-4b-grpo
#   OUTPUT_DIR       Where train.py wrote results (default /data/sentinel_outputs)
#   ARTIFACTS_DIR    Where artifacts are staged in the repo (default outputs/proof_pack)
#   COMMIT_MSG       Custom commit message
#
set -uo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/data/sentinel_outputs}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-outputs/proof_pack}"
COMMIT_MSG="${COMMIT_MSG:-chore(training): publish 200-step GRPO proof pack}"

echo "=========================================="
echo "  push_artifacts.sh"
echo "=========================================="
echo "OUTPUT_DIR    = $OUTPUT_DIR"
echo "ARTIFACTS_DIR = $ARTIFACTS_DIR"
echo "HF_REPO       = ${HF_REPO:-(skipped)}"
echo ""

# ----------------------------------------------------------------------
# 1. Sanity checks
# ----------------------------------------------------------------------
if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "[skip] GITHUB_TOKEN not set — cannot push to GitHub."
  echo "        Set GITHUB_TOKEN, GITHUB_REPO_URL, GIT_USER_NAME, GIT_USER_EMAIL."
fi

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "[fatal] OUTPUT_DIR does not exist: $OUTPUT_DIR"
  exit 1
fi

# ----------------------------------------------------------------------
# 2. Stage artifacts (small/important files only — never the full base model)
# ----------------------------------------------------------------------
mkdir -p "$ARTIFACTS_DIR"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -e "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    cp -r "$src" "$dst"
    echo "  + $src -> $dst"
  else
    echo "  - $src (not present, skipped)"
  fi
}

echo "=== Staging artifacts ==="
copy_if_exists "$OUTPUT_DIR/final"                                  "$ARTIFACTS_DIR/final"
copy_if_exists "$OUTPUT_DIR/monitoring/training_metrics.jsonl"      "$ARTIFACTS_DIR/training_metrics.jsonl"
copy_if_exists "$OUTPUT_DIR/monitoring/stack_versions.json"         "$ARTIFACTS_DIR/stack_versions.json"
copy_if_exists "$OUTPUT_DIR/rollout_audits"                         "$ARTIFACTS_DIR/rollout_audits"
copy_if_exists "$OUTPUT_DIR/warm_start_4b/summary.json"             "$ARTIFACTS_DIR/warm_start_summary.json"
copy_if_exists "outputs/warm_start/summary.json"                    "$ARTIFACTS_DIR/warm_start_summary.json"
copy_if_exists "outputs/reward_curves"                              "$ARTIFACTS_DIR/reward_curves"
copy_if_exists "$OUTPUT_DIR/train_run.log"                          "$ARTIFACTS_DIR/train_run.log"

# Auto-generated README for the proof pack
cat > "$ARTIFACTS_DIR/README.md" << EOF
# Sentinel — Training Proof Pack

Auto-generated on $(date -u '+%Y-%m-%d %H:%M:%S UTC').

## Contents
- \`final/\` — trained LoRA adapter (load with PEFT on top of \`unsloth/Qwen3-4B-bnb-4bit\`)
- \`reward_curves/\` — PNG plot + dashboard
- \`training_metrics.jsonl\` — per-batch metrics (reward, KL, grad_norm, completion length, ...)
- \`rollout_audits/\` — sampled rollouts with full prompt + completion + reward breakdown
- \`stack_versions.json\` — exact training stack
- \`warm_start_summary.json\` — SFT warm-start config & dataset preview
- \`train_run.log\` — full stdout/stderr from training

## Quick reward summary

\`\`\`
$(if [ -f "$OUTPUT_DIR/monitoring/training_metrics.jsonl" ]; then
    head -1 "$OUTPUT_DIR/monitoring/training_metrics.jsonl" 2>/dev/null | python -c "import sys,json; d=json.loads(sys.stdin.read()); print(f'first batch reward_mean: {d.get(\"reward_mean\",\"?\")}')" 2>/dev/null || echo "(could not parse first batch)"
    tail -1 "$OUTPUT_DIR/monitoring/training_metrics.jsonl" 2>/dev/null | python -c "import sys,json; d=json.loads(sys.stdin.read()); print(f'last  batch reward_mean: {d.get(\"reward_mean\",\"?\")}')" 2>/dev/null || echo "(could not parse last batch)"
  else
    echo "(training_metrics.jsonl not found)"
  fi)
\`\`\`
EOF

# ----------------------------------------------------------------------
# 3. Configure git and push to GitHub
# ----------------------------------------------------------------------
if [ -n "${GITHUB_TOKEN:-}" ] && [ -n "${GITHUB_REPO_URL:-}" ]; then
  echo ""
  echo "=== Git push to GitHub ==="
  git config user.name  "${GIT_USER_NAME:-Sentinel Bot}"
  git config user.email "${GIT_USER_EMAIL:-sentinel-bot@users.noreply.github.com}"

  AUTHED_URL="$(echo "$GITHUB_REPO_URL" | sed -e "s|https://|https://x-access-token:${GITHUB_TOKEN}@|")"
  git remote set-url origin "$AUTHED_URL" 2>/dev/null || git remote add origin "$AUTHED_URL"

  git add "$ARTIFACTS_DIR" 2>/dev/null
  if git diff --cached --quiet; then
    echo "[skip] No changes to commit."
  else
    git commit -m "$COMMIT_MSG" || echo "[warn] commit failed (probably no changes)"
    CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
    if git push origin "$CURRENT_BRANCH" 2>&1; then
      echo "PUSHED to GitHub on branch $CURRENT_BRANCH"
    else
      echo "[warn] git push failed — check GITHUB_TOKEN scope (needs repo) and GITHUB_REPO_URL"
    fi
  fi
else
  echo "[skip] GitHub push (missing GITHUB_TOKEN or GITHUB_REPO_URL)"
fi

# ----------------------------------------------------------------------
# 4. Push LoRA to HuggingFace Hub
# ----------------------------------------------------------------------
if [ -n "${HF_TOKEN:-}" ] && [ -n "${HF_REPO:-}" ] && [ -d "$OUTPUT_DIR/final" ]; then
  echo ""
  echo "=== HuggingFace Hub upload ==="
  python - << PYEOF
import os
from huggingface_hub import HfApi, create_repo

repo_id = os.environ['HF_REPO']
token   = os.environ['HF_TOKEN']
src     = os.path.join(os.environ.get('OUTPUT_DIR', '/data/sentinel_outputs'), 'final')

print(f"Creating/ensuring repo: {repo_id}")
create_repo(repo_id, token=token, exist_ok=True, private=False)

print(f"Uploading {src} -> {repo_id}")
HfApi().upload_folder(
    folder_path=src,
    repo_id=repo_id,
    token=token,
    commit_message="Upload trained LoRA adapter (200-step GRPO)"
)
print(f"DONE: https://huggingface.co/{repo_id}")
PYEOF
else
  echo "[skip] HF Hub push (missing HF_TOKEN, HF_REPO, or final/ dir)"
fi

echo ""
echo "=========================================="
echo "  Done"
echo "=========================================="
