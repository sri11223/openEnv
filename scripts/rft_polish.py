"""
RFT (Rejection-sampling Fine-Tuning) polish pass for the trained Sentinel LoRA.

Pipeline:
    1. Load the 200-step GRPO LoRA from $LORA_PATH on top of Qwen3-4B-bnb-4bit.
    2. Generate N rollouts per Sentinel task with the trained policy.
    3. Score each rollout with the real env reward + count false positives
       from the audit trail.
    4. Keep ONLY the rollouts with `score >= MIN_SCORE` AND `fp <= MAX_FP`.
    5. SFT (UnslothTrainer) for `EPOCHS` epochs on those high-quality rollouts.
    6. Save the polished LoRA to $RFT_OUTPUT_DIR/final.
    7. Optionally upload to the HuggingFace Hub.

This is the technique competing teams use to push reward 0.30 -> 0.55+.

ENV VARS:
    LORA_PATH         existing GRPO LoRA  (default /data/sentinel_outputs/final)
    MODEL_NAME        base model          (default unsloth/Qwen3-4B-bnb-4bit)
    RFT_OUTPUT_DIR    where to save       (default /data/sentinel_outputs_rft)
    NUM_ROLLOUTS_PER_TASK   per-task generation count (default 20)
    MAX_NEW_TOKENS    cap on each rollout (default 512)
    GEN_TEMPERATURE   sampling temp       (default 0.7)
    GEN_TOP_P         nucleus p           (default 0.9)
    MIN_SCORE         keep filter (>=)    (default 0.55)
    MAX_FP            keep filter (<=)    (default 3)
    EPOCHS            SFT epochs          (default 2)
    SFT_LR            SFT learning rate   (default 5e-6)
    HF_TOKEN          HF write token (optional)
    HF_REPO           HF repo id          (optional)

Output:
    $RFT_OUTPUT_DIR/final/                  polished LoRA adapter
    $RFT_OUTPUT_DIR/rollouts.jsonl          all rollouts with scores
    $RFT_OUTPUT_DIR/sft_dataset.jsonl       filtered (kept) rollouts
    $RFT_OUTPUT_DIR/rft_summary.json        run summary statistics
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

# Make sure repo root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import Trainer, TrainingArguments

from training.episodes import run_episode_with_completion
from training.prompts import build_prompt_record

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("rft_polish")

# ---------------------------------------------------------------------------
# Config (env-driven so judges can rerun)
# ---------------------------------------------------------------------------
LORA_PATH               = os.environ.get("LORA_PATH", "/data/sentinel_outputs/final")
MODEL_NAME              = os.environ.get("MODEL_NAME", "unsloth/Qwen3-4B-bnb-4bit")
RFT_OUTPUT_DIR          = os.environ.get("RFT_OUTPUT_DIR", "/data/sentinel_outputs_rft")
NUM_ROLLOUTS_PER_TASK   = int(os.environ.get("NUM_ROLLOUTS_PER_TASK", "20"))
MAX_NEW_TOKENS          = int(os.environ.get("MAX_NEW_TOKENS", "512"))
GEN_TEMPERATURE         = float(os.environ.get("GEN_TEMPERATURE", "0.7"))
GEN_TOP_P               = float(os.environ.get("GEN_TOP_P", "0.9"))
MIN_SCORE               = float(os.environ.get("MIN_SCORE", "0.55"))
MAX_FP                  = int(os.environ.get("MAX_FP", "3"))
EPOCHS                  = int(os.environ.get("EPOCHS", "2"))
SFT_LR                  = float(os.environ.get("SFT_LR", "5e-6"))
HF_TOKEN                = os.environ.get("HF_TOKEN", "")
HF_REPO                 = os.environ.get("HF_REPO", "")
MAX_SEQ_LENGTH          = int(os.environ.get("MAX_SEQ_LENGTH", "4096"))
REUSE_RFT_ROLLOUTS      = os.environ.get("REUSE_RFT_ROLLOUTS", "").lower() in {"1", "true", "yes", "y"}

SENTINEL_TASKS = [
    "basic_oversight",
    "fleet_monitoring_conflict",
    "adversarial_worker",
    "multi_crisis_command",
]


def banner(msg: str) -> None:
    bar = "=" * 70
    logger.info(bar)
    logger.info("  %s", msg)
    logger.info(bar)


def count_false_positives(history: List[Dict[str, Any]]) -> int:
    """Count audit entries where Sentinel blocked a SAFE action."""
    fp = 0
    for entry in history:
        audit = entry.get("audit") or {}
        decision = audit.get("sentinel_decision")
        was_misb = audit.get("was_misbehavior")
        if (decision is not None and decision != "APPROVE") and not was_misb:
            fp += 1
    return fp


def load_existing_rollouts(path: Path) -> List[Dict[str, Any]]:
    """Load saved rollout JSONL and recompute keep/drop with current thresholds."""
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        score = float(row.get("score") or 0.0)
        fp = int(row.get("fp") or 0)
        row["score"] = score
        row["fp"] = fp
        row["kept"] = score >= MIN_SCORE and fp <= MAX_FP
        rows.append(row)
    return rows


def resolve_tokenizer_eos(tokenizer) -> str | None:
    """Resolve an EOS token that actually exists in the tokenizer vocab."""
    candidates = [
        getattr(tokenizer, "eos_token", None),
        "<|im_end|>",
        "<|endoftext|>",
    ]
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for token in candidates:
        if not token:
            continue
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            token_id = None
        if token_id is not None and token_id != unk_id:
            return token
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        try:
            return tokenizer.convert_ids_to_tokens(eos_id)
        except Exception:
            return None
    return None


def build_causal_lm_dataset(tokenizer, dataset: Dataset) -> Dataset:
    """Tokenize text rows for plain HF Trainer causal-LM fine-tuning."""
    eos_token = resolve_tokenizer_eos(tokenizer)
    if eos_token:
        tokenizer.eos_token = eos_token
    if tokenizer.pad_token_id is None and eos_token:
        tokenizer.pad_token = eos_token
        logger.info("Using eos token as pad token for RFT SFT: %s", eos_token)

    def tokenize_batch(batch):
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )
        encoded["labels"] = [ids.copy() for ids in encoded["input_ids"]]
        return encoded

    return dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)


def build_causal_lm_collator(tokenizer):
    """Pad inputs and mask padded labels for causal-LM SFT."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    def collate(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = min(MAX_SEQ_LENGTH, max(len(feature["input_ids"]) for feature in features))
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            input_ids = list(feature["input_ids"][:max_len])
            attention_mask = list(feature.get("attention_mask", [1] * len(input_ids))[:max_len])
            labels = list(feature["labels"][:max_len])
            pad_len = max_len - len(input_ids)
            if pad_len > 0:
                input_ids.extend([pad_id] * pad_len)
                attention_mask.extend([0] * pad_len)
                labels.extend([-100] * pad_len)
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}

    return collate


def build_sft_trainer(model, tokenizer, dataset: Dataset, output_dir: Path) -> Trainer:
    """Create a plain HF Trainer to avoid TRL EOS-token version bugs."""
    eos_token = resolve_tokenizer_eos(tokenizer)
    if eos_token:
        tokenizer.eos_token = eos_token
        logger.info("Preparing plain HF Trainer with tokenizer eos_token=%s", eos_token)
    tokenized = build_causal_lm_dataset(tokenizer, dataset)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=SFT_LR,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=False,
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        seed=42,
    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=build_causal_lm_collator(tokenizer),
    )


# ---------------------------------------------------------------------------
# 1. Load base model + existing LoRA in fp16 for inference
# ---------------------------------------------------------------------------
def load_policy():
    banner("Loading base model + GRPO LoRA")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = MODEL_NAME,
        max_seq_length  = MAX_SEQ_LENGTH,
        dtype           = torch.float16,
        load_in_4bit    = True,
    )
    if Path(LORA_PATH).exists():
        logger.info("Loading LoRA adapter from %s", LORA_PATH)
        model = PeftModel.from_pretrained(model, LORA_PATH, is_trainable=True)
        # Coerce LoRA to fp16 to match bnb-4bit compute dtype (avoids matmul errors)
        for name, p in model.named_parameters():
            if "lora_" in name and p.dtype != torch.float16:
                p.data = p.data.to(torch.float16)
    else:
        logger.warning("LORA_PATH %s does not exist, using base model only", LORA_PATH)

    FastLanguageModel.for_inference(model)
    return model, tokenizer


# ---------------------------------------------------------------------------
# 2. Generate rollouts and 3. Score them
# ---------------------------------------------------------------------------
def generate_and_score(model, tokenizer) -> List[Dict[str, Any]]:
    banner(f"Generating {NUM_ROLLOUTS_PER_TASK} rollouts x {len(SENTINEL_TASKS)} tasks")
    all_rollouts: List[Dict[str, Any]] = []

    for task_id in SENTINEL_TASKS:
        for variant_seed in range(NUM_ROLLOUTS_PER_TASK):
            try:
                record = build_prompt_record(
                    task_id=task_id,
                    sentinel_task_ids=SENTINEL_TASKS,
                    variant_seed=variant_seed % 5,  # 5 variants cycled
                    memory_context="",
                )
            except Exception as exc:
                logger.warning("prompt build failed for %s seed %d: %s",
                               task_id, variant_seed, exc)
                continue

            prompt = record["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS).to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens   = MAX_NEW_TOKENS,
                    temperature      = GEN_TEMPERATURE,
                    top_p            = GEN_TOP_P,
                    do_sample        = True,
                    pad_token_id     = tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            completion = tokenizer.decode(
                out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

            try:
                score, history = run_episode_with_completion(
                    completion, task_id, variant_seed % 5, SENTINEL_TASKS,
                    model_steps_limit=3,
                )
            except Exception as exc:
                logger.warning("scoring failed for %s seed %d: %s",
                               task_id, variant_seed, exc)
                score, history = 0.0, []

            fp = count_false_positives(history)

            rollout = {
                "task_id":      task_id,
                "variant_seed": variant_seed % 5,
                "prompt":       prompt,
                "completion":   completion,
                "score":        float(score),
                "fp":           int(fp),
                "kept":         (score >= MIN_SCORE and fp <= MAX_FP),
            }
            all_rollouts.append(rollout)

            logger.info(
                "[%s seed=%d]  score=%.3f  fp=%d  %s",
                task_id, variant_seed % 5, score, fp,
                "KEEP" if rollout["kept"] else "drop",
            )

    return all_rollouts


# ---------------------------------------------------------------------------
# 4. Filter and 5. SFT
# ---------------------------------------------------------------------------
def filter_and_sft(model, tokenizer, all_rollouts: List[Dict[str, Any]]) -> Dict[str, Any]:
    kept = [r for r in all_rollouts if r["kept"]]
    banner(
        f"Filtered: {len(kept)} kept / {len(all_rollouts)} total "
        f"(score >= {MIN_SCORE}, fp <= {MAX_FP})"
    )

    if len(kept) < 4:
        logger.error(
            "Only %d rollouts passed the filter; need at least 4 for stable SFT. "
            "Aborting RFT to avoid producing a worse model.", len(kept)
        )
        return {"status": "skipped_too_few_rollouts", "kept": len(kept), "total": len(all_rollouts)}

    # Build chat-style training texts: prompt + completion
    rows = []
    for r in kept:
        full_text = r["prompt"] + r["completion"] + tokenizer.eos_token
        rows.append({"text": full_text})
    ds = Dataset.from_list(rows)

    # Switch model back to training mode (Unsloth toggles this on for_inference)
    from unsloth import FastLanguageModel
    FastLanguageModel.for_training(model)

    sft_output = Path(RFT_OUTPUT_DIR) / "sft_run"
    sft_output.mkdir(parents=True, exist_ok=True)

    trainer = build_sft_trainer(model, tokenizer, ds, sft_output)

    banner(f"Starting SFT on {len(kept)} kept rollouts for {EPOCHS} epochs (lr={SFT_LR})")
    trainer.train()

    # Save final polished LoRA
    final_dir = Path(RFT_OUTPUT_DIR) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Saved RFT-polished LoRA to %s", final_dir)

    return {
        "status":   "ok",
        "kept":     len(kept),
        "total":    len(all_rollouts),
        "epochs":   EPOCHS,
        "lr":       SFT_LR,
        "saved_to": str(final_dir),
    }


# ---------------------------------------------------------------------------
# 6. Optional HF Hub push
# ---------------------------------------------------------------------------
def maybe_push_to_hub() -> None:
    final_dir = Path(RFT_OUTPUT_DIR) / "final"
    if not (HF_TOKEN and HF_REPO and final_dir.exists()):
        logger.info("Skipping HF Hub push (missing HF_TOKEN/HF_REPO or no final/ dir)")
        return

    banner(f"Uploading {final_dir} -> https://huggingface.co/{HF_REPO}")
    from huggingface_hub import HfApi, create_repo
    create_repo(HF_REPO, token=HF_TOKEN, exist_ok=True, private=False)
    HfApi().upload_folder(
        folder_path     = str(final_dir),
        repo_id         = HF_REPO,
        token           = HF_TOKEN,
        commit_message  = "Upload RFT-polished LoRA (rejection-sampling fine-tune)",
    )
    logger.info("Upload complete: https://huggingface.co/%s", HF_REPO)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    banner("RFT Polish — config")
    for k, v in {
        "LORA_PATH":             LORA_PATH,
        "MODEL_NAME":            MODEL_NAME,
        "RFT_OUTPUT_DIR":        RFT_OUTPUT_DIR,
        "NUM_ROLLOUTS_PER_TASK": NUM_ROLLOUTS_PER_TASK,
        "MAX_NEW_TOKENS":        MAX_NEW_TOKENS,
        "GEN_TEMPERATURE":       GEN_TEMPERATURE,
        "GEN_TOP_P":             GEN_TOP_P,
        "MIN_SCORE":             MIN_SCORE,
        "MAX_FP":                MAX_FP,
        "EPOCHS":                EPOCHS,
        "SFT_LR":                SFT_LR,
        "HF_REPO":               HF_REPO or "(skip)",
        "REUSE_RFT_ROLLOUTS":    REUSE_RFT_ROLLOUTS,
    }.items():
        logger.info("  %-22s = %s", k, v)

    Path(RFT_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_policy()

    # Persist all rollouts (for proof pack)
    rollouts_file = Path(RFT_OUTPUT_DIR) / "rollouts.jsonl"
    if REUSE_RFT_ROLLOUTS and rollouts_file.exists():
        all_rollouts = load_existing_rollouts(rollouts_file)
        logger.info("Reusing %d saved rollouts from %s", len(all_rollouts), rollouts_file)
    else:
        all_rollouts = generate_and_score(model, tokenizer)

    with rollouts_file.open("w") as fh:
        for r in all_rollouts:
            fh.write(json.dumps(r) + "\n")
    logger.info("Wrote %d rollouts to %s", len(all_rollouts), rollouts_file)

    # Per-task summary BEFORE filtering
    by_task = defaultdict(list)
    for r in all_rollouts:
        by_task[r["task_id"]].append(r)
    banner("Per-task generation stats")
    for task_id, rs in by_task.items():
        scores = [r["score"] for r in rs]
        fps    = [r["fp"]    for r in rs]
        kept   = sum(1 for r in rs if r["kept"])
        logger.info(
            "  %-30s  n=%2d  mean_score=%.3f  mean_fp=%.1f  kept=%d",
            task_id, len(rs), sum(scores)/max(1, len(rs)), sum(fps)/max(1, len(rs)), kept,
        )

    # SFT on the kept rollouts
    sft_summary = filter_and_sft(model, tokenizer, all_rollouts)

    # Persist filtered SFT dataset for transparency
    kept_file = Path(RFT_OUTPUT_DIR) / "sft_dataset.jsonl"
    with kept_file.open("w") as fh:
        for r in all_rollouts:
            if r["kept"]:
                fh.write(json.dumps(r) + "\n")
    logger.info("Wrote %d kept samples to %s", sum(1 for r in all_rollouts if r["kept"]), kept_file)

    # Final summary
    summary = {
        "config": {
            "LORA_PATH":           LORA_PATH,
            "MODEL_NAME":          MODEL_NAME,
            "NUM_ROLLOUTS_PER_TASK": NUM_ROLLOUTS_PER_TASK,
            "MIN_SCORE":           MIN_SCORE,
            "MAX_FP":              MAX_FP,
            "EPOCHS":              EPOCHS,
            "SFT_LR":              SFT_LR,
        },
        "rollout_stats": {
            "total":            len(all_rollouts),
            "kept":             sum(1 for r in all_rollouts if r["kept"]),
            "mean_score_total": sum(r["score"] for r in all_rollouts) / max(1, len(all_rollouts)),
            "mean_fp_total":    sum(r["fp"]    for r in all_rollouts) / max(1, len(all_rollouts)),
            "mean_score_kept":  (
                sum(r["score"] for r in all_rollouts if r["kept"]) /
                max(1, sum(1 for r in all_rollouts if r["kept"]))
            ),
            "mean_fp_kept":     (
                sum(r["fp"]    for r in all_rollouts if r["kept"]) /
                max(1, sum(1 for r in all_rollouts if r["kept"]))
            ),
            "task_breakdown":   {
                t: {
                    "n":          len(rs),
                    "mean_score": sum(r["score"] for r in rs) / max(1, len(rs)),
                    "mean_fp":    sum(r["fp"]    for r in rs) / max(1, len(rs)),
                    "kept":       sum(1 for r in rs if r["kept"]),
                }
                for t, rs in by_task.items()
            },
        },
        "sft": sft_summary,
    }
    summary_file = Path(RFT_OUTPUT_DIR) / "rft_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote summary to %s", summary_file)

    maybe_push_to_hub()

    banner("RFT polish complete")
    logger.info("Final LoRA: %s/final", RFT_OUTPUT_DIR)
    logger.info("Summary:    %s", summary_file)
    if HF_REPO:
        logger.info("HF Hub:     https://huggingface.co/%s", HF_REPO)


if __name__ == "__main__":
    main()
