"""
train.py - GRPO Fine-tuning for OpenEnv (IRT / SENTINEL)
==============================================================
Runnable training script. Uses TRL GRPOTrainer + Unsloth (optional) + curriculum.

HOW TO RUN:
    # Minimum (T4 / A10G, no Unsloth):
    python train.py

    # With Unsloth (A100 / H100, 2x faster):
    USE_UNSLOTH=1 python train.py

    # Override model and steps:
    MODEL_NAME=unsloth/Qwen3-30B-A3B-bnb-4bit TRAIN_STEPS=200 python train.py

    # Resume from checkpoint:
    RESUME_FROM=outputs/checkpoints/checkpoint-100 python train.py

ENV VARS:
    MODEL_NAME      HuggingFace model ID (default: unsloth/Qwen3-30B-A3B-bnb-4bit)
    HF_TOKEN        HuggingFace token (for gated models)
    GROQ_API_KEY    Groq API key (for LLM judge panel, optional)
    WANDB_PROJECT   W&B project name (optional, set to "" to disable)
    TRAIN_STEPS     Number of GRPO training steps (default: 200)
    NUM_GENERATIONS G rollouts per prompt (default: 4)
    USE_UNSLOTH     Set to "1" to use Unsloth (requires unsloth installed)
    RESUME_FROM     Path to checkpoint to resume from
    OUTPUT_DIR      Where to save checkpoints (default: outputs/checkpoints)
    LR              Learning rate (default: 5e-6)
    KL_COEF         KL penalty coefficient (default: 0.04)
    LORA_R          LoRA rank (default: 16)
    TRAIN_MONITOR_DIR   Structured metrics output dir (default: outputs/monitoring)
    WARM_START_STEPS    Optional small warm-start steps before GRPO (default: 0)
    WARM_START_LR       Learning rate for warm-start stage (default: 2e-5)
    WARM_START_ONLY     Set to "1" to stop after warm-start
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import TrainerCallback

# Re-export from extracted modules for backward compatibility
from training.metrics import (
    safe_ratio as _safe_ratio,
    _increment_counter,
    _normalize_completion_text,
    _extract_completion_choice,
    _shannon_entropy_from_labels,
    summarize_sentinel_history as _summarize_sentinel_history,
    aggregate_batch_metrics as _aggregate_batch_metrics,
    completion_diversity_metrics as _completion_diversity_metrics,
    productive_signal_metrics as _productive_signal_metrics,
    training_coverage_metrics as _training_coverage_metrics,
    zero_gradient_group_metrics as _zero_gradient_group_metrics,
    frontier_scenario_keys as _frontier_scenario_keys,
    set_thresholds as _set_metric_thresholds,
)
from training.monitoring import (
    TrainingMonitor,
    GRPOStabilityCallback,
    RolloutAuditSampler,
    _truncate_text,
    _audit_priority,
)
from training.prompts import (
    build_system_prompt,
    scenario_to_prompt,
    sentinel_obs_to_prompt,
    sentinel_adversarial_case_to_prompt,
    build_prompt_record as _build_prompt_record_impl,
    memory_context_for_task as _memory_context_for_task,
    load_or_create_sentinel_adversarial_cases as _load_or_create_sentinel_adversarial_cases_impl,
    AdaptivePromptState as _AdaptivePromptStateBase,
    AdaptivePromptDataset,
    WarmStartDataset,
    build_grpo_dataset as _build_grpo_dataset_impl,
)
from training.episodes import (
    parse_action as _parse_action,
    greedy_fallback_action as _greedy_fallback_action,
    greedy_fallback_sentinel_decision as _greedy_fallback_sentinel_decision,
    run_episode_with_completion as _run_episode_with_completion_impl,
    _run_irt_episode,
    _run_sentinel_episode,
    run_sentinel_adversarial_case as _run_sentinel_adversarial_case,
    grpo_reward_fn as _grpo_reward_fn_impl,
    trajectory_summary_from_history as _trajectory_summary_from_history,
    mistakes_from_history as _mistakes_from_history,
    mistake_cards_from_history as _mistake_cards_from_history,
    successes_from_history as _successes_from_history,
)
from training.curriculum import CURRICULUM_FRONTIER_FAILURE_RATE

MODEL_NAME      = os.getenv("MODEL_NAME", "unsloth/Qwen3-30B-A3B-bnb-4bit")
HF_TOKEN        = os.getenv("HF_TOKEN", "")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
WANDB_PROJECT   = os.getenv("WANDB_PROJECT", "").strip()
TRAIN_STEPS     = int(os.getenv("TRAIN_STEPS", "100"))
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "2"))
USE_UNSLOTH     = os.getenv("USE_UNSLOTH", "1") == "1"
RESUME_FROM     = os.getenv("RESUME_FROM", "")
OUTPUT_DIR      = os.getenv("OUTPUT_DIR", "outputs/checkpoints")
LR              = float(os.getenv("LR", "5e-6"))
KL_COEF         = float(os.getenv("KL_COEF", "0.04"))
LORA_R          = int(os.getenv("LORA_R", "16"))
MAX_NEW_TOKENS  = int(os.getenv("MAX_NEW_TOKENS", "512"))
PROMPT_DATASET_SIZE = int(os.getenv("PROMPT_DATASET_SIZE", str(max(512, TRAIN_STEPS * 8))))
USE_LLM_PANEL   = bool(GROQ_API_KEY)                  # auto-enable if key available
USE_CURRICULUM  = os.getenv("USE_CURRICULUM", "1") == "1"
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.7"))
GEN_TOP_P       = float(os.getenv("GEN_TOP_P", "1.0"))
USE_SENTINEL    = os.getenv("USE_SENTINEL", "0") == "1"  # Enable SENTINEL training
USE_AGENT_MEMORY = os.getenv("USE_AGENT_MEMORY", "1") == "1"
USE_FEEDBACK_MEMORY = os.getenv("USE_FEEDBACK_MEMORY", "1") == "1" and USE_AGENT_MEMORY
USE_SENTINEL_ADVERSARIAL = os.getenv("USE_SENTINEL_ADVERSARIAL", "1") == "1"
SENTINEL_ADVERSARIAL_PATH = os.getenv(
    "SENTINEL_ADVERSARIAL_PATH",
    "outputs/sentinel_adversarial_cases.json",
)
SENTINEL_FEEDBACK_MEMORY_PATH = os.getenv(
    "SENTINEL_FEEDBACK_MEMORY_PATH",
    "outputs/sentinel_feedback_memory.json",
)
TRAIN_MONITOR_DIR = os.getenv("TRAIN_MONITOR_DIR", "outputs/monitoring")
WARM_START_STEPS = int(os.getenv("WARM_START_STEPS", "0"))
WARM_START_LR = float(os.getenv("WARM_START_LR", "2e-5"))
WARM_START_DATASET_SIZE = int(os.getenv("WARM_START_DATASET_SIZE", "24"))
WARM_START_OUTPUT_DIR = os.getenv("WARM_START_OUTPUT_DIR", "outputs/warm_start")
WARM_START_ONLY = os.getenv("WARM_START_ONLY", "0") == "1"
ROLLOUT_AUDIT_DIR = os.getenv("ROLLOUT_AUDIT_DIR", os.path.join(TRAIN_MONITOR_DIR, "rollout_audits"))
ROLLOUT_AUDIT_EVERY = int(os.getenv("ROLLOUT_AUDIT_EVERY", "10"))
ROLLOUT_AUDIT_SAMPLES = int(os.getenv("ROLLOUT_AUDIT_SAMPLES", "2"))
REWARD_SCHEDULE_MODE = os.getenv("REWARD_SCHEDULE_MODE", os.getenv("REWARD_PROFILE", "dynamic"))
MODEL_STEPS_LIMIT = int(os.getenv("MODEL_STEPS_LIMIT", "1"))
KL_TARGET = float(os.getenv("KL_TARGET", "0.08"))
KL_ADAPTIVE = os.getenv("KL_ADAPTIVE", "1") == "1"
KL_LOW_FACTOR = float(os.getenv("KL_LOW_FACTOR", "1.5"))
KL_HIGH_FACTOR = float(os.getenv("KL_HIGH_FACTOR", "1.5"))
KL_BETA_UP_MULT = float(os.getenv("KL_BETA_UP_MULT", "2.0"))
KL_BETA_DOWN_MULT = float(os.getenv("KL_BETA_DOWN_MULT", "0.5"))
KL_MIN_BETA = float(os.getenv("KL_MIN_BETA", "0.005"))
KL_MAX_BETA = float(os.getenv("KL_MAX_BETA", "0.5"))
KL_HARD_STOP_ENABLED = os.getenv("KL_HARD_STOP_ENABLED", "0") == "1"
KL_HARD_STOP_MULT = float(os.getenv("KL_HARD_STOP_MULT", "3.0"))
ZERO_SIGNAL_REWARD_THRESHOLD = float(os.getenv("ZERO_SIGNAL_REWARD_THRESHOLD", "0.05"))
TRIVIAL_REWARD_THRESHOLD = float(os.getenv("TRIVIAL_REWARD_THRESHOLD", "0.95"))

TASK_IDS = [
    "severity_classification",
    "root_cause_analysis",
    "full_incident_management",
]

SENTINEL_TASK_IDS = [
    "basic_oversight",
    "fleet_monitoring_conflict",
    "adversarial_worker",
    "multi_crisis_command",
]


def _parse_task_filter(env_name: str, allowed: List[str]) -> List[str]:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return list(allowed)
    selected = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [task_id for task_id in selected if task_id not in allowed]
    if unknown:
        raise ValueError(
            f"{env_name} contains unknown task id(s): {unknown}. "
            f"Allowed: {allowed}"
        )
    return selected or list(allowed)


TASK_IDS = _parse_task_filter("IRT_TASKS", TASK_IDS)
SENTINEL_TASK_IDS = _parse_task_filter("SENTINEL_TASKS", SENTINEL_TASK_IDS)

# Select task set based on USE_SENTINEL flag
ACTIVE_TASK_IDS = SENTINEL_TASK_IDS if USE_SENTINEL else TASK_IDS

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("outputs/reward_curves", exist_ok=True)
os.makedirs(TRAIN_MONITOR_DIR, exist_ok=True)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUTPUT_DIR, "train.log")),
    ],
)
logger = logging.getLogger(__name__)


def _package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "missing"


def collect_training_stack_versions() -> Dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model_name": MODEL_NAME,
        "use_unsloth": USE_UNSLOTH,
        "cuda_available": cuda_available,
        "bf16_available": bool(cuda_available and torch.cuda.is_bf16_supported()),
        "train_steps": TRAIN_STEPS,
        "warm_start_steps": WARM_START_STEPS,
        "reward_schedule_mode": REWARD_SCHEDULE_MODE,
        "memory": {
            "agent_memory_enabled": USE_AGENT_MEMORY,
            "feedback_memory_enabled": USE_FEEDBACK_MEMORY,
        },
        "productive_signal_thresholds": {
            "zero_signal_reward_threshold": ZERO_SIGNAL_REWARD_THRESHOLD,
            "trivial_reward_threshold": TRIVIAL_REWARD_THRESHOLD,
        },
        "adaptive_curriculum": {
            "frontier_failure_rate": CURRICULUM_FRONTIER_FAILURE_RATE,
        },
        "kl_control": {
            "initial_beta": KL_COEF,
            "target": KL_TARGET,
            "adaptive": KL_ADAPTIVE,
            "low_factor": KL_LOW_FACTOR,
            "high_factor": KL_HIGH_FACTOR,
            "beta_up_mult": KL_BETA_UP_MULT,
            "beta_down_mult": KL_BETA_DOWN_MULT,
            "min_beta": KL_MIN_BETA,
            "max_beta": KL_MAX_BETA,
            "hard_stop_enabled": KL_HARD_STOP_ENABLED,
            "hard_stop_mult": KL_HARD_STOP_MULT,
        },
        "packages": {
            "torch": getattr(torch, "__version__", "missing"),
            "bitsandbytes": _package_version("bitsandbytes"),
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
            "trl": _package_version("trl"),
            "datasets": _package_version("datasets"),
            "matplotlib": _package_version("matplotlib"),
            "wandb": _package_version("wandb"),
            "openenv-core": _package_version("openenv-core"),
            "unsloth": _package_version("unsloth"),
        },
    }

# ---------------------------------------------------------------------------
# W&B setup (optional)
# ---------------------------------------------------------------------------

wandb_enabled = bool(WANDB_PROJECT) and WANDB_PROJECT.lower() not in {"0", "false", "none", "disabled"}
if wandb_enabled:
    try:
        import wandb
        wandb.init(project=WANDB_PROJECT, config={
            "model": MODEL_NAME,
            "train_steps": TRAIN_STEPS,
            "num_generations": NUM_GENERATIONS,
            "lr": LR,
            "kl_coef": KL_COEF,
            "lora_r": LORA_R,
            "use_llm_panel": USE_LLM_PANEL,
        })
        logger.info("W&B enabled: project=%s", WANDB_PROJECT)
    except ImportError:
        wandb_enabled = False
        logger.warning("wandb not installed -- logging disabled")
    except Exception as exc:
        wandb_enabled = False
        logger.warning("wandb init skipped: %s", exc)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer():
    """Load model + tokenizer. Uses Unsloth if USE_UNSLOTH=1, else standard HF.

    When Unsloth is enabled:
      - 12x faster MoE training via Triton kernels (torch._grouped_mm)
      - 3x faster inference via fused attention (FastLanguageModel.for_inference)
      - >35% less VRAM via 4-bit quantization + gradient checkpointing
    """
    if USE_UNSLOTH:
        logger.info("Loading model with Unsloth: %s", MODEL_NAME)
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name           = MODEL_NAME,
            max_seq_length       = 4096,
            dtype                = None,
            load_in_4bit         = True,
            token                = HF_TOKEN or None,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r                            = LORA_R,
            target_modules               = ["q_proj","k_proj","v_proj","o_proj",
                                            "gate_proj","up_proj","down_proj"],
            lora_alpha                   = LORA_R,
            lora_dropout                 = 0,
            bias                         = "none",
            use_gradient_checkpointing   = "unsloth",
            random_state                 = 42,
        )
        # Enable Unsloth fast inference (2-3x speedup for generation)
        # GRPOTrainer internally handles train/eval mode toggling, but
        # setting this up front ensures optimized attention kernels are
        # compiled and ready for the first rollout batch.
        try:
            FastLanguageModel.for_inference(model)
            logger.info("Unsloth fast inference enabled (fused attention kernels)")
        except Exception as exc:
            logger.warning("Unsloth fast inference setup skipped: %s", exc)
    else:
        logger.info("Loading model with standard HF: %s", MODEL_NAME)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        cuda_available = torch.cuda.is_available()
        bf16_available = cuda_available and torch.cuda.is_bf16_supported()

        load_kwargs: Dict[str, Any] = {
            "torch_dtype": torch.bfloat16 if bf16_available else (torch.float16 if cuda_available else torch.float32),
            "device_map" : "auto" if cuda_available else None,
        }
        if "bnb-4bit" in MODEL_NAME or "4bit" in MODEL_NAME:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit              = True,
                bnb_4bit_use_double_quant = True,
                bnb_4bit_quant_type       = "nf4",
                bnb_4bit_compute_dtype    = torch.bfloat16,
            )
            load_kwargs.pop("torch_dtype", None)

        if HF_TOKEN:
            load_kwargs["token"] = HF_TOKEN

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN or None)
        model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)

        lora_config = LoraConfig(
            r              = LORA_R,
            lora_alpha     = LORA_R,
            target_modules = ["q_proj","k_proj","v_proj","o_proj"],
            lora_dropout   = 0.05,
            bias           = "none",
            task_type      = "CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if RESUME_FROM:
        logger.info("Resuming from checkpoint: %s", RESUME_FROM)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, RESUME_FROM)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Backward-compatible re-exports for tests
AdaptivePromptState = _AdaptivePromptStateBase
build_prompt_record = _build_prompt_record_impl
build_grpo_dataset = _build_grpo_dataset_impl
_load_or_create_sentinel_adversarial_cases = _load_or_create_sentinel_adversarial_cases_impl
_aggregate_batch_metrics = _aggregate_batch_metrics
_sentinel_history_entry = None  # re-exported below

def _sentinel_history_entry_fn(decision, result):
    from training.episodes import _sentinel_history_entry as _she
    return _she(decision, result)

_sentinel_history_entry = _sentinel_history_entry_fn

# Thin wrappers delegating to extracted modules
# ---------------------------------------------------------------------------

# Prompt construction
def _build_system_prompt(task_id, memory_context=""):
    return build_system_prompt(task_id, SENTINEL_TASK_IDS, memory_context)

def _scenario_to_prompt(scenario, task_id, memory_context=""):
    return scenario_to_prompt(scenario, task_id, SENTINEL_TASK_IDS, memory_context)

def _sentinel_obs_to_prompt(obs, task_id, memory_context=""):
    return sentinel_obs_to_prompt(obs, task_id, SENTINEL_TASK_IDS, memory_context)

# Episode execution
def run_episode_with_completion(completion_text, task_id, variant_seed):
    return _run_episode_with_completion_impl(
        completion_text, task_id, variant_seed, SENTINEL_TASK_IDS,
        model_steps_limit=MODEL_STEPS_LIMIT,
    )

def grpo_reward_fn(prompts, completions, **kwargs):
    return _grpo_reward_fn_impl(
        prompts, completions,
        sentinel_task_ids=SENTINEL_TASK_IDS,
        active_task_ids=list(ACTIVE_TASK_IDS),
        use_llm_panel=USE_LLM_PANEL,
        groq_api_key=GROQ_API_KEY,
        wandb_enabled=wandb_enabled,
        model_steps_limit=MODEL_STEPS_LIMIT,
        **kwargs,
    )

# Warm-start helpers
def _warm_start_counterfactual(obs, decision):
    constitution = obs.constitutional_assessment or {}
    violations = list((constitution.get("constitutional_violations") or {}).keys())
    if decision.get("decision") == "APPROVE":
        return "If approved, the proposal stays within the current safe operating envelope."
    if violations:
        return "If approved unchanged, this proposal would likely violate " + ", ".join(violations) + " and increase operational risk."
    return "If approved unchanged, this proposal could bypass oversight without sufficient justification."

def _warm_start_sentinel_decision_for_observation(obs):
    decision = dict(_greedy_fallback_sentinel_decision(obs, []))
    violations = sorted((obs.constitutional_assessment.get("constitutional_violations") or {}).keys())
    decision.setdefault("worker_message", "Approved. Execute carefully and preserve evidence.")
    for key in ["required_evidence", "redirect_action", "reassign_to", "flag_severity", "suggested_action_type", "suggested_target"]:
        decision.setdefault(key, [] if key == "required_evidence" else None)
    decision.setdefault("suggested_parameters", {})
    decision["constitutional_violations"] = violations
    decision["counterfactual"] = _warm_start_counterfactual(obs, decision)
    return decision

def _build_warm_start_examples(task_ids, memory_context="", memory=None, feedback_memory=None, max_examples=None, max_seeds=3):
    if max_examples is None: max_examples = WARM_START_DATASET_SIZE
    records = []
    for task_id in task_ids:
        for seed in range(max_seeds):
            task_memory = _memory_context_for_task(memory, feedback_memory, task_id, memory_context)
            if task_id in SENTINEL_TASK_IDS:
                from sentinel.environment import SentinelEnv
                env = SentinelEnv()
                obs = env.reset(task_id=task_id, variant_seed=seed)
                prompt = _sentinel_obs_to_prompt(obs, task_id, task_memory)
                response = _warm_start_sentinel_decision_for_observation(obs)
            else:
                from src.environment import IncidentResponseEnv
                env = IncidentResponseEnv()
                obs = env.reset(task_id=task_id, variant_seed=seed)
                prompt = _scenario_to_prompt(env._scenario, task_id, task_memory)
                response = _greedy_fallback_action(env, obs, [])
            records.append({"task_id": task_id, "variant_seed": seed, "text": prompt + json.dumps(response, sort_keys=True)})
            if len(records) >= max_examples: return records
    if records and len(records) < max_examples:
        cycled = []
        idx = 0
        while len(records) + len(cycled) < max_examples:
            cycled.append(dict(records[idx % len(records)]))
            idx += 1
        records.extend(cycled)
    return records[:max_examples]

def _run_small_warm_start(model, tokenizer, prompt_state):
    from transformers import Trainer, TrainingArguments
    output_dir = Path(WARM_START_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = _build_warm_start_examples(task_ids=list(ACTIVE_TASK_IDS), memory_context=prompt_state.memory_context, memory=prompt_state.memory, feedback_memory=prompt_state.feedback_memory, max_examples=max(1, WARM_START_DATASET_SIZE))
    if not examples: raise RuntimeError("Warm-start requested, but no warm-start examples could be built.")
    preview = [{"task_id": r["task_id"], "variant_seed": r["variant_seed"], "text_preview": str(r["text"])[:240]} for r in examples[:5]]
    (output_dir / "dataset_preview.json").write_text(json.dumps(preview, indent=2), encoding="utf-8")
    dataset = WarmStartDataset([r["text"] for r in examples], tokenizer)
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=WARM_START_LR,
        max_steps=max(1, WARM_START_STEPS),
        num_train_epochs=1,
        logging_steps=1,
        save_strategy="no",
        remove_unused_columns=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to="wandb" if wandb_enabled else "none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    summary = {"enabled": True, "steps": max(1, WARM_START_STEPS), "learning_rate": WARM_START_LR, "dataset_size": len(examples), "output_dir": str(output_dir), "saved_model_path": str(final_dir), "task_ids": list(ACTIVE_TASK_IDS)}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Warm-start complete: steps=%d dataset=%d saved=%s", summary["steps"], summary["dataset_size"], final_dir)
    return summary

def train():
    logger.info("=" * 60)
    logger.info("OpenEnv GRPO Training")
    logger.info("Model:      %s", MODEL_NAME)
    logger.info("Steps:      %d", TRAIN_STEPS)
    logger.info("G:          %d rollouts/prompt", NUM_GENERATIONS)
    logger.info("LR:         %g", LR)
    logger.info("KL coef:    %g", KL_COEF)
    logger.info("LoRA r:     %d", LORA_R)
    logger.info("LLM panel:  %s", USE_LLM_PANEL)
    logger.info("Curriculum: %s", USE_CURRICULUM)
    logger.info("Sampling:   temperature=%.2f top_p=%.2f", GEN_TEMPERATURE, GEN_TOP_P)
    logger.info("Episode:    MODEL_STEPS_LIMIT=%d  MAX_NEW_TOKENS=%d", MODEL_STEPS_LIMIT, MAX_NEW_TOKENS)
    logger.info("EvalMinDif: %s", os.getenv("EVAL_MIN_DIFFICULTY", "0.0"))
    logger.info("Warm start: %s", WARM_START_STEPS if WARM_START_STEPS > 0 else "disabled")
    logger.info("Reward schedule: %s", REWARD_SCHEDULE_MODE if USE_SENTINEL else "n/a")
    logger.info(
        "KL control: target=%s adaptive=%s beta=%s [%s, %s]",
        KL_TARGET,
        KL_ADAPTIVE,
        KL_COEF,
        KL_MIN_BETA,
        KL_MAX_BETA,
    )
    logger.info(
        "Rollout audit: every %s batch(es), %s sample(s)",
        ROLLOUT_AUDIT_EVERY if ROLLOUT_AUDIT_EVERY > 0 else "disabled",
        ROLLOUT_AUDIT_SAMPLES,
    )
    logger.info("Output:     %s", OUTPUT_DIR)
    logger.info("=" * 60)

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Load curriculum and agent memory
    from training.curriculum import get_curriculum
    from training.memory import (
        load_agent_memory, build_memory_context, maybe_consolidate_memory,
        record_episode as mem_record_episode, save_agent_memory,
        memory_summary as summarize_agent_memory, new_agent_memory,
    )
    from sentinel.feedback import (
        load_feedback_memory,
        empty_feedback_memory,
        record_episode_feedback,
        save_feedback_memory,
    )
    from sentinel.rewards import reset_reward_weights, scheduled_reward_weights, set_reward_weights

    curriculum = get_curriculum(active_task_ids=ACTIVE_TASK_IDS) if USE_CURRICULUM else None
    memory = load_agent_memory() if USE_AGENT_MEMORY else new_agent_memory()
    feedback_memory = (
        load_feedback_memory(SENTINEL_FEEDBACK_MEMORY_PATH)
        if USE_FEEDBACK_MEMORY
        else empty_feedback_memory()
    )
    memory_ctx = build_memory_context(memory) if USE_AGENT_MEMORY else ""
    prompt_state = _AdaptivePromptStateBase(
        task_ids=list(ACTIVE_TASK_IDS),
        sentinel_task_ids=list(SENTINEL_TASK_IDS),
        curriculum=curriculum,
        memory=memory,
        feedback_memory=feedback_memory,
        memory_context=memory_ctx,
        memory_enabled=USE_AGENT_MEMORY,
        max_seeds=5,
        use_sentinel=USE_SENTINEL,
        use_feedback_memory=USE_FEEDBACK_MEMORY,
        use_llm_panel=USE_LLM_PANEL,
        groq_api_key=GROQ_API_KEY,
        sentinel_adversarial_path=SENTINEL_ADVERSARIAL_PATH,
        sentinel_feedback_memory_path=SENTINEL_FEEDBACK_MEMORY_PATH,
        use_sentinel_adversarial=USE_SENTINEL_ADVERSARIAL,
    )
    if USE_SENTINEL and USE_SENTINEL_ADVERSARIAL:
        prompt_state.refresh_adversarial_cases()

    train_dataset = AdaptivePromptDataset(
        state=prompt_state,
        total_samples=PROMPT_DATASET_SIZE,
    )
    training_monitor = TrainingMonitor(TRAIN_MONITOR_DIR)
    training_monitor.write_stack_versions(collect_training_stack_versions())
    rollout_auditor = RolloutAuditSampler(
        output_dir=ROLLOUT_AUDIT_DIR,
        every=ROLLOUT_AUDIT_EVERY,
        sample_limit=ROLLOUT_AUDIT_SAMPLES,
    )

    warm_start_summary: Optional[Dict[str, Any]] = None
    warm_start_path = os.path.join(WARM_START_OUTPUT_DIR, "final")
    if WARM_START_STEPS > 0 and os.path.isdir(warm_start_path):
        logger.info("Warm-start checkpoint found at %s — SKIPPING (saves ~20 min)", warm_start_path)
        # Reload the warm-start LoRA weights
        try:
            from peft import PeftModel
            if not hasattr(model, "peft_config"):
                model = PeftModel.from_pretrained(model, warm_start_path)
            logger.info("Loaded warm-start LoRA from %s", warm_start_path)
        except Exception as exc:
            logger.warning("Could not reload warm-start LoRA: %s (continuing with base model)", exc)
        warm_start_summary = {"saved_model_path": warm_start_path, "skipped": True}
    elif WARM_START_STEPS > 0:
        warm_start_summary = _run_small_warm_start(model, tokenizer, prompt_state)
        if WARM_START_ONLY:
            return warm_start_summary["saved_model_path"]

    # GRPO config
    from trl import GRPOConfig, GRPOTrainer

    grpo_config = GRPOConfig(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = 1,
        per_device_train_batch_size = NUM_GENERATIONS,
        gradient_accumulation_steps = 1,
        num_generations             = NUM_GENERATIONS,
        max_completion_length       = MAX_NEW_TOKENS,
        learning_rate               = LR,
        beta                        = KL_COEF,
        temperature                 = GEN_TEMPERATURE,
        top_p                       = GEN_TOP_P,
        logging_steps               = 1,
        save_steps                  = 25,
        save_total_limit            = 4,
        dataloader_num_workers      = 0,
        bf16                        = torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16                        = torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to                   = "wandb" if wandb_enabled else "none",
        max_steps                   = TRAIN_STEPS,
        chat_template_kwargs        = {"enable_thinking": False},
    )



    # Wrap reward fn to inject curriculum-selected task_ids and seeds
    def reward_fn_with_curriculum(prompts, completions, **kwargs):
        # Extract task_id and variant_seed from dataset columns if available
        t_ids   = kwargs.get("task_id", [ACTIVE_TASK_IDS[0]] * len(prompts))
        v_seeds = kwargs.get("variant_seed", [0] * len(prompts))
        adv_cases = kwargs.get("adversarial_case", [""] * len(prompts))
        curriculum_snapshot = curriculum.summary() if curriculum else None
        reward_schedule: Optional[Dict[str, Any]] = None
        if USE_SENTINEL:
            current_batch_index = training_monitor.batch_index + 1
            progress = min(1.0, current_batch_index / max(1, TRAIN_STEPS))
            reward_schedule = scheduled_reward_weights(
                progress=progress,
                mode=REWARD_SCHEDULE_MODE,
            )
            set_reward_weights(reward_schedule["weights"])

        rewards, histories = grpo_reward_fn(
            prompts      = prompts,
            completions  = completions,
            task_id      = t_ids,
            variant_seed = v_seeds,
            adversarial_case = adv_cases,
            return_histories = True,
            **{k: v for k, v in kwargs.items() if k not in ("task_id", "variant_seed", "adversarial_case")},
        )

        for i, r in enumerate(rewards):
            t_id = t_ids[i] if i < len(t_ids) else ACTIVE_TASK_IDS[0]
            seed = v_seeds[i] if i < len(v_seeds) else 0
            history = histories[i] if i < len(histories) else []
            prompt_state.update_after_episode(
                task_id=t_id,
                variant_seed=seed,
                reward=r,
                history=history,
                mem_record_episode=mem_record_episode,
                record_episode_feedback=record_episode_feedback,
                save_agent_memory=save_agent_memory,
                save_feedback_memory=save_feedback_memory,
                maybe_consolidate_memory=maybe_consolidate_memory,
            )

        nonlocal memory
        memory = prompt_state.memory
        nonlocal feedback_memory
        feedback_memory = prompt_state.feedback_memory

        monitor_summary = training_monitor.log_batch(
            sentinel_task_ids=list(SENTINEL_TASK_IDS),
            rewards=rewards,
            histories=histories,
            task_ids=[str(task_id) for task_id in t_ids],
            variant_seeds=[int(seed) for seed in v_seeds],
            completions=[str(completion) for completion in completions],
            prompts=[str(prompt) for prompt in prompts],
            adversarial_cases=[str(case) for case in adv_cases],
            curriculum_summary=curriculum_snapshot,
            prompt_refreshes=prompt_state.prompt_refreshes,
            reward_schedule=reward_schedule,
            memory_summary={
                "agent_memory_enabled": USE_AGENT_MEMORY,
                "feedback_memory_enabled": USE_FEEDBACK_MEMORY,
                **summarize_agent_memory(memory),
            },
        )
        audit_path = rollout_auditor.record_batch(
            sentinel_task_ids=list(SENTINEL_TASK_IDS),
            active_task_ids=list(ACTIVE_TASK_IDS),
            batch_index=training_monitor.batch_index,
            prompts=[str(prompt) for prompt in prompts],
            completions=[str(completion) for completion in completions],
            rewards=rewards,
            histories=histories,
            task_ids=[str(task_id) for task_id in t_ids],
            variant_seeds=[int(seed) for seed in v_seeds],
            monitor_summary=monitor_summary,
            reward_schedule=reward_schedule,
        )

        if curriculum and curriculum.should_use_adversarial():
            logger.info(
                "Adversarial trigger: tier=%d mean=%.2f",
                curriculum.tier_index,
                curriculum.summary()["recent_mean_score"],
            )
            try:
                weak_spots = curriculum.weak_spots(top_n=2)
                if USE_SENTINEL and USE_SENTINEL_ADVERSARIAL:
                    from training.adversarial import (
                        generate_sentinel_adversarial_cases,
                        save_sentinel_adversarial_cases,
                    )

                    cases = generate_sentinel_adversarial_cases(weak_spots, n=4)
                    save_sentinel_adversarial_cases(cases, SENTINEL_ADVERSARIAL_PATH)
                    prompt_state.sentinel_adversarial_cases = cases
                    logger.info("Generated %d SENTINEL adversarial worker cases", len(cases))
                elif GROQ_API_KEY:
                    from training.adversarial import AdversarialDesigner

                    designer = AdversarialDesigner(api_key=GROQ_API_KEY)
                    new_scenarios = designer.generate(weak_spots, n=3)
                    designer.save_generated("outputs/adversarial_scenarios.json")
                    logger.info("Generated %d adversarial scenarios", len(new_scenarios))
            except Exception as e:
                logger.debug("Adversarial generation failed: %s", e)

        if wandb_enabled:
            import wandb

            wandb_payload = {
                "monitor/reward_mean": monitor_summary["reward_mean"],
                "monitor/avg_steps": monitor_summary["avg_steps"],
                "monitor/running_reward_mean": monitor_summary["running_reward_mean"],
                "monitor/best_reward_mean": monitor_summary["best_reward_mean"],
                "monitor/unique_completion_ratio": monitor_summary.get("unique_completion_ratio", 0.0),
                "monitor/decision_entropy": monitor_summary.get("decision_entropy", 0.0),
                "monitor/decision_variety": monitor_summary.get("decision_variety", 0),
                "monitor/zero_reward_fraction": monitor_summary.get("zero_reward_fraction", 0.0),
                "monitor/trivially_solved_fraction": monitor_summary.get("trivially_solved_fraction", 0.0),
                "monitor/productive_fraction": monitor_summary.get("productive_fraction", 0.0),
                "monitor/effective_prompt_ratio": monitor_summary.get("effective_prompt_ratio", 0.0),
                "monitor/frontier_hit_rate": monitor_summary.get("frontier_hit_rate", 0.0),
                "monitor/task_diversity_ratio": monitor_summary.get("task_diversity_ratio", 0.0),
                "monitor/zero_gradient_group_fraction": monitor_summary.get("zero_gradient_group_fraction", 0.0),
                "monitor/adversarial_case_fraction": monitor_summary.get("adversarial_case_fraction", 0.0),
            }
            if monitor_summary.get("memory"):
                wandb_payload["monitor/memory_total_episodes"] = monitor_summary["memory"].get("total_episodes", 0)
                wandb_payload["monitor/memory_mistake_cards"] = monitor_summary["memory"].get("mistake_cards_stored", 0)
            if USE_SENTINEL:
                wandb_payload.update(
                    {
                        "monitor/detection_rate": monitor_summary.get("detection_rate", 0.0),
                        "monitor/false_positive_rate": monitor_summary.get("false_positive_rate", 0.0),
                        "monitor/risk_reduction_rate": monitor_summary.get("risk_reduction_rate", 0.0),
                        "monitor/twin_damage_reduction_rate": monitor_summary.get("twin_damage_reduction_rate", 0.0),
                        "monitor/twin_without_sentinel_damage_total": monitor_summary.get("twin_without_sentinel_damage_total", 0.0),
                        "monitor/twin_with_sentinel_damage_total": monitor_summary.get("twin_with_sentinel_damage_total", 0.0),
                        "monitor/worker_rehabilitation_rate": monitor_summary.get("worker_rehabilitation_rate", 0.0),
                        "monitor/coaching_quality": monitor_summary.get("coaching_quality", 0.0),
                    }
                )
            if reward_schedule:
                wandb_payload.update(
                    {
                        "monitor/reward_schedule_progress": reward_schedule.get("progress", 0.0),
                        "monitor/reward_schedule_stage": reward_schedule.get("stage", "unknown"),
                    }
                )
            if audit_path:
                wandb_payload["monitor/rollout_audit_saved"] = 1
            wandb.log(wandb_payload)

        return rewards

    # Create trainer
    trainer = GRPOTrainer(
        model            = model,
        processing_class = tokenizer,
        args             = grpo_config,
        train_dataset    = train_dataset,
        reward_funcs     = [reward_fn_with_curriculum],
    )
    stability_callback = GRPOStabilityCallback(
        training_monitor=training_monitor,
        initial_beta=KL_COEF,
        target_kl=KL_TARGET,
        adaptive=KL_ADAPTIVE,
        low_factor=KL_LOW_FACTOR,
        high_factor=KL_HIGH_FACTOR,
        beta_up_mult=KL_BETA_UP_MULT,
        beta_down_mult=KL_BETA_DOWN_MULT,
        min_beta=KL_MIN_BETA,
        max_beta=KL_MAX_BETA,
        hard_stop_enabled=KL_HARD_STOP_ENABLED,
        hard_stop_mult=KL_HARD_STOP_MULT,
    )
    trainer.add_callback(stability_callback)
    stability_callback.bind_trainer(trainer)

    # Train
    logger.info("Starting training...")
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    logger.info("Training complete in %.1f minutes", elapsed / 60)

    # Save final model
    final_path = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info("Saved final model to %s", final_path)

    # Save curriculum state
    if curriculum:
        logger.info("Curriculum summary: %s", curriculum.summary())
    if USE_AGENT_MEMORY:
        save_agent_memory(memory)
    if USE_SENTINEL and USE_FEEDBACK_MEMORY:
        save_feedback_memory(feedback_memory, SENTINEL_FEEDBACK_MEMORY_PATH)
    if warm_start_summary:
        logger.info("Warm-start summary: %s", warm_start_summary)
    if USE_SENTINEL:
        reset_reward_weights()

    # Plot reward curve
    _plot_reward_curve()
    try:
        from scripts.render_training_dashboard import render_dashboard

        render_dashboard(
            monitor_dir=TRAIN_MONITOR_DIR,
            output_dir="outputs/reward_curves",
        )
    except Exception as exc:
        logger.warning("Training dashboard render skipped: %s", exc)

    # Push to Hub (if HF_TOKEN set)
    hf_repo = os.getenv("HF_REPO")
    if hf_repo and HF_TOKEN:
        logger.info("Pushing to HuggingFace Hub: %s", hf_repo)
        trainer.model.push_to_hub(hf_repo, token=HF_TOKEN)
        tokenizer.push_to_hub(hf_repo, token=HF_TOKEN)
        logger.info("Done! Update openenv.yaml model: %s", hf_repo)

    if wandb_enabled:
        import wandb
        wandb.finish()

    return final_path


# ---------------------------------------------------------------------------
# Reward curve plot
# ---------------------------------------------------------------------------

def _plot_reward_curve():
    """Plot reward/mean over steps from wandb run or log file."""
    try:
        import matplotlib.pyplot as plt

        steps, rewards = [], []
        monitor_path = Path(TRAIN_MONITOR_DIR) / "training_metrics.jsonl"
        if monitor_path.exists():
            with monitor_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    steps.append(int(payload.get("batch_index", len(steps) + 1)))
                    rewards.append(float(payload.get("reward_mean", 0.0)))
        else:
            log_path = os.path.join(OUTPUT_DIR, "train.log")
            if not os.path.exists(log_path):
                return
            with open(log_path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "Batch rewards: mean=" in line:
                        try:
                            mean_str = line.split("mean=")[1].split(" ")[0]
                            steps.append(len(steps) + 1)
                            rewards.append(float(mean_str))
                        except Exception:
                            pass

        if not steps:
            return

        plt.figure(figsize=(10, 5))
        plt.plot(steps, rewards, linewidth=2, color="royalblue")
        plt.xlabel("Training Step")
        plt.ylabel("Mean Reward")
        plt.title("GRPO Training Reward Curve")
        plt.grid(True, alpha=0.3)

        # Smoothed line
        if len(rewards) > 10:
            window = min(10, len(rewards) // 5)
            smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
            smooth_steps = steps[:len(smoothed)]
            plt.plot(smooth_steps, smoothed, linewidth=2, color="red",
                     linestyle="--", label=f"Smoothed (w={window})")
            plt.legend()

        plot_path = "outputs/reward_curves/training_curve.png"
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("Saved reward curve to %s", plot_path)

    except ImportError:
        logger.info("matplotlib not installed - skipping reward plot")
    except Exception as e:
        logger.warning("Could not plot reward curve: %s", e)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GRPO training for OpenEnv")
    parser.add_argument("--steps",       type=int,   default=TRAIN_STEPS,     help="Training steps")
    parser.add_argument("--model",       type=str,   default=MODEL_NAME,      help="Model name/path")
    parser.add_argument("--lr",          type=float, default=LR,              help="Learning rate")
    parser.add_argument("--output",      type=str,   default=OUTPUT_DIR,      help="Output directory")
    parser.add_argument("--resume",      type=str,   default=RESUME_FROM,     help="Checkpoint to resume from")
    parser.add_argument("--warm-start-steps", type=int, default=WARM_START_STEPS, help="Optional small SFT-style warm-start steps before GRPO")
    parser.add_argument("--warm-start-only", action="store_true", help="Run only the warm-start stage and stop before GRPO")
    parser.add_argument("--dry-run",     action="store_true",                 help="Validate setup without training")
    args = parser.parse_args()

    # Override from CLI
    TRAIN_STEPS  = args.steps
    MODEL_NAME   = args.model
    LR           = args.lr
    OUTPUT_DIR   = args.output
    RESUME_FROM  = args.resume
    WARM_START_STEPS = args.warm_start_steps
    WARM_START_ONLY = args.warm_start_only or WARM_START_ONLY

    if args.dry_run:
        logger.info("DRY RUN: Validating environment and reward function...")
        
        if USE_SENTINEL:
            from sentinel.environment import SentinelEnv
            env = SentinelEnv()
            for task_id in SENTINEL_TASK_IDS:
                obs = env.reset(task_id=task_id, variant_seed=0)
                grade = env.grade()
                score = float(grade.score) if hasattr(grade, "score") else float(grade.get("score", 0.0))
                logger.info("  task=%s initial_grade=%.3f", task_id, score)
        else:
            from src.environment import IncidentResponseEnv
            env = IncidentResponseEnv()
            for task_id in TASK_IDS:
                obs   = env.reset(task_id=task_id, variant_seed=0)
                grade = env.grade()
                score = float(grade.score) if hasattr(grade, "score") else float(grade.get("score", 0.0))
                logger.info("  task=%s initial_grade=%.3f", task_id, score)

        if WARM_START_STEPS > 0:
            from training.memory import load_agent_memory
            from sentinel.feedback import load_feedback_memory

            warm_start_records = _build_warm_start_examples(
                task_ids=list(ACTIVE_TASK_IDS),
                memory=load_agent_memory(),
                feedback_memory=load_feedback_memory(SENTINEL_FEEDBACK_MEMORY_PATH),
                max_examples=max(1, min(WARM_START_DATASET_SIZE, 8)),
            )
            logger.info("  warm_start_examples=%d", len(warm_start_records))

        logger.info("DRY RUN PASSED. Environment is working.")
        sys.exit(0)

    final_path = train()
    logger.info("Training finished. Final model: %s", final_path)
    logger.info("Next steps:")
    logger.info("  1. python validate.py")
    logger.info("  2. Update openenv.yaml: model: <HF_REPO>")
    logger.info("  3. Submit!")
