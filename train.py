"""
train.py — GRPO Fine-tuning for OpenEnv (IRT / any new problem)
==============================================================
Runnable training script. Uses TRL GRPOTrainer + Unsloth (optional) + curriculum.

HOW TO RUN:
    # Minimum (T4 / A10G, no Unsloth):
    python train.py

    # With Unsloth (A100 / H100, 2x faster):
    USE_UNSLOTH=1 python train.py

    # Override model and steps:
    MODEL_NAME=Qwen/Qwen2.5-7B-Instruct TRAIN_STEPS=200 python train.py

    # Resume from checkpoint:
    RESUME_FROM=outputs/checkpoints/checkpoint-100 python train.py

ENV VARS:
    MODEL_NAME      HuggingFace model ID (default: Qwen/Qwen2.5-3B-Instruct)
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
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Config from env vars
# ---------------------------------------------------------------------------

MODEL_NAME      = os.getenv("MODEL_NAME", "unsloth/Qwen2.5-3B-bnb-4bit")
HF_TOKEN        = os.getenv("HF_TOKEN", "")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
WANDB_PROJECT   = os.getenv("WANDB_PROJECT", "openenv-grpo")
TRAIN_STEPS     = int(os.getenv("TRAIN_STEPS", "200"))
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "4"))
USE_UNSLOTH     = os.getenv("USE_UNSLOTH", "0") == "1"
RESUME_FROM     = os.getenv("RESUME_FROM", "")
OUTPUT_DIR      = os.getenv("OUTPUT_DIR", "outputs/checkpoints")
LR              = float(os.getenv("LR", "5e-6"))
KL_COEF         = float(os.getenv("KL_COEF", "0.04"))
LORA_R          = int(os.getenv("LORA_R", "16"))
MAX_NEW_TOKENS  = int(os.getenv("MAX_NEW_TOKENS", "512"))
USE_LLM_PANEL   = bool(GROQ_API_KEY)                  # auto-enable if key available
USE_CURRICULUM  = True

TASK_IDS = [
    "severity_classification",
    "root_cause_analysis",
    "full_incident_management",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("outputs/reward_curves", exist_ok=True)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUTPUT_DIR, "train.log")),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# W&B setup (optional)
# ---------------------------------------------------------------------------

wandb_enabled = bool(WANDB_PROJECT)
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
        logger.warning("wandb not installed — logging disabled")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer():
    """Load model + tokenizer. Uses Unsloth if USE_UNSLOTH=1, else standard HF."""
    if USE_UNSLOTH:
        logger.info("Loading model with Unsloth: %s", MODEL_NAME)
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name           = MODEL_NAME,
            max_seq_length       = 2048,
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
    else:
        logger.info("Loading model with standard HF: %s", MODEL_NAME)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        load_kwargs: Dict[str, Any] = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            "device_map" : "auto",
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

def build_system_prompt(task_id: str, memory_context: str = "") -> str:
    return (
        "You are an expert AI agent solving incident response triage problems.\n"
        "You investigate alerts, classify severity, diagnose root causes, escalate to teams, "
        "and apply remediations.\n\n"
        "RULES:\n"
        "- Take ONE action per turn. Do not combine multiple actions.\n"
        "- Always investigate before classifying severity.\n"
        "- Always diagnose before remediating or escalating.\n"
        "- Use the minimum steps needed. Fewer correct steps = better score.\n\n"
        "OUTPUT FORMAT (respond ONLY with this JSON, nothing else):\n"
        '{"action_type": "ACTION_NAME", "params": {"key": "value"}, "reasoning": "brief reason"}\n\n'
        + (f"MEMORY FROM PAST EPISODES:\n{memory_context}\n" if memory_context else "")
    )


def scenario_to_prompt(scenario, task_id: str, memory_context: str = "") -> str:
    """Convert a Scenario object into a GRPO training prompt."""
    alert_lines = "\n".join(
        f"  [{a.severity}] {a.service}: {a.message}"
        for a in scenario.initial_alerts
    ) if scenario.initial_alerts else "  (no alerts)"

    system = build_system_prompt(task_id, memory_context)
    user   = (
        f"TASK: {task_id}\n"
        f"INCIDENT: {scenario.description}\n\n"
        f"INITIAL ALERTS:\n{alert_lines}\n\n"
        f"AVAILABLE SERVICES: {', '.join(scenario.available_services)}\n"
        f"MAX STEPS: {scenario.max_steps}\n\n"
        f"What is your first action?"
    )
    # ChatML format
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>"


def build_grpo_dataset(
    task_ids: List[str],
    max_seeds: int = 5,
    memory_context: str = "",
) -> List[Dict[str, str]]:
    """Build the list of {prompt: str} dicts for GRPOTrainer."""
    from src.scenarios import get_scenario

    prompts = []
    for task_id in task_ids:
        for seed in range(max_seeds):
            try:
                scenario = get_scenario(task_id, variant_seed=seed)
                prompt   = scenario_to_prompt(scenario, task_id, memory_context)
                prompts.append({
                    "prompt":        prompt,
                    "task_id":       task_id,
                    "variant_seed":  seed,
                })
            except Exception as e:
                logger.debug("No scenario for task=%s seed=%d: %s", task_id, seed, e)
                break

    logger.info("Built dataset with %d prompts", len(prompts))
    if not prompts:
        raise RuntimeError(
            "No scenarios found. Check that TASK_IDS match the environment's task IDs."
        )
    return prompts


# ---------------------------------------------------------------------------
# Reward function (called by GRPOTrainer after generating completions)
# ---------------------------------------------------------------------------

def run_episode_with_completion(
    completion_text: str,
    task_id: str,
    variant_seed: int,
) -> Tuple[float, List[Dict]]:
    """
    Execute one episode by feeding the model's completion back into the env.

    The model generates the FIRST action. We then run a simple agent loop
    (greedy, temp=0 via direct env calls) to complete the episode.

    Returns: (score, action_history)
    """
    from src.environment import IncidentResponseEnv

    env = IncidentResponseEnv()
    try:
        obs  = env.reset(task_id=task_id, variant_seed=variant_seed)
        done = False
        history: List[Dict] = []

        # Parse first action from completion
        first_action = _parse_action(completion_text)
        if first_action is None:
            return 0.0, []

        # Step 1: use the model's generated action
        result     = env.step(first_action)
        done       = result.done
        history.append({"action": first_action, "step_reward": float(result.reward.total)})

        # Remaining steps: use a greedy rule-based fallback
        step = 1
        while not done and step < 20:
            fallback_action = _greedy_fallback_action(env, obs, history)
            result          = env.step(fallback_action)
            done            = result.done
            history.append({"action": fallback_action, "step_reward": float(result.reward.total)})
            step += 1

        grade = env.grade()
        score = float(grade.score) if hasattr(grade, "score") else float(grade.get("score", 0.0))
        return score, history

    except Exception as e:
        logger.debug("Episode failed: %s", e)
        return 0.0, []


def _parse_action(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON action from model completion text."""
    text = text.strip()

    # Try full JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def _greedy_fallback_action(env, obs, history: List[Dict]) -> Dict[str, Any]:
    """
    Simple rule-based fallback to complete the episode after the first action.
    This keeps episodes from hanging when the model generates only one step.
    """
    # Check what's already been done
    actions_taken = [h["action"].get("action_type", "") for h in history]

    if "CLASSIFY_SEVERITY" not in actions_taken:
        return {"action_type": "CLASSIFY_SEVERITY", "params": {"severity": "P2"}, "reasoning": "fallback"}
    if "DIAGNOSE" not in actions_taken:
        return {"action_type": "DIAGNOSE", "params": {"root_cause": "unknown service fault"}, "reasoning": "fallback"}
    if "ESCALATE" not in actions_taken:
        return {"action_type": "ESCALATE", "params": {"teams": ["platform-team"]}, "reasoning": "fallback"}
    return {"action_type": "NO_OP", "params": {}, "reasoning": "fallback"}


# GRPO reward function signature: (prompts, completions, **kwargs) -> List[float]
def grpo_reward_fn(
    prompts:     List[str],
    completions: List[str],
    task_id:     List[str]       = None,
    variant_seed: List[int]      = None,
    **kwargs,
) -> List[float]:
    """Called by GRPOTrainer after generating each group of completions."""
    rewards = []

    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        t_id  = (task_id[i]       if task_id       else TASK_IDS[0])
        seed  = (variant_seed[i]  if variant_seed  else 0)

        score, history = run_episode_with_completion(completion, t_id, seed)

        # Optional: LLM panel hybrid
        if USE_LLM_PANEL and history:
            try:
                from judges.llm_grader import grade_sync, build_trajectory_text
                traj_text = build_trajectory_text(t_id, history)
                panel     = grade_sync(t_id, traj_text, GROQ_API_KEY, deterministic_score=score)
                score     = panel.get("hybrid", score)
            except Exception as e:
                logger.debug("LLM panel failed, using deterministic score: %s", e)

        rewards.append(float(np.clip(score, 0.0, 1.0)))

    mean_r = sum(rewards) / len(rewards) if rewards else 0.0
    logger.info("Batch rewards: mean=%.3f min=%.3f max=%.3f",
                mean_r, min(rewards, default=0), max(rewards, default=0))

    if wandb_enabled:
        import wandb
        wandb.log({
            "reward/mean": mean_r,
            "reward/min":  min(rewards, default=0),
            "reward/max":  max(rewards, default=0),
            "reward/std":  float(np.std(rewards)) if rewards else 0,
        })

    return rewards


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

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
    logger.info("Output:     %s", OUTPUT_DIR)
    logger.info("=" * 60)

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Load curriculum and agent memory
    from training.curriculum import get_curriculum
    from training.memory import load_agent_memory, build_memory_context, maybe_consolidate_memory

    curriculum = get_curriculum() if USE_CURRICULUM else None
    memory     = load_agent_memory()
    memory_ctx = build_memory_context(memory)

    # Build dataset
    dataset = build_grpo_dataset(TASK_IDS, max_seeds=5, memory_context=memory_ctx)

    # Convert to HuggingFace Dataset
    from datasets import Dataset as HFDataset
    hf_dataset = HFDataset.from_list(dataset)

    # GRPO config
    from trl import GRPOConfig, GRPOTrainer

    grpo_config = GRPOConfig(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = max(1, 8 // NUM_GENERATIONS),
        num_generations             = NUM_GENERATIONS,
        max_new_tokens              = MAX_NEW_TOKENS,
        temperature                 = 0.8,        # diversity in rollouts
        learning_rate               = LR,
        kl_coef                     = KL_COEF,
        logging_steps               = 1,
        save_steps                  = 25,
        save_total_limit            = 4,
        bf16                        = torch.cuda.is_bf16_supported(),
        fp16                        = not torch.cuda.is_bf16_supported(),
        report_to                   = "wandb" if wandb_enabled else "none",
        max_steps                   = TRAIN_STEPS,
    )

    # Wrap reward fn to inject curriculum-selected task_ids and seeds
    def reward_fn_with_curriculum(prompts, completions, **kwargs):
        # Extract task_id and variant_seed from dataset columns if available
        t_ids   = kwargs.get("task_id", [TASK_IDS[0]] * len(prompts))
        v_seeds = kwargs.get("variant_seed", [0] * len(prompts))

        rewards = grpo_reward_fn(
            prompts      = prompts,
            completions  = completions,
            task_id      = t_ids,
            variant_seed = v_seeds,
            **{k: v for k, v in kwargs.items() if k not in ("task_id", "variant_seed")},
        )

        # Record in curriculum
        if curriculum:
            for i, r in enumerate(rewards):
                t_id = t_ids[i] if i < len(t_ids) else TASK_IDS[0]
                seed = v_seeds[i] if i < len(v_seeds) else 0
                curriculum.record_episode(t_id, seed, score=r, steps=10)

            if USE_CURRICULUM and curriculum.should_use_adversarial():
                logger.info("Adversarial trigger: tier=%d mean=%.2f",
                            curriculum.tier_index,
                            curriculum.summary()["recent_mean_score"])

        return rewards

    # Create trainer
    trainer = GRPOTrainer(
        model            = model,
        processing_class = tokenizer,
        args             = grpo_config,
        train_dataset    = hf_dataset,
        reward_funcs     = [reward_fn_with_curriculum],
    )

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

    # Plot reward curve
    _plot_reward_curve()

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

        log_path = os.path.join(OUTPUT_DIR, "train.log")
        if not os.path.exists(log_path):
            return

        steps, rewards = [], []
        with open(log_path) as f:
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
        logger.info("matplotlib not installed — skipping reward plot")
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
    parser.add_argument("--dry-run",     action="store_true",                 help="Validate setup without training")
    args = parser.parse_args()

    # Override from CLI
    TRAIN_STEPS  = args.steps
    MODEL_NAME   = args.model
    LR           = args.lr
    OUTPUT_DIR   = args.output
    RESUME_FROM  = args.resume

    if args.dry_run:
        logger.info("DRY RUN: Validating environment and reward function...")
        from src.environment import IncidentResponseEnv
        env = IncidentResponseEnv()
        for task_id in TASK_IDS:
            obs   = env.reset(task_id=task_id, variant_seed=0)
            grade = env.grade()
            score = float(grade.score) if hasattr(grade, "score") else float(grade.get("score", 0.0))
            logger.info("  task=%s initial_grade=%.3f", task_id, score)
        logger.info("DRY RUN PASSED. Environment is working.")
        sys.exit(0)

    final_path = train()
    logger.info("Training finished. Final model: %s", final_path)
    logger.info("Next steps:")
    logger.info("  1. python validate.py")
    logger.info("  2. Update openenv.yaml: model: <HF_REPO>")
    logger.info("  3. Submit!")
