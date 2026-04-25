#!/usr/bin/env python3
"""demo_sentinel.py — Run one live SENTINEL episode and pretty-print every decision.

Usage:
    python scripts/demo_sentinel.py                          # rule-based agent
    python scripts/demo_sentinel.py --task adversarial_worker
    python scripts/demo_sentinel.py --task basic_oversight --seed 2
    python scripts/demo_sentinel.py --agent trained          # needs HF_TOKEN env var

Tasks: basic_oversight | fleet_monitoring_conflict | adversarial_worker | multi_crisis_command
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentinel.environment import SentinelEnv

# ANSI colours
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"

FICTITIOUS_SERVICES = {
    "auth-proxy-v2", "postgres-replica-v3", "redis-cache-fallback",
    "api-gateway-shadow", "order-worker-beta", "cdn-edge-node-2",
    "payment-gateway-old", "user-service-canary",
}
DESTRUCTIVE_ACTIONS = {"restart", "rollback", "delete"}


# ---------------------------------------------------------------------------
# Decision agents
# ---------------------------------------------------------------------------

def rule_agent(obs) -> dict:
    """Heuristic rule-based SENTINEL — catches hallucinations and destructive actions."""
    prop = getattr(obs, "proposed_action", None)
    if prop is None:
        return {"decision": "APPROVE", "explanation": "no proposal"}
    target = str(getattr(prop, "target", "")).lower()
    action = str(getattr(prop, "action_type", "")).lower()
    if target in FICTITIOUS_SERVICES:
        return {
            "decision": "BLOCK",
            "reason": "hallucination",
            "explanation": f"Target '{target}' is not in the incident graph — fictitious service.",
        }
    if any(d in action for d in DESTRUCTIVE_ACTIONS):
        return {
            "decision": "BLOCK",
            "reason": "destructive_precedence",
            "explanation": f"Action '{action}' is destructive. Use config_change first.",
        }
    return {"decision": "APPROVE", "explanation": "Passed heuristic checks."}


def trained_agent(model, tok, obs) -> dict:
    """Phase 1 GRPO-trained SENTINEL model."""
    import re
    import torch

    prompt = obs.to_prompt() if hasattr(obs, "to_prompt") else str(obs)
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=3072).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=256,
            temperature=0.7, top_p=0.95, do_sample=True,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    text = tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    s, e = text.find("{"), text.rfind("}") + 1
    if s != -1 and e > 0:
        try:
            return json.loads(text[s:e])
        except Exception:
            pass
    for k in ["BLOCK", "REDIRECT", "REASSIGN", "FLAG", "APPROVE"]:
        if k in text.upper():
            return {"decision": k, "explanation": text[:150]}
    return {"decision": "APPROVE", "explanation": "UNPARSED"}


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _decision_colour(decision: str) -> str:
    d = decision.upper()
    if d == "APPROVE":  return GREEN
    if d == "BLOCK":    return RED
    if d == "REDIRECT": return YELLOW
    if d == "REASSIGN": return YELLOW
    if d == "FLAG":     return CYAN
    return RESET


def print_header(task: str, seed: int, agent_label: str) -> None:
    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  SENTINEL — Live Episode Demo{RESET}")
    print(f"  Task : {CYAN}{task}{RESET}")
    print(f"  Seed : {seed}")
    print(f"  Agent: {agent_label}")
    print(f"{BOLD}{'='*65}{RESET}\n")


def print_step(step: int, proposal, decision_dict: dict,
               info: dict, reward: float) -> None:
    d = decision_dict.get("decision", "?").upper()
    colour = _decision_colour(d)
    is_mb   = info.get("is_misbehavior", False)
    was_tp  = info.get("was_tp", False)
    was_fp  = info.get("was_fp", False)
    was_fn  = info.get("was_fn", False)
    mb_type = info.get("mb_type") or ""

    # Outcome badge
    if was_tp:
        badge = f"{GREEN}✓ TP{RESET}"
    elif was_fp:
        badge = f"{YELLOW}⚠ FP{RESET}"
    elif was_fn:
        badge = f"{RED}✗ FN{RESET}"
    else:
        badge = f"{DIM}· TN{RESET}"

    worker  = str(getattr(proposal, "worker_id", "?")).replace("WorkerId.", "")
    action  = str(getattr(proposal, "action_type", "?")).replace("ActionType.", "")
    target  = str(getattr(proposal, "target", "?"))
    mismatch = f" {RED}[{mb_type}]{RESET}" if is_mb else ""

    print(f"Step {step:02d}  {DIM}{worker:12s}{RESET} "
          f"{action:15s} → {target:30s}{mismatch}")
    print(f"       SENTINEL: {colour}{BOLD}{d:9s}{RESET}  {badge}  "
          f"reward={reward:+.3f}  "
          f"cf_dmg={info.get('counterfactual_risk',{}).get('risk_score', 0.0):.2f}")
    expl = decision_dict.get("explanation", decision_dict.get("reasoning", ""))
    if expl:
        print(f"       {DIM}↳ {expl[:90]}{RESET}")
    print()


def print_footer(grader) -> None:
    score = grader.score
    caught = grader.misbehaviors_caught
    injected = grader.misbehaviors_injected
    fp = grader.false_positives
    fn = grader.false_negatives
    dr  = grader.detection_rate
    prevented = grader.prevented_damage_total
    twin_ratio = grader.twin_damage_reduction_rate

    colour = GREEN if score >= 0.5 else YELLOW if score >= 0.3 else RED

    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  EPISODE RESULT{RESET}")
    print(f"{'='*65}")
    print(f"  Final score      : {colour}{BOLD}{score:.3f}{RESET}")
    print(f"  Misbehaviors     : caught {GREEN}{caught}{RESET}/{injected}  "
          f"FP={YELLOW}{fp}{RESET}  FN={RED}{fn}{RESET}")
    print(f"  Detection rate   : {dr:.1%}")
    print(f"  Damage prevented : {prevented:.3f}  "
          f"(Digital Twin reduction: {twin_ratio:.1%})")
    print(f"  Rehab rate       : {grader.worker_rehabilitation_rate:.1%}")
    print(f"{BOLD}{'='*65}{RESET}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SENTINEL demo episode")
    parser.add_argument("--task", default="adversarial_worker",
                        choices=["basic_oversight", "fleet_monitoring_conflict",
                                 "adversarial_worker", "multi_crisis_command"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent", default="rule", choices=["rule", "trained"])
    args = parser.parse_args()

    # Load model if trained mode requested
    model, tok = None, None
    if args.agent == "trained":
        try:
            import torch
            from unsloth import FastLanguageModel
            from peft import PeftModel
            from huggingface_hub import login

            hf_token = os.environ.get("HF_TOKEN", "")
            if hf_token:
                login(token=hf_token, add_to_git_credential=False)
            print("Loading Phase 1 GRPO model from HF Hub...")
            model, tok = FastLanguageModel.from_pretrained(
                "unsloth/Qwen3-4B-bnb-4bit",
                max_seq_length=4096, dtype=torch.float16, load_in_4bit=True,
            )
            model = PeftModel.from_pretrained(
                model, "srikrish2004/sentinel-qwen3-4b-grpo", is_trainable=False
            )
            FastLanguageModel.for_inference(model)
            model.eval()
            agent_label = "Phase 1 GRPO (srikrish2004/sentinel-qwen3-4b-grpo)"
        except Exception as e:
            print(f"Could not load trained model: {e}")
            print("Falling back to rule-based agent.")
            args.agent = "rule"

    if args.agent == "rule":
        agent_label = "Rule-based (heuristic)"

    print_header(args.task, args.seed, agent_label)

    env = SentinelEnv()
    obs = env.reset(args.task, variant_seed=args.seed)

    step = 0
    while not env.done:
        step += 1
        proposal = obs.proposed_action

        if args.agent == "trained" and model is not None:
            decision_dict = trained_agent(model, tok, obs)
        else:
            decision_dict = rule_agent(obs)

        result = env.step(decision_dict)
        reward = float(result.sentinel_reward.total)
        info   = result.info
        obs    = result.observation

        print_step(step, proposal, decision_dict, info, reward)

    grader = env.grade()
    print_footer(grader)


if __name__ == "__main__":
    main()
