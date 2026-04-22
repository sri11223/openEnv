from __future__ import annotations

from pathlib import Path

import proof_pack
from sentinel.environment import SentinelEnv
from sentinel.feedback import build_feedback_summary, empty_feedback_memory, record_feedback_event
from src.tasks import get_all_tasks
from training.adversarial import AdversarialDesigner
from training.curriculum import CurriculumController
from training.memory import load_agent_memory, record_episode
from train import (
    AdaptivePromptState,
    _aggregate_batch_metrics,
    _build_warm_start_examples,
    _sentinel_history_entry,
    _warm_start_sentinel_decision_for_observation,
)


def test_curriculum_filters_to_active_sentinel_tasks(tmp_path: Path):
    state_path = tmp_path / "curriculum_sentinel.json"
    ctrl = CurriculumController(
        state_path=str(state_path),
        active_task_ids=["basic_oversight", "fleet_monitoring_conflict"],
    )

    for _ in range(5):
        task_id, variant_seed = ctrl.select_episode()
        assert task_id in {"basic_oversight", "fleet_monitoring_conflict"}
        assert variant_seed in {0, 1, 2}


def test_adaptive_prompt_state_uses_updated_memory_within_run(tmp_path: Path):
    memory = load_agent_memory(str(tmp_path / "agent_memory.json"))
    state = AdaptivePromptState(
        task_ids=["basic_oversight"],
        memory=memory,
        feedback_memory=empty_feedback_memory(),
        memory_context="",
        max_seeds=1,
    )

    first = state.next_prompt_record()
    assert "Blocked too late" not in first["prompt"]

    state.memory = record_episode(
        state.memory,
        {
            "task_id": "basic_oversight",
            "score": 0.25,
            "steps": 4,
            "trajectory_summary": "Allowed a hallucinated service before intervening.",
            "mistakes": ["Blocked too late on a hallucinated service proposal."],
            "successes": [],
        },
    )
    second = state.next_prompt_record()
    assert "Blocked too late on a hallucinated service proposal." in second["prompt"]


def test_adversarial_designer_has_deterministic_fallback(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)

    designer = AdversarialDesigner(api_key="")
    scenarios = designer.generate([("severity_classification", 0)], n=1)

    assert len(scenarios) == 1
    scenario = scenarios[0]
    assert scenario["scenario_id"].startswith("fallback_")
    assert scenario["task_id"] == "severity_classification"
    assert scenario["correct_root_cause_service"]
    assert scenario["initial_alerts"]


def test_feedback_memory_persists_successful_rehabilitation():
    memory = empty_feedback_memory()
    updated = record_feedback_event(
        memory,
        {
            "task_id": "basic_oversight",
            "incident_id": "INC-1",
            "incident_label": "Incident 1",
            "worker_id": "worker_app",
            "decision": "BLOCK",
            "reason": "reward_hacking",
            "action_type": "classify",
            "target": "payment-gateway",
            "is_misbehavior": True,
            "revision_attempted": True,
            "revision_approved": True,
            "revised_by": "worker_app",
            "revised_action_type": "investigate",
            "revised_target": "payment-gateway",
            "executed_action_source": "revised",
        },
    )

    summary = build_feedback_summary(updated, worker_id="worker_app", task_id="basic_oversight")
    assert summary["worker_rehabilitations"]
    assert "recovered safely" in summary["worker_rehabilitations"][0]


def test_public_sentinel_task_schema_matches_corrective_loop():
    sentinel_task = next(task for task in get_all_tasks() if task.task_id == "basic_oversight")
    schema = sentinel_task.action_schema

    assert {"required": ["decision"]} in schema["anyOf"]
    assert {"required": ["action"]} in schema["anyOf"]
    assert "worker_message" in schema["properties"]
    assert "required_evidence" in schema["properties"]
    assert "suggested_action_type" in schema["properties"]
    assert "suggested_target" in schema["properties"]
    assert "suggested_parameters" in schema["properties"]


def test_warm_start_decision_contains_rich_oversight_fields():
    env = SentinelEnv()
    obs = env.reset("adversarial_worker", variant_seed=0)

    decision = _warm_start_sentinel_decision_for_observation(obs)

    assert decision["decision"] in {"APPROVE", "BLOCK", "REDIRECT", "REASSIGN", "FLAG"}
    assert "worker_message" in decision
    assert "required_evidence" in decision
    assert "constitutional_violations" in decision
    assert "counterfactual" in decision


def test_build_warm_start_examples_respects_requested_size(tmp_path: Path):
    records = _build_warm_start_examples(
        task_ids=["basic_oversight"],
        memory=load_agent_memory(str(tmp_path / "agent_memory.json")),
        feedback_memory=empty_feedback_memory(),
        max_examples=3,
        max_seeds=1,
    )

    assert len(records) == 3
    assert all(record["task_id"] == "basic_oversight" for record in records)
    assert all("<|assistant|>" in record["text"] for record in records)


def test_aggregate_batch_metrics_surfaces_sentinel_monitoring():
    env = SentinelEnv()
    obs = env.reset("basic_oversight", variant_seed=0)
    decision = _warm_start_sentinel_decision_for_observation(obs)
    result = env.step(decision)

    metrics = _aggregate_batch_metrics(
        rewards=[float(result.sentinel_reward.total)],
        histories=[[_sentinel_history_entry(decision, result)]],
        task_ids=["basic_oversight"],
        variant_seeds=[0],
        curriculum_summary={"tier": "warmup"},
        prompt_refreshes=1,
    )

    assert "detection_rate" in metrics
    assert "false_positive_rate" in metrics
    assert "risk_reduction_rate" in metrics
    assert "worker_rehabilitation_rate" in metrics
    assert "basic_oversight" in metrics["per_task"]


def test_proof_pack_prefers_structured_monitoring_file(tmp_path: Path, monkeypatch):
    metrics_path = tmp_path / "training_metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            [
                '{"batch_index": 1, "reward_mean": 0.15}',
                '{"batch_index": 2, "reward_mean": 0.35}',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(proof_pack, "MONITOR_CANDIDATES", [metrics_path])

    rewards, source = proof_pack._load_reward_points([])

    assert rewards == [0.15, 0.35]
    assert source == str(metrics_path)
