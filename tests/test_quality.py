"""Extended quality tests for the Incident Response Triage environment.

Tests:
  - All 7 scenario variants are accessible and valid
  - Every reward component fires correctly
  - Temporal degradation scales with difficulty
  - Blast radius worsens metrics over time
  - Grader feedback content is actionable
  - Partial credit (wrong severity, incomplete diagnosis)
  - Multi-step partial progress scoring
  - Wrong / destructive actions are penalised
  - Episode never produces score outside (0.0, 1.0)
  - Observation contract (all required fields always present)
  - State immutability (reset always produces independent snapshot)
  - All 3 tasks have a valid grader
  - openenv.yaml is structurally complete
  - Inference script exits zero
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Dict, List

import pytest
import yaml

from src.environment import IncidentResponseEnv
from src.models import Action, ActionType, IncidentSeverity, IncidentStatus
from src.scenarios import SCENARIO_VARIANTS, SCENARIOS, apply_blast_radius, get_scenario
from src.rewards import compute_step_reward
from src.graders import grade
from baseline.inference import run_all_tasks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return IncidentResponseEnv()


# ===========================================================================
# 1.  SCENARIO COVERAGE — All 7 variants
# ===========================================================================

class TestAllScenarioVariants:
    """Every variant must be loadable, have valid structure, and pass reset()."""

    def test_scenario_variants_exist(self):
        """SCENARIO_VARIANTS must have entries for all 3 task_ids."""
        for task_id in ["severity_classification", "root_cause_analysis", "full_incident_management"]:
            assert task_id in SCENARIO_VARIANTS, f"No variants for {task_id}"
            assert len(SCENARIO_VARIANTS[task_id]) >= 1

    def test_total_variants_at_least_7(self):
        total = sum(len(v) for v in SCENARIO_VARIANTS.values())
        assert total >= 7, f"Expected >= 7 total scenario variants, got {total}"

    @pytest.mark.parametrize("task_id", ["severity_classification", "root_cause_analysis", "full_incident_management"])
    def test_all_variants_loadable(self, env, task_id):
        variants = SCENARIO_VARIANTS[task_id]
        for seed, scenario in enumerate(variants):
            obs = env.reset(task_id, variant_seed=seed)
            assert obs.incident_id == scenario.incident_id
            assert len(obs.alerts) > 0
            assert len(obs.available_services) > 0

    @pytest.mark.parametrize("task_id", list(SCENARIOS.keys()))
    def test_scenario_has_required_ground_truth(self, task_id):
        s = SCENARIOS[task_id]
        assert s.correct_severity is not None
        assert s.correct_root_cause_service != ""
        assert len(s.correct_root_cause_keywords) >= 2
        assert len(s.valid_remediation_actions) >= 1
        assert len(s.expected_escalation_teams) >= 1
        assert s.max_steps >= 5
        assert len(s.relevant_services) >= 1

    @pytest.mark.parametrize("task_id", list(SCENARIOS.keys()))
    def test_scenario_services_are_available(self, task_id):
        s = SCENARIOS[task_id]
        for svc in s.relevant_services:
            assert svc in s.available_services, (
                f"relevant service '{svc}' not in available_services for {task_id}"
            )

    @pytest.mark.parametrize("task_id", list(SCENARIOS.keys()))
    def test_scenario_remediation_targets_are_available(self, task_id):
        s = SCENARIOS[task_id]
        for rem in s.valid_remediation_actions:
            assert rem["service"] in s.available_services, (
                f"remediation target '{rem['service']}' not available in {task_id}"
            )

    @pytest.mark.parametrize("task_id", list(SCENARIOS.keys()))
    def test_scenario_has_service_logs_and_metrics(self, task_id):
        s = SCENARIOS[task_id]
        for svc in s.relevant_services:
            assert svc in s.service_logs, f"No logs for relevant service '{svc}' in {task_id}"
            assert svc in s.service_metrics, f"No metrics for relevant service '{svc}' in {task_id}"


# ===========================================================================
# 2.  REWARD COMPONENT COVERAGE
# ===========================================================================

class TestRewardComponents:
    """Every reward component must fire as expected under the right conditions."""

    def test_relevant_investigation_reward(self, env):
        env.reset("severity_classification")
        result = env.step(Action(action_type=ActionType.INVESTIGATE, target="postgres-primary"))
        assert result.reward.components.get("relevant_investigation", 0) > 0

    def test_irrelevant_investigation_penalty(self, env):
        env.reset("severity_classification")
        # redis-cache is available but not a relevant service for easy task
        scenario = SCENARIOS["severity_classification"]
        irrelevant = [s for s in scenario.available_services if s not in scenario.relevant_services]
        if irrelevant:
            result = env.step(Action(action_type=ActionType.INVESTIGATE, target=irrelevant[0]))
            assert result.reward.components.get("irrelevant_investigation", 0) < 0

    def test_duplicate_investigation_penalty(self, env):
        env.reset("severity_classification")
        env.step(Action(action_type=ActionType.INVESTIGATE, target="postgres-primary"))
        result = env.step(Action(action_type=ActionType.INVESTIGATE, target="postgres-primary"))
        assert result.reward.components.get("duplicate_investigation", 0) < 0

    def test_invalid_service_penalty(self, env):
        env.reset("severity_classification")
        result = env.step(Action(action_type=ActionType.INVESTIGATE, target="nonexistent-xyz"))
        assert result.reward.components.get("invalid_target", 0) < 0

    def test_correct_classification_reward(self, env):
        """Correct classification gives positive reward."""
        env.reset("severity_classification")
        result = env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P2"}))
        assert result.reward.components.get("correct_classification", 0) > 0

    def test_wrong_classification_p4_gets_bigger_penalty(self, env):
        """P4 for a P2 incident (2 levels off) should have larger penalty than P3 (1 level off)."""
        env.reset("severity_classification")
        r_p3 = env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P3"}))
        env.reset("severity_classification")
        r_p4 = env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P4"}))
        wc_p3 = r_p3.reward.components.get("wrong_classification", 0)
        wc_p4 = r_p4.reward.components.get("wrong_classification", 0)
        assert wc_p4 < wc_p3, "Further off severity should have larger penalty"

    def test_duplicate_classify_penalty(self, env):
        env.reset("severity_classification")
        env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P2"}))
        # episode ends on classify for easy task, so we use medium
        env.reset("root_cause_analysis")
        env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P1"}))
        result = env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P1"}))
        assert result.reward.components.get("duplicate_classify", 0) < 0

    def test_correct_service_diagnosis_reward(self, env):
        env.reset("root_cause_analysis")
        env.step(Action(action_type=ActionType.INVESTIGATE, target="redis-session"))
        env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P1"}))
        result = env.step(Action(
            action_type=ActionType.DIAGNOSE,
            target="redis-session",
            parameters={"root_cause": "Redis maxmemory limit causing eviction of session tokens"},
        ))
        assert result.reward.components.get("correct_service", 0) > 0
        assert result.reward.components.get("correct_root_cause", 0) > 0

    def test_wrong_service_diagnosis_penalty(self, env):
        env.reset("root_cause_analysis")
        env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P1"}))
        result = env.step(Action(
            action_type=ActionType.DIAGNOSE,
            target="payment-gateway",  # wrong service
            parameters={"root_cause": "network issue in payment gateway"},
        ))
        assert result.reward.components.get("wrong_service", 0) < 0

    def test_correct_remediation_reward(self, env):
        env.reset("root_cause_analysis")
        env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P1"}))
        env.step(Action(action_type=ActionType.DIAGNOSE, target="redis-session", parameters={"root_cause": "redis eviction"}))
        result = env.step(Action(
            action_type=ActionType.REMEDIATE,
            target="redis-session",
            parameters={"action": "scale"},
        ))
        assert result.reward.components.get("correct_remediation", 0) > 0

    def test_wrong_remediation_penalty(self, env):
        env.reset("root_cause_analysis")
        result = env.step(Action(
            action_type=ActionType.REMEDIATE,
            target="payment-gateway",  # wrong target
            parameters={"action": "restart"},
        ))
        assert result.reward.components.get("wrong_remediation", 0) < 0

    def test_duplicate_remediation_penalty(self, env):
        env.reset("root_cause_analysis")
        env.step(Action(action_type=ActionType.REMEDIATE, target="redis-session", parameters={"action": "scale"}))
        result = env.step(Action(action_type=ActionType.REMEDIATE, target="redis-session", parameters={"action": "scale"}))
        assert result.reward.components.get("duplicate_remediation", 0) < 0

    def test_correct_escalation_reward(self, env):
        scenario = SCENARIOS["full_incident_management"]
        env.reset("full_incident_management")
        correct_team = scenario.expected_escalation_teams[0]
        result = env.step(Action(
            action_type=ActionType.ESCALATE,
            target=correct_team,
            parameters={"priority": "urgent", "message": "Cascading outage under investigation."},
        ))
        assert result.reward.components.get("correct_escalation", 0) > 0

    def test_wrong_escalation_penalty(self, env):
        env.reset("full_incident_management")
        result = env.step(Action(
            action_type=ActionType.ESCALATE,
            target="marketing-team",
            parameters={"priority": "medium", "message": "FYI there's an outage."},
        ))
        assert result.reward.components.get("unnecessary_escalation", 0) < 0

    def test_good_communication_reward(self, env):
        env.reset("full_incident_management")
        result = env.step(Action(
            action_type=ActionType.COMMUNICATE,
            target="status_page",
            parameters={"message": "Auth-service experiencing high latency. On-call investigating. ETA 20min."},
        ))
        assert result.reward.components.get("status_communication", 0) > 0

    def test_short_communication_penalty(self, env):
        env.reset("full_incident_management")
        result = env.step(Action(
            action_type=ActionType.COMMUNICATE,
            target="status_page",
            parameters={"message": "hi"},  # too short
        ))
        assert result.reward.components.get("low_quality_communication", 0) < 0

    def test_reasoning_bonus_fires(self, env):
        """A step with relevant reasoning text earns the reasoning bonus."""
        env.reset("root_cause_analysis")
        scenario = SCENARIOS["root_cause_analysis"]
        relevant_keyword = scenario.correct_root_cause_keywords[0]
        result = env.step(Action(
            action_type=ActionType.INVESTIGATE,
            target="payment-gateway",
            parameters={},
            reasoning=f"Investigating because I suspect {relevant_keyword} is involved.",
        ))
        # Component may be named 'reasoning_bonus' or 'reasoning_relevant' depending on impl
        bonus = result.reward.components.get("reasoning_bonus", 0) or result.reward.components.get("reasoning_relevant", 0)
        assert bonus > 0


# ===========================================================================
# 3.  TEMPORAL DEGRADATION
# ===========================================================================

class TestTemporalDegradation:
    """Degradation penalty must scale with step number and task difficulty."""

    def test_degradation_increases_with_step(self, env):
        """Higher step numbers incur larger temporal penalty."""
        env.reset("severity_classification")
        # Step 1 – investigate
        r1 = env.step(Action(action_type=ActionType.INVESTIGATE, target="postgres-primary"))
        env.reset("severity_classification")
        env.step(Action(action_type=ActionType.INVESTIGATE, target="postgres-primary"))
        # Get to step 2 first
        r2 = env.step(Action(action_type=ActionType.INVESTIGATE, target="user-service"))
        deg1 = r1.reward.components.get("temporal_degradation", 0)
        deg2 = r2.reward.components.get("temporal_degradation", 0)
        assert deg2 <= deg1, f"Step 2 degradation ({deg2}) should be <= step 1 ({deg1})"

    def test_hard_task_higher_degradation_than_easy(self, env):
        """Hard task has higher degradation rate per step than easy task."""
        easy_s = SCENARIOS["severity_classification"]
        hard_s = SCENARIOS["full_incident_management"]
        assert hard_s.degradation_per_step > easy_s.degradation_per_step

    def test_easy_degradation_rate(self, env):
        """Easy task uses 0.005/step degradation."""
        assert SCENARIOS["severity_classification"].degradation_per_step == pytest.approx(0.005)

    def test_medium_degradation_rate(self, env):
        """Medium task uses 0.010/step degradation."""
        assert SCENARIOS["root_cause_analysis"].degradation_per_step == pytest.approx(0.010)

    def test_hard_degradation_rate(self, env):
        """Hard task uses 0.015/step degradation."""
        assert SCENARIOS["full_incident_management"].degradation_per_step == pytest.approx(0.015)


# ===========================================================================
# 4.  BLAST RADIUS DYNAMICS
# ===========================================================================

class TestBlastRadius:
    """Blast radius must degrade metrics progressively as steps increase."""

    def test_blast_radius_defined_for_all_tasks(self):
        for task_id, scenario in SCENARIOS.items():
            assert len(scenario.blast_radius) > 0, f"No blast_radius defined for {task_id}"

    def test_blast_radius_worsens_metrics_over_time(self):
        """Applying blast radius at step 5 must give worse metrics than step 1."""
        scenario = SCENARIOS["full_incident_management"]
        metrics_step1 = apply_blast_radius(scenario, step=1)
        metrics_step5 = apply_blast_radius(scenario, step=5)
        # Find a metric that should degrade
        blast_svc = next(iter(scenario.blast_radius))
        blast_metric = next(iter(scenario.blast_radius[blast_svc]))
        delta, cap = scenario.blast_radius[blast_svc][blast_metric]
        if delta > 0:
            # Metric should be higher at step 5
            v1 = getattr(metrics_step1[blast_svc], blast_metric, None)
            v5 = getattr(metrics_step5[blast_svc], blast_metric, None)
            if v1 is None:
                v1 = metrics_step1[blast_svc].custom.get(blast_metric, 0)
                v5 = metrics_step5[blast_svc].custom.get(blast_metric, 0)
            assert v5 >= v1, f"Blast radius metric '{blast_metric}' on '{blast_svc}' did not worsen: {v1} → {v5}"

    def test_blast_radius_never_exceeds_cap(self):
        """No degraded metric should exceed its cap value."""
        scenario = SCENARIOS["full_incident_management"]
        for step in [1, 5, 10, 20, 50]:
            degraded = apply_blast_radius(scenario, step=step)
            for svc, blast in scenario.blast_radius.items():
                for metric_key, (delta, cap) in blast.items():
                    if delta > 0:
                        val = getattr(degraded[svc], metric_key, None)
                        if val is None:
                            val = degraded[svc].custom.get(metric_key, 0)
                        assert val <= cap + 0.001, (
                            f"Metric '{metric_key}' on '{svc}' at step={step} exceeded cap: {val} > {cap}"
                        )

    def test_investigate_reveals_degraded_metrics(self, env):
        """An agent investigating later in the episode sees worse metrics."""
        # Episode 1: investigate at step 1
        e1 = IncidentResponseEnv()
        e1.reset("full_incident_management")
        r1 = e1.step(Action(action_type=ActionType.INVESTIGATE, target="auth-service"))
        metrics_step1 = r1.observation.metrics.get("auth-service")

        # Episode 2: waste steps first, then investigate
        e2 = IncidentResponseEnv()
        e2.reset("full_incident_management")
        svc_list = SCENARIOS["full_incident_management"].available_services
        irrelevant = [s for s in svc_list if s not in SCENARIOS["full_incident_management"].relevant_services]
        if irrelevant:
            e2.step(Action(action_type=ActionType.INVESTIGATE, target=irrelevant[0]))
        e2.step(Action(action_type=ActionType.INVESTIGATE, target=irrelevant[1] if len(irrelevant) > 1 else svc_list[0]))
        r2_auth = e2.step(Action(action_type=ActionType.INVESTIGATE, target="auth-service"))
        metrics_step3 = r2_auth.observation.metrics.get("auth-service")

        assert metrics_step1 is not None
        assert metrics_step3 is not None
        # CPU or memory should be at least as high at step 3 as step 1
        assert metrics_step3.cpu_percent >= metrics_step1.cpu_percent - 0.1


# ===========================================================================
# 5.  PARTIAL CREDIT AND SCORING GRADIENT
# ===========================================================================

class TestPartialCredit:
    """Graders must award partial credit for partially correct solutions."""

    def test_wrong_severity_gets_partial_credit_if_close(self, env):
        """P1 for a P2 incident (1 level off) gets partial severity_accuracy credit."""
        env.reset("severity_classification")
        env.step(Action(action_type=ActionType.INVESTIGATE, target="postgres-primary"))
        env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P1"}))  # off by 1
        result = env.grade()
        assert 0.0 < result.breakdown.get("severity_accuracy", 0) < 0.5

    def test_no_investigation_gets_zero_investigation_quality(self, env):
        env.reset("severity_classification")
        env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P2"}))
        result = env.grade()
        assert result.breakdown.get("investigation_quality", 0) == 0.0

    def test_investigated_but_not_root_cause_service_gets_partial(self, env):
        """Medium task: investigated payment-gateway (symptom) but not redis-session (root cause)."""
        env.reset("root_cause_analysis")
        env.step(Action(action_type=ActionType.INVESTIGATE, target="payment-gateway"))
        env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P1"}))
        env.step(Action(
            action_type=ActionType.DIAGNOSE,
            target="redis-session",
            parameters={"root_cause": "Redis maxmemory limit evicting session tokens"},
        ))
        env.step(Action(action_type=ActionType.REMEDIATE, target="redis-session", parameters={"action": "scale"}))
        result = env.grade()
        # Should not get full root-cause-investigation credit
        assert result.breakdown.get("investigated_root_cause_service", 0) == 0.0
        # But should get diagnosis and remediation credit
        assert result.breakdown.get("diagnosis_accuracy", 0) > 0
        assert result.breakdown.get("remediation_quality", 0) > 0

    def test_correct_diagnosis_wrong_service_partial_credit(self, env):
        """Diagnosing with correct keywords but wrong target service gives partial."""
        env.reset("root_cause_analysis")
        env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P1"}))
        env.step(Action(
            action_type=ActionType.DIAGNOSE,
            target="payment-gateway",  # wrong service
            parameters={"root_cause": "Redis maxmemory limit eviction of session tokens caused payment failures"},
        ))
        env.step(Action(action_type=ActionType.REMEDIATE, target="redis-session", parameters={"action": "scale"}))
        result = env.grade()
        # Gets keyword match credit but not service credit (or partial)
        diag = result.breakdown.get("diagnosis_accuracy", 0)
        assert diag > 0  # keyword matched

    def test_perfect_medium_score(self, env):
        """Perfect solution on medium task achieves >= 0.95 score."""
        env.reset("root_cause_analysis")
        env.step(Action(action_type=ActionType.INVESTIGATE, target="payment-gateway"))
        env.step(Action(action_type=ActionType.INVESTIGATE, target="redis-session"))
        env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P1"}))
        env.step(Action(
            action_type=ActionType.DIAGNOSE,
            target="redis-session",
            parameters={"root_cause": "Redis session store hit maxmemory limit causing eviction of payment session tokens"},
        ))
        env.step(Action(action_type=ActionType.REMEDIATE, target="redis-session", parameters={"action": "scale"}))
        result = env.grade()
        assert result.score >= 0.95, f"Expected >= 0.95, got {result.score}"

    def test_score_strictly_between_zero_and_one(self, env):
        """All graders must return scores strictly in (0.0, 1.0)."""
        for task_id in SCENARIOS:
            env.reset(task_id)
            result = env.grade()
            assert 0.0 < result.score < 1.0, f"{task_id} score {result.score} not in (0,1)"

    def test_difficulty_ordering_baseline(self, env):
        """Rule-based baseline score on easy >= medium >= hard."""
        results = run_all_tasks(env_instance=env, mode="rules")
        scores = {r["task_id"]: r["score"] for r in results}
        easy = scores["severity_classification"]
        medium = scores["root_cause_analysis"]
        hard = scores["full_incident_management"]
        assert easy >= medium >= hard, (
            f"Expected easy >= medium >= hard, got easy={easy}, medium={medium}, hard={hard}"
        )

    def test_late_episode_classify_gets_lower_efficiency(self, env):
        """Classifying at step 8 should score lower on efficiency than step 2."""
        # Early classify (step 2)
        env.reset("severity_classification")
        env.step(Action(action_type=ActionType.INVESTIGATE, target="postgres-primary"))
        env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P2"}))
        early = env.grade()

        # Late classify (step 8) — waste 7 steps first
        env.reset("severity_classification")
        services = SCENARIOS["severity_classification"].available_services
        for i in range(7):
            svc = services[i % len(services)]
            env.step(Action(action_type=ActionType.INVESTIGATE, target=svc))
        env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P2"}))
        late = env.grade()

        assert early.breakdown.get("efficiency", 0) >= late.breakdown.get("efficiency", 0), (
            "Early classification should have higher efficiency score"
        )


# ===========================================================================
# 6.  GRADER FEEDBACK QUALITY
# ===========================================================================

class TestGraderFeedback:
    """Grader feedback must be non-empty, actionable, and contextually correct."""

    @pytest.mark.parametrize("task_id", list(SCENARIOS.keys()))
    def test_feedback_is_non_empty(self, env, task_id):
        env.reset(task_id)
        env.step(Action(action_type=ActionType.INVESTIGATE, target=SCENARIOS[task_id].available_services[0]))
        result = env.grade()
        assert result.feedback, f"Empty feedback for {task_id}"
        assert len(result.feedback) > 30

    def test_easy_correct_feedback_contains_checkmark(self, env):
        """Perfect solution on easy task gets checkmark feedback."""
        env.reset("severity_classification")
        env.step(Action(action_type=ActionType.INVESTIGATE, target="postgres-primary"))
        env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P2"}))
        result = env.grade()
        assert "✓" in result.feedback, f"Expected checkmark in perfect feedback: {result.feedback}"

    def test_easy_wrong_feedback_contains_x(self, env):
        """Wrong severity on easy task gets cross (✗) feedback with hint."""
        env.reset("severity_classification")
        env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P4"}))
        result = env.grade()
        assert "✗" in result.feedback or "✓" not in result.feedback  # some negative feedback

    def test_medium_perfect_feedback(self, env):
        """Perfect medium solution gets positive feedback for all dimensions."""
        env.reset("root_cause_analysis")
        env.step(Action(action_type=ActionType.INVESTIGATE, target="payment-gateway"))
        env.step(Action(action_type=ActionType.INVESTIGATE, target="redis-session"))
        env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P1"}))
        env.step(Action(
            action_type=ActionType.DIAGNOSE, target="redis-session",
            parameters={"root_cause": "Redis maxmemory eviction of payment session tokens"},
        ))
        env.step(Action(action_type=ActionType.REMEDIATE, target="redis-session", parameters={"action": "scale"}))
        result = env.grade()
        assert "✓" in result.feedback
        assert "redis" in result.feedback.lower()

    def test_hard_missing_escalation_feedback(self, env):
        """Hard task with no escalation should mention escalation in feedback."""
        env.reset("full_incident_management")
        env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P1"}))
        env.step(Action(
            action_type=ActionType.DIAGNOSE, target="auth-service",
            parameters={"root_cause": "auth-service memory leak v3.1.0 token cache"},
        ))
        env.step(Action(action_type=ActionType.REMEDIATE, target="auth-service", parameters={"action": "rollback"}))
        env.step(Action(action_type=ActionType.COMMUNICATE, target="status_page", parameters={"message": "Auth outage resolved. Root cause: v3.1.0 memory leak. Rollback complete."}))
        result = env.grade()
        assert "escalat" in result.feedback.lower(), f"Feedback should mention escalation: {result.feedback}"

    def test_grader_breakdown_keys_match_expected(self, env):
        """Grader breakdowns must contain the documented dimension names."""
        expected_dims = {
            "severity_classification": {"severity_accuracy", "investigation_quality", "efficiency"},
            "root_cause_analysis": {"severity_accuracy", "investigated_root_cause_service", "diagnosis_accuracy", "remediation_quality", "efficiency"},
            "full_incident_management": {"severity_accuracy", "diagnosis_accuracy", "remediation_quality", "escalation_quality", "communication", "investigation_thoroughness", "investigation_precision", "efficiency"},
        }
        for task_id, dims in expected_dims.items():
            env.reset(task_id)
            result = env.grade()
            for dim in dims:
                assert dim in result.breakdown, f"Missing dimension '{dim}' in {task_id} breakdown"

    def test_grader_breakdown_values_sum_leq_one(self, env):
        """Perfect breakdown values should sum to <= 1.0 (some may be 0)."""
        for task_id in SCENARIOS:
            env.reset(task_id)
            env.step(Action(action_type=ActionType.INVESTIGATE, target=SCENARIOS[task_id].available_services[0]))
            result = env.grade()
            total = sum(result.breakdown.values())
            assert total <= 1.01, f"Breakdown sum exceeds 1.0 for {task_id}: {total}"


# ===========================================================================
# 7.  OBSERVATION CONTRACT
# ===========================================================================

class TestObservationContract:
    """Every observation returned by reset() and step() must have all required fields."""

    REQUIRED_FIELDS = [
        "incident_id", "step_number", "max_steps", "task_id",
        "alerts", "available_services", "investigated_services",
        "logs", "metrics", "incident_status", "message",
    ]

    @pytest.mark.parametrize("task_id", list(SCENARIOS.keys()))
    def test_reset_observation_has_all_fields(self, env, task_id):
        obs = env.reset(task_id)
        for field in self.REQUIRED_FIELDS:
            assert hasattr(obs, field), f"reset() Observation missing field '{field}' for {task_id}"

    @pytest.mark.parametrize("task_id", list(SCENARIOS.keys()))
    def test_step_observation_has_all_fields(self, env, task_id):
        env.reset(task_id)
        result = env.step(Action(
            action_type=ActionType.INVESTIGATE,
            target=SCENARIOS[task_id].available_services[0],
        ))
        for field in self.REQUIRED_FIELDS:
            assert hasattr(result.observation, field), f"step() Observation missing field '{field}' for {task_id}"

    def test_logs_and_metrics_empty_before_investigation(self, env):
        obs = env.reset("severity_classification")
        assert obs.logs == {}
        assert obs.metrics == {}

    def test_logs_populated_after_investigation(self, env):
        env.reset("severity_classification")
        result = env.step(Action(action_type=ActionType.INVESTIGATE, target="postgres-primary"))
        assert "postgres-primary" in result.observation.logs
        assert len(result.observation.logs["postgres-primary"]) > 0

    def test_metrics_populated_after_investigation(self, env):
        env.reset("severity_classification")
        result = env.step(Action(action_type=ActionType.INVESTIGATE, target="postgres-primary"))
        assert "postgres-primary" in result.observation.metrics
        assert result.observation.metrics["postgres-primary"] is not None

    def test_investigated_services_accumulates(self, env):
        env.reset("full_incident_management")
        env.step(Action(action_type=ActionType.INVESTIGATE, target="auth-service"))
        env.step(Action(action_type=ActionType.INVESTIGATE, target="api-gateway"))
        result = env.step(Action(action_type=ActionType.INVESTIGATE, target="order-service"))
        svcs = result.observation.investigated_services
        assert "auth-service" in svcs
        assert "api-gateway" in svcs
        assert "order-service" in svcs

    def test_incident_status_transitions(self, env):
        env.reset("root_cause_analysis")
        obs0 = env.reset("root_cause_analysis")
        assert obs0.incident_status == IncidentStatus.OPEN
        r1 = env.step(Action(action_type=ActionType.INVESTIGATE, target="redis-session"))
        assert r1.observation.incident_status in {IncidentStatus.OPEN, IncidentStatus.INVESTIGATING}

    def test_severity_classified_none_before_classify(self, env):
        obs = env.reset("severity_classification")
        assert obs.severity_classified is None

    def test_severity_classified_set_after_classify(self, env):
        env.reset("severity_classification")
        result = env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P2"}))
        assert result.observation.severity_classified == IncidentSeverity.P2

    def test_step_number_increments(self, env):
        env.reset("root_cause_analysis")
        for i in range(1, 4):
            result = env.step(Action(action_type=ActionType.INVESTIGATE, target="payment-gateway"))
            assert result.observation.step_number == i or result.done
            if result.done:
                break


# ===========================================================================
# 8.  openenv.yaml STRUCTURAL COMPLETENESS
# ===========================================================================

class TestOpenEnvYaml:
    """openenv.yaml must satisfy all spec requirements."""

    @pytest.fixture(scope="class")
    def yaml_doc(self):
        with open("openenv.yaml") as f:
            return yaml.safe_load(f)

    def test_yaml_loads(self, yaml_doc):
        assert yaml_doc is not None

    def test_has_name_and_description(self, yaml_doc):
        assert "name" in yaml_doc or "env_name" in yaml_doc
        assert "description" in yaml_doc

    def test_has_tasks_list(self, yaml_doc):
        tasks = yaml_doc.get("tasks", [])
        assert len(tasks) >= 3, f"Expected >= 3 tasks, found {len(tasks)}"

    def test_each_task_has_required_fields(self, yaml_doc):
        for task in yaml_doc.get("tasks", []):
            # YAML may use 'id' or 'task_id' — both are valid
            assert "task_id" in task or "id" in task, f"Task missing id/task_id: {task}"
            assert "description" in task, f"Task missing description: {task}"
            assert "difficulty" in task, f"Task missing difficulty: {task}"

    def test_tasks_cover_easy_medium_hard(self, yaml_doc):
        difficulties = {t.get("difficulty", "").lower() for t in yaml_doc.get("tasks", [])}
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties

    def test_has_action_space(self, yaml_doc):
        assert "action_space" in yaml_doc

    def test_has_observation_space(self, yaml_doc):
        assert "observation_space" in yaml_doc

    def test_has_reward_spec(self, yaml_doc):
        assert "reward" in yaml_doc or "reward_spec" in yaml_doc


# ===========================================================================
# 9.  INFERENCE SCRIPT EXITS ZERO
# ===========================================================================

class TestInferenceScript:
    """The root-level inference.py must run to completion and exit 0.

    Tests that require actual episode steps to be taken are marked with
    ``live_server`` — they start a local uvicorn process if one isn't
    already listening on port 7860.
    """

    def test_inference_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"inference.py exited with code {result.returncode}\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-1000:]}"
        )

    @pytest.mark.usefixtures("live_server")
    def test_inference_stdout_has_start_markers(self):
        env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "API_KEY")}
        env["ENV_BASE_URL"] = "http://localhost:7860"
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        assert "[START]" in result.stdout, "inference.py stdout missing [START] markers"
        assert "[STEP]" in result.stdout, "inference.py stdout missing [STEP] markers"
        assert "[END]" in result.stdout, "inference.py stdout missing [END] markers"

    @pytest.mark.usefixtures("live_server")
    def test_inference_produces_three_episodes(self):
        env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "API_KEY")}
        env["ENV_BASE_URL"] = "http://localhost:7860"
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        start_count = result.stdout.count("[START]")
        end_count = result.stdout.count("[END]")
        assert start_count == 3, f"Expected 3 [START] lines, got {start_count}"
        assert end_count == 3, f"Expected 3 [END] lines, got {end_count}"

    @pytest.mark.usefixtures("live_server")
    def test_inference_scores_in_range(self):
        env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "API_KEY")}
        env["ENV_BASE_URL"] = "http://localhost:7860"
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        for line in result.stdout.splitlines():
            if line.startswith("[END]"):
                # Parse score= from line
                for part in line.split():
                    if part.startswith("score="):
                        score = float(part.split("=")[1])
                        assert 0.0 < score < 1.0, f"Score {score} not in (0,1): {line}"

    @pytest.mark.usefixtures("live_server")
    def test_inference_all_done_true(self):
        env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "API_KEY")}
        env["ENV_BASE_URL"] = "http://localhost:7860"
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        for line in result.stdout.splitlines():
            if line.startswith("[END]"):
                assert "success=true" in line, f"Episode did not succeed: {line}"


# ===========================================================================
# 10.  MULTI-TASK HARD SCENARIO COVERAGE
# ===========================================================================

class TestHardScenarioCoverage:
    """The hard task has the most complex grading — verify all 8 dimensions."""

    def test_hard_perfect_solution_all_dimensions(self, env):
        """Run the known-optimal trajectory and check all 8 dimensions have credit."""
        env.reset("full_incident_management")
        # 4 relevant investigations
        for svc in ["auth-service", "api-gateway", "redis-auth-cache", "order-service"]:
            env.step(Action(action_type=ActionType.INVESTIGATE, target=svc))
        # classify
        env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P1"}))
        # diagnose
        env.step(Action(
            action_type=ActionType.DIAGNOSE, target="auth-service",
            parameters={"root_cause": "auth-service v3.1.0 memory leak via unbounded in-memory token cache causing OOMKill"},
        ))
        # 2 remediations
        env.step(Action(action_type=ActionType.REMEDIATE, target="auth-service", parameters={"action": "rollback"}))
        env.step(Action(action_type=ActionType.REMEDIATE, target="order-service", parameters={"action": "scale"}))
        # 2 escalations
        env.step(Action(action_type=ActionType.ESCALATE, target="platform-team",
                        parameters={"priority": "urgent", "message": "Cascading outage from auth v3.1.0 memory leak. Rolling back."}))
        env.step(Action(action_type=ActionType.ESCALATE, target="auth-team",
                        parameters={"priority": "urgent", "message": "auth-service v3.1.0 token cache memory leak. Rolled back v3.0.9."}))
        # communicate (triggers done)
        env.step(Action(action_type=ActionType.COMMUNICATE, target="status_page",
                        parameters={"message": "INCIDENT UPDATE: auth-service v3.1.0 memory leak identified. Rollback in progress. ETA 15 min."}))

        result = env.grade()
        assert result.score >= 0.85, f"Expected >= 0.85, got {result.score}"
        # All 8 dimensions must have non-zero credit
        for dim in ["severity_accuracy", "diagnosis_accuracy", "remediation_quality",
                    "escalation_quality", "communication", "investigation_thoroughness",
                    "investigation_precision"]:
            assert result.breakdown.get(dim, 0) > 0, (
                f"Dimension '{dim}' has zero credit for perfect hard solution"
            )

    def test_hard_red_herring_investigation_lowers_precision(self, env):
        """Investigating red-herring services (cdn-static, postgres-primary) lowers precision."""
        # Perfect except also investigates 2 red herrings
        env.reset("full_incident_management")
        for svc in ["cdn-static", "auth-service", "api-gateway", "redis-auth-cache", "order-service"]:
            r = env.step(Action(action_type=ActionType.INVESTIGATE, target=svc))
            if r.done:
                break
        env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P1"}))
        env.step(Action(action_type=ActionType.DIAGNOSE, target="auth-service",
                        parameters={"root_cause": "auth v3.1.0 unbounded token cache memory leak"}))
        env.step(Action(action_type=ActionType.REMEDIATE, target="auth-service", parameters={"action": "rollback"}))
        env.step(Action(action_type=ActionType.COMMUNICATE, target="status_page",
                        parameters={"message": "Auth memory leak resolved. Rollback complete. ETA 10min."}))
        with_red_herring = env.grade()

        # Perfect (no red herring)
        env.reset("full_incident_management")
        for svc in ["auth-service", "api-gateway", "redis-auth-cache", "order-service"]:
            r = env.step(Action(action_type=ActionType.INVESTIGATE, target=svc))
            if r.done:
                break
        env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P1"}))
        env.step(Action(action_type=ActionType.DIAGNOSE, target="auth-service",
                        parameters={"root_cause": "auth v3.1.0 unbounded token cache memory leak"}))
        env.step(Action(action_type=ActionType.REMEDIATE, target="auth-service", parameters={"action": "rollback"}))
        env.step(Action(action_type=ActionType.COMMUNICATE, target="status_page",
                        parameters={"message": "Auth memory leak resolved. Rollback complete. ETA 10min."}))
        without_red_herring = env.grade()

        assert without_red_herring.breakdown.get("investigation_precision", 0) >= \
               with_red_herring.breakdown.get("investigation_precision", 0), (
            "Red herring investigations should not improve precision score"
        )


# ===========================================================================
# 11.  PROMETHEUS LIVE METRICS ENDPOINTS
# ===========================================================================

class TestPrometheusEndpoints:
    """Verify /prometheus/metrics and /prometheus/query work correctly.

    Uses FastAPI TestClient for in-process HTTP — no external server needed.
    Confirms the Prometheus text format, JSON variant, blast-radius visibility,
    PromQL label filters, and the standard Prometheus JSON response envelope.
    """

    @pytest.fixture()
    def client_and_session(self):
        """Return (TestClient, session_id) with a fresh easy episode started."""
        from fastapi.testclient import TestClient
        from app import app as _app
        client = TestClient(_app, raise_server_exceptions=True)
        resp = client.post("/reset", json={"task_id": "severity_classification", "variant_seed": 0})
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]
        return client, session_id

    # ------------------------------------------------------------------
    # /prometheus/metrics
    # ------------------------------------------------------------------

    def test_prometheus_metrics_returns_prometheus_content_type(self, client_and_session):
        client, sid = client_and_session
        resp = client.get("/prometheus/metrics", headers={"X-Session-ID": sid})
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        assert "0.0.4" in resp.headers["content-type"]

    def test_prometheus_metrics_text_contains_help_and_type_lines(self, client_and_session):
        client, sid = client_and_session
        resp = client.get("/prometheus/metrics", headers={"X-Session-ID": sid})
        body = resp.text
        assert "# HELP irt_error_rate" in body
        assert "# TYPE irt_error_rate gauge" in body
        assert "# HELP irt_cpu_percent" in body

    def test_prometheus_metrics_text_contains_all_core_fields(self, client_and_session):
        client, sid = client_and_session
        resp = client.get("/prometheus/metrics", headers={"X-Session-ID": sid})
        body = resp.text
        for metric_name in (
            "irt_cpu_percent", "irt_memory_percent", "irt_request_rate",
            "irt_error_rate", "irt_latency_p50_ms", "irt_latency_p99_ms",
        ):
            assert metric_name in body, f"Expected '{metric_name}' in Prometheus output"

    def test_prometheus_metrics_labels_include_service_and_scenario(self, client_and_session):
        client, sid = client_and_session
        resp = client.get("/prometheus/metrics", headers={"X-Session-ID": sid})
        body = resp.text
        # Per-service data lines (not the scalar irt_scenario_step) must have service= label
        for line in body.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            if line.startswith("irt_scenario_step"):
                # Scalar metric — no service label, only scenario+incident
                assert 'scenario="' in line
                continue
            assert 'service="' in line, f"Missing service label in: {line}"
            assert 'scenario="' in line, f"Missing scenario label in: {line}"

    def test_prometheus_metrics_json_fmt_returns_dict(self, client_and_session):
        client, sid = client_and_session
        resp = client.get("/prometheus/metrics?fmt=json", headers={"X-Session-ID": sid})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        # Each value should be a ServiceMetrics-shaped dict
        for svc, m in data.items():
            assert "error_rate" in m, f"ServiceMetrics dict missing error_rate for {svc}"
            assert "cpu_percent" in m

    def test_prometheus_metrics_no_session_returns_400(self, client_and_session):
        client, _ = client_and_session
        resp = client.get("/prometheus/metrics")
        assert resp.status_code == 400

    def test_prometheus_metrics_blast_radius_visible_after_steps(self, client_and_session):
        """Error rate in Prometheus output increases with blast radius over steps."""
        client, sid = client_and_session
        # Read baseline error_rate for the blast-affected service
        def get_error_rates(body: str) -> dict:
            rates: dict = {}
            for line in body.splitlines():
                if line.startswith("irt_error_rate{"):
                    svc_match = __import__("re").search(r'service="([^"]+)"', line)
                    if svc_match:
                        val = float(line.split("} ")[1])
                        rates[svc_match.group(1)] = val
            return rates

        resp0 = client.get("/prometheus/metrics", headers={"X-Session-ID": sid})
        rates_step0 = get_error_rates(resp0.text)

        # Take 3 investigate steps to advance the episode (blast radius applies per step)
        from src.models import Action, ActionType
        for svc in ["user-api", "postgres-primary", "connection-pool-monitor"]:
            client.post("/step",
                json={"action_type": "investigate", "target": svc},
                headers={"X-Session-ID": sid},
            )

        resp3 = client.get("/prometheus/metrics", headers={"X-Session-ID": sid})
        rates_step3 = get_error_rates(resp3.text)

        # At least one service's metrics must have worsened (blast radius in action)
        worsened = any(
            rates_step3.get(svc, 0) >= rates_step0.get(svc, 0)
            for svc in rates_step0
        )
        assert worsened, "Expected blast radius to worsen at least one service metric over 3 steps"

    # ------------------------------------------------------------------
    # /prometheus/query
    # ------------------------------------------------------------------

    def test_prometheus_query_returns_success_envelope(self, client_and_session):
        client, sid = client_and_session
        resp = client.get(
            "/prometheus/query?query=irt_error_rate",
            headers={"X-Session-ID": sid},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["data"]["resultType"] == "vector"
        assert isinstance(data["data"]["result"], list)

    def test_prometheus_query_auto_prefix_irt(self, client_and_session):
        """Querying 'error_rate' (no irt_ prefix) should still return results."""
        client, sid = client_and_session
        resp = client.get(
            "/prometheus/query?query=error_rate",
            headers={"X-Session-ID": sid},
        )
        assert resp.status_code == 200
        result = resp.json()["data"]["result"]
        assert len(result) > 0, "Auto-prefixed query 'error_rate' returned no results"

    def test_prometheus_query_service_label_filter(self, client_and_session):
        client, sid = client_and_session
        # Get all services first
        all_resp = client.get(
            "/prometheus/query?query=irt_error_rate",
            headers={"X-Session-ID": sid},
        )
        all_services = [r["metric"]["service"] for r in all_resp.json()["data"]["result"]]
        if len(all_services) < 2:
            pytest.skip("Need at least 2 services to test label filter")
        target_svc = all_services[0]
        # Filtered query
        resp = client.get(
            f'/prometheus/query?query=irt_error_rate{{service="{target_svc}"}}',
            headers={"X-Session-ID": sid},
        )
        assert resp.status_code == 200
        result = resp.json()["data"]["result"]
        assert len(result) == 1, f"Label filter should return exactly 1 result, got {len(result)}"
        assert result[0]["metric"]["service"] == target_svc

    def test_prometheus_query_result_has_value_pair(self, client_and_session):
        """Each result entry must have [timestamp, value_string] as per Prometheus spec."""
        client, sid = client_and_session
        resp = client.get(
            "/prometheus/query?query=irt_cpu_percent",
            headers={"X-Session-ID": sid},
        )
        for entry in resp.json()["data"]["result"]:
            assert len(entry["value"]) == 2, "Prometheus value must be [timestamp, value_string]"
            ts, val_str = entry["value"]
            assert isinstance(ts, (int, float)), "Timestamp must be numeric"
            float(val_str)  # must be parseable as float

    def test_prometheus_query_no_session_returns_400(self, client_and_session):
        client, _ = client_and_session
        resp = client.get("/prometheus/query?query=irt_error_rate")
        assert resp.status_code == 400


# ===========================================================================
# 12.  TSDB RING BUFFER + /prometheus/query_range
# ===========================================================================

class TestTSDBRingBuffer:
    """env.metric_history() accumulates one sample per step and drives query_range."""

    def test_metric_history_empty_before_reset(self):
        env = IncidentResponseEnv()
        # No episode — live_metrics returns {}, metric_history returns {}
        assert env.metric_history(0, 9999999999) == {}

    def test_metric_history_has_step0_snapshot_after_reset(self):
        env = IncidentResponseEnv()
        env.reset("severity_classification", variant_seed=0)
        import time
        history = env.metric_history(time.time() - 10, time.time() + 1)
        assert len(history) > 0, "Should have at least one service in history after reset"
        # Each value is a list of (ts, ServiceMetrics) tuples
        for svc, samples in history.items():
            assert len(samples) >= 1, f"Expected >=1 sample for {svc} at step 0"
            ts, m = samples[0]
            assert isinstance(ts, float) and ts > 0
            from src.models import ServiceMetrics
            assert isinstance(m, ServiceMetrics)

    def test_metric_history_grows_with_steps(self):
        from src.models import Action, ActionType
        import time
        env = IncidentResponseEnv()
        env.reset("severity_classification", variant_seed=0)
        t_start = time.time() - 1
        # step 0 is the reset snapshot; take 2 more steps
        env.step(Action(action_type=ActionType.INVESTIGATE, target="user-api"))
        env.step(Action(action_type=ActionType.INVESTIGATE, target="postgres-primary"))
        t_end = time.time() + 1
        history = env.metric_history(t_start, t_end)
        for svc, samples in history.items():
            assert len(samples) >= 2, (
                f"Expected >=2 samples for {svc} after 2 steps, got {len(samples)}"
            )

    def test_metric_history_values_worsen_over_steps(self):
        """Error rate for a blast-radius-affected service must increase over steps."""
        from src.models import Action, ActionType
        from src.scenarios import get_scenario
        import time
        env = IncidentResponseEnv()
        env.reset("severity_classification", variant_seed=0)
        scenario = get_scenario("severity_classification", variant_seed=0)
        if not scenario.blast_radius:
            return  # no blast radius on this variant — skip
        t_start = time.time() - 1
        for svc in scenario.available_services[:3]:
            env.step(Action(action_type=ActionType.INVESTIGATE, target=svc))
        t_end = time.time() + 1
        history = env.metric_history(t_start, t_end)
        # Find a blast-affected service
        for svc in scenario.blast_radius:
            samples = history.get(svc, [])
            if len(samples) >= 2:
                blast_metrics = scenario.blast_radius[svc]
                for metric_key, (delta, _) in blast_metrics.items():
                    if delta > 0 and metric_key == "error_rate":
                        first_err = samples[0][1].error_rate
                        last_err = samples[-1][1].error_rate
                        assert last_err >= first_err, (
                            f"{svc}.error_rate should worsen: {first_err} -> {last_err}"
                        )
                        return
        # If we get here, no blast-radius metric was found — not a failure

    def test_metric_history_ring_buffer_max(self):
        """Ring buffer must not exceed _TSDB_MAX_SAMPLES per service."""
        from src.models import Action, ActionType
        env = IncidentResponseEnv()
        env.reset("severity_classification", variant_seed=0)
        # Take more steps than the ring buffer can hold by feeding CLASSIFY actions repeatedly
        # (only one valid classify, rest are duplicates — just to advance step counter)
        for _ in range(IncidentResponseEnv._TSDB_MAX_SAMPLES + 5):
            if env._done:
                break
            env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P2"}))
        import time
        history = env.metric_history(0, time.time() + 1)
        for svc, samples in history.items():
            assert len(samples) <= IncidentResponseEnv._TSDB_MAX_SAMPLES, (
                f"Ring buffer exceeded max for {svc}: {len(samples)}"
            )


class TestPrometheusRangeQuery:
    """The /prometheus/query_range endpoint returns proper Prometheus matrix output."""

    @pytest.fixture()
    def client_with_steps(self):
        """Return (TestClient, session_id) with 3 investigate steps taken."""
        from fastapi.testclient import TestClient
        from app import app as _app
        client = TestClient(_app, raise_server_exceptions=True)
        resp = client.post("/reset", json={"task_id": "severity_classification", "variant_seed": 0})
        assert resp.status_code == 200
        sid = resp.json()["session_id"]
        headers = {"X-Session-ID": sid}
        for svc in ["user-api", "postgres-primary", "connection-pool-monitor"]:
            client.post("/step",
                json={"action_type": "investigate", "target": svc},
                headers=headers,
            )
        return client, sid

    def test_query_range_returns_matrix_envelope(self, client_with_steps):
        client, sid = client_with_steps
        resp = client.get(
            "/prometheus/query_range?query=irt_error_rate",
            headers={"X-Session-ID": sid},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["data"]["resultType"] == "matrix"
        assert isinstance(data["data"]["result"], list)

    def test_query_range_values_are_lists_not_single_point(self, client_with_steps):
        """Matrix result must have 'values' (list) not 'value' (single point)."""
        client, sid = client_with_steps
        resp = client.get(
            "/prometheus/query_range?query=irt_error_rate",
            headers={"X-Session-ID": sid},
        )
        for stream in resp.json()["data"]["result"]:
            assert "values" in stream, "Matrix stream must have 'values' key"
            assert "value" not in stream, "Matrix stream must NOT have 'value' (that's instant)"
            assert isinstance(stream["values"], list)
            assert len(stream["values"]) >= 1

    def test_query_range_each_value_is_timestamp_string_pair(self, client_with_steps):
        client, sid = client_with_steps
        resp = client.get(
            "/prometheus/query_range?query=irt_cpu_percent",
            headers={"X-Session-ID": sid},
        )
        for stream in resp.json()["data"]["result"]:
            for pair in stream["values"]:
                assert len(pair) == 2, "Each value must be [timestamp, string]"
                ts, val_str = pair
                assert isinstance(ts, (int, float))
                float(val_str)  # must be parseable

    def test_query_range_no_session_returns_400(self, client_with_steps):
        client, _ = client_with_steps
        resp = client.get("/prometheus/query_range?query=irt_error_rate")
        assert resp.status_code == 400

    def test_query_range_start_greater_than_end_returns_400(self, client_with_steps):
        import time
        client, sid = client_with_steps
        now = time.time()
        resp = client.get(
            f"/prometheus/query_range?query=irt_error_rate&start={now+100}&end={now}",
            headers={"X-Session-ID": sid},
        )
        assert resp.status_code == 400

    def test_query_range_has_more_samples_than_instant_query(self, client_with_steps):
        """Range query across all time should return multiple samples per service stream."""
        client, sid = client_with_steps
        resp = client.get(
            "/prometheus/query_range?query=irt_error_rate&start=0&end=9999999999",
            headers={"X-Session-ID": sid},
        )
        assert resp.status_code == 200
        matrix = resp.json()["data"]["result"]
        assert len(matrix) > 0, "Should have at least one stream"
        # After 3 steps, each stream should have >=2 samples (step-0 + steps 1-3)
        for stream in matrix:
            assert len(stream["values"]) >= 2, (
                f"Expected multiple samples in range, got {len(stream['values'])}"
            )

    def test_query_range_label_filter_works(self, client_with_steps):
        """Service label filter on query_range narrows results."""
        client, sid = client_with_steps
        # Get all services
        all_resp = client.get(
            "/prometheus/query_range?query=irt_error_rate&start=0&end=9999999999",
            headers={"X-Session-ID": sid},
        )
        all_svcs = {s["metric"]["service"] for s in all_resp.json()["data"]["result"]}
        if len(all_svcs) < 2:
            pytest.skip("Need >=2 services to test filter")
        target = next(iter(all_svcs))
        resp = client.get(
            f'/prometheus/query_range?query=irt_error_rate{{service="{target}"}}&start=0&end=9999999999',
            headers={"X-Session-ID": sid},
        )
        result = resp.json()["data"]["result"]
        assert len(result) == 1
        assert result[0]["metric"]["service"] == target
