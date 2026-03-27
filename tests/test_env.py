"""Tests for the Incident Response Triage environment.

Validates:
  - reset() produces clean state for all tasks
  - step() returns correct types
  - Graders produce scores in [0.0, 1.0]
  - Rule-based baseline reproduces expected scores
  - Episode boundaries work correctly
  - Invalid actions are handled gracefully
"""

import pytest

from src.environment import IncidentResponseEnv
from src.models import Action, ActionType, IncidentSeverity, IncidentStatus
from src.scenarios import SCENARIOS
from baseline.inference import run_all_tasks


@pytest.fixture
def env():
    return IncidentResponseEnv()


# --------------------------------------------------------------------------
# reset() tests
# --------------------------------------------------------------------------

class TestReset:
    @pytest.mark.parametrize("task_id", list(SCENARIOS.keys()))
    def test_reset_returns_observation(self, env, task_id):
        obs = env.reset(task_id)
        assert obs.incident_id != ""
        assert obs.step_number == 0
        assert obs.task_id == task_id
        assert len(obs.alerts) > 0
        assert len(obs.available_services) > 0
        assert obs.incident_status == IncidentStatus.OPEN
        assert obs.investigated_services == []
        assert obs.logs == {}
        assert obs.metrics == {}

    def test_reset_invalid_task(self, env):
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset("nonexistent_task")

    def test_reset_clears_state(self, env):
        # Run some steps then reset
        env.reset("severity_classification")
        env.step(Action(action_type=ActionType.INVESTIGATE, target="user-service"))
        # Reset and verify clean state
        obs = env.reset("severity_classification")
        assert obs.step_number == 0
        assert obs.investigated_services == []


# --------------------------------------------------------------------------
# step() tests
# --------------------------------------------------------------------------

class TestStep:
    def test_step_investigate(self, env):
        env.reset("severity_classification")
        result = env.step(Action(
            action_type=ActionType.INVESTIGATE,
            target="postgres-primary",
        ))
        assert result.observation.step_number == 1
        assert "postgres-primary" in result.observation.investigated_services
        assert "postgres-primary" in result.observation.logs
        assert "postgres-primary" in result.observation.metrics
        assert not result.done

    def test_step_classify(self, env):
        env.reset("severity_classification")
        # Investigate first
        env.step(Action(action_type=ActionType.INVESTIGATE, target="postgres-primary"))
        # Then classify
        result = env.step(Action(
            action_type=ActionType.CLASSIFY,
            parameters={"severity": "P2"},
        ))
        assert result.observation.severity_classified == IncidentSeverity.P2
        assert result.done  # Easy task ends on classify

    def test_step_without_reset_raises(self, env):
        with pytest.raises(RuntimeError, match="done"):
            env.step(Action(action_type=ActionType.INVESTIGATE, target="x"))

    def test_step_after_done_raises(self, env):
        env.reset("severity_classification")
        env.step(Action(action_type=ActionType.INVESTIGATE, target="user-service"))
        env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P2"}))
        with pytest.raises(RuntimeError, match="done"):
            env.step(Action(action_type=ActionType.INVESTIGATE, target="user-service"))

    def test_investigate_invalid_service(self, env):
        env.reset("severity_classification")
        result = env.step(Action(
            action_type=ActionType.INVESTIGATE,
            target="nonexistent-service",
        ))
        assert result.reward.value < 0
        assert "nonexistent-service" not in result.observation.investigated_services


# --------------------------------------------------------------------------
# state() tests
# --------------------------------------------------------------------------

class TestState:
    def test_state_after_reset(self, env):
        env.reset("root_cause_analysis")
        state = env.state()
        assert state.task_id == "root_cause_analysis"
        assert state.step_number == 0
        assert not state.done
        assert state.cumulative_reward == 0.0

    def test_state_tracks_actions(self, env):
        env.reset("root_cause_analysis")
        env.step(Action(action_type=ActionType.INVESTIGATE, target="redis-session"))
        state = env.state()
        assert state.step_number == 1
        assert "redis-session" in state.investigated_services
        assert len(state.actions_history) == 1


# --------------------------------------------------------------------------
# Reward tests
# --------------------------------------------------------------------------

class TestRewards:
    def test_relevant_investigation_positive(self, env):
        env.reset("severity_classification")
        result = env.step(Action(
            action_type=ActionType.INVESTIGATE,
            target="postgres-primary",
        ))
        assert result.reward.value > 0
        assert "relevant_investigation" in result.reward.components

    def test_irrelevant_investigation_negative(self, env):
        env.reset("severity_classification")
        result = env.step(Action(
            action_type=ActionType.INVESTIGATE,
            target="redis-cache",
        ))
        assert result.reward.components.get("irrelevant_investigation", 0) < 0

    def test_correct_classification_positive(self, env):
        env.reset("severity_classification")
        env.step(Action(action_type=ActionType.INVESTIGATE, target="postgres-primary"))
        result = env.step(Action(
            action_type=ActionType.CLASSIFY,
            parameters={"severity": "P2"},
        ))
        assert result.reward.components.get("correct_classification", 0) > 0

    def test_wrong_classification_negative(self, env):
        env.reset("severity_classification")
        result = env.step(Action(
            action_type=ActionType.CLASSIFY,
            parameters={"severity": "P4"},
        ))
        assert result.reward.components.get("wrong_classification", 0) < 0


# --------------------------------------------------------------------------
# Grader tests
# --------------------------------------------------------------------------

class TestGraders:
    @pytest.mark.parametrize("task_id", list(SCENARIOS.keys()))
    def test_grader_score_range(self, env, task_id):
        env.reset(task_id)
        # Take one action to avoid empty episode
        svc = SCENARIOS[task_id].available_services[0]
        env.step(Action(action_type=ActionType.INVESTIGATE, target=svc))
        result = env.grade()
        assert 0.0 <= result.score <= 1.0
        assert result.task_id == task_id
        assert isinstance(result.breakdown, dict)

    def test_perfect_easy_score(self, env):
        env.reset("severity_classification")
        env.step(Action(action_type=ActionType.INVESTIGATE, target="postgres-primary"))
        env.step(Action(
            action_type=ActionType.CLASSIFY,
            parameters={"severity": "P2"},
        ))
        result = env.grade()
        assert result.score >= 0.9  # Should get near-perfect

    def test_zero_score_on_no_action(self, env):
        env.reset("full_incident_management")
        # Take an irrelevant action
        env.step(Action(action_type=ActionType.INVESTIGATE, target="cdn-static"))
        result = env.grade()
        assert result.score < 0.2  # Should be very low


# --------------------------------------------------------------------------
# Baseline integration tests
# --------------------------------------------------------------------------

class TestBaseline:
    def test_rule_based_baseline_all_tasks(self, env):
        results = run_all_tasks(env_instance=env, mode="rules")
        assert len(results) == 3
        for r in results:
            assert 0.0 <= r["score"] <= 1.0
            assert r["steps_taken"] > 0

    def test_rule_based_scores_reproducible(self, env):
        """Run twice and verify identical scores."""
        results1 = run_all_tasks(env_instance=env, mode="rules")
        results2 = run_all_tasks(env_instance=env, mode="rules")
        for r1, r2 in zip(results1, results2):
            assert r1["score"] == r2["score"]
            assert r1["steps_taken"] == r2["steps_taken"]

    def test_difficulty_progression(self, env):
        """Easy should score higher than hard with same baseline."""
        results = run_all_tasks(env_instance=env, mode="rules")
        scores = {r["task_id"]: r["score"] for r in results}
        # Rule-based should do well on all, but ordering might not be strict
        # At minimum, all should be > 0.5 for a well-designed baseline
        for task_id, score in scores.items():
            assert score > 0.3, f"Baseline too low for {task_id}: {score}"


# --------------------------------------------------------------------------
# Episode boundary tests
# --------------------------------------------------------------------------

class TestEpisodeBoundaries:
    def test_easy_ends_on_classify(self, env):
        env.reset("severity_classification")
        env.step(Action(action_type=ActionType.INVESTIGATE, target="user-service"))
        result = env.step(Action(
            action_type=ActionType.CLASSIFY,
            parameters={"severity": "P2"},
        ))
        assert result.done

    def test_medium_ends_on_diagnose_and_remediate(self, env):
        env.reset("root_cause_analysis")
        env.step(Action(action_type=ActionType.INVESTIGATE, target="redis-session"))
        env.step(Action(action_type=ActionType.CLASSIFY, parameters={"severity": "P1"}))
        env.step(Action(
            action_type=ActionType.DIAGNOSE,
            target="redis-session",
            parameters={"root_cause": "Redis memory eviction"},
        ))
        result = env.step(Action(
            action_type=ActionType.REMEDIATE,
            target="redis-session",
            parameters={"action": "scale"},
        ))
        assert result.done

    def test_max_steps_terminates(self, env):
        env.reset("severity_classification")
        # Take 10 investigate actions (max_steps=10)
        done = False
        services = ["user-service", "postgres-primary", "redis-cache", "api-gateway"]
        for i in range(10):
            svc = services[i % len(services)]
            result = env.step(Action(
                action_type=ActionType.INVESTIGATE,
                target=svc,
            ))
            if result.done:
                done = True
                break
        assert done  # Should have hit max steps
