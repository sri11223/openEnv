"""Pre-submission validation script.

Validates all requirements:
  - openenv.yaml exists and is valid
  - All 3 tasks are defined
  - reset() / step() / state() work correctly
  - Graders produce scores in [0.0, 1.0]
  - Baseline is reproducible
  - Typed models validate
"""

import sys
import traceback
from typing import List, Tuple

from src.environment import IncidentResponseEnv
from src.models import Action, ActionType
from src.scenarios import SCENARIOS
from src.tasks import get_all_tasks
from baseline.inference import run_all_tasks


def _check(name: str, fn) -> Tuple[bool, str]:
    try:
        result = fn()
        return True, result or "OK"
    except Exception as exc:
        return False, f"FAILED: {exc}\n{traceback.format_exc()}"


def validate() -> bool:
    checks: List[Tuple[str, bool, str]] = []
    env = IncidentResponseEnv()

    # 1. openenv.yaml exists
    def check_yaml():
        import yaml
        with open("openenv.yaml") as f:
            data = yaml.safe_load(f)
        assert data["name"] == "incident-response-triage"
        assert len(data["tasks"]) >= 3
        return f"Found {len(data['tasks'])} tasks"

    try:
        ok, msg = _check("openenv.yaml", check_yaml)
    except ImportError:
        # yaml not installed, just check file exists
        import os
        ok = os.path.exists("openenv.yaml")
        msg = "File exists (yaml not installed for full check)"
    checks.append(("openenv.yaml valid", ok, msg))

    # 2. Tasks defined
    def check_tasks():
        tasks = get_all_tasks()
        assert len(tasks) >= 7
        for t in tasks:
            assert t.difficulty in ("easy", "medium", "hard", "expert")
        return f"{len(tasks)} tasks defined"
    ok, msg = _check("Tasks", check_tasks)
    checks.append(("3+ tasks defined", ok, msg))

    # 3. reset() for all tasks
    def check_reset():
        for task_id in SCENARIOS:
            obs = env.reset(task_id)
            assert obs.step_number == 0
            assert len(obs.alerts) > 0
        return "All tasks reset successfully"
    ok, msg = _check("reset()", check_reset)
    checks.append(("reset() works", ok, msg))

    # 4. step() returns correct types
    def check_step():
        env.reset("severity_classification")
        result = env.step(Action(
            action_type=ActionType.INVESTIGATE,
            target="postgres-primary",
        ))
        assert hasattr(result, "observation")
        assert hasattr(result, "reward")
        assert hasattr(result, "done")
        assert hasattr(result, "info")
        assert -1.0 <= result.reward.value <= 1.0
        return "Step returns correct StepResult"
    ok, msg = _check("step()", check_step)
    checks.append(("step() returns StepResult", ok, msg))

    # 5. state() returns correct type
    def check_state():
        env.reset("severity_classification")
        env.step(Action(action_type=ActionType.INVESTIGATE, target="user-service"))
        state = env.state()
        assert state.step_number == 1
        assert state.task_id == "severity_classification"
        return "State snapshot correct"
    ok, msg = _check("state()", check_state)
    checks.append(("state() works", ok, msg))

    # 6. Graders in [0.0, 1.0]
    def check_graders():
        for task_id in SCENARIOS:
            env.reset(task_id)
            svc = SCENARIOS[task_id].available_services[0]
            env.step(Action(action_type=ActionType.INVESTIGATE, target=svc))
            result = env.grade()
            assert 0.0 <= result.score <= 1.0, f"{task_id}: {result.score}"
        return "All graders in [0.0, 1.0]"
    ok, msg = _check("Graders", check_graders)
    checks.append(("Graders score [0.0-1.0]", ok, msg))

    # 7. Baseline reproducible
    def check_baseline():
        r1 = run_all_tasks(env_instance=env, mode="rules")
        r2 = run_all_tasks(env_instance=env, mode="rules")
        for a, b in zip(r1, r2):
            assert a["score"] == b["score"], f"Non-reproducible: {a['task_id']}"
        scores = [r["score"] for r in r1]
        return f"Baseline scores: {[f'{s:.4f}' for s in scores]}"
    ok, msg = _check("Baseline", check_baseline)
    checks.append(("Baseline reproducible", ok, msg))

    # 8. SENTINEL environment (if available)
    def check_sentinel():
        try:
            from sentinel.environment import SentinelEnv
            sent_env = SentinelEnv()
            sent_tasks = ["basic_oversight", "fleet_monitoring_conflict", "adversarial_worker", "multi_crisis_command"]
            for task_id in sent_tasks:
                obs = sent_env.reset(task_id, variant_seed=0)
                assert hasattr(obs, "step_number")
                assert hasattr(obs, "proposed_action")
                grade = sent_env.grade()
                assert 0.0 <= grade.score <= 1.0
            return f"SENTINEL: {len(sent_tasks)} tasks validated"
        except ImportError:
            return "SENTINEL not installed (optional)"
    ok, msg = _check("SENTINEL", check_sentinel)
    checks.append(("SENTINEL environment", ok, msg))

    # Print results
    print("\n" + "=" * 60)
    print("OpenEnv Pre-Submission Validation")
    print("=" * 60)

    all_pass = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {detail[:80]}")
        if not passed:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — fix before submitting")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
