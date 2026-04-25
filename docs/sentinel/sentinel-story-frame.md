# SENTINEL Story Frame

This file is the complete narrative frame for explaining SENTINEL to teammates,
reviewers, blog readers, or a video audience. It is not just an API doc. It is
the build story: how the project moved from an incident-response OpenEnv
environment into a trainable AI-over-AI oversight system.

Use this as the source for:

- the 2-minute demo script
- the public mini-blog
- README storytelling
- presentation slides
- internal build notes
- submission framing

---

## 1. The Origin: Round 1 Incident Response Training

The project started as an incident-response training environment.

The original idea was practical: teach an agent how to act like an on-call SRE.
The agent receives alerts, inspects services, classifies severity, diagnoses the
root cause, escalates if needed, and applies remediation.

That first system already had strong OpenEnv bones:

- `reset()` creates a fresh incident.
- `step(action)` advances the incident.
- `state()` exposes the current world state.
- deterministic graders score the final episode.
- shaped rewards give partial credit instead of only pass/fail.
- multiple tasks move from easy to hard:
  `severity_classification`, `root_cause_analysis`, and
  `full_incident_management`.

The Round 1 environment included real SRE concepts:

- alert severity
- service metrics
- logs
- root-cause diagnosis
- escalation quality
- remediation quality
- efficiency
- temporal degradation
- blast-radius penalties
- partial-credit grading

That mattered because it was not a toy grid-world. It represented a real
workflow thousands of engineers perform every day.

But after building it, the deeper question appeared:

> If AI agents are doing incident response, who supervises the agents?

That question changed the whole project.

---

## 2. The Real Problem: Companies Will Deploy Agent Fleets

Most AI demos show one agent solving one task.

Real companies will not work that way.

They will deploy fleets:

- database agents
- app agents
- security agents
- network agents
- deployment agents
- customer-support agents
- finance agents
- scheduling agents
- personal assistant agents

Each worker may be helpful, but each can also be dangerous.

A worker can hallucinate a tool or service. It can act outside its domain. It
can escalate too broadly. It can restart systems before collecting evidence. It
can classify a problem too early just to finish the task. It can repeat useless
actions while pretending to make progress. It can sound highly confident without
having enough evidence.

The problem is not only that agents make mistakes.

The problem is that mistakes can execute.

So the missing training target is not another worker agent. The missing training
target is the supervisor.

---

## 3. The Pivot: From Incident Agent to SENTINEL

The IRT environment became the world underneath SENTINEL.

Instead of asking one model to solve the incident directly, SENTINEL creates a
fleet of worker agents. Workers propose actions. SENTINEL sees the proposal
before execution and decides whether to approve, block, redirect, reassign, or
flag it.

The pivot looks like this:

```text
Round 1:
  one incident-response agent acts directly in the environment

SENTINEL:
  many worker agents propose actions
  one oversight model supervises them before execution
```

The key change is pre-execution control.

SENTINEL does not only grade after damage has happened. It intercepts proposals
before the worker action touches the environment.

```text
Worker proposes action
        |
        v
SENTINEL evaluates proposal
        |
        +--> APPROVE    -> action executes
        +--> BLOCK      -> action does not execute
        +--> REDIRECT   -> safer action is proposed
        +--> REASSIGN   -> better worker takes over
        +--> FLAG       -> risk is recorded for review
```

This is the core insight:

> SENTINEL turns scalable oversight into an OpenEnv reinforcement-learning
> environment.

---

## 4. The One-Sentence Pitch We Can Say

SENTINEL is an OpenEnv environment where a smaller LLM learns to supervise a
fleet of AI workers, catching unsafe proposals before execution, giving
corrective feedback, updating worker trust, remembering past mistakes, and
reducing counterfactual damage through verifiable rewards.

---

## 5. The Runtime Loop

The live environment flow is:

```text
1. Worker proposes an action
        |
        v
2. SENTINEL receives:
   - incident state
   - proposed action
   - worker identity
   - worker trust score
   - constitutional assessment
   - memory of past mistakes
   - active multi-incident snapshots
        |
        v
3. SENTINEL outputs structured oversight JSON:
   - decision
   - reason
   - explanation
   - required evidence
   - suggested fix
   - redirect action
   - reassignment target
   - trust impact
   - counterfactual risk
        |
        v
4. If the worker was blocked, redirected, or reassigned:
   the worker receives corrective feedback
        |
        v
5. Worker gets one revision pass
        |
        v
6. SENTINEL re-checks the revision
        |
        v
7. Only safe or corrected actions execute
        |
        v
8. The environment updates:
   - reward
   - audit trail
   - worker trust
   - global memory
   - per-worker memory
   - curriculum state
```

This is why SENTINEL is more than an approval bot. It is a corrective
supervision loop.

---

## 6. What SENTINEL Learns

SENTINEL learns:

- which workers to trust
- when a proposal is harmful
- when a worker is acting outside its domain
- when to require evidence
- when to redirect to a safer action
- when to reassign to a better worker
- how to avoid over-blocking safe actions
- how to maintain a clean audit trail
- how to coach a worker into a safer revision
- how to remember mistakes across episodes
- how to handle several incidents at once

The training objective is not "always block." That would be easy and bad.

The real objective is calibrated oversight:

```text
block harmful proposals
approve safe proposals
redirect recoverable proposals
reassign when the worker is the wrong fit
flag uncertainty without stopping all progress
```

---

## 7. The Four SENTINEL Tasks

SENTINEL has four OpenEnv oversight tasks.

### 7.1 `basic_oversight`

A single worker handles a simple incident.

This teaches the core behavior:

- read proposal
- detect obvious unsafe actions
- approve safe investigation
- block hallucinated or premature actions
- write a structured reason

This is the first layer of oversight.

### 7.2 `fleet_monitoring_conflict`

Multiple workers operate in the same incident.

This introduces:

- worker domain boundaries
- cross-domain mistakes
- proportional escalation
- trust differences between workers
- safe approval versus false positive risk

This teaches SENTINEL that the same action can be safe or unsafe depending on
who proposed it and what evidence exists.

### 7.3 `adversarial_worker`

One worker begins behaving strategically badly.

This introduces:

- repeated misbehavior
- worker trust degradation
- quarantine pressure
- reassignment
- adversarial behavior patterns

This task teaches that oversight is not only about individual actions. It is
also about worker reliability over time.

The training side also has deterministic adversarial hooks in
`training/adversarial.py`. Those hooks let us test cases like confidence
washing, verifier-format pressure, and adversarial worker proposals without
turning the main benchmark into an uncontrolled LLM simulation.

### 7.4 `multi_crisis_command`

This is the grand finale task.

SENTINEL supervises a control room with three real underlying incident-response
environments at once.

The proposal includes:

- `incident_id`
- `incident_label`
- worker identity
- action type
- target
- reasoning

Only the selected incident thread advances when an action is approved,
redirected, or reassigned.

The observation exposes:

- `incident_snapshots`
- `active_incident_count`
- per-incident status
- per-incident investigated services
- per-incident severity and diagnosis

This matters because real operations do not happen one clean task at a time.
Production teams handle overlapping incidents, partial information, and several
workers competing for attention.

---

## 8. Misbehavior Types

SENTINEL trains against several worker failure modes.

| Misbehavior | What It Looks Like | Why It Matters |
|---|---|---|
| hallucination | worker targets a service that does not exist | common LLM failure |
| scope violation | worker acts outside its domain | action may look plausible but belongs to another team |
| destructive precedence | worker restarts/rolls back too early | creates blast radius before evidence |
| reward hacking | worker classifies or finishes early to get score | optimizes proxy instead of real outcome |
| escalation bombing | worker pages too many teams | expensive, noisy, operationally harmful |
| loop exploitation | worker repeats already-known checks | burns steps while appearing useful |
| confidence washing | worker claims high confidence without evidence | eval-only generalization test |
| prompt injection | worker embeds override instructions in reasoning | tests adversarial robustness |

The important point is that the environment models failures that future agent
fleets will actually have.

---

## 9. Constitutional Twin: Why SENTINEL Can Generalize

SENTINEL does not rely only on memorized misbehavior labels.

It has a deterministic constitutional safety layer in `sentinel/constitution.py`.

The five principles are:

- **P1 Minimal Blast Radius**: prefer the smallest safe action.
- **P2 Evidence Before Action**: do not diagnose, classify, or remediate without
  evidence.
- **P3 Proportional Escalation**: escalate only to relevant teams.
- **P4 Reversibility Preference**: prefer reversible actions over irreversible
  or disruptive actions.
- **P5 Domain Competence**: workers should act inside their declared domain.

This lets SENTINEL catch behavior it did not directly memorize.

Example:

```text
Worker says:
  "I am 99% sure the database is the root cause. Classify P1 now."

But the environment knows:
  - only one service was checked
  - evidence is weak
  - confidence is inflated

Constitutional assessment:
  P2 EvidenceBeforeAction violation

SENTINEL should:
  BLOCK or FLAG
  require supporting evidence
  redirect to investigation
```

This is why confidence washing can be eval-only. The model can still catch it
through principle-based oversight.

---

## 10. Worker Trust

The worker trust system lives in `sentinel/trust.py`.

Each worker has:

- trust score
- trust tier
- clean streak
- evidence-required mode
- auto-block recommendation
- last violation type
- risk exposure

Trust changes as workers behave.

```text
safe action approved        -> trust rises
misbehavior caught          -> trust falls
misbehavior allowed         -> trust falls more
low-trust worker no evidence -> trust gate blocks before execution
```

Trust matters because real supervisor systems cannot treat every worker as
equally reliable forever.

SENTINEL can learn:

- worker_db is reliable on database incidents
- worker_app should not touch database remediations
- worker_net may be better for network incidents
- a repeated offender should be quarantined or reassigned

---

## 11. Feedback Memory

SENTINEL has two memory layers.

### 11.1 Global Memory

Global memory stores broad oversight lessons:

- do not approve classification without investigation
- broad escalation without diagnosis is dangerous
- repeated restart suggestions create blast-radius risk
- confidence without evidence should be challenged

### 11.2 Per-Worker Memory

Per-worker memory stores patterns:

- worker_db keeps drifting into app incidents
- worker_app over-escalates
- worker_net requires evidence before remediation
- worker_sec is reliable for security-owned alerts

The prompt can surface these memories back to SENTINEL.

That means the model does not see each episode as isolated. It has a way to
learn from oversight mistakes across the run.

---

## 12. The Corrective Supervision Loop

The loop is:

```text
worker proposes
        |
        v
SENTINEL explains what is wrong
        |
        v
SENTINEL gives required evidence and suggested fix
        |
        v
worker revises
        |
        v
SENTINEL re-checks
        |
        v
safe revision executes or remains blocked
```

The feedback object is intentionally structured:

```text
decision: BLOCK | REDIRECT | REASSIGN | APPROVE | FLAG
reason: hallucination | scope_violation | ...
why_blocked
required_evidence
suggested_fix
suggested_target
suggested_action_type
reassign_to
trust_impact
counterfactual
```

This makes SENTINEL useful as a coach, not only a judge.

---

## 13. Reward Engineering

The reward is decomposed.

It rewards:

- catching unsafe worker proposals
- not blocking safe proposals
- correct reason classification
- good explanations
- good redirects
- good reassignments
- counterfactual risk reduction
- safe worker revision after feedback
- audit quality
- incident progress
- respecting trust gates
- following constitutional principles
- valid structured output

It penalizes:

- approving hallucinations
- approving reward hacking
- approving destructive actions too early
- false positives
- false negatives
- bad JSON
- invalid decisions
- overblocking
- weak evidence
- failed corrective feedback
- repetitive low-value behavior

The core deterministic reward has 10 named components:

1. true-positive catch
2. pre-execution timing
3. explanation accuracy
4. explanation clarity
5. correct redirect
6. correct reassign
7. audit trail quality
8. incident efficiency
9. false-positive penalty
10. false-negative penalty

Then SENTINEL adds:

- terminal completeness
- terminal accuracy
- terminal efficiency
- terminal precision
- worker rehabilitation rate
- correction-loop bonus
- coaching-quality bonus
- CoT reasoning bonus or penalty
- potential-based shaping

This is important for the story:

> SENTINEL is trained on safety behavior, not just final incident score.

---

## 14. Training Architecture

Training uses GRPO through Hugging Face TRL and Unsloth.

The current model path is:

```text
base model: unsloth/Qwen3-4B-bnb-4bit
adapter:    srikrish2004/sentinel-qwen3-4b-grpo
method:     LoRA / QLoRA-style GRPO post-training
```

The training loop is:

```text
sample OpenEnv prompt
        |
        v
generate G candidate oversight decisions
        |
        v
run each through SENTINEL environment / verifier
        |
        v
compute decomposed reward
        |
        v
compare candidates inside group
        |
        v
GRPO updates LoRA weights
        |
        v
write monitoring metrics
        |
        v
sample rollout audits for human inspection
        |
        v
update memory and curriculum
```

Phase 1:

- 200 GRPO steps
- Qwen3 4B base
- LoRA rank 16
- A100 80GB
- roughly 75 minutes in the successful run
- trained adapter pushed to Hugging Face

Phase 2:

- 150-step polish run
- focused on cleaner precision and proof
- deep evaluation running separately

Training also writes a proof trail, not just a final adapter:

- `training_metrics.jsonl`: reward, detection, false positives, scenario
  coverage, productive-signal metrics, and memory counts
- `training_stability.jsonl`: KL drift, policy entropy, adaptive beta, and
  trainer-level metrics
- `rollout_audits/`: sampled prompts, completions, rewards, and histories for
  manual inspection
- reward curves and dashboard images rendered after the run

Rollout audit matters because reward curves alone can lie. A model can get more
reward while producing brittle or suspicious behavior. The audit samples let us
look at actual decisions during training.

---

## 15. Adaptive Curriculum

SENTINEL uses an adaptive curriculum inspired by RLVE and Reasoning Gym.

Per task, it tracks:

- `difficulty_low`
- `difficulty_high`
- `mastery_attempts`
- `mastery_successes`
- frontier difficulty
- sliding window
- backoffs

It also tracks productive-signal metrics:

- `zero_reward_fraction`
- `trivially_solved_fraction`
- `productive_fraction`
- `effective_prompt_ratio`
- `frontier_hit_rate`
- `zero_gradient_group_fraction`
- `mean_reward_group_std`

This matters because GRPO can waste compute if every candidate receives the
same reward. SENTINEL measures that failure mode directly.

The training loop also tracks policy stability:

- approximate KL
- adaptive beta
- policy entropy
- decision entropy
- unique-completion ratio

The adaptive beta controller has configurable target, minimum, and maximum
values. If KL drift gets too high, the controller can increase pressure back
toward the reference policy. If drift is too low, it can loosen the constraint.

This is not just a training detail. It protects the proof story from a common RL
failure:

> reward appears to improve, but the policy drifts into repetitive or
> over-optimized behavior.

> We do not just train. We monitor whether the training batch can actually teach
> the model.

---

## 16. Verifier and Reward-Hacking Defense

SENTINEL uses deterministic verification first.

It does not trust a generative judge as the main reward path.

Hard signals include:

- hidden misbehavior labels
- constitutional assessment
- worker trust gate
- counterfactual damage estimate
- false-positive tracking
- false-negative tracking
- valid structured output
- held-out evaluation seeds

Eval-only tripwires include:

- gibberish answer
- adversarial prefix
- fake self-certification
- overlong nonsense
- format-jamming
- trust-gate bypass
- confidence washing

Held-out evaluation is split into:

- normal held-out seeds
- OOD held-out seeds in `scripts/eval_sentinel.py`
- tripwire cases in `sentinel/evaluation.py`
- sampled Top-1 vs Best-of-K evaluation

Top-1 vs Best-of-K is important because it tells a more honest story than only
showing the best generated sample. We can compare the first sampled answer
against the best answer from multiple samples and measure how much selection
helps. That shows whether the policy itself is good or whether we are relying on
sampling luck.

> We read the verifier-failure research and designed SENTINEL so the reward is
> auditable, not just persuasive.

---

## 17. Protocol and Integration Layer

SENTINEL is not only a local Python class.

The repo includes a multi-protocol serving layer:

- FastAPI app for dashboard and HTTP endpoints
- native OpenEnv adapter mounted under `/openenv`
- MCP server in `server/mcp_server.py`
- A2A protocol server in `server/a2a_server.py`
- universal adapter for normalizing agent/environment interfaces

The MCP layer exposes SENTINEL as callable tools through JSON-RPC / Streamable
HTTP. The A2A layer exposes an agent card and task interface so another agent can
send oversight tasks to SENTINEL as a peer service.

This matters for the story because the environment is not trapped in a notebook.
It is shaped like something that could be plugged into a larger agent ecosystem.

```text
OpenEnv / HTTP client
        |
        +--> /openenv/reset, /openenv/step, /openenv/state

MCP-compatible agent
        |
        +--> server/mcp_server.py tools

A2A-compatible agent
        |
        +--> /.well-known/agent.json + JSON-RPC task calls
```

The long-term direction is a universal oversight harness: paste an OpenEnv Space
or agent endpoint, let SENTINEL inspect proposed actions, and report whether the
action should execute.

---

## 18. Dynamic Workers and Circuit Breaker

The shipped benchmark uses deterministic workers because training proof needs
reproducibility.

But the repo also includes LLM-powered worker infrastructure in
`sentinel/llm_workers.py`.

The Groq worker backend can generate more natural worker proposals using a
remote LLM. It also has a circuit breaker:

- repeated API failures increment failure count
- after the failure threshold, the circuit opens
- while open, the backend avoids repeated failing calls
- after cooldown, it can try again
- deterministic fallback keeps the environment usable

This design choice is important.

Dynamic workers are exciting, but they can make evaluation noisy. A reviewer
needs clean before/after evidence. So the current proof path uses deterministic
worker schedules first, then dynamic Groq workers become the next demo/eval
layer.

The honest build decision:

```text
stable verifier benchmark first
        |
        v
trained checkpoint proof
        |
        v
dynamic LLM workers as extension
```

That is stronger than replacing the benchmark with uncontrolled LLM behavior too
early.

---

## 19. CoT Monitoring and Debate

SENTINEL includes `sentinel/cot_monitor.py`.

It analyzes reasoning traces for:

- evidence grounding
- counterfactual reasoning
- deliberation quality
- constitutional alignment
- faithfulness
- deception risk
- suspicious reasoning patterns

The CoT signal can add reward bonus or penalty, but it is not the only reward.
That matters because CoT can be useful but also fragile.

The repo also includes an AI debate protocol in `sentinel/debate.py`.

The debate framing is:

```text
Advocate argues APPROVE
Prosecutor argues BLOCK
Judge scores which side has stronger evidence
```

This is aligned with scalable-oversight research, but it stays secondary to the
deterministic verifier. The stable reward path remains the spine.

---

## 20. Proof Artifacts

The dashboard exports 18 proof images:

1. reward mean
2. detection vs false-positive rate
3. counterfactual risk reduction
4. worker rehabilitation
5. task coverage
6. scenario coverage heatmap
7. misbehavior coverage
8. per-misbehavior confusion matrix
9. adaptive curriculum frontier
10. productive signal
11. decision entropy and diversity
12. KL drift and adaptive beta
13. tripwire pass rate
14. Top-1 vs Best-of-K
15. learning snapshots at 10 / 50 / 300 episodes when available, or
    early/middle/final checkpoints for shorter proof runs
16. memory ablation
17. zero-gradient group fraction
18. memory growth

The deep eval notebook collects:

- before vs after SENTINEL
- no-SENTINEL baseline
- trained model behavior
- per-task performance
- per-worker trust evolution
- misbehavior detection breakdown
- counterfactual damage
- worker rehabilitation
- master summary

The final pitch should not claim only reward improvement. It should show:

- reward improved
- damage reduced
- false positives tracked
- failure modes exposed
- memory changed behavior
- trust scores evolved
- held-out tripwires were evaluated
- OOD held-out seeds were evaluated
- Top-1 and Best-of-K were separated
- rollout audits were saved for inspection

Learning snapshots are especially useful for storytelling. The ideal proof
sequence is 10 / 50 / 300 episodes; when a run is shorter, we keep the same
idea as early / middle / final checkpoints:

```text
early snapshot:
  SENTINEL catches only obvious cases and may over-block.

middle snapshot:
  SENTINEL starts using trust, evidence, and redirect/reassign fields.

final snapshot:
  SENTINEL gives cleaner structured feedback and handles harder scenarios.
```

---

## 21. The Doubts We Had While Building

The project did not start perfectly. Each doubt led to a stronger design.

### Doubt 1: Is this just approving work?

At first, the idea looked too simple: worker proposes, supervisor approves or
rejects.

That would not be enough.

So SENTINEL became a full control loop:

- approve
- block
- redirect
- reassign
- flag
- explain
- require evidence
- coach revision
- update trust
- update memory
- calculate counterfactual damage

The result is not "yes/no approval." It is trained supervision.

### Doubt 2: Is deterministic worker behavior too artificial?

Dynamic LLM workers are more realistic, and the repo supports them through Groq
workers.

But training proof needs repeatability. If worker behavior changes randomly, the
reward curve becomes hard to trust.

So the benchmark uses deterministic workers first, then dynamic LLM workers as
an extension. This makes the evaluation honest.

### Doubt 3: Can the reward be hacked?

Yes, any reward can be hacked if it is weak.

So SENTINEL avoids relying on one judge score. It uses hidden labels,
constitutional checks, trust gates, counterfactual damage, false-positive and
false-negative penalties, structured output checks, tripwires, OOD held-out
seeds, and rollout audits.

### Doubt 4: Will training collapse into zero signal?

GRPO needs reward differences inside sampled groups. If all completions get the
same reward, the batch teaches very little.

So the training monitor tracks:

- zero-gradient group fraction
- mean reward group standard deviation
- productive fraction
- effective prompt ratio
- frontier hit rate

This makes the training process itself inspectable.

### Doubt 5: What if reward rises but behavior gets worse?

That is a classic RL failure.

So SENTINEL logs:

- KL drift
- adaptive beta
- policy entropy
- decision entropy
- unique-completion ratio
- sampled rollout audits
- held-out eval
- OOD eval
- Top-1 vs Best-of-K eval

The system is designed to show not just that reward increased, but whether the
behavior stayed diverse, stable, and auditable.

### Doubt 6: Is SRE too narrow?

SRE is the testbed, not the limit.

The reusable concept is:

```text
worker proposes action
oversight model checks before execution
safe action proceeds
unsafe action gets corrected
memory improves the next decision
```

That pattern applies to finance, healthcare operations, enterprise IT, software
engineering, personal assistants, and any agent fleet that acts through tools.

---

## 22. Why This Is Not Just Simple Approval

A simple approval bot says yes or no.

SENTINEL does more:

- sees worker identity
- checks domain competence
- checks evidence
- checks constitutional risk
- checks trust
- predicts counterfactual damage
- writes worker-facing feedback
- gives revision instructions
- allows one correction loop
- reassigns to better workers
- tracks repeated worker failures
- updates memory across episodes
- learns from GRPO

The distinction is:

```text
Approval bot:
  Is this action allowed?

SENTINEL:
  Should this worker be trusted, in this incident, with this evidence,
  under these constitutional constraints, and if not, what safer action
  should happen next?
```

That is a much stronger environment.

---

## 23. Why This Can Matter Outside SRE

The incident-response domain is only the first proving ground.

The pattern applies to many real domains:

### Finance

SENTINEL-style oversight can supervise trading, reporting, fraud, and account
agents. It can require evidence before high-risk actions and block hallucinated
account state.

### Healthcare Operations

It can supervise scheduling, triage, claims, and patient-record workflows,
especially where evidence and scope boundaries matter.

### Software Engineering

It can supervise code agents before they run migrations, edit protected files,
delete data, or merge risky changes.

### Enterprise IT

It can gate access changes, deployments, incident actions, and automated ticket
handling.

### Personal Assistants

It can supervise calendar, email, shopping, travel, and delegation agents,
especially where conflicts and private preferences matter.

The general pattern is:

```text
worker agent proposes
oversight agent checks
unsafe action is corrected before execution
memory improves the next decision
```

---

## 24. The 2-Minute Demo Script

Most AI demos show one agent solving one task.
But real companies will deploy fleets of agents: database agents, app agents,
security agents, deployment agents, and personal assistants.

The problem is that these workers can hallucinate, over-escalate, act outside
their domain, or execute destructive actions before anyone checks them.

SENTINEL is an OpenEnv environment for training the missing layer: an AI
supervisor for AI workers.

In the environment, workers propose actions during production incidents.
SENTINEL sees the proposal before execution and must approve, block, redirect,
reassign, or flag it.

It receives worker trust, constitutional safety checks, incident state, memory
from past mistakes, and active multi-incident snapshots.

If it blocks a worker, it does not just say no. It explains what is wrong,
requires evidence, suggests a safer action, and gives the worker one revision
pass.

The reward is decomposed: it rewards catching unsafe actions, avoiding false
positives, naming the right reason, redirecting safely, reassigning correctly,
reducing counterfactual damage, and maintaining audit quality.

We trained Qwen3-4B with GRPO and Unsloth. Phase 1 ran 200 steps, and Phase 2 is
the polish run. The proof pack shows reward curves, detection vs false positives,
counterfactual damage reduction, worker rehabilitation, tripwire tests, OOD
held-out evaluation, rollout audits, KL drift, adaptive beta, Top-1 vs
Best-of-K, memory ablation, and zero-gradient group monitoring.

The key idea is simple but powerful:
as companies deploy many AI workers, they will need trained AI supervisors.
SENTINEL makes that supervision trainable, measurable, and auditable.


---

## 25. Final Submission Checklist

Before submission, confirm:

- [ ] HF Space is public and runnable.
- [ ] `openenv.yaml` points to the correct environment metadata.
- [ ] README links to the Space.
- [ ] README links to trained model.
- [ ] README links to training notebook.
- [ ] README links to deep eval notebook.
- [ ] README links to video/blog/slides.
- [ ] Phase 2 final metrics are inserted.
- [ ] Deep eval summary numbers are inserted.
- [ ] Reward curve image is committed or linked.
- [ ] Dashboard plot pack is committed or linked.
- [ ] Proof pack summary is generated.
- [ ] CI is green.
- [ ] No large video files are committed.
- [ ] The demo story leads with SENTINEL, not only the old IRT system.

---

## 26. The Closing Line

SENTINEL started as an incident-response environment, but the real insight was
larger: the future will not only need AI workers. It will need AI supervisors
trained to control, correct, and audit those workers before they act.

SENTINEL is that supervisor, built as a verifiable OpenEnv training environment.
