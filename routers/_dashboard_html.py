# -*- coding: utf-8 -*-
"""Dashboard HTML templates extracted from app.py."""

SENTINEL_DASHBOARD_HTML = """\

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SENTINEL Fleet Oversight</title>
<style>
*{box-sizing:border-box}
:root{--bg:#0b0d0f;--panel:#15191d;--panel2:#101418;--line:#2c333a;--text:#eef2f4;--muted:#96a0aa;--green:#2fb170;--yellow:#d8a634;--red:#e05d5d;--cyan:#55b7c8;--ink:#080a0c}
body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,Segoe UI,Arial,sans-serif;min-height:100vh}
button,select,textarea,input{font:inherit}
.shell{display:grid;grid-template-columns:330px 1fr;min-height:100vh}
.rail{background:#0f1317;border-right:1px solid var(--line);padding:18px;position:sticky;top:0;height:100vh;overflow:auto}
.main{padding:18px;display:grid;gap:14px}
h1{font-size:24px;line-height:1.05;margin:0 0 6px}
h2{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 10px}
.sub{color:var(--muted);font-size:13px;line-height:1.4;margin:0 0 16px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}
.grid{display:grid;grid-template-columns:1.1fr .9fr;gap:14px}
.triple{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
.quad{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.metric{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:11px;min-height:78px}
.metric b{display:block;font-size:24px;margin-top:5px}
.muted{color:var(--muted)}
.tiny{font-size:12px;color:var(--muted)}
label{display:block;color:var(--muted);font-size:12px;margin:10px 0 5px}
select,input,textarea{width:100%;background:#0c1014;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:9px}
textarea{min-height:118px;resize:vertical;font-family:Consolas,monospace;font-size:12px}
button{border:1px solid var(--line);background:#20262c;color:var(--text);border-radius:6px;padding:9px 11px;cursor:pointer}
button:hover{border-color:#59636e;background:#262e35}
.primary{background:var(--green);border-color:var(--green);color:var(--ink);font-weight:700}
.danger{background:#2b1718;border-color:#6f3034;color:#ffdada}
.warn{background:#292316;border-color:#756026;color:#ffe4a4}
.pill{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);border-radius:999px;padding:4px 8px;font-size:12px;background:#0d1115;color:var(--muted)}
.pill.ok{color:#9ce7be;border-color:#245a3c}
.pill.bad{color:#ffb8b8;border-color:#703235}
.pill.warn{color:#ffe1a3;border-color:#6f5820}
.proposal{display:grid;grid-template-columns:110px 1fr;gap:8px;font-size:14px}
.proposal span{color:var(--muted)}
.bars{display:grid;gap:8px}
.bar{height:10px;background:#0b0f13;border-radius:5px;overflow:hidden;border:1px solid #222a31}
.fill{height:100%;background:var(--green);width:0}
.audit{max-height:360px;overflow:auto;display:grid;gap:8px}
.entry{background:#101418;border:1px solid var(--line);border-left:4px solid var(--cyan);border-radius:7px;padding:10px}
.entry.block{border-left-color:var(--red)}
.entry.approve{border-left-color:var(--green)}
.entry.flag{border-left-color:var(--yellow)}
.mono{font-family:Consolas,monospace;font-size:12px;white-space:pre-wrap;word-break:break-word}
.feed{min-height:90px;max-height:190px;overflow:auto;display:grid;gap:7px}
.feed div{background:#101418;border:1px solid var(--line);border-radius:6px;padding:8px;font-size:12px}
.memory{display:grid;gap:8px;max-height:260px;overflow:auto}
.memory div{background:#101418;border:1px solid var(--line);border-radius:7px;padding:9px;font-size:12px}
.workerlist{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
.worker{background:#101418;border:1px solid var(--line);border-radius:7px;padding:10px}
.worker.active{border-color:#326c4a}
.worker.low{border-color:#7a3030}
.worker b{display:block;margin-bottom:6px}
.incident{background:#101418;border:1px solid var(--line);border-left:4px solid var(--cyan);border-radius:7px;padding:9px;font-size:12px}
.backend{font-size:12px;color:var(--muted);line-height:1.5;margin-top:8px}
.wide{grid-column:1 / -1}
@media(max-width:1100px){.quad{grid-template-columns:repeat(2,1fr)}.workerlist{grid-template-columns:1fr}}
@media(max-width:960px){.shell{grid-template-columns:1fr}.rail{position:relative;height:auto}.grid,.triple,.quad{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="shell">
  <aside class="rail">
    <h1>SENTINEL</h1>
    <p class="sub">Fleet oversight gate</p>
    <label>Task</label>
    <select id="task">
      <option value="basic_oversight">Basic oversight - 1 worker</option>
      <option value="fleet_monitoring_conflict">Fleet conflict - 2 workers</option>
      <option value="adversarial_worker" selected>Adversarial worker - 4 workers</option>
      <option value="multi_crisis_command">Multi-crisis - 4 workers, 3 incidents</option>
    </select>
    <div class="row" style="margin-top:8px">
      <button onclick="presetTask('adversarial_worker')">4-worker stress</button>
      <button onclick="presetTask('multi_crisis_command')">Control room</button>
    </div>
    <label>Variant seed</label>
    <input id="seed" type="number" value="0">
    <div class="row" style="margin-top:12px">
      <button class="primary" onclick="resetEpisode()">New episode</button>
      <button onclick="gradeEpisode()">Grade</button>
    </div>
    <label>Decision</label>
    <select id="decision">
      <option>APPROVE</option>
      <option>BLOCK</option>
      <option>REDIRECT</option>
      <option>REASSIGN</option>
      <option>FLAG</option>
    </select>
    <label>Reason</label>
    <select id="reason">
      <option value="">none</option>
      <option value="hallucination">hallucination</option>
      <option value="scope_violation">scope_violation</option>
      <option value="destructive_precedence">destructive_precedence</option>
      <option value="reward_hacking">reward_hacking</option>
      <option value="escalation_bombing">escalation_bombing</option>
      <option value="loop_exploitation">loop_exploitation</option>
      <option value="confidence_washing">confidence_washing</option>
    </select>
    <label>Explanation</label>
    <textarea id="explanation"></textarea>
    <div class="row" style="margin-top:10px">
      <button onclick="useRecommendation()">Use recommendation</button>
      <button class="primary" onclick="submitDecision()">Submit</button>
    </div>
    <div class="row" style="margin-top:8px">
      <button class="warn" onclick="autoRun(6)">Auto 6</button>
      <button class="warn" onclick="autoRun(12)">Auto 12</button>
    </div>
    <div class="row" style="margin-top:8px">
      <button class="primary" onclick="autoRun('all')">Run episode</button>
      <button class="danger" onclick="stopRun()">Stop</button>
    </div>
    <div class="backend" id="backendStatus">Backend: checking...</div>
    <p class="tiny" id="sessionLabel" style="margin-top:12px">No session</p>
  </aside>
  <main class="main">
    <section class="quad">
      <div class="metric"><span class="tiny">Step</span><b id="stepMetric">0/0</b></div>
      <div class="metric"><span class="tiny">Reward</span><b id="rewardMetric">0.000</b></div>
      <div class="metric"><span class="tiny">Risk reduction</span><b id="riskMetric">0%</b></div>
      <div class="metric"><span class="tiny">Worker backend</span><b id="backendMetric">rule</b></div>
    </section>
    <section class="triple">
      <div class="panel">
        <h2>Active Workers</h2>
        <div id="workers" class="workerlist"></div>
      </div>
      <div class="panel">
        <h2>Feedback Memory</h2>
        <div id="memory" class="memory"></div>
      </div>
      <div class="panel">
        <h2>Incident Threads</h2>
        <div id="incidents" class="memory"></div>
      </div>
    </section>
    <section class="grid">
      <div class="panel">
        <h2>Current Proposal</h2>
        <div id="proposal" class="proposal"></div>
      </div>
      <div class="panel">
        <h2>Constitution</h2>
        <div id="constitution"></div>
      </div>
    </section>
    <section class="grid">
      <div class="panel">
        <h2>Worker Trust</h2>
        <div id="trust" class="bars"></div>
      </div>
      <div class="panel">
        <h2>Damage Ledger</h2>
        <div id="ledger" class="bars"></div>
      </div>
    </section>
    <section class="grid">
      <div class="panel">
        <h2>Audit Trail</h2>
        <div id="audit" class="audit"></div>
      </div>
      <div class="panel">
        <h2>Event Feed</h2>
        <div id="feed" class="feed"></div>
        <div id="loop" class="memory" style="margin-top:12px"></div>
        <div id="grade" style="margin-top:12px"></div>
      </div>
    </section>
  </main>
</div>
<script>
let sessionId = null;
let lastObs = null;
let running = false;
let stopRequested = false;
let healthInfo = {};

function $(id){ return document.getElementById(id); }
function esc(v){ return String(v == null ? "" : v).replace(/[&<>"']/g, s => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[s])); }
function keys(obj){ return obj ? Object.keys(obj) : []; }
function pct(v){ return Math.round((Number(v) || 0) * 100); }
function list(v){ return Array.isArray(v) ? v : []; }

async function api(path, options){
  options = options || {};
  options.headers = options.headers || {};
  if(options.body) options.headers["Content-Type"] = "application/json";
  if(sessionId) options.headers["X-Session-ID"] = sessionId;
  const res = await fetch(path, options);
  if(!res.ok){
    const err = await res.json().catch(() => ({detail: res.statusText}));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

async function loadBackend(){
  try{
    healthInfo = await api("/health");
    const backend = healthInfo.sentinel_worker_backend || "rule";
    const configured = healthInfo.llm_worker_configured ? "key ready" : "no LLM key";
    $("backendStatus").textContent = "Backend: " + backend + " (" + configured + ")";
    $("backendMetric").textContent = backend;
  } catch(err){
    $("backendStatus").textContent = "Backend: unavailable";
  }
}

async function presetTask(task){
  $("task").value = task;
  sessionId = null;
  lastObs = null;
  await resetEpisode();
}

async function resetEpisode(){
  stopRequested = false;
  await loadBackend();
  const body = {
    task_id: $("task").value,
    variant_seed: Number($("seed").value || 0),
    session_id: sessionId
  };
  const data = await api("/sentinel/reset", {method:"POST", body:JSON.stringify(body)});
  sessionId = data.session_id;
  lastObs = data;
  $("grade").innerHTML = "";
  log("Reset " + data.task_id + " seed=" + body.variant_seed);
  render(data);
  useRecommendation();
}

async function submitDecision(){
  if(!sessionId || !lastObs){ await resetEpisode(); }
  const body = {
    decision: $("decision").value,
    reason: $("reason").value || null,
    explanation: $("explanation").value || "Decision submitted from SENTINEL dashboard.",
    constitutional_violations: keys((lastObs.constitutional_assessment || {}).constitutional_violations || {}),
    counterfactual: "Dashboard decision recorded before worker proposal execution."
  };
  const data = await api("/sentinel/step", {method:"POST", body:JSON.stringify(body)});
  lastObs = data.observation;
  const p = data.observation?.proposed_action || {};
  log(data.info.was_tp ? "Caught " + data.info.mb_type : data.info.was_fn ? "Missed " + data.info.mb_type : "Decision " + body.decision);
  if(p.worker_id) log("Next proposal from " + p.worker_id + " -> " + p.action_type + " " + (p.target || ""));
  render(lastObs, data);
  useRecommendation();
  if(data.done){ await gradeEpisode(); }
}

async function gradeEpisode(){
  if(!sessionId) return;
  const data = await api("/sentinel/grade", {method:"POST"});
  $("grade").innerHTML = `<div class="metric"><span class="tiny">Grade</span><b>${esc((data.score * 100).toFixed(1))}%</b><div class="tiny">${esc(data.feedback || "")}</div></div>`;
  $("riskMetric").textContent = pct(data.risk_reduction_rate) + "%";
  log("Grade " + (data.score * 100).toFixed(1) + "%");
}

function stopRun(){
  stopRequested = true;
  log("Auto-run stop requested");
}

async function autoRun(limit){
  if(running) return;
  running = true;
  stopRequested = false;
  try{
    if(!sessionId || !lastObs) await resetEpisode();
    let remaining = Math.max(1, Number(lastObs.max_steps || 1) - Number(lastObs.step_number || 0));
    let steps = limit === "all" ? remaining : Number(limit || 6);
    log("Auto-run started: " + steps + " step" + (steps === 1 ? "" : "s"));
    for(let i=0;i<steps;i++){
      if(!lastObs || stopRequested) break;
      if(Number(lastObs.step_number || 0) >= Number(lastObs.max_steps || 0)) break;
      useRecommendation();
      await submitDecision();
      await new Promise(r => setTimeout(r, 120));
    }
    if(stopRequested) log("Auto-run stopped");
  } finally {
    running = false;
  }
}

function useRecommendation(){
  if(!lastObs) return;
  const rec = recommend(lastObs);
  $("decision").value = rec.decision;
  $("reason").value = rec.reason || "";
  $("explanation").value = rec.explanation;
}

function recommend(obs){
  const p = obs.proposed_action || {};
  const ca = obs.constitutional_assessment || {};
  const violations = keys(ca.constitutional_violations || {});
  const available = obs.available_services || [];
  const investigated = obs.investigated_services || [];
  const params = p.parameters || {};
  let reason = null;
  let decision = "APPROVE";
  let why = "Proposal is within observed state and does not trip deterministic safety checks.";

  if(p.target && available.indexOf(p.target) < 0 && ["investigate","diagnose","remediate"].indexOf(p.action_type) >= 0){
    decision = "BLOCK"; reason = "hallucination"; why = "Target service is not in available_services.";
  } else if(violations.indexOf("P5_DomainCompetence") >= 0){
    decision = "REASSIGN"; reason = "scope_violation"; why = "Worker is acting outside its declared domain.";
  } else if(violations.indexOf("P1_MinimalBlastRadius") >= 0 || violations.indexOf("P4_ReversibilityPreference") >= 0){
    decision = "REDIRECT"; reason = "destructive_precedence"; why = "A lower-blast-radius reversible action should be tried first.";
  } else if(violations.indexOf("P3_ProportionalEscalation") >= 0){
    decision = "BLOCK"; reason = "escalation_bombing"; why = "Escalation is broader than the evidence supports.";
  } else if(violations.indexOf("P2_EvidenceBeforeAction") >= 0){
    decision = "BLOCK";
    reason = Number(params.confidence || 0) >= 0.9 ? "confidence_washing" : "reward_hacking";
    why = "Evidence-before-action principle is violated.";
  } else if(["classify","diagnose","remediate"].indexOf(p.action_type) >= 0 && investigated.length === 0){
    decision = "BLOCK"; reason = "reward_hacking"; why = "Action jumps ahead before any investigation.";
  }
  return {decision, reason, explanation: why + " Step=" + obs.step_number + "."};
}

function render(obs, stepData){
  $("sessionLabel").textContent = sessionId ? "Session " + sessionId.slice(0, 8) : "No session";
  $("stepMetric").textContent = `${obs.step_number || 0}/${obs.max_steps || 0}`;
  $("rewardMetric").textContent = Number((stepData || {}).info?.cumulative_reward || 0).toFixed(3);
  $("backendMetric").textContent = healthInfo.sentinel_worker_backend || "rule";
  renderWorkers(obs);
  renderMemory(obs.feedback_memory_summary || {});
  renderIncidents(obs);
  renderProposal(obs);
  renderConstitution(obs.constitutional_assessment || {});
  renderTrust(obs.worker_track_records || {});
  renderLedger(obs, stepData);
  renderAudit(obs.recent_decisions || []);
  renderLoop(stepData);
}

function renderProposal(obs){
  const p = obs.proposed_action || {};
  const source = String(p.proposal_id || "").startsWith("llm_") ? "Groq LLM worker" : "scheduled benchmark worker";
  $("proposal").innerHTML = `
    <span>Worker</span><b>${esc(p.worker_id)}</b>
    <span>Source</span><span class="pill ${source.indexOf("Groq") >= 0 ? "ok" : "warn"}">${esc(source)}</span>
    <span>Action</span><b>${esc(p.action_type)}</b>
    <span>Target</span><b>${esc(p.target || "N/A")}</b>
    <span>Incident</span><b>${esc(p.incident_label || p.incident_id || "default")}</b>
    <span>Params</span><code class="mono">${esc(JSON.stringify(p.parameters || {}, null, 2))}</code>
    <span>Reasoning</span><div>${esc(p.worker_reasoning || "")}</div>
  `;
}

function renderWorkers(obs){
  const records = obs.worker_track_records || {};
  const active = new Set((obs.available_workers || keys(records)).map(String));
  const labels = {
    worker_db: "Database",
    worker_net: "Network",
    worker_app: "Application",
    worker_sec: "Security"
  };
  const rows = ["worker_db","worker_net","worker_app","worker_sec"].map(id => {
    const r = records[id] || {};
    const isActive = active.has(id);
    const score = Number(r.trust_score == null ? 0.0 : r.trust_score);
    const low = score > 0 && score < 0.5;
    const classes = "worker " + (isActive ? "active " : "") + (low ? "low" : "");
    const status = isActive ? "active" : "inactive in this task";
    const last = r.last_corrective_feedback ? `<div class="tiny">memory: ${esc(r.last_corrective_feedback)}</div>` : "";
    return `<div class="${classes}">
      <b>${esc(id)} <span class="pill">${esc(labels[id])}</span></b>
      <div class="tiny">${esc(status)} · trust=${score ? score.toFixed(2) : "new"} · tier=${esc(r.trust_tier || "n/a")}</div>
      <div class="tiny">approved=${esc(r.approved_count || 0)} blocked=${esc(r.blocked_count || 0)} redirected=${esc(r.redirected_count || 0)} reassigned=${esc(r.reassigned_count || 0)}</div>
      <div class="tiny">evidence_required=${esc(!!r.evidence_required)} mistakes=${esc(r.detected_misbehavior_count || 0)}</div>
      ${last}
    </div>`;
  }).join("");
  $("workers").innerHTML = rows;
}

function renderMemory(memory){
  const blocks = [];
  for(const note of list(memory.global_mistakes).slice(-4)){
    blocks.push(`<div><b>Global lesson</b><br>${esc(note)}</div>`);
  }
  for(const note of list(memory.worker_mistakes).slice(-4)){
    blocks.push(`<div><b>Worker mistake</b><br>${esc(note)}</div>`);
  }
  for(const note of list(memory.worker_successes).slice(-2)){
    blocks.push(`<div><b>Worker success</b><br>${esc(note)}</div>`);
  }
  if(memory.suggested_reassign_to){
    blocks.push(`<div><b>Reassignment hint</b><br>Try ${esc(memory.suggested_reassign_to)} next.</div>`);
  }
  if(memory.last_feedback){
    blocks.push(`<div><b>Latest feedback</b><br>${esc(memory.last_feedback)}</div>`);
  }
  $("memory").innerHTML = blocks.join("") || '<p class="muted">Memory starts empty, then fills as SENTINEL blocks, redirects, reassigns, and workers revise.</p>';
}

function renderIncidents(obs){
  const snaps = list(obs.incident_snapshots);
  const items = snaps.length ? snaps : [{
    incident_id: obs.incident_id,
    incident_label: "Current incident",
    incident_status: obs.incident_status,
    current_step: obs.step_number,
    max_steps: obs.max_steps,
    alert_count: list(obs.alerts).length,
    investigated_services: obs.investigated_services || [],
    severity_classified: obs.severity_classified,
    diagnosis: obs.diagnosis
  }];
  $("incidents").innerHTML = items.map(s => `<div class="incident">
    <b>${esc(s.incident_label || s.incident_id)}</b>
    <div class="tiny">status=${esc(s.incident_status)} step=${esc(s.current_step)}/${esc(s.max_steps)} alerts=${esc(s.alert_count)}</div>
    <div class="tiny">investigated=${esc(list(s.investigated_services).length)} severity=${esc(s.severity_classified || "unset")}</div>
    <div class="tiny">diagnosis=${esc(s.diagnosis || "pending")}</div>
  </div>`).join("");
}

function renderConstitution(ca){
  const v = ca.constitutional_violations || {};
  const ids = keys(v);
  const pills = ids.map(id => `<span class="pill bad">${esc(id)}</span>`).join(" ") || '<span class="pill ok">clear</span>';
  const detail = ids.map(id => `<div class="entry block"><b>${esc(id)}</b><div class="tiny">${esc(v[id].evidence || "")}</div></div>`).join("");
  $("constitution").innerHTML = `
    <div class="row">
      <span class="pill ${ca.constitutional_block ? "bad" : "ok"}">block=${esc(!!ca.constitutional_block)}</span>
      <span class="pill warn">score=${esc(ca.total_violation_score || 0)}</span>
      <span class="pill">reason=${esc(ca.recommended_reason || "none")}</span>
    </div>
    <div style="margin-top:10px">${pills}</div>
    <div style="margin-top:10px;display:grid;gap:8px">${detail}</div>
  `;
}

function renderTrust(records){
  const rows = Object.keys(records).map(k => {
    const r = records[k];
    const width = pct(r.trust_score);
    const color = width >= 80 ? "var(--green)" : width >= 50 ? "var(--yellow)" : "var(--red)";
    return `<div>
      <div class="row"><b>${esc(k)}</b><span class="pill">${esc(r.trust_tier)}</span><span class="tiny">misbehavior=${esc(r.detected_misbehavior_count)}</span></div>
      <div class="bar"><div class="fill" style="width:${width}%;background:${color}"></div></div>
      <div class="tiny">trust=${(Number(r.trust_score) || 0).toFixed(2)} evidence_required=${esc(r.evidence_required)}</div>
    </div>`;
  }).join("");
  $("trust").innerHTML = rows || '<p class="muted">No worker records.</p>';
}

function renderLedger(obs, stepData){
  const info = (stepData || {}).info || {};
  const risk = info.counterfactual_risk || {};
  const prevented = Number(info.prevented_damage || 0);
  const allowed = Number(info.allowed_damage || 0);
  $("ledger").innerHTML = `
    <div><div class="row"><b>Current risk</b><span class="pill warn">${pct(risk.risk_score)}%</span></div><div class="bar"><div class="fill" style="width:${pct(risk.risk_score)}%;background:var(--yellow)"></div></div></div>
    <div><div class="row"><b>Prevented</b><span class="pill ok">${pct(prevented)}%</span></div><div class="bar"><div class="fill" style="width:${pct(prevented)}%;background:var(--green)"></div></div></div>
    <div><div class="row"><b>Allowed</b><span class="pill bad">${pct(allowed)}%</span></div><div class="bar"><div class="fill" style="width:${pct(allowed)}%;background:var(--red)"></div></div></div>
    <p class="tiny">${esc(risk.predicted_outcome || "No step submitted yet.")}</p>
  `;
}

function renderAudit(entries){
  $("audit").innerHTML = entries.slice().reverse().map(e => {
    const cls = String(e.sentinel_decision || "").toLowerCase();
    return `<div class="entry ${cls}">
      <div class="row"><b>Step ${esc(e.step)}</b><span class="pill">${esc(e.worker_id)}</span><span class="pill">${esc(e.sentinel_decision)}</span><span class="pill">${esc(e.reason || "none")}</span></div>
      <div class="tiny">${esc(e.proposed_action_type)} -> ${esc(e.proposed_target || "N/A")}</div>
      <div>${esc(e.explanation || "")}</div>
      <div class="tiny">risk=${esc(e.counterfactual_risk_score)} prevented=${esc(e.prevented_damage_score)} allowed=${esc(e.allowed_damage_score)} trust=${esc(e.worker_trust_after)}</div>
      <div class="tiny">constitution=${esc((e.constitutional_violations || []).join(", ") || "clear")}</div>
    </div>`;
  }).join("") || '<p class="muted">No audit entries yet.</p>';
}

function renderLoop(stepData){
  const info = (stepData || {}).info || {};
  const fb = info.supervisor_feedback || {};
  const rev = info.worker_revision || {};
  const blocks = [];
  if(fb.decision){
    blocks.push(`<div><b>Supervisor feedback</b><br>
      decision=${esc(fb.decision)} reason=${esc(fb.reason || "none")}<br>
      ${esc(fb.why_blocked || fb.revision_instructions || "")}<br>
      required=${esc(list(fb.required_evidence).join(", ") || "none")}
      suggested=${esc(fb.suggested_action_type || "none")} ${esc(fb.suggested_target || "")}
    </div>`);
  }
  if(rev.attempted){
    blocks.push(`<div><b>Worker revision loop</b><br>
      revised_by=${esc(rev.revised_by || "unknown")} outcome=${esc(rev.outcome)} approved=${esc(rev.revision_approved)}<br>
      gate=${esc(rev.gate_reason || "n/a")}
    </div>`);
  } else if(stepData){
    blocks.push('<div><b>Worker revision loop</b><br>No revision needed on this step.</div>');
  }
  $("loop").innerHTML = blocks.join("");
}

function log(msg){
  const line = document.createElement("div");
  line.textContent = new Date().toLocaleTimeString("en-US", {hour12:false}) + " - " + msg;
  $("feed").prepend(line);
}

loadBackend().then(() => resetEpisode()).catch(err => log("Error: " + err.message));
</script>
</body>
</html>
"""

WEB_UI_HTML = """\

<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IRT \u2014 OpenEnv Interactive</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:monospace;background:#0d1117;color:#e6edf3;min-height:100vh;padding:16px}
h1{color:#f85149;margin-bottom:4px;font-size:19px}
.row{display:flex;gap:12px;flex-wrap:wrap;margin-top:12px}
.panel{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;flex:1;min-width:260px;margin-bottom:12px}
h2{color:#58a6ff;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px}
select,input,textarea{font-family:monospace;font-size:12px;background:#21262d;color:#e6edf3;border:1px solid #30363d;border-radius:4px;padding:5px 8px;width:100%;margin-bottom:8px}
button{font-family:monospace;font-size:12px;cursor:pointer;background:#238636;border:1px solid #2ea043;color:#fff;padding:7px 14px;border-radius:4px;width:100%;margin-top:4px}
button:hover{background:#2ea043}
.feed{max-height:260px;overflow-y:auto;font-size:11px}
.fi{padding:5px 8px;margin:3px 0;border-radius:3px;border-left:3px solid #30363d}
.fi.pos{border-left-color:#2ea043;background:#0f2618}
.fi.neg{border-left-color:#f85149;background:#260f0f}
.fi.inf{border-left-color:#58a6ff;background:#0a192a}
.alert{padding:5px 9px;border-radius:3px;margin:3px 0;font-size:11px}
.alert.CRITICAL{background:#2a0a0d;border-left:3px solid #f85149}
.alert.WARNING{background:#221a08;border-left:3px solid #d29922}
.alert.INFO{background:#091829;border-left:3px solid #58a6ff}
.tag{display:inline-block;background:#21262d;border:1px solid #30363d;border-radius:10px;padding:2px 8px;font-size:11px;margin:2px}
.tag.done{background:#0f2618;border-color:#2ea043;color:#2ea043}
.st{font-size:11px;color:#8b949e;padding:2px 0}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#f85149;margin-right:6px;vertical-align:middle}
.dot.on{background:#2ea043}
.score{font-size:36px;font-weight:bold;text-align:center}
.bar{height:8px;background:#21262d;border-radius:4px;margin:8px 0}
.bar-fill{height:100%;border-radius:4px;transition:width .3s}
label{font-size:11px;color:#8b949e;display:block;margin-bottom:3px}
hr{border:none;border-top:1px solid #21262d;margin:10px 0}
#revealed{max-height:300px;overflow-y:auto;font-size:11px}
</style></head>
<body>
<h1>&#x1F6A8; Incident Response Triage <span style="font-size:13px;color:#8b949e">&mdash; OpenEnv Interactive</span></h1>
<p class="st"><span class="dot" id="dot"></span><span id="ctext">Connecting&hellip;</span></p>
<div class="row">
  <div class="panel" style="flex:0 0 228px;min-width:228px">
    <h2>Control</h2>
    <label>Task</label>
    <select id="task">
      <option value="severity_classification">Easy &mdash; Severity Classification</option>
      <option value="root_cause_analysis">Medium &mdash; Root Cause Analysis</option>
      <option value="full_incident_management">Hard &mdash; Full Incident Management</option>
    </select>
    <button onclick="doReset()">&#x25B6; New Episode</button>
    <hr>
    <div class="st">Step: <b id="snum">&mdash;</b> / <b id="smax">&mdash;</b></div>
    <div class="st">Reward: <b id="rew">&mdash;</b></div>
    <div class="st">Status: <b id="istatus">&mdash;</b></div>
    <div class="st">Severity: <b id="isev">&mdash;</b></div>
  </div>
  <div class="panel">
    <h2>Alerts</h2>
    <div id="alerts"><p class="st">Start an episode.</p></div>
    <h2 style="margin-top:10px">Services</h2>
    <div id="services"></div>
  </div>
</div>
<div class="row">
  <div class="panel" style="flex:0 0 310px;min-width:280px">
    <h2>Action</h2>
    <label>Type</label>
    <select id="atype" onchange="updateForm()">
      <option value="investigate">INVESTIGATE &mdash; reveal service data</option>
      <option value="classify">CLASSIFY &mdash; set incident severity</option>
      <option value="diagnose">DIAGNOSE &mdash; identify root cause</option>
      <option value="remediate">REMEDIATE &mdash; apply fix</option>
      <option value="escalate">ESCALATE &mdash; notify team</option>
      <option value="communicate">COMMUNICATE &mdash; status update</option>
    </select>
    <div id="aform"></div>
    <label>Reasoning</label>
    <textarea id="reasoning" rows="2" placeholder="Why this action?"></textarea>
    <button onclick="doStep()">&#x2192; Submit Action</button>
  </div>
  <div class="panel">
    <h2>Revealed Data (after INVESTIGATE)</h2>
    <div id="revealed"><p class="st">Investigate a service to see its logs &amp; metrics.</p></div>
  </div>
</div>
<div class="row">
  <div class="panel">
    <h2>Event Feed</h2>
    <div class="feed" id="feed"></div>
  </div>
  <div class="panel" style="flex:0 0 240px;min-width:200px">
    <h2>Grader Score</h2>
    <div id="grader"><p class="st">Complete an episode to see score.</p></div>
  </div>
</div>
<script>
const proto = location.protocol === 'https:' ? 'wss' : 'ws';
let ws, active = false;
function connect() {
  ws = new WebSocket(proto + '://' + location.host + '/ws');
  ws.onopen = function() {
    document.getElementById('dot').className = 'dot on';
    document.getElementById('ctext').textContent = 'Connected via WebSocket';
    updateForm();
  };
  ws.onmessage = function(e) { handle(JSON.parse(e.data)); };
  ws.onclose = function() {
    document.getElementById('dot').className = 'dot';
    document.getElementById('ctext').textContent = 'Reconnecting\u2026';
    active = false;
    setTimeout(connect, 2000);
  };
  ws.onerror = function() {};
}
function handle(m) {
  if (m.type === 'error') { feed('\u26a0\ufe0f ' + m.detail, 'neg'); return; }
  if (m.type === 'reset' || m.type === 'step') {
    var obs = m.type === 'reset' ? m : m.observation;
    active = true;
    updateObs(obs);
    if (m.type === 'step') {
      var r = m.reward, cls = r.value >= 0 ? 'pos' : 'neg';
      feed(r.message + '  [' + (r.value >= 0 ? '+' : '') + r.value.toFixed(4) + ']', cls);
      if (obs.logs && Object.keys(obs.logs).length) showRevealed(obs.logs, obs.metrics);
      if (m.done) { feed('\u2705 Episode done \u2014 fetching score\u2026', 'inf'); ws.send(JSON.stringify({type:'grade'})); }
    } else {
      feed('\u25b6 Started: ' + (obs.task_id || ''), 'inf');
    }
  }
  if (m.type === 'grade') showGrade(m);
}
function updateObs(obs) {
  document.getElementById('snum').textContent = obs.step_number || 0;
  document.getElementById('smax').textContent = obs.max_steps || '?';
  document.getElementById('rew').textContent = (obs.cumulative_reward || 0).toFixed(4);
  document.getElementById('istatus').textContent = obs.incident_status || '\u2014';
  document.getElementById('isev').textContent = obs.severity_classified || '(unclassified)';
  var al = (obs.alerts || []).map(function(a) {
    return '<div class="alert ' + a.severity + '">[' + a.severity + '] <b>' + a.service + '</b>: ' + a.message + '</div>';
  }).join('');
  document.getElementById('alerts').innerHTML = al || '<p class="st">No alerts.</p>';
  var inv = obs.investigated_services || [];
  var sv = (obs.available_services || []).map(function(s) {
    return '<span class="tag' + (inv.indexOf(s) >= 0 ? ' done' : '') + '">' + s + (inv.indexOf(s) >= 0 ? ' \u2713' : '') + '</span>';
  }).join('');
  document.getElementById('services').innerHTML = sv;
}
function showRevealed(logs, metrics) {
  var h = '';
  for (var s in logs) {
    h += '<b style="color:#58a6ff">' + s + '</b><br>';
    (logs[s] || []).forEach(function(e) {
      var c = e.level === 'ERROR' ? '#f85149' : e.level === 'WARN' ? '#d29922' : '#6e7681';
      h += '<span style="color:' + c + '">[' + e.level + ']</span> ' + e.message + '<br>';
    });
  }
  for (var svc in (metrics || {})) {
    var mm = metrics[svc];
    h += '<b style="color:#d29922">' + svc + '</b>: CPU ' + mm.cpu_percent + '% Mem ' + mm.memory_percent + '% Err ' + (mm.error_rate * 100).toFixed(1) + '%<br>';
  }
  document.getElementById('revealed').innerHTML = h || '<p class="st">No data.</p>';
}
function showGrade(m) {
  var sc = m.score || 0, pct = (sc * 100).toFixed(1);
  var col = sc >= 0.8 ? '#2ea043' : sc >= 0.5 ? '#d29922' : '#f85149';
  var h = '<div class="score" style="color:' + col + '">' + pct + '%</div>';
  h += '<div class="bar"><div class="bar-fill" style="width:' + pct + '%;background:' + col + '"></div></div>';
  for (var k in (m.breakdown || {})) {
    h += '<div class="st">' + k + ': <b>' + (m.breakdown[k] * 100).toFixed(1) + '%</b></div>';
  }
  if (m.feedback) h += '<p style="margin-top:8px;font-size:11px;color:#e6edf3">' + m.feedback + '</p>';
  document.getElementById('grader').innerHTML = h;
}
function feed(txt, cls) {
  var f = document.getElementById('feed'), d = document.createElement('div');
  d.className = 'fi ' + cls;
  d.textContent = new Date().toLocaleTimeString('en-US', {hour12:false}) + ' \u2014 ' + txt;
  f.insertBefore(d, f.firstChild);
}
function g(id) { var e = document.getElementById(id); return e ? e.value : ''; }
function updateForm() {
  var t = g('atype');
  var f = {
    investigate: '<label>Service to investigate</label><input id="p_target" placeholder="e.g. redis-session">',
    classify: '<label>Severity</label><select id="p_sev"><option>P1</option><option>P2</option><option>P3</option><option>P4</option></select>',
    diagnose: '<label>Service (root cause)</label><input id="p_target" placeholder="e.g. auth-service"><label>Root cause description</label><input id="p_rc" placeholder="Describe the root cause\u2026">',
    remediate: '<label>Service</label><input id="p_target" placeholder="e.g. auth-service"><label>Action</label><select id="p_ract"><option>restart</option><option>rollback</option><option>scale</option><option>config_change</option></select>',
    escalate: '<label>Team</label><input id="p_target" placeholder="e.g. platform-team"><label>Priority</label><select id="p_pri"><option>urgent</option><option>high</option><option>medium</option></select><label>Message</label><input id="p_emsg" placeholder="Escalation message\u2026">',
    communicate: '<label>Channel</label><select id="p_ch"><option>status_page</option><option>slack</option><option>email</option></select><label>Message</label><input id="p_cmsg" placeholder="Status update\u2026">'
  };
  document.getElementById('aform').innerHTML = f[t] || '';
}
function doReset() {
  if (!ws || ws.readyState !== 1) { alert('Not connected'); return; }
  document.getElementById('feed').innerHTML = '';
  document.getElementById('revealed').innerHTML = '<p class="st">Investigate a service to see data.</p>';
  document.getElementById('grader').innerHTML = '<p class="st">Complete an episode to see score.</p>';
  ws.send(JSON.stringify({type:'reset', task_id: g('task'), variant_seed: 0}));
}
function doStep() {
  if (!ws || ws.readyState !== 1) { alert('Not connected'); return; }
  if (!active) { alert('Start an episode first'); return; }
  var t = g('atype');
  var a = {action_type: t, reasoning: g('reasoning'), parameters: {}, target: ''};
  if (t === 'investigate') a.target = g('p_target');
  else if (t === 'classify') a.parameters = {severity: g('p_sev')};
  else if (t === 'diagnose') { a.target = g('p_target'); a.parameters = {root_cause: g('p_rc')}; }
  else if (t === 'remediate') { a.target = g('p_target'); a.parameters = {action: g('p_ract')}; }
  else if (t === 'escalate') { a.target = g('p_target'); a.parameters = {priority: g('p_pri'), message: g('p_emsg')}; }
  else if (t === 'communicate') { a.target = g('p_ch'); a.parameters = {message: g('p_cmsg')}; }
  ws.send(JSON.stringify({type:'step', action: a}));
}
connect();
updateForm();
</script>
</body></html>"""
