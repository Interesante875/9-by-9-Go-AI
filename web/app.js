// go9 web GUI controller: game flow, AI moves, live analysis and the training dashboard.
"use strict";

const VISITS = [1, 16, 32, 64, 100, 200, 400, 800, 1600, 3200, 6400, 12800];
const MAX_ANALYSIS_VISITS = 50000;

const $ = (id) => document.getElementById(id);
const board = new BoardView($("board"));
const wrCanvas = $("wrgraph");
Charts.attachHover(wrCanvas);

let state = null;          // latest game state from the server
let analysis = null;       // analysis of the current position (if any)
let aiAnalysis = null;     // the AI's search for its previous move
let analyzing = false;
let aiPending = false;
let info = null;
let chain = Promise.resolve();

// ---------------------------------------------------------------- server helpers
function enqueue(fn) {
  const p = chain.then(fn, fn);
  chain = p.catch(() => {});
  return p;
}
async function api(path, method = "POST", body = null) {
  const res = await fetch(path, { method, body, headers: body ? { "Content-Type": "text/plain" } : {} });
  const text = await res.text();
  let data = null;
  try { data = JSON.parse(text); } catch (_) { data = text; }
  if (!res.ok) throw new Error((data && data.error) || text || res.statusText);
  return data;
}
function flash(msg, isError = false) {
  const el = $("status");
  el.textContent = msg;
  el.style.color = isError ? "var(--danger)" : "var(--accent2)";
  if (isError) setTimeout(() => updateStatus(), 2500);
}

// ---------------------------------------------------------------- preferences
const PREF_KEYS = ["optCand", "optOwn", "optPolicy", "optNums", "optScore", "optCoords"];
function loadPrefs() {
  try {
    const p = JSON.parse(localStorage.getItem("go9prefs") || "{}");
    for (const k of PREF_KEYS) if (k in p) $(k).checked = p[k];
    if (p.visits) $("visits").value = p.visits;
    if (p.ctrlBlack) $("ctrlBlack").value = p.ctrlBlack;
    if (p.ctrlWhite) $("ctrlWhite").value = p.ctrlWhite;
  } catch (_) { /* storage unavailable */ }
}
function savePrefs() {
  try {
    const p = {};
    for (const k of PREF_KEYS) p[k] = $(k).checked;
    p.visits = $("visits").value;
    p.ctrlBlack = $("ctrlBlack").value;
    p.ctrlWhite = $("ctrlWhite").value;
    localStorage.setItem("go9prefs", JSON.stringify(p));
  } catch (_) { /* ignore */ }
}
function applyOpts() {
  board.opts = {
    cand: $("optCand").checked, own: $("optOwn").checked, policy: $("optPolicy").checked,
    nums: $("optNums").checked, score: $("optScore").checked, coords: $("optCoords").checked,
  };
  board.wood = null;
  board.draw();
  savePrefs();
}
function visits() { return VISITS[Number($("visits").value) - 1]; }
function controller(color) { return color === 1 ? $("ctrlBlack").value : $("ctrlWhite").value; }

// ---------------------------------------------------------------- rendering
function currentEval() {
  if (analysis && analysis.visits > 0) return { wr: analysis.winrate_black, score: analysis.score_black, visits: analysis.visits };
  if (aiAnalysis && state && aiAnalysis.move_number === state.move_number - 1)
    return { wr: aiAnalysis.winrate_black, score: aiAnalysis.score_black, visits: aiAnalysis.visits };
  if (state && state.winrate_history.length) {
    const i = state.winrate_history.length - 1;
    return { wr: state.winrate_history[i], score: state.score_history[i], visits: 0 };
  }
  return null;
}

function updateStatus() {
  if (!state) return;
  const el = $("status");
  el.style.color = "var(--accent2)";
  const who = state.to_move === 1 ? "Black" : "White";
  if (state.over) {
    const r = state.result;
    let txt = "Game over: " + r;
    if (r.endsWith("+R")) txt = (r[0] === "B" ? "White" : "Black") + " resigned — " + (r[0] === "B" ? "Black" : "White") + " wins";
    else if (r !== "?") txt = `Game over (area score): ${r[0] === "B" ? "Black" : r[0] === "W" ? "White" : "Nobody"} wins ${r}`;
    el.textContent = txt;
  } else if (aiPending) {
    el.textContent = `${who} (AI) is thinking… ${visits()} visits`;
  } else {
    const passNote = state.passes === 1 ? " · opponent passed (passing now ends the game)" : "";
    el.textContent = `${who} to play${controller(state.to_move) === "ai" ? " (AI)" : ""}${passNote}`;
  }
}

function render() {
  if (!state) return;
  board.state = state;
  board.analysis = analysis && analysis.move_number === state.move_number ? analysis : null;
  board.interactive = !state.over && controller(state.to_move) === "human" && !aiPending;
  board.draw();
  $("capB").textContent = state.captures.black;
  $("capW").textContent = state.captures.white;
  $("komiLabel").textContent = `+${state.komi} komi`;
  $("pBlack").classList.toggle("tomove", !state.over && state.to_move === 1);
  $("pWhite").classList.toggle("tomove", !state.over && state.to_move === 2);
  $("thinking").classList.toggle("show", aiPending);
  $("thinkingText").textContent = `AI is thinking (${visits()} visits)…`;
  $("btnAnalyze").classList.toggle("on", analyzing);
  $("btnAnalyze").textContent = analyzing ? "■ Stop analysis" : "▶ Analyze";
  for (const id of ["btnPass", "btnResign", "btnAI"]) $(id).disabled = state.over || aiPending;
  $("btnUndo").disabled = aiPending || state.moves.length === 0;

  const ev = currentEval();
  if (ev && ev.wr !== null) {
    $("evalFill").style.width = (ev.wr * 100).toFixed(1) + "%";
    const leader = ev.wr >= 0.5 ? "Black" : "White";
    $("evalWinrate").textContent = `${leader} ${(Math.max(ev.wr, 1 - ev.wr) * 100).toFixed(1)}%`;
    const sc = ev.score || 0;
    $("evalScore").textContent = (sc >= 0 ? "B+" : "W+") + Math.abs(sc).toFixed(1);
    $("evalVisits").textContent = ev.visits ? `${ev.visits.toLocaleString()} visits` : "network only";
  }
  updateStatus();
  renderMoveTable();
  renderGraph();
}

function renderMoveTable() {
  const tb = $("moveRows");
  let an = board.analysis, note = "";
  if (!an && aiAnalysis && state && aiAnalysis.move_number === state.move_number - 1) {
    an = aiAnalysis;
    note = "AI's search for its last move";
  }
  if (!an || !an.moves || !an.moves.length) {
    tb.innerHTML = `<tr><td colspan="7" class="muted">Press “Analyze” or let the AI move.</td></tr>`;
    return;
  }
  const best = an.moves[0].winrate;
  const rows = an.moves.slice(0, 10).map((m, i) => {
    const c = candidateColor(i, best - m.winrate);
    const lead = (m.score_lead >= 0 ? "+" : "") + m.score_lead.toFixed(1);
    return `<tr data-i="${i}"><td><span class="dot" style="background:rgb(${c})"></span>${i + 1}</td>
      <td><b>${m.move}</b></td><td>${(m.winrate * 100).toFixed(1)}</td><td>${lead}</td>
      <td>${fmtVisits(m.visits)}</td><td>${(m.prior * 100).toFixed(1)}%</td>
      <td class="pv">${m.pv.slice(0, 8).join(" ")}</td></tr>`;
  });
  if (note) rows.unshift(`<tr><td colspan="7" class="muted">${note}</td></tr>`);
  tb.innerHTML = rows.join("");
  tb.querySelectorAll("tr[data-i]").forEach((tr) => {
    const m = an.moves[Number(tr.dataset.i)];
    tr.addEventListener("mouseenter", () => { if (an === board.analysis) board.setPV(m.pv); });
    tr.addEventListener("mouseleave", () => board.setPV(null));
    tr.addEventListener("click", () => { if (an === board.analysis) humanPlay(m.a); });
  });
}

function renderGraph() {
  const wh = state.winrate_history.map((v) => (v === null ? null : v * 100));
  const sh = state.score_history;
  const xs = wh.map((_, i) => i);
  Charts.draw(wrCanvas, {
    series: [
      { name: "Black win %", color: "#e6e8ec", x: xs, y: wh, width: 2 },
      { name: "score lead (B)", color: "rgba(54,194,240,0.75)", x: xs, y: sh, axis: "right", dash: true, width: 1.2 },
    ],
    yRange: [0, 100], hline: 50, marker: state.move_number, xLabel: "move", legend: true,
    area: { base: 50, above: "rgba(0,0,0,0.55)", below: "rgba(240,240,236,0.18)" },
  });
  const ev = currentEval();
  $("graphInfo").textContent = ev && ev.wr !== null ? `move ${state.move_number}` : "";
}

function setState(s, an = undefined) {
  const moved = !state || s.move_number !== state.move_number || s.moves.join() !== state.moves.join();
  state = s;
  if (an !== undefined) analysis = an;
  else if (moved) analysis = null;
  render();
  if (moved) setTimeout(maybeAIMove, controller(1) === "ai" && controller(2) === "ai" ? 250 : 30);
}

// ---------------------------------------------------------------- actions
function maybeAIMove() {
  if (!state || state.over || aiPending) return;
  if (controller(state.to_move) !== "ai") return;
  aiMove();
}

function aiMove() {
  if (!state || state.over || aiPending) return;
  aiPending = true;
  render();
  enqueue(async () => {
    try {
      const mn = state.move_number;
      const res = await api(`/api/genmove?visits=${visits()}`);
      aiPending = false;
      if (res.analysis) { aiAnalysis = res.analysis; aiAnalysis.move_number = mn; }
      setState(res.state);
      if (res.move === "pass") flash(`${mn % 2 === 0 ? "Black" : "White"} (AI) passes`);
    } catch (e) {
      aiPending = false;
      render();
      flash(e.message, true);
    }
  });
}

function humanPlay(a) {
  if (!state || state.over || aiPending) return;
  if (controller(state.to_move) !== "human") { flash("It's the AI's turn", true); return; }
  if (a !== 81 && !state.legal[a]) { flash(`Illegal move ${actionToGtp(a)}`, true); return; }
  enqueue(async () => {
    try {
      const res = await api(`/api/play?move=${actionToGtp(a)}`);
      setState(res.state);
    } catch (e) { flash(e.message, true); }
  });
}

function undo() {
  if (!state || aiPending) return;
  // Against the AI, undo back to the human's previous turn.
  let n = 1;
  const other = 3 - state.to_move;
  if (controller(state.to_move) === "human" && controller(other) === "ai" && state.moves.length >= 2) n = 2;
  if (state.over && state.result.endsWith("+R")) n = 1;
  enqueue(async () => {
    try { setState((await api(`/api/undo?n=${n}`)).state); } catch (e) { flash(e.message, true); }
  });
}

function toggleAnalysis() {
  analyzing = !analyzing;
  render();
  if (analyzing) analysisLoop();
}

async function analysisLoop() {
  let step = 64;
  while (analyzing) {
    if (!state || state.over || aiPending || (analysis && analysis.move_number === state.move_number && analysis.visits >= MAX_ANALYSIS_VISITS)) {
      await new Promise((r) => setTimeout(r, 300));
      continue;
    }
    const t0 = performance.now();
    await enqueue(async () => {
      if (!analyzing || aiPending) return;
      try {
        const res = await api(`/api/analyze?visits=${step}&max=${MAX_ANALYSIS_VISITS}`);
        res.analysis.move_number = res.state.move_number;
        if (state && res.state.move_number === state.move_number) {
          analysis = res.analysis;
          state = res.state;
          render();
        }
      } catch (e) { flash(e.message, true); analyzing = false; render(); }
    });
    // Aim for ~250 ms per request so the GUI stays responsive on CPU and GPU alike.
    const dt = performance.now() - t0;
    step = Math.max(16, Math.min(2000, Math.round(step * (250 / Math.max(20, dt)))));
    await new Promise((r) => setTimeout(r, 20));
  }
}

function newGame() {
  analysis = null;
  aiAnalysis = null;
  const komi = parseFloat($("komiInput").value || "7");
  enqueue(async () => {
    try { setState((await api(`/api/new?komi=${komi}`)).state); flash("New game"); } catch (e) { flash(e.message, true); }
  });
}

async function downloadSgf() {
  const text = await (await fetch("/api/sgf")).text();
  const blob = new Blob([text], { type: "application/x-go-sgf" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `go9-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.sgf`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ---------------------------------------------------------------- wiring
board.onClick = (a) => humanPlay(a);
$("btnPass").onclick = () => humanPlay(81);
$("btnUndo").onclick = undo;
$("btnAI").onclick = aiMove;
$("btnResign").onclick = () => enqueue(async () => {
  try { setState((await api("/api/resign")).state); } catch (e) { flash(e.message, true); }
});
$("btnNew").onclick = newGame;
$("btnAnalyze").onclick = toggleAnalysis;
$("btnSgf").onclick = downloadSgf;
$("sgfFile").onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const text = await f.text();
  enqueue(async () => {
    try { analysis = null; setState((await api("/api/sgf", "POST", text)).state); flash(`Loaded ${f.name}`); } catch (err) { flash(err.message, true); }
  });
  e.target.value = "";
};
$("btnReload").onclick = () => enqueue(async () => {
  try {
    analysis = null;
    setState((await api("/api/reload")).state);
    await loadInfo();
    flash("Model reloaded");
  } catch (e) { flash(e.message, true); }
});
for (const k of PREF_KEYS) $(k).onchange = applyOpts;
$("visits").oninput = () => { $("visitsLabel").textContent = visits(); savePrefs(); render(); };
$("ctrlBlack").onchange = $("ctrlWhite").onchange = () => { savePrefs(); render(); maybeAIMove(); };
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "p" || e.key === "P") humanPlay(81);
  else if (e.key === "u" || e.key === "U" || (e.key === "z" && (e.ctrlKey || e.metaKey))) undo();
  else if (e.key === "a" || e.key === "A") toggleAnalysis();
  else if (e.key === " ") { e.preventDefault(); aiMove(); }
});
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
  document.querySelectorAll(".tabpane").forEach((p) => p.classList.toggle("active", p.id === "tab-" + t.dataset.tab));
  if (t.dataset.tab === "train") refreshTraining();
  else render();
}));
window.addEventListener("resize", () => { if (state) renderGraph(); drawTrainingCharts(); });

async function loadInfo() {
  info = await api("/api/info", "GET");
  const name = info.model ? info.model : "untrained network (random weights)";
  $("modelChip").textContent = `${name} · ${info.blocks}×${info.channels} · ${info.device}`;
  $("modelChip").title = $("modelChip").textContent;
  if (info.run_dir) $("trainInfo").textContent = `Run directory: ${info.run_dir} — refreshes every 15 s while auto-refresh is on.`;
}

// ---------------------------------------------------------------- training dashboard
let metrics = null, evalRows = null;
const C = { blue: "#36c2f0", green: "#7be08f", orange: "#f5a65b", red: "#ef6a5b", purple: "#b18cff", gray: "#c8ccd4", yellow: "#f2d45c" };

async function refreshTraining() {
  try {
    const csv = await (await fetch("/api/metrics?file=metrics")).text();
    if (csv.startsWith("{")) throw new Error(JSON.parse(csv).error);
    metrics = Charts.parseCsv(csv).rows;
  } catch (e) {
    metrics = null;
    $("trainInfo").textContent = e.message;
  }
  try {
    const csv = await (await fetch("/api/metrics?file=eval")).text();
    evalRows = csv.startsWith("{") ? [] : Charts.parseCsv(csv).rows;
  } catch (_) { evalRows = []; }
  try {
    const log = await (await fetch("/api/log")).text();
    if (!log.startsWith("{")) {
      const el = $("trainLog");
      const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
      el.textContent = log.split("\n").slice(-200).join("\n");
      if (atBottom) el.scrollTop = el.scrollHeight;
    }
  } catch (_) { /* ignore */ }
  drawTrainingCharts();
}

function col(name) {
  return { x: metrics.map((r) => r.iter), y: metrics.map((r) => r[name]) };
}
function drawTrainingCharts() {
  if (!$("tab-train").classList.contains("active")) return;
  const smooth = $("optSmooth").checked;
  const charts = {};
  document.querySelectorAll("canvas[data-chart]").forEach((c) => { charts[c.dataset.chart] = c; if (!c._hooked) { Charts.attachHover(c); c._hooked = true; } });
  if (!metrics || !metrics.length) {
    for (const c of Object.values(charts)) Charts.draw(c, { series: [], empty: "no training data yet" });
    $("kpis").innerHTML = "";
    return;
  }
  const S = (name, label, color, extra = {}) => ({ name: label, color, ...col(name), ...extra });
  const base = { smooth, xLabel: "iter" };
  Charts.draw(charts.total, { ...base, series: [S("loss_total", "total loss", C.blue)] });
  Charts.draw(charts.policy, { ...base, series: [S("loss_policy", "policy CE", C.orange), S("policy_kl", "KL(π_mcts‖p)", C.yellow, { axis: "right" })] });
  Charts.draw(charts.value, { ...base, series: [S("loss_value", "value MSE", C.green), S("loss_ownership", "ownership", C.purple), S("loss_score", "score", C.gray)] });
  Charts.draw(charts.ppo, { ...base, series: [S("loss_ppo", "PPO surrogate", C.red), S("entropy", "entropy", C.blue, { axis: "right" })] });
  const er = (evalRows || []).filter((r) => r.opponent === 0 || r.opponent === -1);
  Charts.draw(charts.elo, {
    smooth: false, xLabel: "iter", series: [{ name: "Elo", color: C.green, x: er.map((r) => r.iter), y: er.map((r) => r.elo), points: true },
      { name: "score vs prev (%)", color: C.gray, x: er.map((r) => r.iter), y: er.map((r) => r.score * 100), axis: "right", dash: true, points: true }],
    empty: "evaluation runs every eval_every iterations",
  });
  Charts.draw(charts.games, { ...base, series: [S("avg_game_len", "avg game length", C.orange), { name: "Black win %", color: C.gray, x: col("black_winrate").x, y: col("black_winrate").y.map((v) => v === null ? null : v * 100), axis: "right" }] });
  Charts.draw(charts.buffer, { ...base, smooth: false, series: [S("buffer", "replay buffer", C.purple), S("lr", "learning rate", C.yellow, { axis: "right" })] });
  Charts.draw(charts.speed, { ...base, series: [S("evals_per_sec", "NN evals / s", C.blue), S("selfplay_sec", "self-play s / iter", C.gray, { axis: "right" })] });

  const last = metrics[metrics.length - 1];
  const lastLoss = [...metrics].reverse().find((r) => r.loss_total !== null);
  const kpi = (k, v) => `<div class="kpi"><div class="k">${k}</div><div class="v">${v}</div></div>`;
  $("kpis").innerHTML = [
    kpi("iteration", last.iter),
    kpi("training time", `${last.hours.toFixed(2)} h`),
    kpi("self-play games", Charts.fmt(last.games_total)),
    kpi("positions", Charts.fmt(last.samples_total)),
    kpi("SGD steps", Charts.fmt(last.steps_total)),
    kpi("Elo", Math.round(last.elo)),
    kpi("total loss", lastLoss ? lastLoss.loss_total.toFixed(3) : "—"),
    kpi("NN evals / s", Math.round(last.evals_per_sec)),
  ].join("");
}
$("optSmooth").onchange = drawTrainingCharts;
$("btnRefresh").onclick = refreshTraining;
setInterval(() => { if ($("optAuto").checked && $("tab-train").classList.contains("active")) refreshTraining(); }, 15000);

// ---------------------------------------------------------------- boot
loadPrefs();
$("visitsLabel").textContent = visits();
applyOpts();
loadInfo().catch((e) => flash(e.message, true));
api("/api/state", "GET").then((r) => { $("komiInput").value = r.state.komi; setState(r.state); }).catch((e) => flash(e.message, true));
