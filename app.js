const API = ""; // same origin; set to "http://localhost:5000" if hosting the frontend separately
const LABELS = {study_hours:"Study hours",attendance:"Attendance",previous_grade:"Previous grade",
  assignments_completed:"Assignments",sleep_hours:"Sleep",participation:"Participation",
  parental_support:"Parental support",extracurricular:"Extracurriculars",internet_access:"Internet access",tutoring:"Tutoring"};
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
let batchRows = [];

// ---- Tabs
$$(".tab").forEach(t => t.onclick = () => {
  $$(".tab").forEach(x => x.classList.toggle("active", x === t));
  $$(".panel").forEach(p => p.classList.toggle("active", p.id === t.dataset.tab));
  if (t.dataset.tab === "insights") loadInsights();
});

// ---- Live slider values
$$("input[type=range]").forEach(r => {
  const out = $("#v_" + r.name);
  r.oninput = () => out.textContent = r.value;
});
$("#resetBtn").onclick = () => setTimeout(() => $$("input[type=range]").forEach(r => $("#v_" + r.name).textContent = r.value));

// ---- Predict
$("#predictForm").onsubmit = async e => {
  e.preventDefault();
  const err = $("#formError"), btn = e.submitter;
  err.hidden = true; btn.disabled = true; btn.textContent = "Predicting…";
  const body = Object.fromEntries(new FormData(e.target));
  try {
    const res = await fetch(API + "/api/predict", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed");
    showResult(data);
  } catch (ex) {
    err.textContent = ex.message.includes("fetch") ? "Cannot reach the server. Is the backend running?" : ex.message;
    err.hidden = false;
  } finally { btn.disabled = false; btn.textContent = "Predict Performance"; }
};

function showResult(d) {
  $("#placeholder").hidden = true; $("#resultBody").hidden = false;
  const circ = 326.7, fg = $("#gFg");
  fg.style.strokeDashoffset = circ - circ * d.predicted_score / 100;
  fg.style.stroke = d.predicted_score >= 70 ? "var(--good)" : d.predicted_score >= 50 ? "var(--warn)" : "var(--bad)";
  animateNumber($("#scoreVal"), d.predicted_score);
  $("#gradeVal").textContent = d.grade;
  $("#probVal").textContent = d.pass_probability + "%";
  $("#riskVal").textContent = d.risk_level;
  $("#riskChip").className = "chip " + d.risk_level;
  $("#tips").innerHTML = d.recommendations.map(t => `<li>${esc(t)}</li>`).join("");
  const max = Math.max(1, ...Object.values(d.factor_impact).map(Math.abs));
  $("#impact").innerHTML = Object.entries(d.factor_impact).sort((a,b) => Math.abs(b[1]) - Math.abs(a[1])).map(([k,v]) => {
    const w = Math.abs(v) / max * 50;
    return `<div class="irow"><span>${LABELS[k]}</span><div class="track"><div class="fill ${v>=0?"pos":"neg"}" style="width:${w}%;${v>=0?"left:50%":`left:${50-w}%`}"></div></div><span>${v>0?"+":""}${v}</span></div>`;
  }).join("");
}
function animateNumber(el, to) {
  const start = performance.now();
  (function step(t) {
    const p = Math.min((t - start) / 800, 1);
    el.textContent = (to * p).toFixed(1);
    if (p < 1) requestAnimationFrame(step);
  })(start);
}

// ---- Insights
let insightsLoaded = false;
async function loadInsights() {
  if (insightsLoaded) return;
  try {
    const m = await (await fetch(API + "/api/metrics")).json();
    insightsLoaded = true;
    const stat = (l, v) => `<div class="stat"><small>${l}</small><b>${v}</b></div>`;
    $("#modelStats").innerHTML =
      stat("Accuracy", (m.classification.accuracy * 100).toFixed(1) + "%") +
      stat("F1 score", m.classification.f1) + stat("R² (score model)", m.regression.r2) +
      stat("Avg. error (MAE)", "±" + m.regression.mae + " pts") + stat("Training rows", m.dataset.rows.toLocaleString());
    const fi = Object.entries(m.feature_importance), top = fi[0][1];
    $("#fiChart").innerHTML = fi.map(([k,v]) =>
      `<div class="irow"><span>${LABELS[k]}</span><div class="track"><div class="fill" style="width:${v/top*100}%"></div></div><span>${(v*100).toFixed(1)}%</span></div>`).join("");
    const [[tn, fp], [fn, tp]] = m.classification.confusion_matrix;
    const cell = (n, l, c) => `<div style="background:${c}22;color:${c}">${n}<small>${l}</small></div>`;
    $("#cm").innerHTML = cell(tn,"True Fail","#16a34a") + cell(fp,"False Pass","#dc2626") + cell(fn,"False Fail","#dc2626") + cell(tp,"True Pass","#16a34a");
    $("#dsInfo").textContent = `${m.dataset.rows.toLocaleString()} student records (${(m.dataset.pass_rate*100).toFixed(0)}% pass rate)`;
    $("#colList").textContent = m.features.join(", ");
  } catch { $("#modelStats").innerHTML = '<p class="error">Could not load model metrics.</p>'; }
}
loadInsights().then(() => {});

// ---- Batch
$("#batchBtn").onclick = async () => {
  const f = $("#csvFile").files[0], err = $("#batchError");
  err.hidden = true;
  if (!f) { err.textContent = "Please choose a CSV file first."; err.hidden = false; return; }
  const fd = new FormData(); fd.append("file", f);
  try {
    const res = await fetch(API + "/api/batch", {method:"POST", body:fd});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    batchRows = data.results; renderBatch(data.summary);
  } catch (ex) { err.textContent = ex.message; err.hidden = false; }
};
function renderBatch(s) {
  $("#batchOut").hidden = false;
  const stat = (l, v) => `<div class="stat"><small>${l}</small><b>${v}</b></div>`;
  $("#batchStats").innerHTML = stat("Students", s.total) + stat("Avg. predicted score", s.avg_score) +
    stat("High risk", s.high_risk) + stat("Medium risk", s.medium_risk) + stat("Low risk", s.low_risk);
  drawTable();
}
function drawTable() {
  const q = $("#batchSearch").value.toLowerCase(), r = $("#riskFilter").value;
  $("#batchTable tbody").innerHTML = batchRows
    .filter(x => x.name.toLowerCase().includes(q) && (!r || x.risk_level === r))
    .map(x => `<tr><td>${esc(x.name)}</td><td>${x.predicted_score}</td><td>${x.grade}</td><td>${x.pass_probability}%</td><td><span class="pill ${x.risk_level}">${x.risk_level}</span></td></tr>`).join("");
}
$("#batchSearch").oninput = drawTable; $("#riskFilter").onchange = drawTable;
$("#exportBtn").onclick = () => {
  const csv = "name,predicted_score,grade,pass_probability,risk_level\n" +
    batchRows.map(x => [`"${x.name.replace(/"/g,'""')}"`, x.predicted_score, x.grade, x.pass_probability, x.risk_level].join(",")).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], {type:"text/csv"})); a.download = "edupredict_results.csv"; a.click();
};
