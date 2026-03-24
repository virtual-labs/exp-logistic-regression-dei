/* ===== ML Virtual Lab – script.js ===== */
/* Loads real CSV data from car_data.csv and Dengue_dataset.csv */
/* All 3 panels visible simultaneously; W/B sliders for both experiments */

const $ = (id) => document.getElementById(id);

const linearCanvas = $('linearCanvas');
const bridgeCanvas = $('bridgeCanvas');
const logisticCanvas = $('logisticCanvas');
const tooltip = $('tooltip');

const linearCtx = linearCanvas.getContext('2d');
const bridgeCtx = bridgeCanvas.getContext('2d');
const logisticCtx = logisticCanvas.getContext('2d');

/* W / B sliders */
const linWSlider = $('linW');
const linBSlider = $('linB');
const linWValueEl = $('linWValue');
const linBValueEl = $('linBValue');
const logWSlider = $('logW');
const logBSlider = $('logB');
const logWValueEl = $('logWValue');
const logBValueEl = $('logBValue');

const linearFeatureSelect = $('linearFeatureSelect');
const logisticFeatureSelect = $('logisticFeatureSelect');
const linearSampleSelect = $('linearSampleSelect');
const logisticSampleSelect = $('logisticSampleSelect');

const trainBtn = $('trainBtn');
const resetLinearBtn = $('resetLinearBtn');
const autoBridgeBtn = $('autoBridgeBtn');
const resetBridgeBtn = $('resetBridgeBtn');
const resetLogisticBtn = $('resetLogisticBtn');
const trainLogisticBtn = $('trainLogisticBtn');

const linearFormulaEl = $('linearFormula');
const logisticFormulaEl = $('logisticFormula');
const bridgeFormulaEl = $('bridgeFormula');
const bridgeTextEl = $('bridgeText');
const bridgeDetailsEl = $('bridgeDetails');

/* Metric table cells */
const linMSEEl = $('linMSE');
const linRMSEEl = $('linRMSE');
const linR2El = $('linR2');
const logAccuracyEl = $('logAccuracy');
const boundaryEl = $('decisionBoundary');
const logLossEl = $('logLoss');

const linearSampleInfoEl = $('linearSampleInfo');
const logisticSampleInfoEl = $('logisticSampleInfo');

const linearDataInfoEl = $('linearDataInfo');
const logisticDataInfoEl = $('logisticDataInfo');

const linGuideEl = $('linGuide');
const logGuideEl = $('logGuide');

const linearHoverInfo = $('linearHoverInfo');
const linearHoverContent = $('linearHoverContent');
const logisticHoverInfo = $('logisticHoverInfo');
const logisticHoverContent = $('logisticHoverContent');

/* ── CSV parsing ── */
function parseCSV(text) {
  const lines = text.trim().split('\n');
  const headers = lines[0].split(',').map(h => h.trim());
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const vals = lines[i].split(',');
    if (vals.length < headers.length) continue;
    const obj = {};
    headers.forEach((h, j) => {
      const v = vals[j].trim();
      const n = Number(v);
      obj[h] = isNaN(n) ? v : n;
    });
    rows.push(obj);
  }
  return { headers, rows };
}

/* ── Data containers ── */
let carData = [];
let dengueData = [];
let linearRecords = [];
let logisticRecords = [];

const LINEAR_FEATURES = ['Present_Price', 'Year', 'Kms_Driven'];
const LOGISTIC_FEATURES = ['Platelets', 'Hematocrit', 'WBC'];

const INIT_W = 1.0;
const INIT_B = 0.0;

const state = {
  linear:  { w: INIT_W, b: INIT_B },
  logistic: { w: INIT_W, b: INIT_B },
  linearFeature: 'Present_Price',
  logisticFeature: 'Platelets',
  trainingLinear: false,
  trainingLogistic: false,
  linearTrained: false,
  logisticTrained: false,
  linearTrainStep: 0,
  logisticTrainStep: 0,
  bridgeAnimT: 0,
  bridgeAutoMode: false,
  bridgeTransitioning: false,
  testPoint: null,
  rafId: null
};

/* ── Helpers ── */
function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
function sigmoid(z) { return 1 / (1 + Math.exp(-clamp(z, -500, 500))); }
function mean(a) { return a.reduce((s, v) => s + v, 0) / a.length; }
function std(a) { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length) || 1; }

function fmtTick(v) {
  const av = Math.abs(v);
  if (av >= 1000) return Math.round(v).toLocaleString('en-US');
  if (av >= 100) return v.toFixed(1).replace(/\.0$/, '');
  if (av >= 10) return v.toFixed(2).replace(/0+$/, '').replace(/\.$/, '');
  return v.toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
}

function getLinearXValues() { return linearRecords.map(r => r[state.linearFeature]); }
function getLogisticXValues() { return logisticRecords.map(r => r[state.logisticFeature]); }

function linearXNorm(rawX) { const v = getLinearXValues(); return (rawX - mean(v)) / std(v); }
function logisticXNorm(rawX) { const v = getLogisticXValues(); return (rawX - mean(v)) / std(v); }

function linearPredict(rawX) { return state.linear.w * linearXNorm(rawX) + state.linear.b; }
function logisticPredict(rawX) { return sigmoid(state.logistic.w * logisticXNorm(rawX) + state.logistic.b); }

/* ── Fitting targets (OLS / logit closed-form) ── */
function computeLinearTarget() {
  const xs = getLinearXValues();
  const mu = mean(xs), sd = std(xs);
  const xn = linearRecords.map(r => (r[state.linearFeature] - mu) / sd);
  const ys = linearRecords.map(r => r.Selling_Price);
  const xb = mean(xn), yb = mean(ys);
  let num = 0, den = 0;
  for (let i = 0; i < xn.length; i++) { num += (xn[i] - xb) * (ys[i] - yb); den += (xn[i] - xb) ** 2; }
  const w = den ? num / den : INIT_W;
  const b = yb - w * xb;
  return { w, b };
}

function computeLogisticTarget() {
  const xs = getLogisticXValues();
  const mu = mean(xs), sd = std(xs);
  const xn = [], zs = [];
  for (const r of logisticRecords) {
    xn.push((r[state.logisticFeature] - mu) / sd);
    const p = clamp(r.Dengue === 1 ? 0.985 : 0.015, 0.001, 0.999);
    zs.push(Math.log(p / (1 - p)));
  }
  const xb = mean(xn), zb = mean(zs);
  let num = 0, den = 0;
  for (let i = 0; i < xn.length; i++) { num += (xn[i] - xb) * (zs[i] - zb); den += (xn[i] - xb) ** 2; }
  const w = den ? num / den : INIT_W;
  const b = zb - w * xb;
  return { w, b };
}

const MAX_TRAIN_STEPS = 120;

function gradientStepLinear() {
  const target = computeLinearTarget();
  const ease = 0.045;
  state.linear.w += (target.w - state.linear.w) * ease;
  state.linear.b += (target.b - state.linear.b) * ease;
  state.linearTrainStep++;

  /* Sync sliders to current w,b during training */
  syncLinearSliders();

  if (Math.abs(target.w - state.linear.w) < 0.005 && Math.abs(target.b - state.linear.b) < 0.005 || state.linearTrainStep >= MAX_TRAIN_STEPS) {
    state.linear.w = target.w;
    state.linear.b = target.b;
    state.trainingLinear = false;
    state.linearTrained = true;
    syncLinearSliders();
    setLinearTrainedUI();
    enableLinearSamples();
  }
}

function gradientStepLogistic() {
  const target = computeLogisticTarget();
  const ease = 0.045;
  state.logistic.w += (target.w - state.logistic.w) * ease;
  state.logistic.b += (target.b - state.logistic.b) * ease;
  state.logisticTrainStep++;

  syncLogisticSliders();

  if (Math.abs(target.w - state.logistic.w) < 0.005 && Math.abs(target.b - state.logistic.b) < 0.005 || state.logisticTrainStep >= MAX_TRAIN_STEPS) {
    state.logistic.w = target.w;
    state.logistic.b = target.b;
    state.trainingLogistic = false;
    state.logisticTrained = true;
    syncLogisticSliders();
    setLogisticTrainedUI();
    enableLogisticSamples();
  }
}

/* ── Slider sync helpers ── */
function syncLinearSliders() {
  linWSlider.value = state.linear.w;
  linBSlider.value = state.linear.b;
  linWValueEl.textContent = state.linear.w.toFixed(2);
  linBValueEl.textContent = state.linear.b.toFixed(2);
}

function syncLogisticSliders() {
  logWSlider.value = state.logistic.w;
  logBSlider.value = state.logistic.b;
  logWValueEl.textContent = state.logistic.w.toFixed(2);
  logBValueEl.textContent = state.logistic.b.toFixed(2);
}

/* ── Button state helpers ── */
function setLinearTrainedUI() {
  trainBtn.classList.add('trained-done');
  trainBtn.disabled = true;
  trainBtn.textContent = 'Trained';
  resetLinearBtn.classList.add('highlight');
  linearFeatureSelect.disabled = true;
  linWSlider.disabled = true;
  linBSlider.disabled = true;
}

function clearLinearTrainedUI() {
  trainBtn.classList.remove('trained-done');
  trainBtn.disabled = false;
  trainBtn.textContent = 'Fit Optimal Line (Train Data)';
  resetLinearBtn.classList.remove('highlight');
  linearFeatureSelect.disabled = false;
  linWSlider.disabled = false;
  linBSlider.disabled = false;
}

function setLogisticTrainedUI() {
  trainLogisticBtn.classList.add('trained-done');
  trainLogisticBtn.disabled = true;
  trainLogisticBtn.textContent = 'Trained';
  resetLogisticBtn.classList.add('highlight');
  logisticFeatureSelect.disabled = true;
  logWSlider.disabled = true;
  logBSlider.disabled = true;
}

function clearLogisticTrainedUI() {
  trainLogisticBtn.classList.remove('trained-done');
  trainLogisticBtn.disabled = false;
  trainLogisticBtn.textContent = 'Fit Optimal Curve (Train Data)';
  resetLogisticBtn.classList.remove('highlight');
  logisticFeatureSelect.disabled = false;
  logWSlider.disabled = false;
  logBSlider.disabled = false;
}

/* ── Metrics ── */
function linearMSE() {
  let s = 0;
  for (const r of linearRecords) { const e = linearPredict(r[state.linearFeature]) - r.Selling_Price; s += e * e; }
  return s / linearRecords.length;
}

function linearRMSE() { return Math.sqrt(linearMSE()); }

function linearR2() {
  const ys = linearRecords.map(r => r.Selling_Price);
  const yBar = mean(ys);
  let ssTot = 0, ssRes = 0;
  for (const r of linearRecords) {
    const y = r.Selling_Price;
    ssTot += (y - yBar) ** 2;
    ssRes += (y - linearPredict(r[state.linearFeature])) ** 2;
  }
  return ssTot ? 1 - ssRes / ssTot : 0;
}

function logisticAccuracy() {
  let correct = 0;
  for (const r of logisticRecords) {
    const pred = logisticPredict(r[state.logisticFeature]) >= 0.5 ? 1 : 0;
    if (pred === r.Dengue) correct++;
  }
  return correct / logisticRecords.length;
}

function logisticLogLoss() {
  let s = 0;
  const eps = 1e-15;
  for (const r of logisticRecords) {
    const p = clamp(logisticPredict(r[state.logisticFeature]), eps, 1 - eps);
    const y = r.Dengue;
    s += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
  }
  return s / logisticRecords.length;
}

/* ── Drawing ── */
function drawAxes(ctx, canvas, xRange, yRange, labels) {
  const w = canvas.width, h = canvas.height;
  const m = { top: 22, right: 16, bottom: 40, left: 54 };
  const iw = w - m.left - m.right, ih = h - m.top - m.bottom;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = '#e7edf7'; ctx.lineWidth = 1;
  for (let i = 0; i <= 6; i++) {
    const x = m.left + (i / 6) * iw;
    ctx.beginPath(); ctx.moveTo(x, m.top); ctx.lineTo(x, h - m.bottom); ctx.stroke();
  }
  for (let i = 0; i <= 6; i++) {
    const y = m.top + (i / 6) * ih;
    ctx.beginPath(); ctx.moveTo(m.left, y); ctx.lineTo(w - m.right, y); ctx.stroke();
  }

  ctx.strokeStyle = '#394d63'; ctx.lineWidth = 1.4;
  ctx.beginPath(); ctx.moveTo(m.left, h - m.bottom); ctx.lineTo(w - m.right, h - m.bottom); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(m.left, h - m.bottom); ctx.lineTo(m.left, m.top); ctx.stroke();

  ctx.fillStyle = '#1a2a3a'; ctx.font = 'bold 12px Segoe UI';
  ctx.textAlign = 'center';
  ctx.fillText(labels.x, m.left + iw / 2, h - 4);
  ctx.save(); ctx.translate(12, m.top + ih / 2 + 24); ctx.rotate(-Math.PI / 2);
  ctx.fillText(labels.y, 0, 0); ctx.restore();

  ctx.fillStyle = '#3a4f65'; ctx.font = 'bold 10.5px Segoe UI';
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (let i = 0; i <= 6; i++) {
    const x = m.left + (i / 6) * iw;
    ctx.fillText(fmtTick(xRange[0] + (i / 6) * (xRange[1] - xRange[0])), x, h - m.bottom + 4);
  }
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (let i = 0; i <= 6; i++) {
    const y = h - m.bottom - (i / 6) * ih;
    ctx.fillText(fmtTick(yRange[0] + (i / 6) * (yRange[1] - yRange[0])), m.left - 6, y);
  }

  return {
    m,
    toPx: (x, y) => ({
      x: m.left + ((x - xRange[0]) / (xRange[1] - xRange[0])) * iw,
      y: h - m.bottom - ((y - yRange[0]) / (yRange[1] - yRange[0])) * ih
    }),
    fromPx: (px, py) => ({
      x: xRange[0] + ((px - m.left) / iw) * (xRange[1] - xRange[0]),
      y: yRange[0] + ((h - m.bottom - py) / ih) * (yRange[1] - yRange[0])
    })
  };
}

function getLinearRanges() {
  const xs = getLinearXValues();
  const ys = linearRecords.map(r => r.Selling_Price);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const xPadL = (xMax - xMin) * 0.08;
  const xR = [xMin - xPadL, xMax + xPadL];

  /* Include predictions across the x-range so the line always stays in bounds */
  const ysPred = [];
  for (let i = 0; i <= 20; i++) {
    ysPred.push(linearPredict(xR[0] + (i / 20) * (xR[1] - xR[0])));
  }
  const yAll = [...ys, ...ysPred];
  const yLow = Math.min(...yAll), yHigh = Math.max(...yAll);
  const yPad = Math.max(0.5, (yHigh - yLow) * 0.15);
  return { x: xR, y: [yLow - yPad, yHigh + yPad] };
}

function getLogisticRanges() {
  const xs = getLogisticXValues();
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const pad = (xMax - xMin) * 0.06;
  return { x: [xMin - pad, xMax + pad], y: [-0.05, 1.05] };
}

function drawLinearGraph(ctx, canvas) {
  const ranges = getLinearRanges();
  const axis = drawAxes(ctx, canvas, ranges.x, ranges.y, {
    x: state.linearFeature,
    y: 'Selling_Price'
  });

  /* Data points */
  for (const r of linearRecords) {
    const p = axis.toPx(r[state.linearFeature], r.Selling_Price);
    ctx.fillStyle = '#6366f1';
    ctx.beginPath(); ctx.arc(p.x, p.y, 4.5, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#4338ca'; ctx.lineWidth = 0.8;
    ctx.beginPath(); ctx.arc(p.x, p.y, 4.5, 0, Math.PI * 2); ctx.stroke();
  }

  /* Regression line (clipped to plot area) */
  ctx.save();
  ctx.beginPath();
  ctx.rect(axis.m.left, axis.m.top, canvas.width - axis.m.left - axis.m.right, canvas.height - axis.m.top - axis.m.bottom);
  ctx.clip();
  ctx.strokeStyle = '#e8563a'; ctx.lineWidth = 2.6;
  ctx.beginPath();
  for (let i = 0; i <= 160; i++) {
    const xRaw = ranges.x[0] + (i / 160) * (ranges.x[1] - ranges.x[0]);
    const p = axis.toPx(xRaw, linearPredict(xRaw));
    if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
  }
  ctx.stroke();
  ctx.restore();

  /* Predicted markers after training */
  if (state.linearTrained) {
    for (const r of linearRecords) {
      const pred = linearPredict(r[state.linearFeature]);
      const pp = axis.toPx(r[state.linearFeature], pred);
      ctx.fillStyle = '#f59e0b';
      ctx.fillRect(pp.x - 2.5, pp.y - 2.5, 5, 5);
    }
  }

  /* Test point */
  if (state.testPoint && state.testPoint.source === 'linear') {
    const p = axis.toPx(state.testPoint.xValue, state.testPoint.yValue);
    ctx.fillStyle = '#ff7f0e';
    ctx.beginPath(); ctx.arc(p.x, p.y, 5.5, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.8;
    ctx.beginPath(); ctx.arc(p.x, p.y, 5.5, 0, Math.PI * 2); ctx.stroke();
  }

  return axis;
}

function drawBridgeGraph() {
  const axis = drawAxes(bridgeCtx, bridgeCanvas, [-3, 3], [-0.1, 1.1], {
    x: 'z = wx + b',
    y: 'Output'
  });

  const t = state.bridgeAnimT;

  bridgeCtx.strokeStyle = `rgba(124,58,237,${1 - t * 0.4})`;
  bridgeCtx.lineWidth = 2.8;
  bridgeCtx.beginPath();
  for (let i = 0; i <= 180; i++) {
    const x = -3 + (i / 180) * 6;
    const yLin = clamp(0.5 + x * 0.18, -0.1, 1.1);
    const ySig = sigmoid(x * 2.5);
    const y = (1 - t) * yLin + t * ySig;
    const p = axis.toPx(x, y);
    if (i === 0) bridgeCtx.moveTo(p.x, p.y); else bridgeCtx.lineTo(p.x, p.y);
  }
  bridgeCtx.stroke();

  if (t > 0.3) {
    bridgeCtx.strokeStyle = `rgba(236,72,153,${Math.min(1, (t - 0.3) * 2)})`;
    bridgeCtx.setLineDash([6, 4]);
    const p05l = axis.toPx(-3, 0.5);
    const p05r = axis.toPx(3, 0.5);
    bridgeCtx.beginPath(); bridgeCtx.moveTo(p05l.x, p05l.y); bridgeCtx.lineTo(p05r.x, p05r.y); bridgeCtx.stroke();
    bridgeCtx.setLineDash([]);
  }

  bridgeCtx.fillStyle = '#1f2a37'; bridgeCtx.font = 'bold 11px Segoe UI'; bridgeCtx.textAlign = 'left';
  if (t < 0.3) bridgeCtx.fillText('Linear: z = wx + b (unbounded)', axis.m.left + 6, axis.m.top + 14);
  else if (t < 0.9) bridgeCtx.fillText('Transforming through sigmoid...', axis.m.left + 6, axis.m.top + 14);
  else bridgeCtx.fillText('Sigmoid: p = 1/(1+e\u207B\u1DBD) \u2192 probability [0,1]', axis.m.left + 6, axis.m.top + 14);
}

function drawLogisticGraph(ctx, canvas) {
  const ranges = getLogisticRanges();
  const axis = drawAxes(ctx, canvas, ranges.x, ranges.y, {
    x: state.logisticFeature,
    y: 'Probability / Class'
  });

  /* Gradient fill under sigmoid */
  const grad = ctx.createLinearGradient(0, axis.m.top, 0, canvas.height - axis.m.bottom);
  grad.addColorStop(0, 'rgba(244,63,94,0.14)');
  grad.addColorStop(1, 'rgba(16,185,129,0.14)');
  ctx.fillStyle = grad;
  ctx.beginPath();
  for (let i = 0; i <= 180; i++) {
    const x = ranges.x[0] + (i / 180) * (ranges.x[1] - ranges.x[0]);
    const pt = axis.toPx(x, logisticPredict(x));
    if (i === 0) ctx.moveTo(pt.x, pt.y); else ctx.lineTo(pt.x, pt.y);
  }
  ctx.lineTo(axis.toPx(ranges.x[1], 0).x, axis.toPx(ranges.x[1], 0).y);
  ctx.lineTo(axis.toPx(ranges.x[0], 0).x, axis.toPx(ranges.x[0], 0).y);
  ctx.closePath(); ctx.fill();

  /* Sigmoid curve */
  ctx.strokeStyle = '#0891b2'; ctx.lineWidth = 2.6;
  ctx.beginPath();
  for (let i = 0; i <= 180; i++) {
    const x = ranges.x[0] + (i / 180) * (ranges.x[1] - ranges.x[0]);
    const pt = axis.toPx(x, logisticPredict(x));
    if (i === 0) ctx.moveTo(pt.x, pt.y); else ctx.lineTo(pt.x, pt.y);
  }
  ctx.stroke();

  /* Decision boundary vertical line */
  if (Math.abs(state.logistic.w) > 1e-6) {
    const vals = getLogisticXValues();
    const mu = mean(vals), sd = std(vals);
    const xB = mu + sd * (-state.logistic.b / state.logistic.w);
    if (xB >= ranges.x[0] && xB <= ranges.x[1]) {
      ctx.strokeStyle = '#f43f5e'; ctx.setLineDash([6, 4]);
      const pT = axis.toPx(xB, 1), pB = axis.toPx(xB, 0);
      ctx.beginPath(); ctx.moveTo(pT.x, pT.y); ctx.lineTo(pB.x, pB.y); ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  /* Data points */
  for (const r of logisticRecords) {
    const p = axis.toPx(r[state.logisticFeature], r.Dengue);
    ctx.fillStyle = r.Dengue === 1 ? '#f43f5e' : '#10b981';
    ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, Math.PI * 2); ctx.fill();

    if (state.logisticTrained) {
      const prob = logisticPredict(r[state.logisticFeature]);
      const pp = axis.toPx(r[state.logisticFeature], prob);
      ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.arc(pp.x, pp.y, 3.2, 0, Math.PI * 2); ctx.stroke();
    }
  }

  /* Test point */
  if (state.testPoint && state.testPoint.source === 'logistic') {
    const p = axis.toPx(state.testPoint.xValue, state.testPoint.yValue);
    ctx.fillStyle = '#ff7f0e';
    ctx.beginPath(); ctx.arc(p.x, p.y, 5.5, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.8;
    ctx.beginPath(); ctx.arc(p.x, p.y, 5.5, 0, Math.PI * 2); ctx.stroke();
  }

  return axis;
}

/* ── Sample inspection (only after training) ── */
function enableLinearSamples() {
  linearSampleSelect.disabled = false;
  linearSampleSelect.innerHTML = '<option value="">-- select sample --</option>';
  linearRecords.forEach((r, i) => {
    const opt = document.createElement('option');
    opt.value = String(i);
    const pred = linearPredict(r[state.linearFeature]);
    opt.textContent = `#${i} ${r.Car_Name} \u2192 Pred: ${pred.toFixed(2)}`;
    linearSampleSelect.appendChild(opt);
  });
}

function enableLogisticSamples() {
  logisticSampleSelect.disabled = false;
  logisticSampleSelect.innerHTML = '<option value="">-- select sample --</option>';
  logisticRecords.forEach((r, i) => {
    const opt = document.createElement('option');
    opt.value = String(i);
    const prob = logisticPredict(r[state.logisticFeature]);
    const cls = prob >= 0.5 ? 1 : 0;
    opt.textContent = `#${i} Actual:${r.Dengue} \u2192 Pred:${cls} (p=${prob.toFixed(3)})`;
    logisticSampleSelect.appendChild(opt);
  });
}

function updateLinearSampleInfo() {
  const idx = linearSampleSelect.value;
  if (idx === '' || idx == null) { linearSampleInfoEl.innerHTML = ''; return; }
  const r = linearRecords[Number(idx)];
  const pred = linearPredict(r[state.linearFeature]);
  const residual = r.Selling_Price - pred;
  linearSampleInfoEl.innerHTML = [
    `<div><strong>${r.Car_Name}</strong></div>`,
    `<div>${state.linearFeature}: ${fmtTick(r[state.linearFeature])}</div>`,
    `<div>Actual Selling_Price: ${r.Selling_Price.toFixed(2)}</div>`,
    `<div>Predicted: ${pred.toFixed(2)}</div>`,
    `<div>Residual: ${residual.toFixed(2)}</div>`
  ].join('');
}

function updateLogisticSampleInfo() {
  const idx = logisticSampleSelect.value;
  if (idx === '' || idx == null) { logisticSampleInfoEl.innerHTML = ''; return; }
  const r = logisticRecords[Number(idx)];
  const prob = logisticPredict(r[state.logisticFeature]);
  const cls = prob >= 0.5 ? 1 : 0;
  logisticSampleInfoEl.innerHTML = [
    `<div><strong>${r.Dengue === 1 ? 'Dengue Positive' : 'Dengue Negative'}</strong></div>`,
    `<div>${state.logisticFeature}: ${fmtTick(r[state.logisticFeature])}</div>`,
    `<div>Actual class: ${r.Dengue}</div>`,
    `<div>Predicted probability: ${prob.toFixed(4)}</div>`,
    `<div>Predicted class: ${cls}</div>`
  ].join('');
}

/* ── Formulas & metrics ── */
function updateFormulasAndMetrics() {
  linearFormulaEl.innerHTML = [
    '<div>z = w &middot; x + b</div>',
    `<div><em>x is feature, w is weight, b is bias</em></div>`,
    `<div>w = ${state.linear.w.toFixed(3)}, b = ${state.linear.b.toFixed(3)}</div>`
  ].join('');
  logisticFormulaEl.innerHTML = [
    '<div>z = w &middot; x + b</div>',
    '<div>p = 1 / (1 + e<sup>&minus;z</sup>)</div>',
    `<div><em>x is feature, w is weight, b is bias</em></div>`,
    `<div>w = ${state.logistic.w.toFixed(3)}, b = ${state.logistic.b.toFixed(3)}</div>`
  ].join('');

  /* Linear metrics table */
  linMSEEl.textContent = linearMSE().toFixed(4);
  linRMSEEl.textContent = linearRMSE().toFixed(4);
  linR2El.textContent = linearR2().toFixed(4);

  /* Logistic metrics table */
  logAccuracyEl.textContent = (logisticAccuracy() * 100).toFixed(1) + '%';
  logLossEl.textContent = logisticLogLoss().toFixed(4);

  const vals = getLogisticXValues();
  const mu = mean(vals), sd = std(vals);
  if (Math.abs(state.logistic.w) < 1e-6) {
    boundaryEl.textContent = 'N/A';
  } else {
    const xB = mu + sd * (-state.logistic.b / state.logistic.w);
    boundaryEl.textContent = fmtTick(xB);
  }

  /* Data info */
  linearDataInfoEl.textContent = `${linearRecords.length} sampled points from ${carData.length} rows`;
  logisticDataInfoEl.textContent = `${logisticRecords.length} sampled points from ${dengueData.length} rows`;
}

/* ── Tooltip & hover ── */
function showTooltip(event, html) {
  tooltip.innerHTML = html;
  tooltip.style.display = 'block';
  tooltip.style.left = `${event.clientX + 14}px`;
  tooltip.style.top = `${event.clientY + 14}px`;
}
function hideTooltip() {
  tooltip.style.display = 'none';
}

function clearLinearHover() {
  linearHoverInfo.classList.remove('active');
  linearHoverContent.textContent = 'Move your cursor over any point on the graph to see its details here.';
}

function clearLogisticHover() {
  logisticHoverInfo.classList.remove('active');
  logisticHoverContent.textContent = 'Move your cursor over any point on the graph to see its details here.';
}

function nearestPoint(event, canvas, records, featureKey, yKey) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const mx = (event.clientX - rect.left) * scaleX;
  const my = (event.clientY - rect.top) * scaleY;
  let axis;
  if (canvas === linearCanvas) axis = drawLinearGraph(linearCtx, linearCanvas);
  else axis = drawLogisticGraph(logisticCtx, logisticCanvas);
  let best = null, bestD = Infinity;
  for (const r of records) {
    const p = axis.toPx(r[featureKey], r[yKey]);
    const d = Math.hypot(p.x - mx, p.y - my);
    if (d < bestD) { bestD = d; best = r; }
  }
  return bestD < 18 ? best : null;
}

function onLinearMove(event) {
  const r = nearestPoint(event, linearCanvas, linearRecords, state.linearFeature, 'Selling_Price');
  if (!r) { hideTooltip(); clearLinearHover(); return; }
  const pred = linearPredict(r[state.linearFeature]);
  showTooltip(event, [
    `<strong>${r.Car_Name}</strong>`,
    `${state.linearFeature}: ${fmtTick(r[state.linearFeature])}`,
    `Actual: ${r.Selling_Price.toFixed(2)}`,
    `Predicted: ${pred.toFixed(2)}`
  ].join('<br>'));
  /* Highlight below graph */
  linearHoverInfo.classList.add('active');
  linearHoverContent.textContent = `${r.Car_Name}  |  ${state.linearFeature}: ${fmtTick(r[state.linearFeature])}  |  Actual: ${r.Selling_Price.toFixed(2)}  |  Predicted: ${pred.toFixed(2)}`;
}

function onLogisticMove(event) {
  const r = nearestPoint(event, logisticCanvas, logisticRecords, state.logisticFeature, 'Dengue');
  if (!r) { hideTooltip(); clearLogisticHover(); return; }
  const prob = logisticPredict(r[state.logisticFeature]);
  const cls = prob >= 0.5 ? 1 : 0;
  showTooltip(event, [
    `<strong>${r.Dengue === 1 ? 'Dengue +' : 'Dengue \u2212'}</strong>`,
    `${state.logisticFeature}: ${fmtTick(r[state.logisticFeature])}`,
    `Actual: ${r.Dengue}`,
    `Probability: ${prob.toFixed(4)}`,
    `Predicted class: ${cls}`
  ].join('<br>'));
  /* Highlight below graph */
  logisticHoverInfo.classList.add('active');
  logisticHoverContent.textContent = `${r.Dengue === 1 ? 'Dengue +' : 'Dengue \u2212'}  |  ${state.logisticFeature}: ${fmtTick(r[state.logisticFeature])}  |  Actual: ${r.Dengue}  |  Prob: ${prob.toFixed(4)}  |  Pred: ${cls}`;
}

function onLinearClick(event) {
  const rect = linearCanvas.getBoundingClientRect();
  const axis = drawLinearGraph(linearCtx, linearCanvas);
  const scaleX = linearCanvas.width / rect.width;
  const scaleY = linearCanvas.height / rect.height;
  const p = axis.fromPx((event.clientX - rect.left) * scaleX, (event.clientY - rect.top) * scaleY);
  const ranges = getLinearRanges().x;
  const x = clamp(p.x, ranges[0], ranges[1]);
  const pred = linearPredict(x);
  state.testPoint = { source: 'linear', xValue: x, yValue: pred };
  showTooltip(event, [
    '<strong>Test Point</strong>',
    `${state.linearFeature}: ${fmtTick(x)}`,
    `Predicted Selling_Price: ${pred.toFixed(2)}`
  ].join('<br>'));
  drawAll();
}

function onLogisticClick(event) {
  const rect = logisticCanvas.getBoundingClientRect();
  const axis = drawLogisticGraph(logisticCtx, logisticCanvas);
  const scaleX = logisticCanvas.width / rect.width;
  const scaleY = logisticCanvas.height / rect.height;
  const p = axis.fromPx((event.clientX - rect.left) * scaleX, (event.clientY - rect.top) * scaleY);
  const ranges = getLogisticRanges().x;
  const x = clamp(p.x, ranges[0], ranges[1]);
  const prob = logisticPredict(x);
  const cls = prob >= 0.5 ? 1 : 0;
  state.testPoint = { source: 'logistic', xValue: x, yValue: prob };
  showTooltip(event, [
    '<strong>Test Point</strong>',
    `${state.logisticFeature}: ${fmtTick(x)}`,
    `Probability: ${prob.toFixed(4)}`,
    `Predicted class: ${cls}`
  ].join('<br>'));
  drawAll();
}

/* ── Canvas resize ── */
function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(280, Math.floor(rect.width));
  canvas.height = Math.max(200, Math.floor(rect.height));
}

function resizeAllCanvases() {
  [linearCanvas, bridgeCanvas, logisticCanvas].forEach(c => { if (c) resizeCanvas(c); });
}

/* ── Bridge animation ── */
function startBridgeAuto() {
  state.bridgeAutoMode = true;
  state.bridgeAnimT = 0;
  state.bridgeTransitioning = true;
  autoBridgeBtn.textContent = 'Animating...';
  autoBridgeBtn.disabled = true;
}

function resetBridge() {
  state.bridgeAutoMode = false;
  state.bridgeAnimT = 0;
  state.bridgeTransitioning = false;
  autoBridgeBtn.textContent = 'Auto Animate';
  autoBridgeBtn.disabled = false;
  bridgeDetailsEl.innerHTML = '<p>The linear output is unbounded. The sigmoid squeezes it into [0,&nbsp;1] to get a probability. Click <em>Auto Animate</em> to watch the transformation.</p>';
}

function updateBridgeAnimation() {
  if (!state.bridgeTransitioning) return;
  state.bridgeAnimT += 0.004;
  if (state.bridgeAnimT >= 1) {
    state.bridgeAnimT = 1;
    state.bridgeTransitioning = false;
    state.bridgeAutoMode = false;
    autoBridgeBtn.textContent = 'Auto Animate (Slow)';
    autoBridgeBtn.disabled = false;
    bridgeDetailsEl.innerHTML = '<p>Transformation complete. The sigmoid maps any real number to a probability between 0 and 1, enabling classification.</p>';
  }
}

/* ── Draw all ── */
function drawAll() {
  if (linearRecords.length) drawLinearGraph(linearCtx, linearCanvas);
  drawBridgeGraph();
  if (logisticRecords.length) drawLogisticGraph(logisticCtx, logisticCanvas);
  updateFormulasAndMetrics();
}

/* ── Main loop ── */
function updateLoop() {
  if (state.trainingLinear) gradientStepLinear();
  if (state.trainingLogistic) gradientStepLogistic();
  updateBridgeAnimation();
  drawAll();
  state.rafId = requestAnimationFrame(updateLoop);
}

/* ── Feature selectors ── */
function populateFeatureSelectors() {
  linearFeatureSelect.innerHTML = '';
  LINEAR_FEATURES.forEach(f => {
    const opt = document.createElement('option');
    opt.value = f; opt.textContent = f;
    if (f === state.linearFeature) opt.selected = true;
    linearFeatureSelect.appendChild(opt);
  });

  logisticFeatureSelect.innerHTML = '';
  LOGISTIC_FEATURES.forEach(f => {
    const opt = document.createElement('option');
    opt.value = f; opt.textContent = f;
    if (f === state.logisticFeature) opt.selected = true;
    logisticFeatureSelect.appendChild(opt);
  });
}

/* ── Wire events ── */
function wireEvents() {
  /* W / B sliders – live update */
  linWSlider.addEventListener('input', () => {
    if (state.linearTrained) return;
    state.linear.w = Number(linWSlider.value);
    linWValueEl.textContent = state.linear.w.toFixed(2);
    drawAll();
  });

  linBSlider.addEventListener('input', () => {
    if (state.linearTrained) return;
    state.linear.b = Number(linBSlider.value);
    linBValueEl.textContent = state.linear.b.toFixed(2);
    drawAll();
  });

  logWSlider.addEventListener('input', () => {
    if (state.logisticTrained) return;
    state.logistic.w = Number(logWSlider.value);
    logWValueEl.textContent = state.logistic.w.toFixed(2);
    drawAll();
  });

  logBSlider.addEventListener('input', () => {
    if (state.logisticTrained) return;
    state.logistic.b = Number(logBSlider.value);
    logBValueEl.textContent = state.logistic.b.toFixed(2);
    drawAll();
  });

  /* Feature change */
  linearFeatureSelect.addEventListener('change', () => {
    if (state.trainingLinear) return;
    state.linearFeature = linearFeatureSelect.value;
    state.linear.w = INIT_W; state.linear.b = INIT_B;
    state.linearTrained = false;
    state.linearTrainStep = 0;
    clearLinearTrainedUI();
    syncLinearSliders();
    linearSampleSelect.disabled = true;
    linearSampleSelect.innerHTML = '<option value="">Train first</option>';
    linearSampleInfoEl.innerHTML = '';
    state.testPoint = null;
    drawAll();
  });

  logisticFeatureSelect.addEventListener('change', () => {
    if (state.trainingLogistic) return;
    state.logisticFeature = logisticFeatureSelect.value;
    state.logistic.w = INIT_W; state.logistic.b = INIT_B;
    state.logisticTrained = false;
    state.logisticTrainStep = 0;
    clearLogisticTrainedUI();
    syncLogisticSliders();
    logisticSampleSelect.disabled = true;
    logisticSampleSelect.innerHTML = '<option value="">Train first</option>';
    logisticSampleInfoEl.innerHTML = '';
    state.testPoint = null;
    drawAll();
  });

  linearSampleSelect.addEventListener('change', updateLinearSampleInfo);
  logisticSampleSelect.addEventListener('change', updateLogisticSampleInfo);

  /* Train buttons */
  trainBtn.addEventListener('click', () => {
    if (state.linearTrained) return;
    state.trainingLinear = true;
    trainBtn.textContent = 'Training...';
    trainBtn.disabled = true;
    linearFeatureSelect.disabled = true;
    linWSlider.disabled = true;
    linBSlider.disabled = true;
  });

  trainLogisticBtn.addEventListener('click', () => {
    if (state.logisticTrained) return;
    state.trainingLogistic = true;
    trainLogisticBtn.textContent = 'Training...';
    trainLogisticBtn.disabled = true;
    logisticFeatureSelect.disabled = true;
    logWSlider.disabled = true;
    logBSlider.disabled = true;
  });

  /* Reset buttons */
  resetLinearBtn.addEventListener('click', () => {
    state.trainingLinear = false;
    state.linearTrained = false;
    state.linearTrainStep = 0;
    state.linear.w = INIT_W; state.linear.b = INIT_B;
    clearLinearTrainedUI();
    syncLinearSliders();
    linearSampleSelect.disabled = true;
    linearSampleSelect.innerHTML = '<option value="">Train first</option>';
    linearSampleInfoEl.innerHTML = '';
    state.testPoint = null;
    drawAll();
  });

  resetLogisticBtn.addEventListener('click', () => {
    state.trainingLogistic = false;
    state.logisticTrained = false;
    state.logisticTrainStep = 0;
    state.logistic.w = INIT_W; state.logistic.b = INIT_B;
    clearLogisticTrainedUI();
    syncLogisticSliders();
    logisticSampleSelect.disabled = true;
    logisticSampleSelect.innerHTML = '<option value="">Train first</option>';
    logisticSampleInfoEl.innerHTML = '';
    state.testPoint = null;
    drawAll();
  });

  autoBridgeBtn.addEventListener('click', startBridgeAuto);
  resetBridgeBtn.addEventListener('click', resetBridge);

  /* Canvas interactions */
  linearCanvas.addEventListener('mousemove', onLinearMove);
  logisticCanvas.addEventListener('mousemove', onLogisticMove);
  linearCanvas.addEventListener('mouseleave', () => { hideTooltip(); clearLinearHover(); });
  logisticCanvas.addEventListener('mouseleave', () => { hideTooltip(); clearLogisticHover(); });
  linearCanvas.addEventListener('click', onLinearClick);
  logisticCanvas.addEventListener('click', onLogisticClick);

  window.addEventListener('resize', () => { resizeAllCanvases(); drawAll(); });
}

/* ── Sample data from full CSV ── */
function sampleRecords(rows, n) {
  if (rows.length <= n) return rows.slice();
  const step = rows.length / n;
  const out = [];
  for (let i = 0; i < n; i++) out.push(rows[Math.min(Math.floor(i * step), rows.length - 1)]);
  return out;
}

/* ── Load CSV files and init ── */
async function loadData() {
  const [carText, dengueText] = await Promise.all([
    fetch('./car_data.csv').then(r => r.text()),
    fetch('./Dengue_dataset.csv').then(r => r.text())
  ]);
  const carParsed = parseCSV(carText);
  const dengueParsed = parseCSV(dengueText);

  carData = carParsed.rows;
  dengueData = dengueParsed.rows;

  linearRecords = sampleRecords(carData, 50);
  logisticRecords = sampleRecords(dengueData, 60);
}

async function init() {
  await loadData();
  populateFeatureSelectors();
  syncLinearSliders();
  syncLogisticSliders();
  resizeAllCanvases();
  wireEvents();
  if (state.rafId) cancelAnimationFrame(state.rafId);
  state.rafId = requestAnimationFrame(updateLoop);
}

init();
