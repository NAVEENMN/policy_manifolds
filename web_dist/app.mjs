import {clone, clamp, edgeValue, infer, allEdges, validateSession} from './engine.mjs';
import {environments, environmentCatalog, environmentRollout} from './environments.mjs';
import {createPolicyLoader} from './policy-loader.mjs';
import {mujocoRequest, simulationRequest} from './mujoco-client.mjs';
import {drawSimulationScene} from './simulation-scene.mjs';

const $ = id => document.getElementById(id);
const algorithmNames = {PPO: 'Proximal Policy Optimization', SAC: 'Soft Actor-Critic'};
let env, inputShort, inputNames, units;
const sessions = new Map();
const policyLoader = createPolicyLoader(environments, readPolicy);
const {policies, errors: loadErrors, pending: pendingPolicies} = policyLoader;
let environmentVersion = 0;
let runtimeSession = null, runtimeSequence = 0, inputPage = 0;
const INPUT_PAGE_SIZE = 12;
const cleanLabel = value => String(value).replace(/[<>&\"']/g, '').slice(0, 100);
let runVersion = 0, stepPending = null, runtimeReady = true;
const MINT = '#b7dec9', DIM = '#70877a', GRID = '#2a3b30', ORANGE = '#e2b08d';
let original, model, edges, selected = {l: 0, i: 2, o: 0}, point = 3;
let undoStack = [], redoStack = [], revision = 0, dragging = null;
let running = false, episode = 1, live, baseline, samples = [], baseSamples = [];
let accumulator = 0, lastFrame = 0, endSince = null, lastPaint = 0, resetSeed = 42;
let view = {xmin: -1.65, xmax: 1.65, ymin: -1, ymax: 1};
const PLOT = {x: 48, y: 18, w: 413, h: 224};
let toastTimeout;

function toast(message) {
  $('toast').textContent = message;
  $('toast').hidden = false;
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => $('toast').hidden = true, 3500);
}
function label(edge, short = false) {
  return `${edge.l === 0 ? (short ? inputShort[edge.i] : inputNames[edge.i]) : nodeLabel(edge.l, edge.i)} → ${nodeLabel(edge.l + 1, edge.o)}`;
}
const nodeLabel = (layer, index) => `${layer === model.layers.length ? 'z' : 'h' + layer + '·'}${index + 1}`;
const layerLabel = l => `${l === 0 ? 'State' : 'Hidden ' + l} → ${l === model.layers.length - 1 ? 'latent' : 'hidden ' + (l + 1)}`;
function edgeChanged(e) {
  const a = model.layers[e.l], b = original.layers[e.l];
  return a.baseWeights[e.o][e.i] !== b.baseWeights[e.o][e.i]
    || a.coefficients[e.o][e.i].some((c, k) => c !== b.coefficients[e.o][e.i][k]);
}
function selectedLayer() { return model.layers[selected.l]; }
function selectedCoefficients() { return selectedLayer().coefficients[selected.o][selected.i]; }
function saveUndo() {
  undoStack.push(clone(model.layers));
  if (undoStack.length > 100) undoStack.shift();
  redoStack = [];
}
function changed() {
  revision++;
  $('evaluation-result').hidden = true;
  $('edit-count').textContent = `${edges.filter(edgeChanged).length} edited`;
  $('undo').disabled = undoStack.length === 0;
  $('redo').disabled = redoStack.length === 0;
  syncPointInputs();
  renderCards(); renderEditor(); renderNetwork(); updateStats();
}
function edit(fn, message) {
  saveUndo(); fn(); changed();
  if (message) toast(message);
}
function undo() {
  if (!undoStack.length) return;
  redoStack.push(clone(model.layers)); model.layers = undoStack.pop(); fitView(); changed();
}
function redo() {
  if (!redoStack.length) return;
  undoStack.push(clone(model.layers)); model.layers = redoStack.pop(); fitView(); changed();
}

function fitView() {
  const a = selectedLayer(), b = original.layers[selected.l];
  const xmin = a.gridStart + 2 / a.invH - .45 / a.invH;
  const xmax = a.gridStart + (a.nBases + 1) / a.invH + .45 / a.invH;
  const values = [...selectedCoefficients(), ...b.coefficients[selected.o][selected.i], 0];
  for (let k = 0; k <= 120; k++) {
    const x = xmin + k / 120 * (xmax - xmin);
    values.push(edgeValue(a, selected.o, selected.i, x), edgeValue(b, selected.o, selected.i, x));
  }
  const min = Math.min(...values), max = Math.max(...values), pad = Math.max(.2, (max - min) * .15);
  view = {xmin, xmax, ymin: min - pad, ymax: max + pad};
}
const sx = x => PLOT.x + (x - view.xmin) / (view.xmax - view.xmin) * PLOT.w;
const sy = y => PLOT.y + PLOT.h - (y - view.ymin) / (view.ymax - view.ymin) * PLOT.h;
const shortNumber = x => Math.abs(x) >= 10 ? x.toFixed(1) : Math.abs(x) < .001 ? '0' : x.toFixed(2).replace(/0$/, '');
function curvePath(layer, e, mapX, mapY, xmin, xmax, n = 150) {
  return Array.from({length: n + 1}, (_, k) => {
    const x = xmin + (xmax - xmin) * k / n;
    return `${k ? 'L' : 'M'}${mapX(x).toFixed(2)},${mapY(edgeValue(layer, e.o, e.i, x)).toFixed(2)}`;
  }).join(' ');
}
function renderEditor() {
  const a = selectedLayer(), b = original.layers[selected.l];
  $('edge-title').textContent = label(selected);
  $('edge-layer').textContent = `LAYER ${selected.l + 1} · ${layerLabel(selected.l).toUpperCase()}`;
  let html = '<defs><clipPath id="plot-clip"><rect x="48" y="18" width="413" height="224"/></clipPath></defs>';
  for (let k = 0; k <= 4; k++) {
    const x = view.xmin + (view.xmax - view.xmin) * k / 4;
    const y = view.ymin + (view.ymax - view.ymin) * k / 4;
    html += `<path d="M${sx(x)},18V242 M48,${sy(y)}H461" fill="none" stroke="${GRID}" stroke-width=".7"/>`;
    html += `<text x="${sx(x)}" y="258" text-anchor="middle" fill="${DIM}" font-size="9">${shortNumber(x)}</text><text x="40" y="${sy(y) + 3}" text-anchor="end" fill="${DIM}" font-size="9">${shortNumber(y)}</text>`;
  }
  const axisLabel = selected.l === 0 ? inputNames[selected.i] + (inputNames[selected.i].includes('(normalized)') && units[selected.i] === 'normalized' ? '' : ' (' + units[selected.i] + ')') : 'Hidden activation h' + (selected.i + 1);
  html += `<text x="254" y="275" text-anchor="middle" fill="${DIM}" font-size="9">${axisLabel}</text>`;
  html += `<g clip-path="url(#plot-clip)"><path d="M${sx(0)},18V242 M48,${sy(0)}H461" stroke="#45604d" stroke-width=".8" fill="none"/>`;
  const cx = k => a.gridStart + (k + 2) / a.invH;
  html += `<path d="${selectedCoefficients().map((c, k) => `${k ? 'L' : 'M'}${sx(cx(k))},${sy(c)}`).join(' ')}" stroke="#6f968044" fill="none" stroke-dasharray="3 4"/>`;
  html += `<path d="${curvePath(b, selected, sx, sy, view.xmin, view.xmax)}" stroke="#8a9d90" stroke-width="1.4" fill="none" stroke-dasharray="5 4"/>`;
  html += `<path d="${curvePath(a, selected, sx, sy, view.xmin, view.xmax)}" stroke="${MINT}" stroke-width="2.3" fill="none"/>`;
  html += '<g id="live-marker"></g></g>';
  selectedCoefficients().forEach((c, k) => {
    const x = sx(cx(k)), y = clamp(sy(c), 19, 241);
    html += `<g class="control-point ${k === point ? 'selected' : ''}" data-point="${k}" role="slider" tabindex="0" aria-label="Spline control point ${k + 1}" aria-valuemin="-50" aria-valuemax="50" aria-valuenow="${c.toFixed(4)}" transform="translate(${x},${y})"><title>Control point ${k + 1}: ${c.toFixed(4)}. Drag vertically or use arrow keys.</title><circle class="hit" r="13"/><path class="handle" d="M0,-5.5L5.5,0L0,5.5L-5.5,0Z"/></g>`;
  });
  $('curve-editor').innerHTML = html;
  updateMarker();
}
function updateMarker() {
  if (!live) return;
  const prediction = infer(model, env.observe(live.state));
  const x = prediction.activations[selected.l][selected.i];
  const y = edgeValue(selectedLayer(), selected.o, selected.i, x);
  const marker = $('live-marker');
  if (!marker) return;
  if (x < view.xmin || x > view.xmax) {
    const px = x < view.xmin ? 50 : 459;
    marker.innerHTML = `<text x="${px}" y="32" text-anchor="${x < view.xmin ? 'start' : 'end'}" fill="${ORANGE}" font-size="9">Input ${x.toFixed(2)} outside view</text>`;
  } else {
    marker.innerHTML = `<path d="M${sx(x)},18V242" stroke="${MINT}" stroke-opacity=".15"/><circle cx="${sx(x)}" cy="${sy(y)}" r="4" fill="${MINT}" stroke="#15231a" stroke-width="2"/>`;
  }
}
function syncPointInputs() {
  $('point').value = String(point);
  $('coefficient').value = selectedCoefficients()[point].toFixed(5);
  $('residual').value = selectedLayer().baseWeights[selected.o][selected.i].toFixed(5);
}
function selectEdge(e) {
  if (!e || !model) return;
  selected = {l: e.l, i: e.i, o: e.o};
  if (e.l === 0 && model.architecture[0] > 16) inputPage = Math.floor(e.i / INPUT_PAGE_SIZE);
  renderInputPage();
  point = Math.min(point, selectedLayer().nBases - 1);
  $('point').innerHTML = Array.from({length: selectedLayer().nBases}, (_, k) => `<option value="${k}">P${k + 1}</option>`).join('');
  $('curve-editor').setAttribute('aria-label', `Spline curve editor with ${selectedLayer().nBases} draggable control points`);
  fitView(); syncPointInputs(); renderEditor(); renderCards(); renderNetwork();
}
function renderCards() {
  for (let l = 0; l < model.layers.length; l++) {
    $('layer-' + l).innerHTML = edges.filter(e => e.l === l && (l !== 0 || visibleInputs().includes(e.i))).map(e => {
      const a = model.layers[l], b = original.layers[l];
      const points = Array.from({length: 60}, (_, k) => -1.5 + k / 59 * 3);
      const ys = points.flatMap(x => [edgeValue(a, e.o, e.i, x), edgeValue(b, e.o, e.i, x)]);
      const low = Math.min(0, ...ys), high = Math.max(0, ...ys), range = Math.max(.1, high - low);
      const mx = x => 3 + (x + 1.5) / 3 * 88, my = y => 35 - (y - low) / range * 28;
      const active = selected.l === e.l && selected.i === e.i && selected.o === e.o;
      return `<button class="edge-card" data-edge="${e.id}" aria-label="Edit ${label(e)}" aria-pressed="${active}"><span class="edge-name">${label(e, true)}</span>${edgeChanged(e) ? '<i class="changed"></i>' : ''}<svg viewBox="0 0 94 40" aria-hidden="true"><path d="M3,${my(0)}H91" stroke="#3a4b3f" stroke-width=".5"/><path d="${curvePath(b, e, mx, my, -1.5, 1.5, 50)}" stroke="#75877b" fill="none" stroke-dasharray="2 3" stroke-width=".8"/><path d="${curvePath(a, e, mx, my, -1.5, 1.5, 50)}" stroke="${active ? '#d3eddd' : '#94b79f'}" fill="none" stroke-width="1.4"/></svg></button>`;
    }).join('');
  }
}
function visibleInputs() {
  const count = model.architecture[0], start = count > 16 ? inputPage * INPUT_PAGE_SIZE : 0;
  return Array.from({length: count > 16 ? Math.min(INPUT_PAGE_SIZE, count - start) : count}, (_, i) => start + i);
}
function renderInputPage() {
  const count = model.architecture[0];
  $('input-page-control').hidden = count <= 16;
  $('input-page').innerHTML = Array.from({length: Math.ceil(count / INPUT_PAGE_SIZE)}, (_, i) =>
    `<option value="${i}">Inputs ${i * INPUT_PAGE_SIZE + 1}–${Math.min(count, (i + 1) * INPUT_PAGE_SIZE)} of ${count}</option>`).join('');
  $('input-page').value = String(inputPage);
  $('observations').innerHTML = visibleInputs().map(i => `<div class="observation"><div class="row"><span>${env.observationNames[i]}</span><b id="obs-value-${i}">0.000</b></div><div class="observation-track"><i id="obs-bar-${i}"></i></div></div>`).join('');
}
function renderNetwork() {
  const visible = model.architecture.map((count, l) => l === 0 ? visibleInputs() : Array.from({length: count}, (_, i) => i));
  const coords = visible.map((indices, l) => [28 + l * 168 / model.layers.length,
    new Map(indices.map((index, i) => [index, indices.length === 1 ? 132 : 32 + i * 200 / (indices.length - 1)]))]);
  let html = '';
  for (const e of edges) {
    const [x1, ys1] = coords[e.l], [x2, ys2] = coords[e.l + 1];
    if (!ys1.has(e.i) || !ys2.has(e.o)) continue;
    const y1 = ys1.get(e.i), y2 = ys2.get(e.o);
    const active = selected.l === e.l && selected.i === e.i && selected.o === e.o;
    const path = `M${x1 + 10},${y1}C${x1 + 45},${y1} ${x2 - 45},${y2} ${x2 - 10},${y2}`;
    html += `<g class="network-edge" data-edge="${e.id}" role="button" tabindex="0" aria-label="Select ${label(e)}"><path d="${path}" fill="none" stroke="transparent" stroke-width="12"/><path d="${path}" fill="none" stroke="${active ? MINT : edgeChanged(e) ? ORANGE : '#3c5945'}" stroke-width="${active ? 2.3 : 1.1}"/></g>`;
  }
  coords.forEach(([x, ys], l) => ys.forEach((y, i) => {
    const radius = ys.size > 8 ? 8 : 13;
    html += `<circle cx="${x}" cy="${y}" r="${radius}" fill="#21382a" stroke="#63846c" stroke-width="1"/><text x="${x}" y="${y + 3}" text-anchor="middle" fill="#c4ddcb" font-size="${ys.size > 8 || l === 0 && inputShort[i].length > 3 ? 6 : 9}">${l === 0 ? inputShort[i] : nodeLabel(l, i)}</text>`;
  }));
  $('network').innerHTML = html;
}

function newRun(state) { return {state: state.slice(), steps: 0, reward: 0, stableSteps: 0, done: false, failed: false, recorded: false}; }
function resetEpisode(newSeed = false) {
  runVersion++; stepPending = null;
  if (newSeed) episode = 1;
  const raw = Number($('seed').value);
  if (!Number.isInteger(raw) || raw < 0 || raw > 4294967295) {
    $('seed').value = String(resetSeed); toast('Use a whole reset seed between 0 and 4294967295.');
  } else resetSeed = raw;
  const initial = env.initial(resetSeed);
  live = newRun(initial); baseline = newRun(initial);
  samples = [env.metric(initial)]; baseSamples = [env.metric(initial)];
  accumulator = 0; endSince = null;
  $('episode-label').textContent = `Episode ${String(episode).padStart(2, '0')}`;
  $('scene-message').hidden = true;
  updateStats(); drawScene(); drawHistory(); updateMarker();
  if (env.session) connectRuntime(environmentVersion, runVersion);
}
function advanceRun(run, policy, trace) {
  if (run.done) return;
  const next = env.step(run.state, infer(policy, env.observe(run.state)).action);
  applyStep(run, next, trace);
}
function applyStep(run, next, trace) {
  run.state = next.state; run.scene = next.scene; run.reward += next.reward; run.steps++;
  if (run.steps > env.limit - 50 && env.stable(run.state)) run.stableSteps++;
  const terminated = next.terminated ?? env.terminal(run.state);
  run.failed = terminated && !env.goalTask;
  run.success = terminated && !!env.goalTask;
  run.done = terminated || next.truncated || run.steps >= env.limit;
  trace.push(env.metric(run.state));
}
function advance() {
  if (env.remote) { advanceMuJoCo(); return; }
  advanceRun(live, model, samples); advanceRun(baseline, original, baseSamples);
  finishStep();
}
async function advanceMuJoCo() {
  if (stepPending || !runtimeReady || (live.done && baseline.done)) return;
  const version = environmentVersion, currentRun = runVersion;
  const runs = [[live, model], [baseline, original]].map(([run, policy]) => run.done ? null
    : {state: run.state.slice(), action: infer(policy, env.observe(run.state)).action});
  const request = env.session
    ? simulationRequest(env.id, 'step', {session: runtimeSession, sequence: runtimeSequence, actions: runs.map(run => run?.action ?? null)})
    : mujocoRequest('step', {runs});
  stepPending = request;
  try {
    const next = await request;
    if (version !== environmentVersion || currentRun !== runVersion) return;
    if (env.session) runtimeSequence = next.sequence;
    if (next.runs[0]) applyStep(live, next.runs[0], samples);
    if (next.runs[1]) applyStep(baseline, next.runs[1], baseSamples);
    finishStep(); updateStats(); updateMarker(); drawScene(); drawHistory();
  } catch (error) {
    if (version === environmentVersion && currentRun === runVersion) {
      setRunning(false); runtimeReady = false; showRuntimeError(error.message);
    }
  } finally { if (stepPending === request) stepPending = null; }
}
function finishStep() {
  if (live.done && !live.recorded) {
    live.recorded = true;
    if (env.id === 'pendulum') {
      const result = live.stableSteps >= 45 ? 'Held upright' : 'Episode complete';
      $('last-result').textContent = `Episode ${episode}: return ${live.reward.toFixed(1)} · stable ${live.stableSteps}/50 final steps`;
      $('scene-message').textContent = `${result} · return ${live.reward.toFixed(1)}`;
    } else if (env.session) {
      const result = live.failed ? 'Episode ended' : 'Time limit reached';
      $('last-result').textContent = `Episode ${episode}: return ${live.reward.toFixed(2)} · ${live.steps}/${env.limit} steps`;
      $('scene-message').textContent = `${result} · return ${live.reward.toFixed(2)}`;
    } else if (env.goalTask) {
      const result = live.success ? 'Goal reached' : 'Time limit reached';
      $('last-result').textContent = `Episode ${episode}: return ${live.reward} · ${live.steps}/${env.limit} steps · ${result.toLowerCase()}`;
      $('scene-message').textContent = `${result} · return ${live.reward}`;
    } else if (env.id === 'invertedpendulum') {
      const result = live.failed ? 'Pole angle exceeded 0.2 rad' : 'Balanced for 40 s';
      $('last-result').textContent = `Episode ${episode}: return ${live.reward} / 1000 · ${result.toLowerCase()}`;
      $('scene-message').textContent = `${result} · return ${live.reward}`;
    } else {
      const cause = Math.abs(live.state[0]) > 2.4 ? 'track boundary reached' : 'pole angle exceeded 12°';
      $('last-result').textContent = `Episode ${episode}: ${live.steps}/500 · ${live.failed ? cause : 'balanced for 10 s'}`;
      $('scene-message').textContent = live.failed ? `${cause[0].toUpperCase() + cause.slice(1)} · ${live.steps} steps` : 'Balanced · 500 / 500';
    }
    $('scene-message').hidden = false;
  }
  // Let the ghost finish independently when auto restart is off.
  if (live.done && ($('auto-reset').checked || baseline.done)) {
    if (endSince === null) endSince = performance.now();
    if (!$('auto-reset').checked) setRunning(false);
  }
}
function setRunning(value) {
  running = value && runtimeReady;
  accumulator = 0;
  $('play').textContent = running ? 'Ⅱ Pause' : '▶ Run policy';
  $('run-status').textContent = running ? 'Live simulation' : live?.done ? 'Episode finished' : 'Paused · edit freely';
  $('run-dot').style.background = running ? MINT : '#6e8776';
}
function updateStats() {
  if (!live) return;
  const pendulum = env.id === 'pendulum', continuous = model.actionType === 'continuous';
  const score = run => env.id === 'cartpole' || env.id === 'invertedpendulum' ? `${run.reward}<small> / ${env.limit}</small>` : `${run.reward.toFixed(pendulum || env.session ? 1 : 0)}<small> return</small>`;
  $('reward').innerHTML = score(live);
  $('baseline-reward').innerHTML = score(baseline);
  $('angle').innerHTML = `${env.metric(live.state).toFixed(env.goalTask ? 2 : 1)}<small>${env.metricUnit}</small>`;
  env.observe(live.state).forEach((v, i) => {
    if (!$('obs-value-' + i)) return;
    $('obs-value-' + i).textContent = `${v >= 0 ? '+' : ''}${v.toFixed(3)}`;
    const bar = $('obs-bar-' + i), w = Math.abs(clamp(v / env.observationLimits[i], -1, 1)) * 50;
    bar.style.width = w + '%'; bar.style.left = (v < 0 ? 50 - w : 50) + '%';
  });
  const next = infer(model, env.observe(live.state)), orig = infer(original, env.observe(live.state));
  const fill = $('prob-fill');
  if (continuous && model.actionVector) {
    next.action.forEach((value, i) => {
      const low = model.actionLow[i], high = model.actionHigh[i], zero = -low / (high - low) * 100;
      const position = (value - low) / (high - low) * 100;
      $('action-prob-' + i).textContent = `${value >= 0 ? '+' : ''}${value.toFixed(3)}`;
      const bar = $('action-fill-' + i);
      bar.style.left = `${Math.min(zero, position)}%`; bar.style.width = `${Math.abs(position-zero)}%`;
      $('action-baseline-' + i).style.left = `${(orig.action[i] - low) / (high - low) * 100}%`;
    });
    const delta = Math.max(...next.action.map((v, i) => Math.abs(v - orig.action[i])));
    $('decision-comparison').textContent = `Largest control change: ${delta.toFixed(3)}`;
    $('decision-comparison').style.color = delta < .001 ? '#93ad9c' : ORANGE;
  } else if (continuous) {
    const limit = model.torqueLimit, unit = env.controlUnit;
    $('prob-left').textContent = `${next.action >= 0 ? '+' : ''}${next.action.toFixed(3)} ${unit}`;
    $('prob-right').textContent = `${orig.action >= 0 ? '+' : ''}${orig.action.toFixed(3)} ${unit}`;
    fill.style.left = `${50 + Math.min(0, next.action) / (2 * limit) * 100}%`;
    fill.style.width = `${Math.abs(next.action) / (2 * limit) * 100}%`;
    $('baseline-prob').style.left = `${(orig.action + limit) / (2 * limit) * 100}%`;
    const delta = next.action - orig.action;
    $('decision-comparison').textContent = `Δ ${env.controlName} ${delta >= 0 ? '+' : ''}${delta.toFixed(3)} ${unit} at this state`;
    $('decision-comparison').style.color = Math.abs(delta) < .001 ? '#93ad9c' : ORANGE;
  } else if (env.actions.length > 2) {
    next.probabilities.forEach((probability, i) => {
      $('action-prob-' + i).textContent = (probability * 100).toFixed(1) + '%';
      $('action-fill-' + i).style.width = probability * 100 + '%';
      $('action-baseline-' + i).style.left = orig.probabilities[i] * 100 + '%';
      $('action-row-' + i).classList.toggle('chosen', next.action === i);
      $('action-row-' + i).setAttribute('aria-label', `${env.actions[i]}: ${(probability * 100).toFixed(1)} percent${next.action === i ? ', selected' : ''}`);
    });
    $('decision-comparison').textContent = next.action === orig.action ? 'Original agrees at this state' : 'Action changed at this state';
    $('decision-comparison').style.color = next.action === orig.action ? '#93ad9c' : ORANGE;
  } else {
    $('prob-left').textContent = `${((1 - next.pRight) * 100).toFixed(1)}%`;
    $('prob-right').textContent = `${(next.pRight * 100).toFixed(1)}%`;
    fill.style.left = '0'; fill.style.width = `${(1 - next.pRight) * 100}%`;
    $('baseline-prob').style.left = `${(1 - orig.pRight) * 100}%`;
    $('decision-comparison').textContent = next.action === orig.action ? 'Original agrees at this state' : 'Action changed at this state';
    $('decision-comparison').style.color = next.action === orig.action ? '#93ad9c' : ORANGE;
  }
}
function canvasContext(id) {
  const canvas = $(id), rect = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * dpr)), height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {ctx, w: rect.width, h: rect.height};
}
function drawScene() {
  if (env.session) {
    const {ctx, w, h} = canvasContext('scene');
    drawSimulationScene(ctx, w, h, live, baseline, $('ghost').checked);
    return;
  }
  if (!live) return;
  if (env.id === 'pendulum') { drawPendulum(); return; }
  if (env.id === 'acrobot') { drawAcrobot(); return; }
  if (env.id === 'mountaincar') { drawMountainCar(); return; }
  const {ctx, w, h} = canvasContext('scene');
  ctx.clearRect(0, 0, w, h);
  const inverted = env.id === 'invertedpendulum';
  const ground = h * .69, unit = (w - 56) / (inverted ? 2.8 : 5.6), cartW = Math.max(32, unit * (inverted ? .3 : .52)), cartH = 21;
  ctx.strokeStyle = '#26392d'; ctx.lineWidth = .5;
  for (let x = w / 2 % 28; x < w; x += 28) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h - 36); ctx.stroke(); }
  for (let y = 8; y < h - 36; y += 28) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  ctx.setLineDash([3, 5]); ctx.strokeStyle = '#48634f';
  for (const value of (inverted ? [-1, 1] : [-2.4, 2.4])) { const x = w / 2 + unit * value; ctx.beginPath(); ctx.moveTo(x, 20); ctx.lineTo(x, ground + 15); ctx.stroke(); }
  ctx.setLineDash([]); ctx.strokeStyle = '#6a8d73'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(17, ground + cartH / 2 + 7); ctx.lineTo(w - 17, ground + cartH / 2 + 7); ctx.stroke();
  ctx.font = '9px ui-monospace, monospace'; ctx.fillStyle = '#789482'; ctx.textAlign = 'center';
  for (const value of (inverted ? [-1, -.5, 0, .5, 1] : [-2.4, -1.2, 0, 1.2, 2.4])) {
    const x = w / 2 + unit * value;
    ctx.fillText(`${value === 0 ? '0' : value.toFixed(1)} m`, x, ground + 37);
    ctx.beginPath(); ctx.moveTo(x, ground + 19); ctx.lineTo(x, ground + 23); ctx.stroke();
  }
  function cart(run, ghost) {
    const x = w / 2 + unit * run.state[0], y = ground;
    const angle = env.angle(run.state);
    const pole = Math.min(h * .51, unit * (inverted ? .6 : 1.55)), tx = x + Math.sin(angle) * pole, ty = y - Math.cos(angle) * pole;
    ctx.save(); ctx.globalAlpha = ghost ? .48 : 1;
    ctx.lineWidth = ghost ? 2 : 6; ctx.strokeStyle = ghost ? '#95a99a' : run.failed ? '#e0ae8d' : MINT;
    ctx.setLineDash(ghost ? [5, 5] : []); ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(tx, ty); ctx.stroke();
    ctx.lineWidth = 1.5;
    ctx.fillStyle = ghost ? '#425d4955' : '#335c41'; ctx.strokeStyle = ghost ? '#9cb8a5' : '#a5d5b4';
    ctx.beginPath(); ctx.roundRect(x - cartW / 2, y - cartH / 2, cartW, cartH, 5); ctx.fill(); ctx.stroke();
    ctx.setLineDash([]);
    for (const dx of [-cartW * .31, cartW * .31]) { ctx.fillStyle = ghost ? '#708978' : '#a3cbb0'; ctx.beginPath(); ctx.arc(x + dx, y + cartH / 2 + 3, 4, 0, Math.PI * 2); ctx.fill(); }
    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fillStyle = '#1b3524'; ctx.fill(); ctx.strokeStyle = '#b8dbc1'; ctx.stroke();
    if (!ghost) { ctx.beginPath(); ctx.arc(tx, ty, 4, 0, Math.PI * 2); ctx.fillStyle = run.failed ? ORANGE : '#deeedf'; ctx.fill(); }
    ctx.restore();
  }
  if ($('ghost').checked) cart(baseline, true);
  cart(live, false);
  if (!live.done) {
    const action = infer(model, env.observe(live.state)).action;
    const direction = inverted ? Math.sign(action) : action === 1 ? 1 : -1;
    const x = w / 2 + unit * live.state[0], y = ground + 65;
    ctx.strokeStyle = '#88b599'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x - direction * 9, y); ctx.lineTo(x + direction * 15, y); ctx.lineTo(x + direction * 10, y - 4); ctx.moveTo(x + direction * 15, y); ctx.lineTo(x + direction * 10, y + 4); ctx.stroke();
  }
}
function drawPendulum() {
  const {ctx, w, h} = canvasContext('scene');
  ctx.clearRect(0, 0, w, h);
  const cx = w / 2, cy = h * .48, radius = Math.min(h * .34, w * .29);
  ctx.strokeStyle = '#26392d'; ctx.lineWidth = .5;
  for (let x = cx % 28; x < w; x += 28) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h - 36); ctx.stroke(); }
  for (let y = 8; y < h - 36; y += 28) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  ctx.fillStyle = '#81b79512'; ctx.strokeStyle = '#557760'; ctx.setLineDash([3, 5]);
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.arc(cx, cy, radius + 15, -Math.PI / 2 - Math.PI / 12, -Math.PI / 2 + Math.PI / 12); ctx.closePath(); ctx.fill(); ctx.stroke();
  ctx.beginPath(); ctx.arc(cx, cy, radius, 0, 2 * Math.PI); ctx.strokeStyle = '#3e5847'; ctx.stroke(); ctx.setLineDash([]);
  ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.fillStyle = '#789482';
  ctx.fillText('UPRIGHT TARGET', 18, 24);
  ctx.fillText(`${live.steps} / 200 steps`, 18, 40);
  function rod(run, ghost) {
    // The positive angle and torque direction matches Gymnasium's rendering.
    const x = cx - Math.sin(run.state[0]) * radius, y = cy - Math.cos(run.state[0]) * radius;
    ctx.save(); ctx.globalAlpha = ghost ? .45 : 1; ctx.strokeStyle = ghost ? '#9bb4a3' : MINT;
    ctx.lineWidth = ghost ? 3 : 6; ctx.setLineDash(ghost ? [5, 5] : []);
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(x, y, ghost ? 9 : 11, 0, Math.PI * 2); ctx.fillStyle = ghost ? '#53705e' : '#b7dec9'; ctx.fill();
    ctx.restore();
  }
  if ($('ghost').checked) rod(baseline, true);
  rod(live, false);
  ctx.beginPath(); ctx.arc(cx, cy, 7, 0, Math.PI * 2); ctx.fillStyle = '#1d3526'; ctx.fill(); ctx.strokeStyle = '#a0caae'; ctx.lineWidth = 2; ctx.stroke();
  const torque = infer(model, env.observe(live.state)).action;
  if (!live.done && Math.abs(torque) > .02) {
    const start = .65, end = start - torque / 2 * Math.PI * .9, r = 28;
    ctx.strokeStyle = ORANGE; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(cx, cy, r, start, end, torque > 0); ctx.stroke();
    const ax = cx + Math.cos(end) * r, ay = cy + Math.sin(end) * r;
    const tangent = end + (torque > 0 ? -Math.PI / 2 : Math.PI / 2);
    ctx.beginPath(); ctx.moveTo(ax - 6 * Math.cos(tangent - .5), ay - 6 * Math.sin(tangent - .5)); ctx.lineTo(ax, ay); ctx.lineTo(ax - 6 * Math.cos(tangent + .5), ay - 6 * Math.sin(tangent + .5)); ctx.stroke();
  }
}
function drawAcrobot() {
  const {ctx, w, h} = canvasContext('scene'); ctx.clearRect(0, 0, w, h);
  const unit = Math.min((h - 60) / 4.5, (w - 60) / 4.5), cx = w / 2, cy = (h - 30) / 2;
  ctx.fillStyle = '#a9d9bd08'; ctx.fillRect(16, 10, w - 32, cy - unit - 10);
  ctx.strokeStyle = '#5a8066'; ctx.setLineDash([4, 5]); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(16, cy - unit); ctx.lineTo(w - 16, cy - unit); ctx.stroke(); ctx.setLineDash([]);
  ctx.font = '9px ui-monospace, monospace'; ctx.fillStyle = '#96b4a1';
  ctx.fillText('GOAL: TIP ABOVE +1 m', 18, cy - unit - 10);
  function links(run, ghost) {
    const [a, b] = run.state;
    const x1 = cx + Math.sin(a) * unit, y1 = cy + Math.cos(a) * unit;
    const x2 = x1 + Math.sin(a + b) * unit, y2 = y1 + Math.cos(a + b) * unit;
    ctx.save(); ctx.globalAlpha = ghost ? .45 : 1; ctx.lineCap = 'round';
    ctx.strokeStyle = ghost ? '#9bb4a3' : MINT; ctx.lineWidth = ghost ? 2 : 6; ctx.setLineDash(ghost ? [4, 5] : []);
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.setLineDash([]);
    for (const [x, y] of [[cx, cy], [x1, y1], [x2, y2]]) {
      ctx.beginPath(); ctx.arc(x, y, 5, 0, 2 * Math.PI); ctx.fillStyle = ghost ? '#7e9787' : '#273e30'; ctx.fill(); ctx.lineWidth = 1.5; ctx.stroke();
    }
    if (!ghost) { ctx.fillStyle = run.success ? MINT : ORANGE; ctx.beginPath(); ctx.arc(x2, y2, 3, 0, 2 * Math.PI); ctx.fill(); }
    ctx.restore();
  }
  if ($('ghost').checked) links(baseline, true);
  links(live, false);
  ctx.fillStyle = '#789482'; ctx.fillText(`${live.steps} / ${env.limit} steps`, 18, h - 40);
}

function drawMountainCar() {
  const {ctx, w, h} = canvasContext('scene'); ctx.clearRect(0, 0, w, h);
  const px = x => 24 + (x + 1.2) / 1.8 * (w - 48);
  const py = x => h * .74 - (Math.sin(3 * x) * .45 + .55) * h * .46;
  ctx.beginPath(); ctx.moveTo(px(-1.2), h - 37);
  for (let k = 0; k <= 150; k++) { const x = -1.2 + k / 150 * 1.8; ctx.lineTo(px(x), py(x)); }
  ctx.lineTo(px(.6), h - 37); ctx.closePath(); ctx.fillStyle = '#23372a'; ctx.fill();
  ctx.beginPath();
  for (let k = 0; k <= 150; k++) { const x = -1.2 + k / 150 * 1.8; k ? ctx.lineTo(px(x), py(x)) : ctx.moveTo(px(x), py(x)); }
  ctx.strokeStyle = '#668770'; ctx.lineWidth = 1.5; ctx.stroke();
  const goalX = px(.5), goalY = py(.5);
  ctx.strokeStyle = MINT; ctx.beginPath(); ctx.moveTo(goalX, goalY); ctx.lineTo(goalX, goalY - 37); ctx.stroke();
  ctx.fillStyle = MINT; ctx.beginPath(); ctx.moveTo(goalX, goalY - 37); ctx.lineTo(goalX + 18, goalY - 31); ctx.lineTo(goalX, goalY - 25); ctx.fill();
  function car(run, ghost) {
    const x = run.state[0], slope = -1.35 * Math.cos(3 * x) * h * .46 / ((w - 48) / 1.8);
    ctx.save(); ctx.translate(px(x), py(x)); ctx.rotate(Math.atan(slope)); ctx.globalAlpha = ghost ? .45 : 1;
    ctx.setLineDash(ghost ? [3, 3] : []); ctx.fillStyle = ghost ? '#637d6b44' : '#335c41'; ctx.strokeStyle = ghost ? '#a0b5a7' : MINT; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.roundRect(-14, -17, 28, 12, 3); ctx.fill(); ctx.stroke(); ctx.setLineDash([]);
    for (const dx of [-8, 8]) { ctx.beginPath(); ctx.arc(dx, -4, 3, 0, Math.PI * 2); ctx.fillStyle = ghost ? '#849c8c' : '#c7e1d1'; ctx.fill(); }
    ctx.restore();
  }
  if ($('ghost').checked) car(baseline, true);
  car(live, false);
  ctx.font = '9px ui-monospace, monospace'; ctx.fillStyle = '#789482'; ctx.textAlign = 'left';
  ctx.fillText(`GOAL: x ≥ 0.5 m · ${live.steps} / 200 steps`, 18, 25);
  ctx.textAlign = 'center'; for (const x of [-1.2, -.6, 0, .5]) ctx.fillText(`${x.toFixed(1)} m`, px(x), h - 42);
}

function drawHistory() {
  const {ctx, w, h} = canvasContext('history'); ctx.clearRect(0, 0, w, h);
  const pendulum = env.id === 'pendulum';
  const values = [...samples, ...baseSamples, 0], min = Math.min(...values), max = Math.max(...values);
  const pad = Math.max(.05, (max-min)*.15);
  const [low, high] = env.session ? [min-pad, max+pad] : env.plotRange;
  const ticks = env.session ? [low, (low+high)/2, high] : env.plotTicks;
  const x = i => 28 + i / env.limit * (w - 35), y = a => h - 5 - (a - low) / (high - low) * (h - 10);
  ctx.font = '8px ui-monospace, monospace'; ctx.textAlign = 'left';
  if (pendulum) { ctx.fillStyle = '#81b79516'; ctx.fillRect(28, y(15), w - 28, y(-15) - y(15)); }
  for (const value of ticks) {
    const py = y(value);
    ctx.strokeStyle = value === 0 ? '#354c3d' : '#775e445c'; ctx.lineWidth = .7; ctx.setLineDash(value === 0 ? [] : [4, 5]);
    ctx.beginPath(); ctx.moveTo(27, py); ctx.lineTo(w, py); ctx.stroke(); ctx.fillStyle = '#789180'; ctx.fillText(env.session ? value.toFixed(2) : String(value), 0, py + 3);
  }
  for (const [trace, color, dash] of [[baseSamples, '#6b8473', [3, 4]], [samples, MINT, []]]) {
    ctx.strokeStyle = color; ctx.lineWidth = 1.4; ctx.setLineDash(dash); ctx.beginPath();
    trace.forEach((value, i) => i && (!pendulum || Math.abs(value - trace[i - 1]) < 180) ? ctx.lineTo(x(i), y(value)) : ctx.moveTo(x(i), y(value))); ctx.stroke();
  }
  ctx.setLineDash([]);
}
function frame(now) {
  const elapsed = lastFrame ? Math.min((now - lastFrame) / 1000, .1) : 0;
  lastFrame = now;
  if (model && live) {
    if (running) {
      if (live.done && $('auto-reset').checked) {
        if (endSince === null) endSince = now;
        if (now - endSince > 1200) { episode++; resetEpisode(); }
      } else {
        accumulator = Math.min(.5, accumulator + elapsed * Number($('speed').value));
        let iterations = 0;
        while (accumulator >= env.dt && running && !stepPending && iterations++ < 24) {
          advance(); accumulator -= env.dt;
          if (live.done && $('auto-reset').checked) { accumulator = 0; break; }
        }
      }
    }
    if (now - lastPaint > 40) { updateStats(); updateMarker(); drawHistory(); lastPaint = now; }
    drawScene();
  }
  requestAnimationFrame(frame);
}

function wireUI() {
  $('input-page').onchange = () => {
    inputPage = Number($('input-page').value);
    selectEdge({l: 0, i: inputPage * INPUT_PAGE_SIZE, o: 0});
    updateStats();
  };
  $('environment-select').onchange = () => switchEnvironment($('environment-select').value);
  $('retry-policy').onclick = () => switchEnvironment(env.id, true);
  $('load-policy').onclick = () => $('policy-file').click();
  $('policy-file').onchange = async () => {
    const file = $('policy-file').files[0], target = env;
    if (!file) return;
    try {
      if (file.size > 5_000_000) throw new Error('Actor JSON must be smaller than 5 MB.');
      const contents = await file.text(), raw = JSON.parse(contents);
      if (!raw.checkpoint_sha256) raw.checkpoint_sha256 = await fingerprint(contents);
      const loaded = target.decode(raw);
      policies.set(target.id, loaded); sessions.delete(target.id); loadErrors.delete(target.id);
      renderEnvironmentPicker();
      if (env.id === target.id) switchEnvironment(target.id, true);
      toast(`${target.name} actor loaded`);
    } catch (error) { toast(error.message); }
    $('policy-file').value = '';
  };
  for (const id of ['edge-layers', 'network']) {
    $(id).addEventListener('click', event => {
      const target = event.target.closest('[data-edge]');
      if (target) selectEdge(edges.find(e => e.id === target.dataset.edge));
    });
  }
  $('network').addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.target.closest('[data-edge]')?.dispatchEvent(new MouseEvent('click', {bubbles: true})); }
  });
  const editor = $('curve-editor');
  editor.addEventListener('pointerdown', event => {
    const handle = event.target.closest('[data-point]');
    if (!handle || event.button !== 0) return;
    event.preventDefault(); point = Number(handle.dataset.point);
    dragging = {id: event.pointerId, before: clone(model.layers), started: false};
    editor.setPointerCapture(event.pointerId);
    syncPointInputs(); renderEditor();
  });
  editor.addEventListener('pointermove', event => {
    if (!dragging || dragging.id !== event.pointerId) return;
    if (!dragging.started) { saveUndo(); dragging.started = true; }
    const rect = editor.getBoundingClientRect();
    const y = (event.clientY - rect.top) / rect.height * 280;
    const value = clamp(view.ymin + (PLOT.y + PLOT.h - y) / PLOT.h * (view.ymax - view.ymin), -50, 50);
    selectedCoefficients()[point] = value;
    changed();
  });
  editor.addEventListener('pointerup', event => {
    if (dragging?.id !== event.pointerId) return;
    const moved = dragging.started; dragging = null;
    if (editor.hasPointerCapture(event.pointerId)) editor.releasePointerCapture(event.pointerId);
    if (moved) toast('Curve updated · live on the next action');
  });
  editor.addEventListener('pointercancel', event => {
    if (dragging?.id !== event.pointerId) return;
    if (dragging.started) { model.layers = dragging.before; undoStack.pop(); changed(); }
    dragging = null;
  });
  editor.addEventListener('keydown', event => {
    const handle = event.target.closest('[data-point]');
    if (!handle || !['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault(); point = Number(handle.dataset.point);
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      point = clamp(point + (event.key === 'ArrowRight' ? 1 : -1), 0, selectedLayer().nBases - 1); syncPointInputs(); renderEditor();
    } else {
      edit(() => selectedCoefficients()[point] = clamp(selectedCoefficients()[point] + (event.key === 'ArrowUp' ? 1 : -1) * (event.shiftKey ? .5 : .05), -50, 50));
    }
    editor.querySelector(`[data-point="${point}"]`).focus();
  });
  $('point').addEventListener('change', () => { point = Number($('point').value); syncPointInputs(); renderEditor(); });
  for (const id of ['coefficient', 'residual']) {
    const commit = () => {
      const value = Number($(id).value);
      if ($(id).value.trim() === '' || !Number.isFinite(value) || Math.abs(value) > 50) { toast('Use a number between −50 and 50.'); syncPointInputs(); return; }
      const previous = id === 'coefficient' ? selectedCoefficients()[point] : selectedLayer().baseWeights[selected.o][selected.i];
      if (value === previous || value.toFixed(5) === previous.toFixed(5)) return;
      edit(() => {
        if (id === 'coefficient') selectedCoefficients()[point] = value;
        else selectedLayer().baseWeights[selected.o][selected.i] = value;
        fitView();
      });
    };
    $(id).addEventListener('change', commit);
    $(id).addEventListener('blur', commit);
    $(id).addEventListener('keydown', event => { if (event.key === 'Enter') { event.preventDefault(); commit(); } });
  }
  $('undo').onclick = undo; $('redo').onclick = redo;
  $('fit').onclick = () => { fitView(); renderEditor(); };
  $('reset-edge').onclick = () => edit(() => {
    const source = original.layers[selected.l];
    selectedLayer().coefficients[selected.o][selected.i] = source.coefficients[selected.o][selected.i].slice();
    selectedLayer().baseWeights[selected.o][selected.i] = source.baseWeights[selected.o][selected.i]; fitView();
  }, 'Selected edge restored');
  $('zero-edge').onclick = () => edit(() => {
    selectedCoefficients().fill(0); selectedLayer().baseWeights[selected.o][selected.i] = 0; fitView();
  }, 'Selected edge removed from the live policy');
  $('reset-all').onclick = () => edit(() => { model = clone(original); fitView(); }, 'Trained policy restored · restart for a fresh comparison');
  $('play').onclick = () => {
    if (!running && live.done) { episode++; resetEpisode(); }
    setRunning(!running);
  };
  $('step').onclick = () => { setRunning(false); if (live.done && baseline.done) { episode++; resetEpisode(); } advance(); updateStats(); updateMarker(); drawScene(); drawHistory(); };
  $('restart').onclick = () => { episode++; resetEpisode(); setRunning(running); };
  $('seed').onchange = () => { resetEpisode(true); setRunning(running); };
  $('ghost').onchange = () => { $('ghost-legend').hidden = !$('ghost').checked; drawScene(); };
  $('help').onclick = () => $('guide').showModal();
  $('close-help').onclick = () => $('guide').close();
  $('guide').addEventListener('click', event => { if (event.target === $('guide')) { const r = $('guide').getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) $('guide').close(); } });
  $('export').onclick = () => {
    const data = {format: 'policy-curve-session-v1', environment: env.name, checkpointSha256: original.sha256,
      exportedAt: new Date().toISOString(), seed: resetSeed,
      layers: model.layers.map(l => ({coefficients: l.coefficients, baseWeights: l.baseWeights}))};
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'}));
    const a = document.createElement('a'); a.href = url; a.download = `${env.id}-curve-edits.json`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000); toast('Edits exported');
  };
  $('import').onclick = () => $('import-file').click();
  $('import-file').onchange = async () => {
    const file = $('import-file').files[0]; if (!file) return;
    const importVersion = environmentVersion;
    try {
      if (file.size > 1_000_000) throw new Error('Session file is too large.');
      const data = JSON.parse(await file.text());
      if (importVersion !== environmentVersion) throw new Error('Environment changed. Import into the selected environment again.');
      const imported = validateSession(data, original);
      edit(() => { model = imported; fitView(); }, 'Imported edits applied to the live policy');
    } catch (error) { toast(error.message); }
    $('import-file').value = '';
  };
  $('evaluate').onclick = async () => {
    const button = $('evaluate'), snapshot = clone(model), originalSnapshot = original;
    const snapshotRevision = revision, snapshotEnvironment = env, version = environmentVersion;
    button.disabled = true; button.textContent = 'Comparing…';
    let editedResults = [], originalResults = [];
    try {
      if (snapshotEnvironment.remote) {
        setRunning(false);
        if (stepPending) await stepPending;
        if (version !== environmentVersion) return;
        const result = snapshotEnvironment.session
          ? await simulationRequest(snapshotEnvironment.id, 'compare', {edited: snapshot, original: originalSnapshot, seeds: Array.from({length: 20}, (_, seed) => seed)})
          : await mujocoRequest('compare', {edited: snapshot, original: originalSnapshot, starts: Array.from({length: 20}, (_, seed) => snapshotEnvironment.initial(seed))});
        editedResults = result.edited; originalResults = result.original;
      } else for (let seed = 0; seed < 20; seed++) {
        if (version !== environmentVersion) return;
        const initial = snapshotEnvironment.initial(seed);
        editedResults.push(environmentRollout(snapshot, snapshotEnvironment, initial));
        originalResults.push(environmentRollout(originalSnapshot, snapshotEnvironment, initial));
        if (seed % 2 === 0) await new Promise(resolve => setTimeout(resolve, 0));
      }
      if (version !== environmentVersion) return;
      const summary = results => {
        const rewards = results.map(r => r.reward), mean = rewards.reduce((a, b) => a + b, 0) / rewards.length;
        if (snapshotEnvironment.id === 'pendulum') return `${mean.toFixed(1)} mean return (worst ${Math.min(...rewards).toFixed(1)}; ${results.filter(r => r.stable).length}/20 stable)`;
        if (snapshotEnvironment.session) return `${mean.toFixed(2)} mean return (worst ${Math.min(...rewards).toFixed(2)})`;
        if (snapshotEnvironment.goalTask) return `${mean.toFixed(1)} mean return (worst ${Math.min(...rewards)}; ${results.filter(r => r.success).length}/20 reached goal)`;
        return `${mean.toFixed(1)}/${snapshotEnvironment.limit} mean (worst ${Math.min(...rewards)})`;
      };
      $('evaluation-result').textContent = `${revision === snapshotRevision ? 'Current edits' : 'Edits at comparison start'}: ${summary(editedResults)}. Original: ${summary(originalResults)}. 20 paired app seeds, 0–19.${snapshotEnvironment.id === 'pendulum' ? ' Higher return is better; 0 is ideal. Stable = ≥45 of the last 50 steps within ±15° and |ω| < 1 rad/s.' : ''}`;
      $('evaluation-result').hidden = false;
    } catch (error) { toast('Comparison failed: ' + error.message); }
    finally { if (version === environmentVersion) { button.disabled = false; button.textContent = 'Compare 20 episodes ↗'; } }
  };
  document.addEventListener('keydown', e => {
    if (!model || /INPUT|SELECT|TEXTAREA/.test(e.target.tagName) || $('guide').open) return;
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); e.shiftKey ? redo() : undo(); }
    if (e.code === 'Space' && e.target === document.body) { e.preventDefault(); $('play').click(); }
  });
  window.addEventListener('blur', () => { accumulator = 0; lastFrame = 0; });
}

function renderEnvironmentPicker() {
  const status = id => !environments[id] ? ' · coming later'
    : policies.has(id) ? ' · policy loaded'
    : pendingPolicies.has(id) ? ' · loading…'
    : env?.id === id ? ' · model needed' : '';
  $('environment-select').innerHTML = environmentCatalog.map(group => `<optgroup label="${group.group}">${group.entries.map(([id, name]) => `<option value="${id}" ${environments[id] ? '' : 'disabled'}>${name}${status(id)}</option>`).join('')}</optgroup>`).join('');
  if (env) {
    $('environment-select').value = env.id;
    $('connection').textContent = env.name + status(env.id);
    $('connection-dot').dataset.state = policies.has(env.id) ? 'ready' : pendingPolicies.has(env.id) ? 'loading' : 'missing';
  }
}

function showPolicyAvailability(available, loading = false) {
  for (const name of ['policy-content', 'network-content', 'editor-content']) {
    $(name).hidden = !available; $(name).inert = !available;
  }
  for (const name of ['missing-policy', 'network-empty', 'editor-empty']) $(name).hidden = !!available;
  for (const name of ['play', 'step', 'restart', 'evaluate', 'export', 'import']) $(name).disabled = !available;
  for (const name of ['load-policy', 'retry-policy', 'policy-file-hint']) $(name).hidden = loading;
  $('missing-policy').setAttribute('aria-busy', String(loading));
  $('policy-load-status').textContent = loading ? 'LOADING POLICY' : 'ENVIRONMENT READY';
  $('network-empty').textContent = loading ? 'Loading the selected policy’s network…' : 'Load a saved policy to inspect its network and edge curves.';
}

function showRuntimeError(message) {
  $('runtime-notice').hidden = false;
  $('runtime-detail').textContent = message;
  for (const name of ['play', 'step', 'evaluate']) $(name).disabled = true;
  $('run-status').textContent = 'Simulation unavailable · curve editing ready';
}

async function connectRuntime(version, currentRun = runVersion) {
  if (!env.remote) return;
  for (const name of ['play', 'step', 'evaluate']) $(name).disabled = true;
  const key = env.id, session = env.session, resume = running;
  runtimeReady = false; setRunning(false);
  $('runtime-title').textContent = `${env.physics || 'MuJoCo'} simulation`;
  $('run-status').textContent = 'Connecting to simulator…';
  try {
    const result = session ? await simulationRequest(key, 'reset', {seed: resetSeed, session: runtimeSession}) : await mujocoRequest('status');
    if (version !== environmentVersion || currentRun !== runVersion) {
      if (result.session) simulationRequest(key, 'release', {session: result.session}).catch(() => {});
      return;
    }
    if (session) {
      runtimeSession = result.session; runtimeSequence = result.sequence;
      [live, baseline].forEach((run, i) => {run.state = result.runs[i].state; run.scene = result.runs[i].scene;});
      samples = [env.metric(live.state)]; baseSamples = [env.metric(baseline.state)];
      updateStats(); updateMarker(); drawScene(); drawHistory();
    }
    runtimeReady = true; $('runtime-notice').hidden = true;
    for (const name of ['play', 'step', 'evaluate']) $(name).disabled = false;
    setRunning(resume);
  } catch (error) { if (version === environmentVersion && currentRun === runVersion) showRuntimeError(error.message); }
}

async function switchEnvironment(id, force = false) {
  if ((!force && env?.id === id && (model || pendingPolicies.has(id))) || !environments[id]) return;
  if (env && model && !force) sessions.set(env.id, {model, undoStack, redoStack, selected, point, seed: resetSeed});
  if (env?.session && runtimeSession) simulationRequest(env.id, 'release', {session: runtimeSession}).catch(() => {});
  runtimeSession = null; runtimeSequence = 0;
  dragging = null; environmentVersion++; runVersion++; stepPending = null; running = false; accumulator = 0;
  const version = environmentVersion;
  env = environments[id]; original = policies.get(id); model = null; live = null; baseline = null;
  history.replaceState(null, '', '#' + id);
  runtimeReady = !env.remote; $('runtime-notice').hidden = true;
  inputShort = env.inputShort; inputNames = env.inputNames; units = env.units;
  $('environment-chip').textContent = env.name;
  $('algorithm-name').textContent = algorithmNames[env.algorithm] ? `${env.algorithm} — ${algorithmNames[env.algorithm]}` : env.algorithm;
  $('evaluation-result').hidden = true;
  $('saved-result').hidden = true;
  $('missing-title').textContent = env.name;
  $('missing-description').textContent = env.description;
  $('missing-icon').textContent = id === 'mountaincar' ? '⌃' : id === 'acrobot' ? '⌁' : '◴';
  if (!original) {
    showPolicyAvailability(false, true);
    $('missing-detail').textContent = 'Loading this environment’s saved actor…';
    const request = policyLoader.load(id);
    renderEnvironmentPicker();
    const loaded = await request;
    // A slower previous request must never replace the user's newer selection.
    if (version !== environmentVersion) return;
    original = loaded;
  }
  renderEnvironmentPicker();
  showPolicyAvailability(!!original);
  if (!original) {
    undoStack = []; redoStack = [];
    $('missing-detail').textContent = loadErrors.has(id) ? `Could not load this actor: ${loadErrors.get(id)}` : 'The trained model is not in this local app yet. Load its exported actor JSON to run the simulation and edit its curves.';
    return;
  }
  if (env.session) {
    const labels = original.observationLabels || env.inputNames;
    env.inputNames = labels.map(cleanLabel); env.observationNames = env.inputNames;
    env.metricCaption = env.inputNames[0];
    env.metricName = env.inputNames[0].toUpperCase();
    env.dt = original.environmentMetadata?.simulation_dt || env.dt;
    env.actionLabels = (original.actionLabels || env.actions || original.head.weight.map((_, i) => `Actuator ${i+1}`)).map(cleanLabel);
    inputNames = env.inputNames;
  }
  const demo = original.demo;
  $('saved-result').hidden = !demo?.status;
  $('saved-status').textContent = demo?.status || '';
  $('saved-score').textContent = Number.isFinite(demo?.score) ? `${demo.score.toFixed(2)} return · ${demo.score_label}` : '';
  $('saved-detail').textContent = demo?.detail || '';
  $('saved-result').dataset.status = demo?.status?.includes('met') ? 'met' : 'research';
  const saved = sessions.get(id);
  model = saved?.model ?? clone(original); undoStack = saved?.undoStack ?? []; redoStack = saved?.redoStack ?? [];
  selected = saved?.selected ?? {l: 0, i: Math.min(id === 'pendulum' || env.remote ? 1 : 2, model.layers[0].inputs - 1), o: 0};
  point = saved?.point ?? Math.min(id === 'pendulum' ? 2 : 3, selectedLayer().nBases - 1);
  resetSeed = saved?.seed ?? 42; $('seed').value = String(resetSeed);
  edges = allEdges(model); episode = 1;
  $('architecture').innerHTML = model.architecture.join(' <span>→</span> ');
  inputPage = model.architecture[0] > 16 && selected.l === 0 ? Math.floor(selected.i / INPUT_PAGE_SIZE) : 0;
  $('network-intro').innerHTML = `${edges.length} learned edge curves.<br>Click a connection to edit it.`;
  $('network').setAttribute('aria-label', `Interactive ${model.architecture.join(' to ')} actor network`);
  $('network-key').innerHTML = model.architecture.map((_, l) => `<span>${l === 0 ? 'State' : l === model.layers.length ? 'Latent' : 'Hidden ' + l}</span>`).join('');
  $('parameter-count').textContent = `${model.actorParameters} parameters`;
  const normalized = model.observationNormalization || model.observationScale?.some(x => x !== 1);
  $('model-description').textContent = `${model.layers.length} KAN ${model.layers.length === 1 ? 'layer feeds' : 'layers feed'} a fixed action head.${normalized ? ' Inputs use the saved observation normalization; curves show the processed inputs.' : ''} ${model.actionType === 'continuous' ? model.actionVector ? `${model.head.bias.length} actuator ${model.head.bias.length === 1 ? 'control uses' : 'controls use'} the saved action bounds.` : `The ${env.controlName} is limited to ±${model.torqueLimit}${env.controlUnit ? ' ' + env.controlUnit : ''}.` : 'The highest logit selects the action.'}`;
  inputNames = env.inputNames.map((name, i) => {
    if (id === 'pendulum') return name;
    return name + (model.observationNormalization ? ' (normalized)' : '') + (model.observationScale?.[i] && model.observationScale[i] !== 1 ? ` / ${model.observationScale[i]}` : '');
  });
  units = env.units.map((unit, i) => id !== 'pendulum' && (model.observationNormalization || (model.observationScale?.[i] ?? 1) !== 1) ? 'normalized' : unit);
  renderInputPage();
  $('point').innerHTML = Array.from({length: selectedLayer().nBases}, (_, k) => `<option value="${k}">P${k + 1}</option>`).join('');
  $('curve-editor').setAttribute('aria-label', `Spline curve editor with ${selectedLayer().nBases} draggable control points`);
  $('edge-layers').innerHTML = model.layers.map((layer, l) => `<div class="layer-heading"><span>${String(l + 1).padStart(2, '0')}</span> ${layerLabel(l)} <small>${layer.inputs * layer.outputs} edges</small></div><div class="edge-grid ${layer.inputs * layer.outputs < 4 ? 'compact' : ''}" id="layer-${l}"></div>`).join('');
  $('checkpoint-hash').textContent = 'Checkpoint ' + original.sha256.slice(0, 12);
  $('simulation').setAttribute('aria-label', `Live ${env.name} simulation`);
  $('scene').setAttribute('aria-label', `${env.name} animation showing the edited policy and original policy ghost`);
  $('history-title').textContent = env.historyTitle;
  $('angle-limit').textContent = env.targetLabel;
  $('angle-caption').textContent = env.metricCaption;
  $('angle-label').textContent = env.metricName;
  $('episode-duration').textContent = `${env.limit} steps / episode`;
  $('history').setAttribute('aria-label', `${env.metricCaption}, over time for the live and original policies`);
  const continuous = model.actionType === 'continuous', multi = model.actionVector || env.actions?.length > 2;
  $('action-track').classList.toggle('torque', continuous);
  $('binary-labels').hidden = multi; $('action-track').hidden = multi; $('multi-actions').hidden = !multi;
  $('multi-actions').innerHTML = multi ? (model.actionVector ? env.actionLabels : env.actions).map((name, i) => `<div class="action-row" id="action-row-${i}"><span>${name}</span><div class="action-meter"><div id="action-fill-${i}"></div><i id="action-baseline-${i}" title="Original probability at the live state"></i></div><b id="action-prob-${i}"></b></div>`).join('') : '';
  $('action-left-label').textContent = continuous ? `Live ${env.controlName}` : '← Push left';
  $('action-right-label').textContent = continuous ? `Original ${env.controlName}` : 'Push right →';
  $('action-description').textContent = model.actionVector ? 'Live actuator controls. Amber markers show the original policy at this same state; each row uses its saved action bounds.' : continuous
    ? `Control at the live state. Center is 0; range is −${model.torqueLimit} to +${model.torqueLimit}${env.controlUnit ? ' ' + env.controlUnit : ''}. The amber marker is the original policy.`
    : 'Probabilities at the live state. The largest selects the action. Amber markers show the original policy at this same state.';
  $('baseline-prob').title = `Original policy ${continuous ? env.controlName : 'probability'} at the current live state`;
  $('last-result').textContent = env.description;
  $('evaluate').textContent = 'Compare 20 episodes ↗';
  resetEpisode(); fitView(); changed(); setRunning(false);
  if (!env.session) connectRuntime(version);
}

async function fingerprint(contents) {
  const bytes = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(contents));
  return Array.from(new Uint8Array(bytes), b => b.toString(16).padStart(2, '0')).join('');
}

async function readPolicy(definition) {
  let contents;
  if ($('embedded-models')) {
    const entry = $('embedded-model-' + definition.id);
    if (!entry) return null;
    contents = entry.textContent;
  } else {
    const response = await fetch('./' + definition.file);
    if (response.status === 404) return null;
    if (!response.ok) throw new Error(`Saved actor request failed (${response.status}).`);
    contents = await response.text();
  }
  const raw = JSON.parse(contents);
  if (raw.format === 'inkan-research-actor-v1' && !raw.checkpoint_sha256) raw.checkpoint_sha256 = await fingerprint(contents);
  return raw;
}

function init() {
  for (const name of ['policy-content', 'network-content', 'editor-content']) $(name).inert = true;
  renderEnvironmentPicker(); wireUI();
  $('retry-runtime').onclick = () => env.session ? resetEpisode() : connectRuntime(environmentVersion);
  window.addEventListener('hashchange', () => {
    const id = location.hash.slice(1);
    if (Object.prototype.hasOwnProperty.call(environments, id)) switchEnvironment(id);
  });
  const requested = location.hash.slice(1);
  switchEnvironment(Object.prototype.hasOwnProperty.call(environments, requested) ? requested : 'cartpole');
  requestAnimationFrame(frame);
}
init();
