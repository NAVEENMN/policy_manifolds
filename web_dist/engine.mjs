// The exported actor and Gymnasium CartPole-v1 dynamics. No external runtime.
export const clone = value => JSON.parse(JSON.stringify(value));
export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
export const silu = x => x / (1 + Math.exp(-x));

export function basis(layer, x) {
  const t = (x - layer.gridStart) * layer.invH;
  return Array.from({length: layer.nBases}, (_, k) => {
    const u = t - k;
    const s = Math.max(0, u <= 2 ? u : 4 - u);
    const v = Math.max(0, s - 1);
    return (s <= 1 ? s * s * s : 1 + v * (3 + v * (3 - 3 * v))) / 6;
  });
}

export function edgeValue(layer, output, input, x) {
  return basis(layer, x).reduce((sum, b, k) => sum + b * layer.coefficients[output][input][k], 0)
    + layer.baseWeights[output][input] * silu(x);
}

export function infer(model, observation) {
  // Gym emits float32 observations; retain that conversion in browser inference.
  let values = observation.map((x, i) => {
    let value = model.observationDtype === 'float64' ? x : Math.fround(x);
    const norm = model.observationNormalization;
    if (norm) value = Math.fround(clamp((value - norm.mean[i]) / Math.sqrt(norm.var[i] + norm.epsilon), -norm.clip, norm.clip));
    return Math.fround(value / (model.observationScale?.[i] ?? 1));
  });
  const activations = [values];
  const contributions = [];
  for (const layer of model.layers) {
    const edges = Array.from({length: layer.outputs}, (_, o) =>
      values.map((x, i) => edgeValue(layer, o, i, x)));
    values = edges.map(row => row.reduce((a, b) => a + b, 0));
    contributions.push(edges);
    activations.push(values);
  }
  const logits = model.head.weight.map((row, o) =>
    row.reduce((s, w, i) => s + w * values[i], model.head.bias[o]));
  if (model.actionType === 'continuous') {
    const actions = logits.map((value, i) => {
      const low = model.actionLow?.[i] ?? -model.torqueLimit, high = model.actionHigh?.[i] ?? model.torqueLimit;
      return low + (Math.tanh(value) + 1) * .5 * (high - low);
    });
    return {action: model.actionVector ? actions : actions[0], logits, activations, contributions};
  }
  const max = Math.max(...logits), exp = logits.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0), probabilities = exp.map(x => x / sum);
  return {action: logits.indexOf(max), logits, probabilities, pRight: probabilities[1], activations, contributions};
}

export function initialState(seed) {
  // App-specific reproducible reset RNG; not NumPy's PCG64 seed sequence.
  let a = seed >>> 0;
  return Array.from({length: 4}, () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return (((t ^ t >>> 14) >>> 0) / 4294967296) * 0.1 - 0.05;
  });
}

export function physics(state, action) {
  const [x, velocity, angle, omega] = state;
  const cos = Math.cos(angle), sin = Math.sin(angle);
  const force = action === 1 ? 10 : -10;
  const temp = (force + 0.05 * omega * omega * sin) / 1.1;
  const thetaAcc = (9.8 * sin - cos * temp) / (0.5 * (4 / 3 - 0.1 * cos * cos / 1.1));
  const xAcc = temp - 0.05 * thetaAcc * cos / 1.1;
  return [x + 0.02 * velocity, velocity + 0.02 * xAcc, angle + 0.02 * omega, omega + 0.02 * thetaAcc];
}

export function terminal(state) {
  return Math.abs(state[0]) > 2.4 || Math.abs(state[2]) > 12 * Math.PI / 180;
}

export function rollout(model, start, limit = 500) {
  let state = start.slice(), steps = 0;
  while (steps < limit && (steps === 0 || !terminal(state))) {
    state = physics(state, infer(model, state).action);
    steps++;
  }
  return {steps, state, failed: terminal(state)};
}

export function allEdges(model) {
  return model.layers.flatMap((layer, l) => Array.from({length: layer.inputs}, (_, i) =>
    Array.from({length: layer.outputs}, (_, o) => ({l, i, o, id: `${l}-${i}-${o}`}))).flat());
}

export function validateSession(data, original) {
  const compatible = data?.format === 'cartpole-curve-session-v1' && original.format === 'cartpole-kan-policy-v1'
    || data?.format === 'policy-curve-session-v1' && data.environment === (original.environment || 'CartPole-v1');
  if (!compatible || data.checkpointSha256 !== original.sha256)
    throw new Error('Choose a session exported from this checkpoint.');
  const model = clone(original);
  if (!Array.isArray(data.layers) || data.layers.length !== model.layers.length) throw new Error('Invalid layer count.');
  const number = x => typeof x === 'number' && Number.isFinite(x) && Math.abs(x) <= 50;
  for (let l = 0; l < model.layers.length; l++) {
    const target = model.layers[l], source = data.layers[l];
    if (!source || !Array.isArray(source.coefficients) || !Array.isArray(source.baseWeights)
      || source.coefficients.length !== target.outputs || source.baseWeights.length !== target.outputs)
      throw new Error('Invalid policy dimensions.');
    for (let o = 0; o < target.outputs; o++) {
      if (!Array.isArray(source.coefficients[o]) || source.coefficients[o].length !== target.inputs
        || !Array.isArray(source.baseWeights[o]) || source.baseWeights[o].length !== target.inputs)
        throw new Error('Invalid edge dimensions.');
      for (let i = 0; i < target.inputs; i++) {
        const c = source.coefficients[o][i];
        if (!Array.isArray(c) || c.length !== target.nBases || !c.every(number) || !number(source.baseWeights[o][i]))
          throw new Error('Curve values must be finite numbers between −50 and 50.');
        target.coefficients[o][i] = c.slice();
        target.baseWeights[o][i] = source.baseWeights[o][i];
      }
    }
  }
  return model;
}
