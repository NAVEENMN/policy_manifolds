import {clamp, initialState, physics, terminal, infer} from './engine.mjs';

export const angleNormalize = angle => ((angle + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;

// Catalog entries without a browser environment remain visible but disabled.
export const environmentCatalog = [
  {group: 'Classic Control', entries: [
    ['cartpole', 'CartPole-v1', 'Balance a pole', '⇆'],
    ['pendulum', 'Pendulum-v1', 'Swing up and balance', '◴'],
    ['acrobot', 'Acrobot-v1', 'Swing a double pendulum', '⌁'],
    ['mountaincar', 'MountainCar-v0', 'Build momentum to climb', '⌃'],
    ['mountaincar-continuous', 'MountainCarContinuous-v0'],
  ]},
  {group: 'Box2D', entries: [
    ['lunarlander', 'LunarLander-v3'], ['lunarlander-continuous', 'LunarLanderContinuous-v3'],
    ['bipedalwalker', 'BipedalWalker-v3'], ['carracing', 'CarRacing-v3'],
  ]},
  {group: 'MuJoCo', entries: [
    ['halfcheetah', 'HalfCheetah-v5'], ['hopper', 'Hopper-v5'], ['walker2d', 'Walker2d-v5'],
    ['ant', 'Ant-v5'], ['humanoid', 'Humanoid-v5'], ['swimmer', 'Swimmer-v5'],
    ['reacher', 'Reacher-v5'], ['invertedpendulum', 'InvertedPendulum-v5'],
    ['inverteddoublependulum', 'InvertedDoublePendulum-v5'],
  ]},
];

export function mountainCarStep([position, velocity], action) {
  velocity = clamp(velocity + (action - 1) * .001 - .0025 * Math.cos(3 * position), -.07, .07);
  position = clamp(position + velocity, -1.2, .6);
  if (position <= -1.2 && velocity < 0) velocity = 0;
  return {state: [position, velocity], reward: -1};
}

export const acrobotHeight = ([a, b]) => -Math.cos(a) - Math.cos(a + b);
export function acrobotStep(state, action) {
  // Gymnasium's default "book" dynamics, integrated with RK4 at dt = 0.2.
  const derivative = ([a, b, u, v]) => {
    const d1 = 3.5 + Math.cos(b), d2 = 1.25 + .5 * Math.cos(b);
    const phi2 = 4.9 * Math.cos(a + b - Math.PI / 2);
    const phi1 = -.5 * v * v * Math.sin(b) - v * u * Math.sin(b)
      + 14.7 * Math.cos(a - Math.PI / 2) + phi2;
    const dd2 = (action - 1 + d2 / d1 * phi1 - .5 * u * u * Math.sin(b) - phi2) / (1.25 - d2 * d2 / d1);
    return [u, v, -(d2 * dd2 + phi1) / d1, dd2];
  };
  const add = (k, scale) => state.map((v, i) => v + scale * k[i]);
  const k1 = derivative(state), k2 = derivative(add(k1, .1));
  const k3 = derivative(add(k2, .1)), k4 = derivative(add(k3, .2));
  const next = state.map((v, i) => v + .2 / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
  for (let i = 0; i < 2; i++) {
    while (next[i] > Math.PI) next[i] -= 2 * Math.PI;
    while (next[i] < -Math.PI) next[i] += 2 * Math.PI;
  }
  next[2] = clamp(next[2], -4 * Math.PI, 4 * Math.PI);
  next[3] = clamp(next[3], -9 * Math.PI, 9 * Math.PI);
  return {state: next, reward: acrobotHeight(next) > 1 ? 0 : -1};
}

export function discreteModel(raw, environment, inputs, actions = 3, continuous = false) {
  if (raw.format !== 'inkan-research-actor-v1' || raw.environment !== environment
    || raw.action_mapping !== (continuous ? 'tanh_then_box_rescale' : 'argmax') || raw.image_encoder_required)
    throw new Error(`Choose an exported InKAN actor for ${environment}.`);
  const finite = x => typeof x === 'number' && Number.isFinite(x);
  const vector = (x, n) => Array.isArray(x) && x.length === n && x.every(finite);
  if (!Array.isArray(raw.layers) || raw.layers.length < 1 || raw.layers.length > 4) throw new Error('Invalid KAN layers.');
  let width = inputs, count = 0;
  const layers = raw.layers.map(l => {
    const outputs = l.coefficients?.length, nBases = l.coefficients?.[0]?.[0]?.length;
    if (!Number.isInteger(outputs) || outputs < 1 || outputs > 64 || !Number.isInteger(nBases) || nBases < 4 || nBases > 64
      || !finite(l.grid_start) || !finite(l.inv_h) || l.inv_h <= 0
      || !l.coefficients.every(row => Array.isArray(row) && row.length === width && row.every(c => vector(c, nBases)))
      || !Array.isArray(l.base_weights) || l.base_weights.length !== outputs || !l.base_weights.every(row => vector(row, width)))
      throw new Error('Invalid spline dimensions or values.');
    const layer = {inputs: width, outputs, nBases, coefficients: l.coefficients, baseWeights: l.base_weights, gridStart: l.grid_start, invH: l.inv_h};
    count += outputs * width * (nBases + 1); width = outputs;
    return layer;
  });
  if (!Array.isArray(raw.head?.weight) || raw.head.weight.length !== actions || !raw.head.weight.every(row => vector(row, width)) || !vector(raw.head.bias, actions))
    throw new Error(`Expected a fixed head with ${actions} actions.`);
  const scale = raw.observation_scale ?? Array(inputs).fill(1), norm = raw.observation_normalization;
  if (!vector(scale, inputs) || scale.some(v => v <= 0)) throw new Error('Invalid observation scale.');
  if (norm && (!vector(norm.mean, inputs) || !vector(norm.var, inputs) || norm.var.some(v => v < 0)
    || !finite(norm.epsilon) || norm.epsilon <= 0 || !finite(norm.clip) || norm.clip <= 0)) throw new Error('Invalid observation normalization.');
  if (!/^[a-f0-9]{64}$/.test(raw.checkpoint_sha256 || '')) throw new Error('Missing model fingerprint. Export with app.py or load the actor JSON.');
  return {format: raw.format, environment, sha256: raw.checkpoint_sha256, actionType: 'discrete',
    architecture: [inputs, ...layers.map(l => l.outputs)], actorParameters: count + actions * (width + 1),
    observationScale: scale, observationNormalization: norm, layers, head: raw.head,
    demo: raw.demo, observationLabels: raw.observation_labels, actionLabels: raw.action_labels,
    environmentMetadata: raw.environment_metadata};
}

export function invertedPendulumModel(raw) {
  if (JSON.stringify(raw.action_low) !== '[-3]' || JSON.stringify(raw.action_high) !== '[3]')
    throw new Error('InvertedPendulum requires the saved −3 to +3 action bounds.');
  return {...discreteModel(raw, 'InvertedPendulum-v5', 4, 1, true),
    actionType: 'continuous', torqueLimit: 3, observationDtype: 'float64'};
}

export function pendulumStep(state, action) {
  const [theta, omega] = state, torque = clamp(action, -2, 2);
  const reward = -(angleNormalize(theta) ** 2 + .1 * omega ** 2 + .001 * torque ** 2);
  const nextOmega = clamp(omega + (15 * Math.sin(theta) + 3 * torque) * .05, -8, 8);
  return {state: [theta + nextOmega * .05, nextOmega], reward};
}

export function pendulumModel(raw) {
  if (raw.format !== 'inkan-pendulum-actor-v1') throw new Error('Unsupported Pendulum checkpoint');
  return {
    format: 'inkan-pendulum-actor-v1', environment: raw.environment,
    sha256: raw.checkpoint_sha256, architecture: raw.architecture,
    actorParameters: raw.parameters.actor_deterministic,
    observationScale: raw.observation_scale, actionType: 'continuous', torqueLimit: raw.torque_limit,
    layers: raw.layers.map(l => ({inputs: l.coefficients[0].length, outputs: l.coefficients.length,
      nBases: l.coefficients[0][0].length, coefficients: l.coefficients, baseWeights: l.base_weights,
      gridStart: l.grid_start, invH: l.inv_h})),
    head: raw.mean_head,
    demo: raw.demo,
  };
}

export const environments = {
  cartpole: {
    id: 'cartpole', name: 'CartPole-v1', algorithm: 'PPO', file: 'model.json', dt: .02, limit: 500,
    inputShort: ['x', 'v', 'θ', 'ω'], inputNames: ['Cart position', 'Cart velocity', 'Pole angle', 'Angular velocity'],
    units: ['m', 'm/s', 'rad', 'rad/s'], observationNames: ['Cart position', 'Cart velocity', 'Pole angle', 'Angular velocity'],
    observationLimits: [2.4, 3, .20944, 3], initial: initialState, observe: state => state,
    step: (state, action) => ({state: physics(state, action), reward: 1}), terminal,
    angle: state => state[2], stable: () => false,
    metric: state => state[2] * 180 / Math.PI, metricName: 'POLE ANGLE', metricUnit: '°',
    historyTitle: 'Balance over time', metricCaption: 'Pole angle · degrees', targetLabel: '±12° limit',
    plotRange: [-16, 16], plotTicks: [-12, 0, 12], actions: ['Push left', 'Push right'],
    decode: raw => { if (raw.format !== 'cartpole-kan-policy-v1') throw new Error('Unsupported CartPole checkpoint'); return raw; },
    description: 'Balance the pole with left or right pushes.',
  },
  pendulum: {
    id: 'pendulum', name: 'Pendulum-v1', algorithm: 'SAC', file: 'pendulum.json', dt: .05, limit: 200,
    controlName: 'torque', controlUnit: 'N·m',
    inputShort: ['cos θ', 'sin θ', 'ω/8'], inputNames: ['Cosine of angle', 'Sine of angle', 'Angular velocity / 8'],
    units: ['unitless', 'unitless', 'normalized'], observationNames: ['Cosine of angle', 'Sine of angle', 'Angular velocity'],
    observationLimits: [1, 1, 8],
    initial: seed => { const u = initialState(seed); return [u[0] * 20 * Math.PI, u[1] * 20]; },
    observe: ([theta, omega]) => [Math.cos(theta), Math.sin(theta), omega],
    step: pendulumStep, terminal: () => false,
    angle: state => angleNormalize(state[0]),
    stable: state => Math.abs(angleNormalize(state[0])) < Math.PI / 12 && Math.abs(state[1]) < 1,
    metric: state => angleNormalize(state[0]) * 180 / Math.PI, metricName: 'UPRIGHT ANGLE', metricUnit: '°',
    historyTitle: 'Swing up and settle', metricCaption: 'Angle from upright · degrees', targetLabel: '±15° target',
    plotRange: [-190, 190], plotTicks: [-180, 0, 180],
    decode: pendulumModel,
    description: 'Swing up and hold upright with continuous torque.',
  },
  acrobot: {
    id: 'acrobot', name: 'Acrobot-v1', algorithm: 'PPO', file: 'acrobot.json', dt: .2, limit: 500,
    inputShort: ['cos θ₁', 'sin θ₁', 'cos θ₂', 'sin θ₂', 'ω₁', 'ω₂'],
    inputNames: ['Cosine of joint 1', 'Sine of joint 1', 'Cosine of joint 2', 'Sine of joint 2', 'Joint 1 velocity', 'Joint 2 velocity'],
    units: ['unitless', 'unitless', 'unitless', 'unitless', 'rad/s', 'rad/s'],
    observationNames: ['cos θ₁', 'sin θ₁', 'cos θ₂', 'sin θ₂', 'Joint 1 velocity', 'Joint 2 velocity'],
    observationLimits: [1, 1, 1, 1, 4 * Math.PI, 9 * Math.PI],
    initial: seed => initialState(seed).map(x => Math.fround(x * 2)),
    observe: ([a, b, u, v]) => [Math.cos(a), Math.sin(a), Math.cos(b), Math.sin(b), u, v],
    step: acrobotStep, terminal: state => acrobotHeight(state) > 1, goalTask: true, stable: () => false,
    metric: acrobotHeight, metricName: 'TIP HEIGHT', metricUnit: ' m',
    historyTitle: 'Swing toward the target', metricCaption: 'Tip height above pivot · meters', targetLabel: '>1 m goal',
    plotRange: [-2.2, 2.2], plotTicks: [-2, 0, 1, 2], actions: ['Torque −1', 'No torque', 'Torque +1'],
    decode: raw => discreteModel(raw, 'Acrobot-v1', 6),
    description: 'Swing the tip above the goal line. Fewer steps give a higher return.',
  },
  mountaincar: {
    id: 'mountaincar', name: 'MountainCar-v0', algorithm: 'PPO', file: 'mountaincar.json', dt: 1 / 30, limit: 200,
    inputShort: ['x', 'v'], inputNames: ['Position', 'Velocity'], units: ['m', 'm/step'],
    observationNames: ['Position', 'Velocity'], observationLimits: [1.2, .07],
    initial: seed => [-.5 + initialState(seed)[0] * 2, 0], observe: state => state,
    step: mountainCarStep, terminal: ([x, v]) => x >= .5 && v >= 0, goalTask: true, stable: () => false,
    metric: state => state[0], metricName: 'POSITION', metricUnit: ' m',
    historyTitle: 'Build momentum and climb', metricCaption: 'Car position · meters', targetLabel: '0.5 m goal',
    plotRange: [-1.3, .7], plotTicks: [-1.2, 0, .5], actions: ['Push left', 'Coast', 'Push right'],
    decode: raw => discreteModel(raw, 'MountainCar-v0', 2),
    description: 'Reach the flag in 200 steps. Fewer steps give a higher return.',
  },
  invertedpendulum: {
    id: 'invertedpendulum', name: 'InvertedPendulum-v5', algorithm: 'SAC', file: 'invertedpendulum.json',
    dt: .04, limit: 1000, remote: true, controlName: 'cart control', controlUnit: '',
    inputShort: ['x', 'θ', 'v', 'ω'], inputNames: ['Cart position', 'Pole angle', 'Cart velocity', 'Pole angular velocity'],
    units: ['m', 'rad', 'm/s', 'rad/s'], observationNames: ['Cart position', 'Pole angle', 'Cart velocity', 'Pole angular velocity'],
    observationLimits: [1, .2, 3, 3], initial: seed => initialState(seed).map(value => value * .2), observe: state => state,
    terminal: state => state.some(value => !Number.isFinite(value)) || Math.abs(state[1]) > .2,
    angle: state => state[1], stable: () => false,
    metric: state => state[1] * 180 / Math.PI, metricName: 'POLE ANGLE', metricUnit: '°',
    historyTitle: 'Balance over time', metricCaption: 'Pole angle · degrees', targetLabel: '±0.2 rad limit',
    plotRange: [-16, 16], plotTicks: [-.2 * 180 / Math.PI, 0, .2 * 180 / Math.PI],
    decode: invertedPendulumModel,
    description: 'Balance for 1,000 steps with continuous cart control and MuJoCo physics.',
  },
};

// Server sessions preserve MuJoCo/Box2D state while editable inference stays here.
export const simulationDefinitions = [
  ['lunarlander', 'LunarLander-v3', 8, 4, 'PPO', 1000, .02],
  ['halfcheetah', 'HalfCheetah-v5', 17, 6, 'SAC', 1000, .05],
  ['hopper', 'Hopper-v5', 11, 3, 'SAC', 1000, .008],
  ['walker2d', 'Walker2d-v5', 17, 6, 'SAC', 1000, .008],
  ['ant', 'Ant-v5', 105, 8, 'SAC', 1000, .05],
  ['humanoid', 'Humanoid-v5', 348, 17, 'SAC', 1000, .015],
  ['swimmer', 'Swimmer-v5', 8, 2, 'SAC', 1000, .04],
  ['reacher', 'Reacher-v5', 10, 2, 'SAC', 50, .02],
  ['inverteddoublependulum', 'InvertedDoublePendulum-v5', 9, 1, 'SAC', 1000, .05],
];
for (const [id, name, inputs, actions, algorithm, limit, dt] of simulationDefinitions) {
  const labels = Array.from({length: inputs}, (_, i) => `Observation ${i + 1}`);
  environments[id] = {
    id, name, algorithm, file: id + '.json', dt, limit, remote: true, session: true,
    physics: id === 'lunarlander' ? 'Box2D' : 'MuJoCo', controlName: 'actuator controls', controlUnit: '',
    inputShort: labels.map((_, i) => `x${i + 1}`), inputNames: labels, observationNames: labels,
    units: labels.map(() => 'raw'), observationLimits: labels.map(() => 5),
    initial: () => Array(inputs).fill(0), observe: state => state,
    terminal: () => false, stable: () => false, metric: state => state[0], metricName: 'OBSERVATION 1', metricUnit: '',
    historyTitle: 'Policy response over time', metricCaption: 'First observed feature', targetLabel: 'Live / original',
    plotRange: [-2, 2], plotTicks: [-2, 0, 2],
    actions: algorithm === 'PPO' ? ['No engine', 'Left engine', 'Main engine', 'Right engine'] : undefined,
    decode: raw => {
      const continuous = algorithm === 'SAC';
      const decoded = discreteModel(raw, name, inputs, actions, continuous);
      if (!continuous) return decoded;
      if (!Array.isArray(raw.action_low) || !Array.isArray(raw.action_high)
        || raw.action_low.length !== actions || raw.action_high.length !== actions
        || raw.action_low.some((low, i) => !Number.isFinite(low) || !Number.isFinite(raw.action_high[i]) || low >= raw.action_high[i]))
        throw new Error('Invalid continuous action bounds.');
      return {...decoded, actionType: 'continuous', actionVector: true,
        actionLow: raw.action_low, actionHigh: raw.action_high, observationDtype: 'float64'};
    },
    description: 'Edit the saved policy and compare its behavior with the original. Check the saved evaluation status above.',
  };
}

export function environmentRollout(policy, env, start) {
  let state = start.slice(), reward = 0, steps = 0, stableSteps = 0;
  while (steps < env.limit) {
    const next = env.step(state, infer(policy, env.observe(state)).action);
    state = next.state; reward += next.reward; steps++;
    if (steps > env.limit - 50 && env.stable(state)) stableSteps++;
    if (env.terminal(state)) break;
  }
  return {state, reward, steps, stable: stableSteps >= 45, success: !!env.goalTask && env.terminal(state)};
}
