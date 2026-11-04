"""Deterministic Gymnasium evaluation on explicit reset seeds."""
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
from .pendulum_export import NumpyKANPolicy


class DiscreteActor:
    def __init__(self, data, environment):
        self.data = data
        self.continuous = data.get('action_mapping') == 'tanh_then_box_rescale'
        self.low = np.asarray(data.get('action_low', []))
        self.high = np.asarray(data.get('action_high', []))
        if data.get('format') == 'cartpole-kan-policy-v1':
            if environment != 'CartPole-v1':
                raise ValueError('CartPole actor supplied for a different environment.')
            self.layers = data['layers']; self.head = data['head']
            self.scale = data.get('observationScale', [1.] * 4)
            self.normalization = data.get('observationNormalization')
        else:
            from .classic_export import validate_actor
            validate_actor(data, environment)
            self.layers = [{**l, 'baseWeights': l['base_weights'], 'gridStart': l['grid_start'],
                            'invH': l['inv_h']} for l in data['layers']]
            self.head = data['head']
            self.scale = data.get('observation_scale') or [1.] * len(self.layers[0]['coefficients'][0])
            self.normalization = data.get('observation_normalization')

    def predict(self, observations, deterministic=True):
        x = np.asarray(observations, dtype=np.float64 if self.continuous else np.float32)
        single = x.ndim == 1; x = np.atleast_2d(x)
        if self.normalization:
            n = self.normalization
            x = np.clip((x - np.array(n['mean'])) / np.sqrt(np.array(n['var']) + n['epsilon']), -n['clip'], n['clip']).astype(np.float32)
        x = (x.astype(np.float32) / np.asarray(self.scale, dtype=np.float32)).astype(np.float64)
        for layer in self.layers:
            coeff, base = np.array(layer['coefficients']), np.array(layer['baseWeights'])
            u = (x[:, :, None] - layer['gridStart']) * layer['invH'] - np.arange(coeff.shape[-1])
            s = np.maximum(0, np.minimum(u, 4-u)); v = np.maximum(0, s-1)
            basis = np.where(s <= 1, s**3, 1 + v*(3 + v*(3-3*v))) / 6
            residual = x / (1 + np.exp(np.clip(-x, -700, 700)))
            x = np.einsum('bik,oik->bo', basis, coeff) + np.einsum('bi,oi->bo', residual, base)
        logits = np.einsum('bi,oi->bo', x, np.array(self.head['weight'])) + np.array(self.head['bias'])
        action = self.low + (np.tanh(logits) + 1) * .5 * (self.high - self.low) if self.continuous else np.argmax(logits, axis=1)
        return (action[0] if single else action), None


def load_actor(path, environment):
    path = Path(path); data = json.loads(path.read_text())
    if data.get('format') == 'inkan-pairwise-actor-v1':
        from kan_search.exported_actor import PairwiseActor
        return PairwiseActor(data, environment)
    if environment == 'Pendulum-v1':
        if data.get('environment') != environment:
            raise ValueError('Actor environment does not match Pendulum-v1.')
        return NumpyKANPolicy(path)
    return DiscreteActor(data, environment)


def evaluate(policy, environment, episodes=20, seed=100_000):
    if episodes < 1 or seed < 0:
        raise ValueError('Episodes must be positive and seed nonnegative.')
    if environment == 'InvertedPendulum-v5':
        from .native_inverted import HeadlessInvertedPendulum
        env = HeadlessInvertedPendulum()
    else:
        env = gym.make(environment)
    rows = []
    try:
        for reset_seed in range(seed, seed + episodes):
            obs, _ = env.reset(seed=reset_seed); total = 0.; steps = 0; stable = 0
            while True:
                action, _ = policy.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward); steps += 1
                if environment == 'Pendulum-v1' and steps > 150:
                    stable += int(abs(np.arctan2(obs[1], obs[0])) < np.pi/12 and abs(obs[2]) < 1)
                if terminated or truncated:
                    break
            from .paths import SIMULATIONS
            score_targets = {s[0]: s[5] for s in SIMULATIONS.values()}
            success = (total >= score_targets[environment] if environment in score_targets else stable >= 45 if environment == 'Pendulum-v1' else not terminated if environment in ('CartPole-v1', 'InvertedPendulum-v5') else bool(terminated))
            rows.append({'seed': reset_seed, 'return': total, 'steps': steps, 'success': success})
    finally:
        env.close()
    returns = [row['return'] for row in rows]
    return {'environment': environment, 'episodes': episodes, 'seed_start': seed,
            'mean_return': float(np.mean(returns)), 'std_return': float(np.std(returns)),
            'worst_return': min(returns), 'best_return': max(returns),
            'success_rate': float(np.mean([r['success'] for r in rows])), 'episode_results': rows}
