"""Export only the deterministic InKAN actor; omit SAC critics and exploration head."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


class NumpyKANPolicy:
    """Standalone deterministic Pendulum inference from JSON (NumPy only)."""
    def __init__(self, path):
        self.data = json.loads(Path(path).read_text())
        if self.data['format'] != 'inkan-pendulum-actor-v1':
            raise ValueError('Unsupported actor format')

    def predict(self, observations, deterministic=True):
        x = np.asarray(observations, dtype=np.float64)
        single = x.ndim == 1
        x = np.atleast_2d(x) / np.array(self.data['observation_scale'])
        for layer in self.data['layers']:
            coeff = np.asarray(layer['coefficients'])
            base = np.asarray(layer['base_weights'])
            u = (x[:, :, None] - layer['grid_start']) * layer['inv_h'] - np.arange(coeff.shape[-1])
            s = np.maximum(0, np.where(u <= 2, u, 4-u))
            v = np.maximum(0, s-1)
            bases = np.where(s <= 1, s**3, 1+v*(3+v*(3-3*v)))/6
            residual = x / (1 + np.exp(np.clip(-x, -700, 700)))
            x = (bases[:, None, :, :] * coeff[None, :, :, :]).sum(axis=(2, 3)) + (residual[:, None, :] * base[None, :, :]).sum(axis=2)
        weight = np.asarray(self.data['mean_head']['weight'])
        bias = np.asarray(self.data['mean_head']['bias'])
        mean = (x[:, None, :] * weight[None, :, :]).sum(axis=2) + bias
        action = self.data['torque_limit'] * np.tanh(mean)
        return (action[0] if single else action), None


def export(checkpoint, output):
    import torch
    from stable_baselines3 import SAC
    from .pendulum_policy import parameter_counts
    torch.set_num_threads(1)
    model = SAC.load(checkpoint, device='cpu')
    data = {'format': 'inkan-pendulum-actor-v1', 'environment': 'Pendulum-v1',
            'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            'architecture': [3] + list(model.actor.net_arch),
            'grid_size': model.actor.grid_size,
            'observation_scale': [1., 1., 8.], 'torque_limit': 2.,
            'parameters': parameter_counts(model), 'layers': [],
            'mean_head': {'weight': model.actor.mu.weight.detach().tolist(), 'bias': model.actor.mu.bias.detach().tolist()}}
    for layer in model.actor.latent_pi:
        data['layers'].append({'coefficients': layer.spline_weight.detach().tolist(),
                               'base_weights': layer.base_weight.detach().tolist(),
                               'grid_start': layer.grid_starts[0].item(), 'inv_h': layer.inv_h})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(data, indent=2))
    exported = NumpyKANPolicy(output)
    rng = np.random.default_rng(713)
    angles = rng.uniform(-np.pi, np.pi, 10_000)
    obs = np.column_stack([np.cos(angles), np.sin(angles), rng.uniform(-8, 8, 10_000)]).astype(np.float32)
    expected, _ = model.predict(obs, deterministic=True)
    actual, _ = exported.predict(obs)
    error = float(np.max(np.abs(expected - actual)))
    if not np.isfinite(actual).all() or error > 1e-4:
        raise AssertionError(f'Export differs from PyTorch: max error {error}')
    report = {'observations_checked': len(obs), 'maximum_action_error': error,
              'actor_parameters': data['parameters']['actor_deterministic']}
    Path(str(output) + '.validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
