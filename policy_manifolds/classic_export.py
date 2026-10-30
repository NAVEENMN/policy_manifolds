"""Export discrete Classic Control InKAN actors without importing training scripts.

Raw PPO ZIPs must use the default cubic InKAN grid, SiLU residuals and raw
observations (as in train.py). For kan_search runs, supply best_actor.json:
it includes VecNormalize and scaling.
"""
import hashlib
import io
import json
import math
from pathlib import Path
import re
from zipfile import ZipFile

INPUTS = {'CartPole-v1': 4, 'Acrobot-v1': 6, 'MountainCar-v0': 2, 'InvertedPendulum-v5': 4}
ACTIONS = {'CartPole-v1': 2, 'Acrobot-v1': 3, 'MountainCar-v0': 3, 'InvertedPendulum-v5': 1}
from .paths import SIMULATIONS
INPUTS.update({s[0]: s[1] for s in SIMULATIONS.values()})
ACTIONS.update({s[0]: s[2] for s in SIMULATIONS.values()})
CONTINUOUS = {'InvertedPendulum-v5'} | {s[0] for s in SIMULATIONS.values() if s[3] == 'SAC'}


def validate_actor(data, environment):
    if (data.get('format') != 'inkan-research-actor-v1'
            or data.get('environment') != environment
            or data.get('action_mapping') != ('tanh_then_box_rescale' if environment in CONTINUOUS else 'argmax')
            or data.get('image_encoder_required')):
        raise ValueError(f'Expected an exported InKAN actor for {environment}.')
    if environment == 'InvertedPendulum-v5' and (data.get('action_low') != [-3.] or data.get('action_high') != [3.]):
        raise ValueError('InvertedPendulum actions must use the saved [-3, 3] bounds.')
    width = INPUTS[environment]
    layers = data.get('layers', [])
    if not 1 <= len(layers) <= 4:
        raise ValueError('Expected one to four KAN layers.')

    def vector(values, count):
        return (isinstance(values, list) and len(values) == count
                and all(type(v) in (int, float) and math.isfinite(v) for v in values))

    if environment in CONTINUOUS:
        low, high = data.get('action_low'), data.get('action_high')
        if not vector(low, ACTIONS[environment]) or not vector(high, ACTIONS[environment]) or any(a >= b for a, b in zip(low, high)):
            raise ValueError('Invalid continuous action bounds.')

    for layer in layers:
        coefficients, base = layer['coefficients'], layer['base_weights']
        count = len(coefficients[0][0])
        if (not 1 <= len(coefficients) <= 64 or not 4 <= count <= 64
                or len(base) != len(coefficients)
                or not all(len(row) == width and all(vector(c, count) for c in row) for row in coefficients)
                or not all(vector(row, width) for row in base)
                or not math.isfinite(layer['grid_start'])
                or not math.isfinite(layer['inv_h']) or layer['inv_h'] <= 0):
            raise ValueError('Invalid spline dimensions or values.')
        width = len(coefficients)
    head = data['head']
    if len(head['weight']) != ACTIONS[environment] or not all(vector(row, width) for row in head['weight']) or not vector(head['bias'], ACTIONS[environment]):
        raise ValueError('Linear action head does not match the environment.')
    scale = data.get('observation_scale')
    if scale is not None and (not vector(scale, INPUTS[environment]) or min(scale) <= 0):
        raise ValueError('Invalid observation scale.')
    norm = data.get('observation_normalization')
    if norm is not None and (not vector(norm['mean'], INPUTS[environment])
            or not vector(norm['var'], INPUTS[environment]) or min(norm['var']) < 0
            or not math.isfinite(norm['epsilon']) or norm['epsilon'] <= 0
            or not math.isfinite(norm['clip']) or norm['clip'] <= 0):
        raise ValueError('Invalid observation normalization.')


def export_classic(source, output, environment):
    source, output = Path(source), Path(output)
    contents = source.read_bytes()
    if source.suffix.lower() == '.json':
        data = json.loads(contents)
    elif source.suffix.lower() == '.zip':
        import torch
        with ZipFile(source) as archive:
            state = torch.load(io.BytesIO(archive.read('policy.pth')), map_location='cpu', weights_only=True)
            metadata = json.loads(archive.read('data'))
        if any('features_extractor.' in name for name in state):
            raise ValueError('This checkpoint has observation preprocessing. Supply its exported actor.json to preserve normalization.')
        if list(source.parent.glob('*vecnormalize*')) or (source.parent / 'normalization.pkl').exists():
            raise ValueError('Observation normalization found. Supply actor.json, which includes those statistics.')
        prefixes = [name.removesuffix('.grid_starts') for name in state
                    if re.fullmatch(r'mlp_extractor\.policy_net(?:\.\d+)?\.grid_starts', name)]
        prefixes.sort(key=lambda name: int(name.rsplit('.', 1)[-1]) if name.rsplit('.', 1)[-1].isdigit() else -1)
        if not prefixes:
            raise ValueError('No InKAN actor layers found in checkpoint.')
        layers, width, parameters = [], INPUTS[environment], 0
        architecture = [width]
        for prefix in prefixes:
            weights, grid = state[prefix + '.weight'], state[prefix + '.grid_starts']
            count = len(grid)
            if count < 4 or weights.ndim != 2 or weights.shape[1] != width * (count + 1):
                raise ValueError('Checkpoint dimensions do not match this environment.')
            inv_h = (count - 3) / 2
            expected = torch.arange(count, dtype=grid.dtype) / inv_h - 1 - 3 / inv_h
            if not torch.allclose(grid, expected, atol=1e-6, rtol=1e-6):
                raise ValueError('Non-default spline grid: supply exported actor.json with its exact inverse spacing.')
            layers.append({'inputs': width, 'outputs': weights.shape[0], 'n_bases': count,
                           'grid_start': grid[0].item(), 'inv_h': inv_h,
                           'coefficients': weights[:, :width * count].reshape(-1, width, count).tolist(),
                           'base_weights': weights[:, width * count:].tolist()})
            parameters += weights.numel()
            width = weights.shape[0]; architecture.append(width)
        weight, bias = state['action_net.weight'], state['action_net.bias']
        parameters += weight.numel() + bias.numel()
        data = {'format': 'inkan-research-actor-v1', 'environment': environment,
                'architecture': architecture, 'basis': 'cubic_uniform_b_spline_plus_silu',
                'layers': layers, 'head': {'weight': weight.tolist(), 'bias': bias.tolist()},
                'action_mapping': 'argmax', 'image_encoder_required': False,
                'observation_scale': [1.] * INPUTS[environment],
                'parameters': {'actor_parameters': parameters},
                'training_steps': metadata.get('num_timesteps'), 'checkpoint': source.name}
    else:
        raise ValueError('Supply an exported actor JSON or an unnormalized InKAN PPO ZIP.')
    validate_actor(data, environment)
    data['architecture'] = [INPUTS[environment]] + [len(layer['coefficients']) for layer in data['layers']]
    # Bind imported curve sessions to these exact source weights and preprocessing.
    data['checkpoint_sha256'] = hashlib.sha256(contents).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)
    return data
