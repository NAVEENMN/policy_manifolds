"""Export the trained 4→2→2 actor for the browser, without unpickling policy classes."""
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent


def export_policy(checkpoint=None, output=None):
    import torch

    checkpoint = Path(checkpoint or ROOT / 'models/CartPole-v1/ppo_kan_cartpole_2x2.zip')
    output = Path(output or ROOT / 'models/CartPole-v1/actor.json')
    with ZipFile(checkpoint) as archive:
        state = torch.load(io.BytesIO(archive.read('policy.pth')),
                           map_location='cpu', weights_only=True)
        metadata = json.loads(archive.read('data'))
    layers = []
    for index, inputs in enumerate((4, 2)):
        prefix = f'mlp_extractor.policy_net.{index}'
        weights = state[f'{prefix}.weight']
        grid = state[f'{prefix}.grid_starts']
        if tuple(weights.shape) != (2, inputs * 9) or len(grid) != 8:
            raise ValueError('Expected a 4→2→2 InKAN actor with grid_size=5 and cubic splines.')
        if not torch.allclose(grid, torch.arange(8) * 0.4 - 2.2, atol=1e-6):
            raise ValueError('Expected the checkpoint’s uniform [-1, 1] spline grid.')
        layers.append({
            'inputs': inputs, 'outputs': 2, 'gridStart': grid[0].item(),
            'invH': 2.5, 'nBases': 8,
            'coefficients': weights[:, :inputs * 8].reshape(2, inputs, 8).tolist(),
            'baseWeights': weights[:, inputs * 8:].tolist(),
        })
    model = {
        'format': 'cartpole-kan-policy-v1',
        'checkpoint': checkpoint.name,
        'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'trainingSteps': metadata['num_timesteps'],
        'architecture': [4, 2, 2], 'actorParameters': 114, 'totalParameters': 225,
        'layers': layers,
        'head': {'weight': state['action_net.weight'].tolist(),
                 'bias': state['action_net.bias'].tolist()},
        'observationNames': ['Cart position', 'Cart velocity', 'Pole angle', 'Pole angular velocity'],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2) + '\n')
    return model


if __name__ == '__main__':
    model = export_policy()
    print(f"Exported {model['checkpoint']}: 12 actor edges, 114 actor parameters.")
