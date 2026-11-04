"""Export checkpoints and install a selected model for the browser app."""
import hashlib
import json
from pathlib import Path
import shutil
from .paths import ENVIRONMENTS, MODELS, WEB


def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def cartpole_browser_actor(data, checkpoint_name):
    result = {'format': 'cartpole-kan-policy-v1', 'environment': 'CartPole-v1',
              'sha256': data['checkpoint_sha256'], 'checkpoint': checkpoint_name,
              'architecture': data['architecture'],
              'actorParameters': data['parameters']['actor_parameters'],
              'observationScale': data.get('observation_scale') or [1.] * 4,
              'head': data['head'],
              'layers': [{'inputs': l['inputs'], 'outputs': l['outputs'], 'nBases': l['n_bases'],
                          'gridStart': l['grid_start'], 'invH': l['inv_h'],
                          'coefficients': l['coefficients'], 'baseWeights': l['base_weights']} for l in data['layers']]}
    if data.get('observation_normalization'):
        result['observationNormalization'] = data['observation_normalization']
    return result


def export_model(environment, source, output):
    source, output = Path(source), Path(output)
    if environment not in ENVIRONMENTS:
        raise ValueError('Unsupported demo environment.')
    if source.suffix == '.json':
        from .evaluation import load_actor
        load_actor(source, environment)
        data = json.loads(source.read_text())
        if data.get('format') == 'inkan-research-actor-v1':
            data.setdefault('checkpoint_sha256', hashlib.sha256(source.read_bytes()).hexdigest())
            if environment == 'CartPole-v1':
                data = cartpole_browser_actor(data, data.get('checkpoint', source.name))
        atomic_json(output, data)
        return data
    if ENVIRONMENTS[environment]['algorithm'] == 'SAC' and environment != 'Pendulum-v1':
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import VecNormalize
        from kan_search.io import export_actor
        import pickle
        normalization = source.with_name('normalization.pkl')
        if not normalization.exists():
            normalization = source.with_name(source.stem.replace('_model', '_normalization') + '.pkl')
        if not normalization.exists():
            raise ValueError('Supply the exported actor JSON or the matching normalization.pkl alongside this checkpoint.')
        with normalization.open('rb') as stream:
            vec = pickle.load(stream)
        if not isinstance(vec, VecNormalize):
            raise ValueError('Expected the saved VecNormalize snapshot.')
        model = SAC.load(source, device='cpu')
        export_actor(model, vec, output, environment)
        data = json.loads(output.read_text())
        environment_file = source.parent / 'environment.json'
        if not environment_file.exists():
            environment_file = source.parent.parent.parent / 'environment.json'
        if environment_file.exists():
            data['environment_metadata'] = json.loads(environment_file.read_text())
            data['observation_labels'] = data['environment_metadata']['observation_labels']
            data['action_labels'] = data['environment_metadata']['action_labels']
        data['checkpoint_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        atomic_json(output, data)
        return data
    if environment == 'Pendulum-v1':
        from .pendulum_export import export
        export(source, output)
        return json.loads(output.read_text())
    from .classic_export import export_classic
    data = export_classic(source, output, environment)
    if environment == 'CartPole-v1':
        data = cartpole_browser_actor(data, source.name)
        atomic_json(output, data)
    return data


def prepare_web(models=MODELS, web=WEB, refresh=False, preserve_bundled=False):
    models, web = Path(models), Path(web); web.mkdir(parents=True, exist_ok=True)
    ready = []
    for environment, spec in ENVIRONMENTS.items():
        folder = models / environment; actor = folder / 'actor.json'; checkpoint = folder / 'model.zip'
        if checkpoint.exists() and (refresh or not actor.exists()):
            export_model(environment, checkpoint, actor)
        destination = web / spec['file']
        if actor.exists():
            data = json.loads(actor.read_text())
            if data.get('format') == 'inkan-pairwise-actor-v1':
                if destination.exists(): destination.unlink()
                continue  # The browser currently edits univariate curves only.
            metadata = folder / 'metadata.json'
            if metadata.exists():
                details = json.loads(metadata.read_text())
                data['demo'] = details.get('demo', {'algorithm': details.get('algorithm', spec['algorithm'])})
            atomic_json(destination, data)
            ready.append(environment)
        elif destination.exists():
            if preserve_bundled:
                ready.append(environment)
            else:
                destination.unlink()  # Custom collections must not inherit unrelated actors.
    return ready


def promote(environment, checkpoint, actor, metrics, models=MODELS):
    if json.loads(Path(actor).read_text()).get('format') == 'inkan-pairwise-actor-v1':
        raise ValueError('2D actors can be trained/exported/validated, but app installation needs a 2D surface editor.')
    folder = Path(models) / environment; folder.mkdir(parents=True, exist_ok=True)
    # Validate the export before replacing any selected model.
    from .evaluation import load_actor
    load_actor(actor, environment)
    copies = [(checkpoint, 'model.zip'), (actor, 'actor.json')]
    if ENVIRONMENTS[environment]['algorithm'] == 'SAC' and environment != 'Pendulum-v1':
        checkpoint = Path(checkpoint)
        normalization = checkpoint.with_name('normalization.pkl')
        if not normalization.exists():
            normalization = checkpoint.with_name(checkpoint.stem.replace('_model', '_normalization') + '.pkl')
        if not normalization.exists():
            raise ValueError('The selected MuJoCo checkpoint needs its matching normalization snapshot.')
        copies.append((normalization, 'normalization.pkl'))
    for source, name in copies:
        temporary = folder / (name + '.tmp'); shutil.copy2(source, temporary); temporary.replace(folder / name)
    atomic_json(folder / 'metadata.json', {'environment': environment, 'checkpoint': 'model.zip',
        'actor': 'actor.json', 'sha256': hashlib.sha256((folder/'model.zip').read_bytes()).hexdigest(),
        'evaluation': metrics})
