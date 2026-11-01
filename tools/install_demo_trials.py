"""Select complete 1D trials by selection-validation return and package demo actors.

Run from this checkout: python tools/install_demo_trials.py --results /path/to/results
The sources are never modified. Selection does not use held-out test scores.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = ['Hopper-v5', 'Walker2d-v5', 'Humanoid-v5', 'Ant-v5', 'Swimmer-v5', 'Reacher-v5']
REQUIRED = ['result.json', 'best_model.zip', 'best_actor.json', 'best_normalization.pkl',
            'best_validation.json', 'learning_curve.csv']


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(results):
    inventory, selections = [], []
    for env in ENVIRONMENTS:
        candidates = []
        for trial in sorted((results / env / 'trials').iterdir()):
            if not trial.is_dir():
                continue
            result = read(trial / 'result.json')
            missing = [f for f in REQUIRED if not (trial / f).is_file()]
            score = result.get('validation_mean_return')
            eligible = (not missing and result.get('status') == 'complete'
                        and result.get('actor_kind') == '1d'
                        and isinstance(score, (int, float)) and math.isfinite(score))
            inventory.append({'environment': env, 'trial': trial.name, 'status': result['status'],
                'architecture': result['architecture'], 'grid_size': result['grid_size'],
                'training_seed': result['training_seed'], 'actor_parameters': result.get('actor_parameters'),
                'validation_episodes': result.get('validation_episodes'),
                'validation_mean_return': score, 'target': result['target'],
                'qualifies_validation': result.get('qualifies_validation', False),
                'eligible_for_demo': eligible, 'missing_files': ';'.join(missing), 'selected': False})
            if eligible:
                candidates.append((trial, result))
        if not candidates:
            raise ValueError(f'No complete 1D trial available for {env}')
        trial, result = max(candidates, key=lambda row: (row[1]['validation_mean_return'], row[0].name))
        actor = read(trial / 'best_actor.json')
        if (actor.get('format') != 'inkan-research-actor-v1' or actor['environment'] != env
                or actor['action_mapping'] != 'tanh_then_box_rescale' or not actor.get('observation_normalization')):
            raise ValueError(f'Unsupported or unnormalized actor: {trial}')
        arch = [actor['layers'][0]['inputs'], *[l['outputs'] for l in actor['layers']]]
        if arch != actor['architecture'] or arch[1:] != [int(x) for x in result['architecture'].split(',')]:
            raise ValueError(f'Architecture metadata mismatch: {trial}')
        count = sum(l['inputs'] * l['outputs'] * (l['n_bases'] + 1) for l in actor['layers'])
        count += len(actor['head']['weight']) * (arch[-1] + 1)
        if count != result['actor_parameters'] or count != actor['parameters']['actor_parameters']:
            raise ValueError(f'Actor parameter count mismatch: {trial}')
        if any(l['n_bases'] != result['grid_size'] + 3 for l in actor['layers']):
            raise ValueError(f'Grid metadata mismatch: {trial}')
        siblings = [r for _, r in candidates if r['grid_size'] == result['grid_size']
                    and r['architecture'] == result['architecture']]
        seeds = {r['training_seed'] for r in siblings}
        passed = {r['training_seed'] for r in siblings if r['qualifies_validation']}
        status = 'Target met · one training seed' if result['qualifies_validation'] else 'Below target'
        if len(passed) > 1:
            status = f'Target met · {len(passed)} training seeds'
        reliability = (f'{len(passed)}/{len(seeds)} training seeds passed at this grid.'
                       + (' Not reliably solved across seeds.' if passed and passed != seeds else ''))
        demo = {'algorithm': result['algorithm'], 'status': status,
                'score': result['validation_mean_return'], 'score_label': f"{result['validation_episodes']} validation episodes",
                'target': result['target'], 'target_met': bool(result['qualifies_validation']),
                'trial': trial.name, 'training_seed': result['training_seed'], 'grid_size': result['grid_size'],
                'training_seeds_evaluated': len(seeds), 'training_seeds_passed': len(passed),
                'detail': f"Target {result['target']:g} · seed {result['training_seed']} · grid {result['grid_size']} · {result['best_checkpoint_steps']:,} checkpoint steps. {reliability}",
                'selection': 'Highest selection-validation mean return among complete exported 1D trials; no held-out test selection.'}
        actor['checkpoint_sha256'] = digest(trial / 'best_model.zip')
        actor['demo'] = demo
        record = {'environment': env, 'trial': trial.name, 'source': f'results/{env}/trials/{trial.name}',
                  'architecture': arch, 'head_architecture': [arch[-1], len(actor['head']['bias'])],
                  'actor_parameters': count, 'demo': demo,
                  'files': {name: digest(trial / name) for name in REQUIRED}}
        selections.append((trial, actor, record))
        next(r for r in inventory if r['environment'] == env and r['trial'] == trial.name)['selected'] = True
    return inventory, selections


def install(results, root):
    inventory, selections = collect(results)  # Validate the entire selection before writing.
    for trial, actor, record in selections:
        env = record['environment']; key = env.split('-v')[0].lower()
        dest = root / 'models' / env
        dest.mkdir(parents=True, exist_ok=True)
        for source, filename in [('best_model.zip', 'model.zip'), ('best_normalization.pkl', 'normalization.pkl'),
                                 ('best_validation.json', 'best_validation.json'), ('result.json', 'result.json'),
                                 ('learning_curve.csv', 'learning_curve.csv')]:
            shutil.copy2(trial / source, dest / filename)
        write(dest / 'actor.json', actor)
        write(dest / 'environment.json', actor['environment_metadata'])
        write(dest / 'metadata.json', record)
        write(root / 'web_dist' / f'{key}.json', actor)
        curve = root / 'web_dist' / 'results' / f'{key}-learning-curve.csv'
        curve.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trial / 'learning_curve.csv', curve)
    write_csv(root / 'models' / 'width2_trials.csv', inventory)
    write_csv(root / 'web_dist' / 'results' / 'width2_trials.csv', inventory)
    write(root / 'models' / 'width2_selection.json', {
        'selection_rule': 'Maximize result.json validation_mean_return among complete 1D exports.',
        'trial_count': len(inventory), 'eligible_count': sum(r['eligible_for_demo'] for r in inventory),
        'note': 'Validation selection scores, not held-out test or independent reproduction. A demo winner does not establish reliability across training seeds.',
        'models': [r for _, _, r in selections]})
    return selections


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    for _, _, record in install(args.results.resolve(), args.output.resolve()):
        print(record['environment'], record['trial'], record['demo']['score'])
