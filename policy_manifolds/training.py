"""Train one architecture, saving every seed and its validation history."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from .paths import ENVIRONMENTS, MODELS
from .exporting import atomic_json, export_model, promote


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env', required=True, choices=[name for name, spec in ENVIRONMENTS.items() if not spec.get('research_only')])
    p.add_argument('--arch', help='KAN widths after the observations, e.g. 4,4 or 2')
    p.add_argument('--grid', type=int)
    p.add_argument('--steps', type=int)
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--eval-every', type=int, default=10_000)
    p.add_argument('--eval-episodes', type=int, default=20)
    p.add_argument('--test-episodes', type=int, default=20)
    p.add_argument('--target', type=float, help='Explicit mean-return criterion for promotion')
    p.add_argument('--output', type=Path, help='New run directory; defaults to models/<env>/runs/<timestamp>')
    p.add_argument('--promote', action='store_true', help='Install the best validation seed only if every seed passes held-out evaluation')
    p.add_argument('--save-replay', action='store_true', help='Also retain the potentially large SAC replay buffer')
    p.add_argument('--smoke', action='store_true', help='64-step plumbing check; never promotes a demo model')
    return p


def run_seed(args, spec, directory, seed):
    import torch
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env
    from .classic_policy import ClassicKANPolicy
    from .pendulum_policy import KANSACPolicy, parameter_counts
    from .evaluation import evaluate

    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory/'config.json', {**vars(args), 'output': str(args.output), 'training_seed': seed, 'spec': spec})
    torch.set_num_threads(1)
    env = make_vec_env(args.env, n_envs=1 if spec['algorithm']=='PPO' or args.smoke else 4, seed=seed)
    if spec['algorithm'] == 'PPO':
        model = PPO(ClassicKANPolicy, env, seed=seed, device='cpu', verbose=0,
                    learning_rate=3e-4, n_steps=32 if args.smoke else 2048,
                    batch_size=32 if args.smoke else 64, n_epochs=1 if args.smoke else 10,
                    gamma=.99, gae_lambda=.95, clip_range=.2, ent_coef=.01 if args.env=='MountainCar-v0' else 0.,
                    policy_kwargs={'widths': args.arch, 'grid_size': args.grid})
        actor_parameters = sum(p.numel() for module in [model.policy.mlp_extractor.policy_net, model.policy.action_net] for p in module.parameters())
    else:
        model = SAC(KANSACPolicy, env, seed=seed, device='cpu', verbose=0, learning_rate=3e-4,
                    buffer_size=256 if args.smoke else 200_000, learning_starts=0 if args.smoke else 2000,
                    batch_size=8 if args.smoke else 256, train_freq=1, gradient_steps=1, gamma=.99, tau=.005,
                    policy_kwargs={'net_arch': {'pi': args.arch, 'qf': [64, 64]}, 'grid_size': args.grid})
        actor_parameters = parameter_counts(model)['actor_deterministic']
    training_parameters = sum(p.numel() for p in model.policy.parameters())

    class SaveAndEvaluate(BaseCallback):
        def __init__(self):
            super().__init__(); self.next_eval = args.eval_every; self.best = None; self.last_eval = -1

        def assess(self):
            # Save first: even an interrupted/failed evaluation retains the trained weights.
            self.model.save(directory/'last_model.zip')
            metrics = evaluate(self.model, args.env, args.eval_episodes, 20_000)
            record = {'steps': self.num_timesteps, 'mean_return': metrics['mean_return'],
                      'worst_return': metrics['worst_return'], 'success_rate': metrics['success_rate']}
            path = directory/'learning_curve.csv'; exists = path.exists()
            with path.open('a', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(record))
                if not exists: writer.writeheader()
                writer.writerow(record); stream.flush()
            if self.best is None or metrics['mean_return'] > self.best['mean_return']:
                self.best = metrics
                self.model.save(directory/'best_model.zip')
                atomic_json(directory/'best_validation.json', metrics)
            self.last_eval = self.num_timesteps
            print(json.dumps({'environment': args.env, 'seed': seed, **record}), flush=True)

        def _on_step(self):
            if self.num_timesteps >= self.next_eval:
                self.assess(); self.next_eval = self.num_timesteps + args.eval_every
            return True

    callback = SaveAndEvaluate()
    try:
        model.learn(total_timesteps=args.steps, callback=callback)
        model.save(directory/'last_model.zip')
        if callback.last_eval != model.num_timesteps: callback.assess()
        if args.save_replay and spec['algorithm']=='SAC': model.save_replay_buffer(directory/'replay_buffer.pkl')
        cls = SAC if spec['algorithm']=='SAC' else PPO
        best = cls.load(directory/'best_model.zip', device='cpu')
        test = evaluate(best, args.env, args.test_episodes, 100_000)
        export_model(args.env, directory/'best_model.zip', directory/'actor.json')
        result = {'environment': args.env, 'status': 'smoke_only' if args.smoke else 'complete',
                  'seed': seed, 'architecture': args.arch, 'grid_size': args.grid,
                  'training_steps': model.num_timesteps, 'actor_parameters': actor_parameters,
                  'training_parameters': training_parameters, 'checkpoint': 'best_model.zip',
                  'actor': 'actor.json', 'validation': callback.best, 'test': test}
        atomic_json(directory/'result.json', result)
        plot_learning_curve(directory)
        return result
    except BaseException as error:
        model.save(directory/'last_model.zip')
        atomic_json(directory/'interrupted.json', {'environment': args.env, 'seed': seed,
                    'steps': model.num_timesteps, 'error_type': type(error).__name__})
        raise
    finally:
        env.close()


def plot_learning_curve(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (directory/'learning_curve.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    ax.plot([int(r['steps']) for r in rows], [float(r['mean_return']) for r in rows], marker='o', color='#438a69')
    ax.set(xlabel='Training environment steps', ylabel='Validation mean return', title='Policy Manifolds · learning curve')
    ax.grid(alpha=.2)
    fig.savefig(directory/'learning_curve.png', dpi=140)
    plt.close(fig)


def main(argv=None):
    import math
    p = parser(); args = p.parse_args(argv); spec = ENVIRONMENTS[args.env]
    try:
        args.arch = list(map(int, args.arch.split(','))) if args.arch else spec['widths']
        if not args.arch or len(args.arch)>4 or min(args.arch)<1 or max(args.arch)>64: raise ValueError()
    except ValueError:
        p.error('Use one to four widths between 1 and 64, separated by commas.')
    args.grid = spec['grid'] if args.grid is None else args.grid
    args.steps = spec['steps'] if args.steps is None else args.steps
    args.target = spec['target'] if args.target is None else args.target
    if min(args.grid,args.steps,args.eval_every,args.eval_episodes,args.test_episodes)<1 or args.grid>61 or min(args.seeds)<0 or len(set(args.seeds))!=len(args.seeds) or not math.isfinite(args.target):
        p.error('Counts/grid must be positive, grid at most 61, seeds unique/nonnegative, and target finite.')
    if args.smoke and args.promote: p.error('Smoke fixtures cannot be promoted as trained demos.')
    if args.smoke: args.steps=64; args.eval_every=64; args.eval_episodes=2; args.test_episodes=2
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    args.output = args.output or MODELS/args.env/'runs'/stamp
    if args.output.exists(): p.error('Output already exists. Choose a new run directory to preserve previous checkpoints.')
    args.output.mkdir(parents=True)
    results = []
    for seed in args.seeds:
        result = run_seed(args, spec, args.output/f'seed-{seed}', seed); results.append(result)
        atomic_json(args.output/'results.json', results)
        with (args.output/'results.csv').open('w', newline='') as stream:
            fields=['environment','seed','status','actor_parameters','training_parameters','training_steps','validation_mean_return','test_mean_return','test_success_rate','checkpoint']
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for row in results:
                writer.writerow({**{key:row[key] for key in fields[:6]}, 'training_steps':row['training_steps'],
                    'validation_mean_return':row['validation']['mean_return'], 'test_mean_return':row['test']['mean_return'],
                    'test_success_rate':row['test']['success_rate'], 'checkpoint':f"seed-{row['seed']}/best_model.zip"})
    if args.promote:
        passed = all(r['test']['mean_return']>=args.target and (args.env!='Pendulum-v1' or r['test']['success_rate']>=.95) for r in results)
        if passed:
            selected=max(results,key=lambda r:r['validation']['mean_return']); folder=args.output/f"seed-{selected['seed']}"
            promote(args.env, folder/'best_model.zip', folder/'actor.json', selected['test'])
            print(f'Installed selected actor in {MODELS/args.env}',flush=True)
        else:
            print('Saved every run; the requested evaluation criterion was not met, so the selected demo was not replaced.',flush=True)
    print(f'Checkpoints, CSV results, and learning curves: {args.output}',flush=True)
    return results
