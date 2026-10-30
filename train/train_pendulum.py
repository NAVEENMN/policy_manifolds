"""Train and save small InKAN Pendulum controllers with reproducible validation.

Example: venv/bin/python train_pendulum.py --arch 1,1 --grid 2 --steps 100000 --seed 0 --output models/Pendulum-v1/runs/new_run
The deployment actor is KAN; SAC's two MLP critics are training-only.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from pendulum_kan import KANSACPolicy, evaluate, parameter_counts

ROOT = Path(__file__).resolve().parent


class ValidationCallback(BaseCallback):
    def __init__(self, output, interval, episodes, stop_on_success=False):
        super().__init__()
        self.output, self.interval, self.episodes = output, interval, episodes
        self.next_eval = interval
        self.best_score = -float('inf')
        self.history = []
        self.started = time.monotonic()
        self.stop_on_success = stop_on_success

    def _init_callback(self):
        self.next_eval = (self.model.num_timesteps // self.interval + 1) * self.interval

    def _on_step(self):
        if self.num_timesteps < self.next_eval:
            return True
        self.next_eval += self.interval
        result = evaluate(self.model, range(1000, 1000 + self.episodes))
        record = {'steps': self.num_timesteps, 'seconds': round(time.monotonic() - self.started, 1), **result}
        self.history.append(record)
        if result['mean_reward'] > self.best_score:
            self.best_score = result['mean_reward']
            self.model.save(self.output / 'best_model')
            (self.output / 'best_validation.json').write_text(json.dumps(record, indent=2))
        (self.output / 'learning_curve.json').write_text(json.dumps(self.history, indent=2))
        print(json.dumps({k: record[k] for k in ['steps', 'seconds', 'mean_reward', 'worst_reward', 'success_rate']}), flush=True)
        return not (self.stop_on_success and result['mean_reward'] >= -200 and result['success_rate'] >= .95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arch', default='1,1', help='KAN widths after the 3 observations, e.g. 1 or 2,1')
    parser.add_argument('--grid', type=int, default=2)
    parser.add_argument('--steps', type=int, default=100_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--envs', type=int, default=4)
    parser.add_argument('--critic', type=int, default=64)
    parser.add_argument('--eval-every', type=int, default=10_000)
    parser.add_argument('--eval-episodes', type=int, default=32)
    parser.add_argument('--test-episodes', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-on-success', action='store_true')
    args = parser.parse_args()
    widths = [int(x) for x in args.arch.split(',')]
    if not widths or min(widths) < 1 or args.grid < 1:
        parser.error('Widths and grid size must be positive.')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = args.output or ROOT / 'models/Pendulum-v1/runs' / f"sac_3-{'-'.join(map(str,widths))}_g{args.grid}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'config.json').exists() and not args.resume:
        parser.error(f'{output} already contains a run; use a new output directory or --resume.')
    env = make_vec_env('Pendulum-v1', n_envs=args.envs, seed=args.seed)
    if args.resume:
        model = SAC.load(args.resume, env=env, device='cpu')
        replay = args.resume.parent / 'replay_buffer.pkl'
        if replay.exists():
            model.load_replay_buffer(replay)
    else:
        model = SAC(KANSACPolicy, env, seed=args.seed, device='cpu', verbose=0,
                    learning_rate=args.learning_rate, buffer_size=200_000, learning_starts=2000,
                    batch_size=256, train_freq=1, gradient_steps=1, gamma=.99, tau=.005,
                    policy_kwargs={'net_arch': {'pi': widths, 'qf': [args.critic, args.critic]}, 'grid_size': args.grid})
    config = {**vars(args), 'arch': widths, 'output': str(output), 'resume': str(args.resume) if args.resume else None,
              'initial_timesteps': model.num_timesteps,
              'algorithm': 'SAC', 'observation_scale': [1, 1, 8], 'parameters': parameter_counts(model),
              'success_definition': 'At least 90% of final 50 steps within 15 degrees upright and abs angular velocity < 1 rad/s.',
              'selection_target': 'validation mean return >= -200 and success rate >= 0.95'}
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    print(json.dumps({'run': str(output), 'parameters': config['parameters']}), flush=True)
    callback = ValidationCallback(output, args.eval_every, args.eval_episodes, args.stop_on_success)
    try:
        model.learn(total_timesteps=args.steps, callback=callback, reset_num_timesteps=not bool(args.resume))
        model.save(output / 'last_model')
        model.save_replay_buffer(output / 'replay_buffer.pkl')
        if not (output / 'best_model.zip').exists():
            model.save(output / 'best_model')
        best = SAC.load(output / 'best_model.zip', device='cpu')
        result = evaluate(best, range(20_000, 20_000 + args.test_episodes))
        result.update(parameters=parameter_counts(best), checkpoint='best_model.zip', num_timesteps=best.num_timesteps)
        (output / 'test_results.json').write_text(json.dumps(result, indent=2))
        print('HELD_OUT ' + json.dumps({k: result[k] for k in ['mean_reward', 'std_reward', 'worst_reward', 'success_rate', 'num_timesteps']}), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
