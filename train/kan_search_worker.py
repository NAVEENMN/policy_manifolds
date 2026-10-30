"""KAN vs MLP smallest-network search for any Gymnasium environment.

Usage: python3 kan_search_worker.py --envs MountainCar-v0 Acrobot-v1 --output results/
"""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Callable, Tuple

import gymnasium as gym
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import torch.nn as nn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env

from inkan import KANLayer

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# --- KAN policy factories ---

class KANPPOBase(ActorCriticPolicy):
    def __init__(self, obs_space, act_space, lr_schedule, *a, **kw):
        kw["ortho_init"] = False
        super().__init__(obs_space, act_space, lr_schedule, *a, **kw)


def make_kan_extractor(layers, grid_size):
    """Build a nn.Module KAN extractor from a list of (in, out) dims."""
    class Ext(nn.Module):
        def __init__(self, feature_dim):
            super().__init__()
            dims = [feature_dim] + layers
            pi_layers = []
            vf_layers = []
            for i in range(len(dims) - 1):
                pi_layers.append(KANLayer(dims[i], dims[i+1], grid_size=grid_size, compile_basis=False))
                vf_layers.append(KANLayer(dims[i], dims[i+1], grid_size=grid_size, compile_basis=False))
            self.policy_net = nn.Sequential(*pi_layers) if len(pi_layers) > 1 else pi_layers[0]
            self.value_net = nn.Sequential(*vf_layers) if len(vf_layers) > 1 else vf_layers[0]
            self.latent_dim_pi = dims[-1]
            self.latent_dim_vf = dims[-1]

        def forward(self, f):
            return self.forward_actor(f), self.forward_critic(f)
        def forward_actor(self, f):
            return self.policy_net(f)
        def forward_critic(self, f):
            return self.value_net(f)
    return Ext


def make_kan_policy(layers, grid_size):
    ext_cls = make_kan_extractor(layers, grid_size)
    class P(KANPPOBase):
        def _build_mlp_extractor(self):
            self.mlp_extractor = ext_cls(self.features_dim)
    return P


def count_params(model):
    return sum(p.numel() for p in model.policy.parameters() if p.requires_grad)


# --- Environment configs ---

ENV_CONFIGS = {
    "MountainCar-v0":            {"algo": "PPO", "steps": 1_000_000, "target": -110, "discrete": True},
    "MountainCarContinuous-v0":  {"algo": "PPO", "steps": 500_000,   "target": 90,   "discrete": False},
    "Acrobot-v1":                {"algo": "PPO", "steps": 500_000,   "target": -100,  "discrete": True},
    "LunarLander-v3":            {"algo": "PPO", "steps": 1_000_000, "target": 200,   "discrete": True},
    "BipedalWalker-v3":          {"algo": "PPO", "steps": 2_000_000, "target": 300,   "discrete": False},
    "CartPole-v1":               {"algo": "PPO", "steps": 100_000,   "target": 499,   "discrete": True},
    "Pendulum-v1":               {"algo": "PPO", "steps": 200_000,   "target": -200,  "discrete": False},
}

# Architectures to search (ordered small to large)
KAN_ARCHS = [
    # (hidden_layers, grid_size, label)
    ([2],     2, "KAN g2 1L-2"),
    ([2],     5, "KAN g5 1L-2"),
    ([2, 2],  2, "KAN g2 2L-2"),
    ([4],     2, "KAN g2 1L-4"),
    ([2, 2],  5, "KAN g5 2L-2"),
    ([4],     5, "KAN g5 1L-4"),
    ([4, 4],  2, "KAN g2 2L-4"),
    ([4, 4],  5, "KAN g5 2L-4"),
    ([8],     2, "KAN g2 1L-8"),
    ([8],     5, "KAN g5 1L-8"),
    ([8, 8],  2, "KAN g2 2L-8"),
    ([8, 8],  5, "KAN g5 2L-8"),
    ([16, 16], 2, "KAN g2 2L-16"),
    ([16, 16], 5, "KAN g5 2L-16"),
    ([32, 32], 5, "KAN g5 2L-32"),
]

MLP_ARCHS = [
    ([2, 2],   "MLP 2x2"),
    ([4, 4],   "MLP 4x4"),
    ([8, 8],   "MLP 8x8"),
    ([16, 16], "MLP 16x16"),
    ([32, 32], "MLP 32x32"),
    ([64, 64], "MLP 64x64"),
]

SEEDS = [0, 1, 2]


def run_search(env_name, output_dir, models_dir=None):
    cfg = ENV_CONFIGS[env_name]
    target = cfg["target"]
    steps = cfg["steps"]
    obs_dim = gym.make(env_name).observation_space.shape[0]

    results = []
    best_kan = None
    best_mlp = None

    print(f"\n{'#'*60}")
    print(f"# {env_name}  obs={obs_dim}  target={target}  steps={steps}")
    print(f"{'#'*60}\n")

    # --- KAN search ---
    for layers, grid_size, label in KAN_ARCHS:
        policy_cls = make_kan_policy(layers, grid_size)
        rewards = []
        params = None

        for seed in SEEDS:
            env = make_vec_env(env_name, n_envs=2, seed=seed)
            model = PPO(
                policy_cls, env, verbose=0, seed=seed,
                learning_rate=3e-4, n_steps=2048, batch_size=64,
                n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                ent_coef=0.01,
            )
            if params is None:
                params = count_params(model)
            model.learn(total_timesteps=steps)
            eval_env = gym.make(env_name)
            mean_r, _ = evaluate_policy(model, eval_env, n_eval_episodes=20)
            rewards.append(mean_r)
            eval_env.close()
            env.close()

            # Early stop if first seed fails badly
            if seed == 0 and ((cfg["target"] > 0 and mean_r < target * 0.5) or
                              (cfg["target"] < 0 and mean_r < target * 2)):
                print(f"  {label:25s}  params={params:6d}  seed0={mean_r:.1f} — skipping remaining seeds")
                break

        avg = sum(rewards) / len(rewards)
        worst = min(rewards)
        passed = all(r >= target for r in rewards) if cfg["target"] > 0 else all(r >= target for r in rewards)

        row = {"env": env_name, "type": "KAN", "arch": label, "params": params,
               "avg": round(avg, 1), "worst": round(worst, 1), "passed": passed,
               "seeds": [round(r, 1) for r in rewards]}
        results.append(row)
        status = "PASS" if passed else "fail"
        print(f"  {label:25s}  params={params:6d}  avg={avg:8.1f}  worst={worst:8.1f}  [{status}]  {[f'{r:.0f}' for r in rewards]}")

        if passed and (best_kan is None or params < best_kan["params"]):
            best_kan = row
            # Save the model
            env = gym.make(env_name)
            model = PPO(policy_cls, env, verbose=0, seed=SEEDS[rewards.index(max(rewards))],
                        learning_rate=3e-4, n_steps=2048, batch_size=64,
                        n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)
            model.learn(total_timesteps=steps)
            model_folder = Path(models_dir or Path(__file__).resolve().parent / 'models') / env_name
            model_folder.mkdir(parents=True, exist_ok=True)
            model.save(model_folder / f"{env_name}_best_kan.zip")
            env.close()

    # --- MLP search ---
    for layers, label in MLP_ARCHS:
        rewards = []
        params = None

        for seed in SEEDS:
            env = make_vec_env(env_name, n_envs=2, seed=seed)
            model = PPO(
                "MlpPolicy", env, verbose=0, seed=seed,
                learning_rate=3e-4, n_steps=2048, batch_size=64,
                n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                ent_coef=0.01,
                policy_kwargs=dict(net_arch=layers, activation_fn=nn.Tanh),
            )
            if params is None:
                params = count_params(model)
            model.learn(total_timesteps=steps)
            eval_env = gym.make(env_name)
            mean_r, _ = evaluate_policy(model, eval_env, n_eval_episodes=20)
            rewards.append(mean_r)
            eval_env.close()
            env.close()

        avg = sum(rewards) / len(rewards)
        worst = min(rewards)
        passed = all(r >= target for r in rewards)

        row = {"env": env_name, "type": "MLP", "arch": label, "params": params,
               "avg": round(avg, 1), "worst": round(worst, 1), "passed": passed,
               "seeds": [round(r, 1) for r in rewards]}
        results.append(row)
        status = "PASS" if passed else "fail"
        print(f"  {label:25s}  params={params:6d}  avg={avg:8.1f}  worst={worst:8.1f}  [{status}]  {[f'{r:.0f}' for r in rewards]}")

        if passed and (best_mlp is None or params < best_mlp["params"]):
            best_mlp = row

    # --- Summary ---
    print(f"\n--- {env_name} Summary ---")
    if best_kan:
        print(f"  Smallest KAN: {best_kan['arch']} ({best_kan['params']} params, avg={best_kan['avg']})")
    else:
        print(f"  No KAN architecture passed target={target}")
    if best_mlp:
        print(f"  Smallest MLP: {best_mlp['arch']} ({best_mlp['params']} params, avg={best_mlp['avg']})")
    else:
        print(f"  No MLP architecture passed target={target}")

    return results, best_kan, best_mlp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", required=True)
    parser.add_argument("--output", default="kan_search_results")
    parser.add_argument("--models-dir", default=str(Path(__file__).resolve().parent / 'models'))
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    all_results = []

    for env_name in args.envs:
        if env_name not in ENV_CONFIGS:
            print(f"Unknown env: {env_name}, skipping")
            continue
        results, best_kan, best_mlp = run_search(env_name, args.output, args.models_dir)
        all_results.extend(results)

        # Write CSV incrementally
        csv_path = os.path.join(args.output, "results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["env", "type", "arch", "params", "avg", "worst", "passed", "seeds"])
            writer.writeheader()
            writer.writerows(all_results)

    # Final summary
    summary_path = os.path.join(args.output, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()
