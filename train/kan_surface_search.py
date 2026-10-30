"""KAN surface (dim=2) search for unsolved environments.

Uses tensor-product B-spline surfaces that learn S(x,y) per edge pair.
Input features are paired up, processed by dim=2 KANLayers, then
optionally followed by dim=1 KANLayers.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from itertools import combinations
from typing import Tuple

import gymnasium as gym
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env

from inkan import KANLayer

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class PairSurfaceLayer(nn.Module):
    """Pairs up input features and processes each pair with a dim=2 surface KANLayer.

    For obs_dim=N, creates N//2 surface layers (each takes 2 inputs),
    plus handles odd remainder with a dim=1 layer if needed.
    Each pair outputs `pair_out` features, so total output = (N//2) * pair_out [+ pair_out if odd].
    """

    def __init__(self, obs_dim, pair_out=1, grid_size=5, compile_basis=False):
        super().__init__()
        self.obs_dim = obs_dim
        self.pair_out = pair_out
        self.n_pairs = obs_dim // 2
        self.has_remainder = obs_dim % 2 == 1

        # One surface layer per pair of features
        self.surface_layers = nn.ModuleList([
            KANLayer(2, pair_out, grid_size=grid_size, dim=2, compile_basis=compile_basis)
            for _ in range(self.n_pairs)
        ])

        # Handle odd feature with a 1D KAN
        if self.has_remainder:
            self.remainder_layer = KANLayer(1, pair_out, grid_size=grid_size, dim=1, compile_basis=compile_basis)

        self.output_dim = (self.n_pairs + (1 if self.has_remainder else 0)) * pair_out

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.surface_layers):
            pair_input = x[:, 2*i:2*i+2]
            outputs.append(layer(pair_input))
        if self.has_remainder:
            outputs.append(self.remainder_layer(x[:, -1:]))
        return torch.cat(outputs, dim=-1)


class SurfaceKANExtractor(nn.Module):
    """MLP extractor using surface layers for the first layer, then optional dim=1 layers."""

    def __init__(self, feature_dim, pair_out=1, hidden_dims=None, grid_size=5, compile_basis=False):
        super().__init__()
        # First layer: pair surfaces
        pi_surface = PairSurfaceLayer(feature_dim, pair_out, grid_size, compile_basis)
        vf_surface = PairSurfaceLayer(feature_dim, pair_out, grid_size, compile_basis)
        surface_out = pi_surface.output_dim

        if hidden_dims:
            # Add dim=1 KAN layers after the surface layer
            pi_layers = [pi_surface]
            vf_layers = [vf_surface]
            prev_dim = surface_out
            for h in hidden_dims:
                pi_layers.append(KANLayer(prev_dim, h, grid_size=grid_size, compile_basis=compile_basis))
                vf_layers.append(KANLayer(prev_dim, h, grid_size=grid_size, compile_basis=compile_basis))
                prev_dim = h
            self.policy_net = nn.Sequential(*pi_layers)
            self.value_net = nn.Sequential(*vf_layers)
            self.latent_dim_pi = prev_dim
            self.latent_dim_vf = prev_dim
        else:
            self.policy_net = pi_surface
            self.value_net = vf_surface
            self.latent_dim_pi = surface_out
            self.latent_dim_vf = surface_out

    def forward(self, f):
        return self.forward_actor(f), self.forward_critic(f)

    def forward_actor(self, f):
        return self.policy_net(f)

    def forward_critic(self, f):
        return self.value_net(f)


class PPOBase(ActorCriticPolicy):
    def __init__(self, o, a, lr, *a2, **kw):
        kw["ortho_init"] = False
        super().__init__(o, a, lr, *a2, **kw)


def make_surface_policy(pair_out, hidden_dims, grid_size):
    class P(PPOBase):
        def _build_mlp_extractor(self):
            self.mlp_extractor = SurfaceKANExtractor(
                self.features_dim, pair_out=pair_out, hidden_dims=hidden_dims,
                grid_size=grid_size, compile_basis=False
            )
    return P


def count_params(model):
    return sum(p.numel() for p in model.policy.parameters() if p.requires_grad)


# Environments to test (the unsolved ones + some solved for comparison)
ENV_CONFIGS = {
    "BipedalWalker-v3":          {"steps": 2_000_000, "target": 300, "ent": 0.001},
    "Pendulum-v1":               {"steps": 200_000,   "target": -200, "ent": 0.01},
    "MountainCarContinuous-v0":  {"steps": 500_000,   "target": 90,  "ent": 0.01},
    "HalfCheetah-v5":            {"steps": 1_000_000, "target": 4800, "ent": 0.0},
    "Hopper-v5":                 {"steps": 1_000_000, "target": 3800, "ent": 0.0},
    "Walker2d-v5":               {"steps": 1_000_000, "target": 4000, "ent": 0.0},
    "Swimmer-v5":                {"steps": 500_000,   "target": 360,  "ent": 0.0},
    "Reacher-v5":                {"steps": 500_000,   "target": -3.75, "ent": 0.0},
    "CartPole-v1":               {"steps": 100_000,   "target": 499,  "ent": 0.0},
    "MountainCar-v0":            {"steps": 1_000_000, "target": -110, "ent": 0.01},
    "Acrobot-v1":                {"steps": 500_000,   "target": -100, "ent": 0.0},
    "LunarLander-v3":            {"steps": 1_000_000, "target": 200,  "ent": 0.01},
}

# Surface architectures to search (pair_out, hidden_dims, grid_size, label)
SURFACE_ARCHS = [
    (1, None,  2, "Surf p1 g2"),
    (1, None,  5, "Surf p1 g5"),
    (2, None,  2, "Surf p2 g2"),
    (2, None,  5, "Surf p2 g5"),
    (1, [2],   2, "Surf p1→2 g2"),
    (1, [2],   5, "Surf p1→2 g5"),
    (2, [4],   2, "Surf p2→4 g2"),
    (2, [4],   5, "Surf p2→4 g5"),
    (4, None,  5, "Surf p4 g5"),
    (4, [4],   5, "Surf p4→4 g5"),
    (4, [8],   5, "Surf p4→8 g5"),
    (8, None,  5, "Surf p8 g5"),
    (8, [8],   5, "Surf p8→8 g5"),
    (8, [16],  5, "Surf p8→16 g5"),
]

SEEDS = [0, 1, 2]


def run_search(env_name, output_dir):
    cfg = ENV_CONFIGS[env_name]
    target = cfg["target"]
    steps = cfg["steps"]
    ent_coef = cfg["ent"]
    obs_dim = gym.make(env_name).observation_space.shape[0]
    n_pairs = obs_dim // 2 + (1 if obs_dim % 2 else 0)

    results = []
    best = None

    print(f"\n{'#'*60}")
    print(f"# {env_name}  obs={obs_dim}  pairs={n_pairs}  target={target}  steps={steps}")
    print(f"# Using dim=2 tensor-product B-spline surfaces")
    print(f"{'#'*60}\n", flush=True)

    for pair_out, hidden_dims, grid_size, label in SURFACE_ARCHS:
        policy_cls = make_surface_policy(pair_out, hidden_dims, grid_size)
        rewards = []
        params = None

        for seed in SEEDS:
            env = make_vec_env(env_name, n_envs=2, seed=seed)
            model = PPO(
                policy_cls, env, verbose=0, seed=seed,
                learning_rate=3e-4, n_steps=2048, batch_size=64,
                n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                ent_coef=ent_coef,
            )
            if params is None:
                params = count_params(model)
            model.learn(total_timesteps=steps)
            eval_env = gym.make(env_name)
            mean_r, _ = evaluate_policy(model, eval_env, n_eval_episodes=20)
            rewards.append(mean_r)
            eval_env.close()
            env.close()

            # Early stop on clear failure
            if seed == 0 and target > 0 and mean_r < target * 0.3:
                print(f"  {label:25s}  params={params:6d}  seed0={mean_r:.1f} — skipping", flush=True)
                break
            if seed == 0 and target < 0 and mean_r < target * 3:
                print(f"  {label:25s}  params={params:6d}  seed0={mean_r:.1f} — skipping", flush=True)
                break

        avg = sum(rewards) / len(rewards)
        worst = min(rewards)
        is_better = (target > 0 and all(r >= target for r in rewards)) or \
                    (target < 0 and all(r >= target for r in rewards))

        row = {"env": env_name, "type": "Surface", "arch": label, "params": params,
               "avg": round(avg, 1), "worst": round(worst, 1), "passed": is_better,
               "seeds": [round(r, 1) for r in rewards]}
        results.append(row)
        status = "PASS" if is_better else "fail"
        print(f"  {label:25s}  params={params:6d}  avg={avg:8.1f}  worst={worst:8.1f}  [{status}]  {[f'{r:.0f}' for r in rewards]}", flush=True)

        if is_better and (best is None or params < best["params"]):
            best = row

    print(f"\n--- {env_name} Surface Summary ---")
    if best:
        print(f"  Smallest surface: {best['arch']} ({best['params']} params, avg={best['avg']})")
    else:
        print(f"  No surface architecture passed target={target}")

    return results, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", required=True)
    parser.add_argument("--output", default="surface_results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    all_results = []

    for env_name in args.envs:
        if env_name not in ENV_CONFIGS:
            print(f"Unknown env: {env_name}, skipping")
            continue
        results, best = run_search(env_name, args.output)
        all_results.extend(results)

        csv_path = os.path.join(args.output, "surface_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["env", "type", "arch", "params", "avg", "worst", "passed", "seeds"])
            writer.writeheader()
            writer.writerows(all_results)

    summary_path = os.path.join(args.output, "surface_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()
