"""Solve the quick-win environments using SAC for continuous control.

Quick wins:
1. HalfCheetah-v5 — was 92% with PPO, try SAC + more steps
2. BipedalWalker-v3 — was 93% with PPO, try SAC
3. Reacher-v5 — close, SAC + more steps
4. MountainCarContinuous-v0 — PPO failed, SAC should work
5. Swimmer-v5 — SAC + longer training
"""

import json
import os
from pathlib import Path

import gymnasium as gym
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.sac.policies import SACPolicy

from inkan import KANLayer

torch.set_num_threads(2)
torch.set_num_interop_threads(1)


# --- KAN SAC Policy ---

class KANActor(nn.Module):
    """KAN-based actor for SAC. Outputs to latent, SB3 adds the squashed Gaussian head."""
    def __init__(self, obs_dim, hidden_dims, latent_dim, grid_size=5):
        super().__init__()
        dims = [obs_dim] + hidden_dims + [latent_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(KANLayer(dims[i], dims[i+1], grid_size=grid_size, compile_basis=False))
        self.net = nn.Sequential(*layers) if len(layers) > 1 else layers[0]
        self.latent_dim = latent_dim

    def forward(self, x):
        return self.net(x)


def make_sac_policy_kwargs(actor_hidden, actor_latent, critic_hidden, grid_size=5):
    """Create policy_kwargs for SAC with KAN actor and MLP critic.

    SB3's SAC uses net_arch for both actor and critic. We use a custom
    approach: KAN actor via _build override, MLP critic for stability.
    """
    return {
        "net_arch": {"pi": actor_hidden + [actor_latent], "qf": critic_hidden},
        "grid_size": grid_size,
    }


class KANSACPolicy(SACPolicy):
    """SAC policy with KAN actor, MLP critic."""
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, grid_size=5, **kwargs):
        self._kan_pi_arch = net_arch.get("pi", [64]) if net_arch else [64]
        self._kan_qf_arch = net_arch.get("qf", [256, 256]) if net_arch else [256, 256]
        self._grid_size = grid_size
        # Override net_arch for critic only
        kwargs["net_arch"] = {"pi": [64, 64], "qf": self._kan_qf_arch}
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def make_actor(self, *args, **kwargs):
        # Use default actor but we'll replace the network
        actor = super().make_actor(*args, **kwargs)
        return actor

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # Replace actor's latent_pi with KAN
        obs_dim = self.observation_space.shape[0]
        act_dim = self.action_space.shape[0]
        kan_dims = self._kan_pi_arch
        # Build KAN layers for actor mu network
        layers = []
        prev = obs_dim
        for d in kan_dims:
            layers.append(KANLayer(prev, d, grid_size=self._grid_size, compile_basis=False))
            prev = d
        self.actor.latent_pi = nn.Sequential(*layers)
        # Update mu/log_std to match new latent dim
        self.actor.mu = nn.Linear(prev, act_dim)
        self.actor.log_std = nn.Linear(prev, act_dim)


def count_actor_params(model):
    """Count only actor (policy) parameters."""
    total = 0
    for name, p in model.policy.named_parameters():
        if 'actor' in name and p.requires_grad:
            total += p.numel()
    return total


def count_all_params(model):
    return sum(p.numel() for p in model.policy.parameters() if p.requires_grad)


# --- Environment configs ---

ENVS = {
    "HalfCheetah-v5": {
        "target": 4800,
        "steps": 3_000_000,
        "archs": [
            ([8], 5, "8 g5"),
            ([16], 5, "16 g5"),
            ([8, 8], 5, "8-8 g5"),
            ([16, 16], 5, "16-16 g5"),
            ([32], 5, "32 g5"),
            ([32, 32], 5, "32-32 g5"),
        ],
        "critic": [256, 256],
    },
    "BipedalWalker-v3": {
        "target": 300,
        "steps": 2_000_000,
        "archs": [
            ([8], 5, "8 g5"),
            ([16], 5, "16 g5"),
            ([8, 8], 5, "8-8 g5"),
            ([16, 16], 5, "16-16 g5"),
            ([32, 32], 5, "32-32 g5"),
        ],
        "critic": [256, 256],
    },
    "Reacher-v5": {
        "target": -3.75,
        "steps": 1_000_000,
        "archs": [
            ([4], 2, "4 g2"),
            ([4], 5, "4 g5"),
            ([8], 5, "8 g5"),
            ([4, 4], 5, "4-4 g5"),
            ([8, 8], 5, "8-8 g5"),
        ],
        "critic": [64, 64],
    },
    "MountainCarContinuous-v0": {
        "target": 90,
        "steps": 500_000,
        "archs": [
            ([2], 2, "2 g2"),
            ([4], 2, "4 g2"),
            ([2, 2], 2, "2-2 g2"),
            ([4], 5, "4 g5"),
            ([4, 4], 5, "4-4 g5"),
            ([8, 8], 5, "8-8 g5"),
        ],
        "critic": [64, 64],
    },
    "Swimmer-v5": {
        "target": 360,
        "steps": 2_000_000,
        "archs": [
            ([4], 2, "4 g2"),
            ([8], 5, "8 g5"),
            ([4, 4], 5, "4-4 g5"),
            ([8, 8], 5, "8-8 g5"),
            ([16, 16], 5, "16-16 g5"),
            ([32, 32], 5, "32-32 g5"),
        ],
        "critic": [64, 64],
    },
}

SEEDS = [0, 1, 2]


def run_env(env_name, output_dir):
    cfg = ENVS[env_name]
    target = cfg["target"]
    steps = cfg["steps"]
    critic = cfg["critic"]
    obs_dim = gym.make(env_name).observation_space.shape[0]
    act_dim = gym.make(env_name).action_space.shape[0]

    print(f"\n{'#'*60}", flush=True)
    print(f"# {env_name}  obs={obs_dim}  act={act_dim}  target={target}  steps={steps}  algo=SAC", flush=True)
    print(f"{'#'*60}\n", flush=True)

    results = []
    best = None

    for kan_arch, grid_size, label in cfg["archs"]:
        rewards = []
        actor_params = None
        total_params = None

        for seed in SEEDS:
            env = make_vec_env(env_name, n_envs=2, seed=seed)
            try:
                model = SAC(
                    KANSACPolicy, env, verbose=0, seed=seed, device="cpu",
                    learning_rate=3e-4, buffer_size=300_000, learning_starts=5000,
                    batch_size=256, train_freq=1, gradient_steps=1,
                    gamma=0.99, tau=0.005,
                    policy_kwargs={
                        "net_arch": {"pi": kan_arch, "qf": critic},
                        "grid_size": grid_size,
                    },
                )
                if actor_params is None:
                    actor_params = count_actor_params(model)
                    total_params = count_all_params(model)
                model.learn(total_timesteps=steps)
                eval_env = gym.make(env_name)
                mean_r, _ = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
                rewards.append(mean_r)
                eval_env.close()

                # Early stop on clear failure
                if seed == 0:
                    if target > 0 and mean_r < target * 0.3:
                        print(f"  KAN {label:15s}  actor={actor_params:6d}  seed0={mean_r:.1f} — skip", flush=True)
                        env.close()
                        break
                    if target < 0 and mean_r < target * 3:
                        print(f"  KAN {label:15s}  actor={actor_params:6d}  seed0={mean_r:.1f} — skip", flush=True)
                        env.close()
                        break
            except Exception as e:
                print(f"  KAN {label:15s}  ERROR: {e}", flush=True)
                rewards.append(float('-inf'))
            finally:
                env.close()

        if not rewards:
            continue

        avg = sum(rewards) / len(rewards)
        worst = min(rewards)
        passed = all(r >= target for r in rewards)

        row = {"env": env_name, "arch": f"KAN {label}", "actor_params": actor_params,
               "total_params": total_params, "avg": round(avg, 1), "worst": round(worst, 1),
               "passed": passed, "seeds": [round(r, 1) for r in rewards]}
        results.append(row)
        status = "PASS" if passed else "fail"
        print(f"  KAN {label:15s}  actor={actor_params:6d}  avg={avg:8.1f}  worst={worst:8.1f}  [{status}]  {[f'{r:.0f}' for r in rewards]}", flush=True)

        if passed and (best is None or actor_params < best["actor_params"]):
            best = row
            # Save best model
            env = make_vec_env(env_name, n_envs=2, seed=SEEDS[rewards.index(max(rewards))])
            model = SAC(
                KANSACPolicy, env, verbose=0, seed=SEEDS[rewards.index(max(rewards))], device="cpu",
                learning_rate=3e-4, buffer_size=300_000, learning_starts=5000,
                batch_size=256, train_freq=1, gradient_steps=1,
                gamma=0.99, tau=0.005,
                policy_kwargs={"net_arch": {"pi": kan_arch, "qf": critic}, "grid_size": grid_size},
            )
            model.learn(total_timesteps=steps)
            save_dir = Path(output_dir) / env_name
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save(save_dir / "best_sac_kan")
            env.close()
            print(f"  -> Saved to {save_dir}/best_sac_kan.zip", flush=True)

    print(f"\n--- {env_name} Summary ---", flush=True)
    if best:
        print(f"  Best: {best['arch']} (actor={best['actor_params']} params, avg={best['avg']})", flush=True)
    else:
        print(f"  No architecture passed target={target}", flush=True)

    return results, best


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=list(ENVS.keys()))
    parser.add_argument("--output", default="quick_wins_results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    all_results = []
    summary = {}

    for env_name in args.envs:
        if env_name not in ENVS:
            print(f"Unknown: {env_name}")
            continue
        results, best = run_env(env_name, args.output)
        all_results.extend(results)
        summary[env_name] = best

        # Save incrementally
        with open(os.path.join(args.output, "results.json"), "w") as f:
            json.dump(all_results, f, indent=2)

    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nAll results saved to {args.output}/", flush=True)


if __name__ == "__main__":
    main()
