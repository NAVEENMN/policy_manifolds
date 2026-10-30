"""Find smallest KAN network for MountainCar-v0 and compare with MLP."""

import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from inkan import KANLayer
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable


class PPOBase(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *a, **kw):
        kw["ortho_init"] = False
        super().__init__(observation_space, action_space, lr_schedule, *a, **kw)


def make_kan_single(latent, grid_size=5):
    class Ext(nn.Module):
        def __init__(self, fd):
            super().__init__()
            self.latent_dim_pi = latent
            self.latent_dim_vf = latent
            self.policy_net = KANLayer(fd, latent, grid_size=grid_size)
            self.value_net = KANLayer(fd, latent, grid_size=grid_size)
        def forward(self, f): return self.forward_actor(f), self.forward_critic(f)
        def forward_actor(self, f): return self.policy_net(f)
        def forward_critic(self, f): return self.value_net(f)
    class P(PPOBase):
        def _build_mlp_extractor(self): self.mlp_extractor = Ext(self.features_dim)
    return P


def make_kan_two(h, l, grid_size=5):
    class Ext(nn.Module):
        def __init__(self, fd):
            super().__init__()
            self.latent_dim_pi = l
            self.latent_dim_vf = l
            self.policy_net = nn.Sequential(KANLayer(fd, h, grid_size=grid_size), KANLayer(h, l, grid_size=grid_size))
            self.value_net = nn.Sequential(KANLayer(fd, h, grid_size=grid_size), KANLayer(h, l, grid_size=grid_size))
        def forward(self, f): return self.forward_actor(f), self.forward_critic(f)
        def forward_actor(self, f): return self.policy_net(f)
        def forward_critic(self, f): return self.value_net(f)
    class P(PPOBase):
        def _build_mlp_extractor(self): self.mlp_extractor = Ext(self.features_dim)
    return P


def count_params(model):
    return sum(p.numel() for p in model.policy.parameters() if p.requires_grad)


# MountainCar-v0: obs=2 (position, velocity), action=Discrete(3) (left, none, right)
# Reward: -1 per step, 0 at goal. Max 200 steps. Solved ~ -110.
# Hard exploration problem for PPO — needs tuning.

configs = [
    # KAN single layer
    ("KAN 2->2 (1L)",      make_kan_single(2),      {}),
    ("KAN 2->4 (1L)",      make_kan_single(4),      {}),
    ("KAN 2->8 (1L)",      make_kan_single(8),      {}),
    # KAN two layers
    ("KAN 2->2->2",        make_kan_two(2, 2),      {}),
    ("KAN 2->4->4",        make_kan_two(4, 4),      {}),
    ("KAN 2->8->8",        make_kan_two(8, 8),      {}),
    ("KAN 2->16->16",      make_kan_two(16, 16),    {}),
    # MLP baselines
    ("MLP 2->2->2",        "MlpPolicy", dict(net_arch=[2, 2], activation_fn=nn.Tanh)),
    ("MLP 2->8->8",        "MlpPolicy", dict(net_arch=[8, 8], activation_fn=nn.Tanh)),
    ("MLP 2->16->16",      "MlpPolicy", dict(net_arch=[16, 16], activation_fn=nn.Tanh)),
    ("MLP 2->32->32",      "MlpPolicy", dict(net_arch=[32, 32], activation_fn=nn.Tanh)),
    ("MLP 2->64->64",      "MlpPolicy", dict(net_arch=[64, 64], activation_fn=nn.Tanh)),
]

SEEDS = list(range(5))
TIMESTEPS = 300_000  # MountainCar needs more exploration

env = gym.make("MountainCar-v0")
results = []

for name, policy, kwargs in configs:
    rewards = []
    params = None
    for seed in SEEDS:
        model = PPO(
            policy, env, verbose=0, seed=seed,
            learning_rate=3e-4, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.01,  # encourage exploration
            policy_kwargs=kwargs,
        )
        if params is None:
            params = count_params(model)
        model.learn(total_timesteps=TIMESTEPS)
        mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=20)
        rewards.append(mean_r)

    avg = sum(rewards) / len(rewards)
    best = max(rewards)
    worst = min(rewards)
    results.append((name, params, rewards, avg, best, worst))
    print(f"{name:25s}  params={params:5d}  avg={avg:8.1f}  best={best:8.1f}  worst={worst:8.1f}  seeds={[f'{r:.0f}' for r in rewards]}")

env.close()

print(f"\n{'='*95}")
print(f"{'Name':25s}  {'Params':>6s}  {'Avg':>8s}  {'Best':>8s}  {'Worst':>8s}  Seeds")
print(f"{'-'*95}")
for name, params, rewards, avg, best, worst in results:
    print(f"{name:25s}  {params:6d}  {avg:8.1f}  {best:8.1f}  {worst:8.1f}  {[f'{r:.0f}' for r in rewards]}")

# Save best small KAN that solved it (avg > -160)
good_kans = [(n, p, a) for n, p, r, a, b, w in results if 'KAN' in n and a > -160]
if good_kans:
    best_kan = min(good_kans, key=lambda x: x[1])
    print(f"\nSmallest good KAN: {best_kan[0]} ({best_kan[1]} params, avg={best_kan[2]:.1f})")
