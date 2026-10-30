"""Compare minimal KAN architectures vs MLP baselines on CartPole."""

import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from train_kan_ppo import KANActorCriticPolicy, KANMlpExtractor
from inkan import KANLayer


# --- KAN policy factories ---

def make_kan_single_layer(latent_dim, grid_size=5):
    """Single KAN layer: obs(4) -> latent_dim, no hidden layer."""

    class SingleKANExtractor(nn.Module):
        def __init__(self, feature_dim):
            super().__init__()
            self.latent_dim_pi = latent_dim
            self.latent_dim_vf = latent_dim
            self.policy_net = KANLayer(feature_dim, latent_dim, grid_size=grid_size)
            self.value_net = KANLayer(feature_dim, latent_dim, grid_size=grid_size)

        def forward(self, features):
            return self.forward_actor(features), self.forward_critic(features)

        def forward_actor(self, features):
            return self.policy_net(features)

        def forward_critic(self, features):
            return self.value_net(features)

    class Policy(KANActorCriticPolicy):
        def _build_mlp_extractor(self):
            self.mlp_extractor = SingleKANExtractor(self.features_dim)

    return Policy


def make_kan_two_layer(hidden, latent, grid_size=5):
    """Two KAN layers: obs(4) -> hidden -> latent."""

    class TwoKANExtractor(nn.Module):
        def __init__(self, feature_dim):
            super().__init__()
            self.latent_dim_pi = latent
            self.latent_dim_vf = latent
            self.policy_net = nn.Sequential(
                KANLayer(feature_dim, hidden, grid_size=grid_size),
                KANLayer(hidden, latent, grid_size=grid_size),
            )
            self.value_net = nn.Sequential(
                KANLayer(feature_dim, hidden, grid_size=grid_size),
                KANLayer(hidden, latent, grid_size=grid_size),
            )

        def forward(self, features):
            return self.forward_actor(features), self.forward_critic(features)

        def forward_actor(self, features):
            return self.policy_net(features)

        def forward_critic(self, features):
            return self.value_net(features)

    class Policy(KANActorCriticPolicy):
        def _build_mlp_extractor(self):
            self.mlp_extractor = TwoKANExtractor(self.features_dim)

    return Policy


# --- MLP policy factories ---

def make_mlp_policy(net_arch, activation_fn=nn.Tanh):
    """Standard MLP via SB3's built-in net_arch."""
    return dict(net_arch=net_arch, activation_fn=activation_fn)


def count_params(model):
    return sum(p.numel() for p in model.policy.parameters() if p.requires_grad)


# --- Configs ---

configs = [
    # (name, policy_class_or_str, policy_kwargs, description)

    # Single KAN layer experiments
    ("KAN 4->2 (1 layer)",        make_kan_single_layer(2),   {},  "8 KAN edges"),
    ("KAN 4->4 (1 layer)",        make_kan_single_layer(4),   {},  "16 KAN edges"),

    # Two KAN layer experiments
    ("KAN 4->2->2 (2 layers)",    make_kan_two_layer(2, 2),   {},  "24 KAN edges"),

    # MLP baselines with comparable or larger sizes
    ("MLP 4->2->2",     "MlpPolicy", make_mlp_policy([2, 2]),        "tiny MLP"),
    ("MLP 4->4->4",     "MlpPolicy", make_mlp_policy([4, 4]),        "small MLP"),
    ("MLP 4->8->8",     "MlpPolicy", make_mlp_policy([8, 8]),        "medium MLP"),
    ("MLP 4->32->32",   "MlpPolicy", make_mlp_policy([32, 32]),      "large MLP"),
    ("MLP 4->64->64",   "MlpPolicy", make_mlp_policy([64, 64]),      "default SB3"),
]

SEEDS = [0, 1, 2]
TIMESTEPS = 100_000

env = gym.make("CartPole-v1")
results = []

for name, policy, kwargs, desc in configs:
    rewards = []
    params = None
    for seed in SEEDS:
        model = PPO(
            policy, env, verbose=0, seed=seed,
            learning_rate=3e-4, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            policy_kwargs=kwargs,
        )
        if params is None:
            params = count_params(model)
        model.learn(total_timesteps=TIMESTEPS)
        mean_r, _ = evaluate_policy(model, env, n_eval_episodes=20)
        rewards.append(mean_r)

    avg = sum(rewards) / len(rewards)
    worst = min(rewards)
    results.append((name, desc, params, rewards, avg, worst))
    print(f"{name:30s}  params={params:6d}  seeds={[f'{r:.0f}' for r in rewards]}  avg={avg:.1f}  worst={worst:.0f}")

env.close()

print(f"\n{'='*80}")
print(f"{'Name':30s}  {'Params':>7s}  {'Seed 0':>7s}  {'Seed 1':>7s}  {'Seed 2':>7s}  {'Avg':>7s}  {'Worst':>7s}")
print(f"{'-'*80}")
for name, desc, params, rewards, avg, worst in results:
    print(f"{name:30s}  {params:7d}  {rewards[0]:7.0f}  {rewards[1]:7.0f}  {rewards[2]:7.0f}  {avg:7.1f}  {worst:7.0f}")
