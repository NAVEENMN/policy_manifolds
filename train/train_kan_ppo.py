"""
PPO on CartPole-v1 with InKAN layers replacing the default MLP.

KANLayer is a drop-in replacement for nn.Linear that learns nonlinear
B-spline activation functions on each edge, so no explicit activation
functions are needed between layers.
"""

from typing import Callable, Tuple
from pathlib import Path

import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy

from inkan import KANLayer


class KANMlpExtractor(nn.Module):
    """Replaces the default MlpExtractor with InKAN layers."""

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
        grid_size: int = 5,
    ):
        super().__init__()

        # SB3 reads these to size action_net and value_net
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy (actor) network: KAN layers instead of Linear+ReLU
        self.policy_net = nn.Sequential(
            KANLayer(feature_dim, 64, grid_size=grid_size),
            KANLayer(64, last_layer_dim_pi, grid_size=grid_size),
        )

        # Value (critic) network
        self.value_net = nn.Sequential(
            KANLayer(feature_dim, 64, grid_size=grid_size),
            KANLayer(64, last_layer_dim_vf, grid_size=grid_size),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class KANActorCriticPolicy(ActorCriticPolicy):
    """ActorCriticPolicy that uses KAN layers instead of MLP."""

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal init (not meaningful for spline weights)
        kwargs["ortho_init"] = False
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = KANMlpExtractor(self.features_dim)


def main():
    env = gym.make("CartPole-v1")

    model = PPO(
        KANActorCriticPolicy,
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )

    print("Training PPO with InKAN layers on CartPole-v1...")
    model.learn(total_timesteps=100_000)

    # Evaluate
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
    print(f"Evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # Save
    save_path = Path(__file__).resolve().parent / "models/CartPole-v1/ppo_kan_cartpole"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
