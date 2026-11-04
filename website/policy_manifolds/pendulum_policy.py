"""Small InKAN actors for Pendulum; the training critics are separate MLPs."""
import numpy as np
import torch
from torch import nn
from inkan import KANLayer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import Actor, SACPolicy


class PendulumFeatures(BaseFeaturesExtractor):
    """Fixed physical scaling, included in the saved policy (no fitted statistics)."""
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=3)
        self.register_buffer('scale', torch.tensor([1., 1., 8.]))

    def forward(self, observations):
        return observations / self.scale


class KANActor(Actor):
    def __init__(self, *args, grid_size=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_size = grid_size
        dimensions = [self.features_dim] + list(self.net_arch)
        self.latent_pi = nn.Sequential(*[
            KANLayer(a, b, grid_size=grid_size, compile_basis=False)
            for a, b in zip(dimensions[:-1], dimensions[1:])
        ])

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(), 'grid_size': self.grid_size}


class KANSACPolicy(SACPolicy):
    def __init__(self, *args, grid_size=5, **kwargs):
        self.grid_size = grid_size
        kwargs.setdefault('features_extractor_class', PendulumFeatures)
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor=None):
        kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return KANActor(**kwargs, grid_size=self.grid_size).to(self.device)

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(), 'grid_size': self.grid_size}


def parameter_counts(model):
    count = lambda module: sum(p.numel() for p in module.parameters())
    return {'actor_deterministic': count(model.actor.latent_pi) + count(model.actor.mu),
            'actor_training': count(model.actor), 'critics_training': count(model.critic),
            'critic_targets': count(model.critic_target)}


def evaluate(model, seeds, batch_size=32):
    """Evaluate deterministic torque on explicit, reproducible Gymnasium resets."""
    import gymnasium as gym
    seeds = list(seeds)
    returns, stable_fractions, end_angles = [], [], []
    for start in range(0, len(seeds), batch_size):
        batch = seeds[start:start + batch_size]
        env = gym.vector.SyncVectorEnv([lambda: gym.make('Pendulum-v1') for _ in batch])
        obs, _ = env.reset(seed=batch)
        rewards = np.zeros(len(batch))
        stable = np.zeros(len(batch))
        for t in range(200):
            actions, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(actions)
            rewards += reward
            angle = np.arctan2(obs[:, 1], obs[:, 0])
            if t >= 150:
                stable += (np.abs(angle) < np.deg2rad(15)) & (np.abs(obs[:, 2]) < 1.)
        returns.extend(rewards.tolist())
        stable_fractions.extend((stable / 50).tolist())
        end_angles.extend(np.rad2deg(angle).tolist())
        env.close()
    successes = [fraction >= .9 for fraction in stable_fractions]
    return {'episodes': len(seeds), 'seed_start': seeds[0], 'seed_end': seeds[-1],
            'mean_reward': float(np.mean(returns)), 'std_reward': float(np.std(returns)),
            'worst_reward': float(np.min(returns)), 'best_reward': float(np.max(returns)),
            'success_rate': float(np.mean(successes)),
            'mean_final50_upright_fraction': float(np.mean(stable_fractions)),
            'returns': returns, 'final50_upright_fractions': stable_fractions,
            'end_angles_degrees': end_angles}
