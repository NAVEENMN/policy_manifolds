"""Search for the smallest KAN network that can solve CartPole."""

from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from train_kan_ppo import KANActorCriticPolicy, KANMlpExtractor


def make_policy_class(hidden_dim, latent_dim, grid_size=5):
    """Create a policy class with the given architecture."""

    class SmallKANMlpExtractor(KANMlpExtractor):
        def __init__(self, feature_dim, **kwargs):
            super().__init__(
                feature_dim,
                last_layer_dim_pi=latent_dim,
                last_layer_dim_vf=latent_dim,
                grid_size=grid_size,
            )
            # Rebuild with smaller hidden dim
            from inkan import KANLayer
            from torch import nn

            self.policy_net = nn.Sequential(
                KANLayer(feature_dim, hidden_dim, grid_size=grid_size),
                KANLayer(hidden_dim, latent_dim, grid_size=grid_size),
            )
            self.value_net = nn.Sequential(
                KANLayer(feature_dim, hidden_dim, grid_size=grid_size),
                KANLayer(hidden_dim, latent_dim, grid_size=grid_size),
            )

    class SmallKANPolicy(KANActorCriticPolicy):
        def _build_mlp_extractor(self):
            self.mlp_extractor = SmallKANMlpExtractor(self.features_dim)

    return SmallKANPolicy


# Architectures to try: (hidden_dim, latent_dim)
configs = [
    (32, 32),
    (16, 16),
    (8, 8),
    (4, 4),
    (2, 2),
]

env = gym.make("CartPole-v1")

for hidden, latent in configs:
    edges = 4 * hidden + hidden * latent  # per network
    total_edges = edges * 2
    print(f"\n{'='*50}")
    print(f"Architecture: 4 -> {hidden} -> {latent}  ({total_edges} KAN edges total)")
    print(f"{'='*50}")

    policy_cls = make_policy_class(hidden, latent)
    model = PPO(
        policy_cls, env, verbose=0,
        learning_rate=3e-4, n_steps=2048, batch_size=64,
        n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
    )
    model.learn(total_timesteps=100_000)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
    print(f"Result: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    if mean_reward >= 499:
        save_path = Path(__file__).resolve().parent / "models/CartPole-v1" / f"ppo_kan_cartpole_{hidden}x{latent}.zip"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        print(f"Saved: {save_path}")

env.close()
