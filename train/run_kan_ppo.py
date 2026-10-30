"""Load saved KAN-PPO model and run CartPole with visual rendering."""

from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO

# Must import so the custom policy class is available for deserialization
from train_kan_ppo import KANActorCriticPolicy

env = gym.make("CartPole-v1", render_mode="human")
model = PPO.load(Path(__file__).resolve().parent / "models/CartPole-v1/ppo_kan_cartpole.zip", env=env)

print("Running CartPole with trained KAN-PPO agent...")
for episode in range(5):
    obs, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Episode {episode + 1}: reward = {total_reward:.0f}")

env.close()
print("Done.")
