from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'models'
WEB = ROOT / 'web/dist'
ENVIRONMENTS = {
    'CartPole-v1': {'algorithm': 'PPO', 'widths': [2, 2], 'grid': 5, 'steps': 100_000, 'target': 499., 'file': 'model.json'},
    'Pendulum-v1': {'algorithm': 'SAC', 'widths': [1, 1], 'grid': 2, 'steps': 100_000, 'target': -200., 'file': 'pendulum.json'},
    'Acrobot-v1': {'algorithm': 'PPO', 'widths': [2, 2], 'grid': 5, 'steps': 200_000, 'target': -100., 'file': 'acrobot.json'},
    'MountainCar-v0': {'algorithm': 'PPO', 'widths': [4, 4], 'grid': 5, 'steps': 300_000, 'target': -110., 'file': 'mountaincar.json'},
    'InvertedPendulum-v5': {'algorithm': 'SAC', 'widths': [1], 'grid': 1, 'steps': 200_000, 'target': 950., 'file': 'invertedpendulum.json', 'research_only': True},
}

# Observation dimensions come from the collected Gymnasium v5 environment
# metadata (Ant includes contact forces; Humanoid excludes unused fields).
SIMULATIONS = {
    'lunarlander': ('LunarLander-v3', 8, 4, 'PPO', 1000, 200.),
    'halfcheetah': ('HalfCheetah-v5', 17, 6, 'SAC', 1000, 4800.),
    'hopper': ('Hopper-v5', 11, 3, 'SAC', 1000, 3800.),
    'walker2d': ('Walker2d-v5', 17, 6, 'SAC', 1000, 4000.),
    'ant': ('Ant-v5', 105, 8, 'SAC', 1000, 6000.),
    'humanoid': ('Humanoid-v5', 348, 17, 'SAC', 1000, 6000.),
    'swimmer': ('Swimmer-v5', 8, 2, 'SAC', 1000, 360.),
    'reacher': ('Reacher-v5', 10, 2, 'SAC', 50, -3.75),
    'inverteddoublependulum': ('InvertedDoublePendulum-v5', 9, 1, 'SAC', 1000, 9100.),
}
for key, (name, inputs, actions, algorithm, limit, target) in SIMULATIONS.items():
    ENVIRONMENTS[name] = {'algorithm': algorithm, 'file': key + '.json',
                          'target': target, 'research_only': True}
