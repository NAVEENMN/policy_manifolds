# Policy Manifolds

Interpretable reinforcement learning with minimal Kolmogorov-Arnold Network (KAN) policies.

KAN policies replace standard MLP layers with learnable B-spline activations on each edge. The result: controllers with 22 to 8,437 parameters that solve standard benchmarks, versus the 5,000 to 160,000 parameters typical of MLP policies.

**Live demo**: [policymanifolds.com](https://policymanifolds.com)

## Results

| Environment | KAN Params | Score | Target | MLP Params | Reduction |
|---|---:|---:|---:|---:|---:|
| InvertedPendulum-v5 | 22 | 1,000 | 950 | - | - |
| Pendulum-v1 | 26 | -185 | -200 | 8,963 (fails) | - |
| InvertedDblPendulum-v5 | 47 | 9,360 | 9,100 | - | - |
| Acrobot-v1 | 81 | -84 | -100 | 29 | 0.4x |
| HalfCheetah-v5 | 97 | 7,582 | 4,800 | 71,942 | 742x |
| CartPole-v1 | 114 | 500 | 500 | 386 | 3.4x |
| MountainCar-v0 | 747 | -101 | -110 | 8,964 (fails) | - |
| LunarLander-v3 | 1,188 | 269 | 200 | 4,996 | 4.2x |
| Humanoid-v5 | 8,437 | 6,194 | 6,000 | 159,505 | 19x |

## Quick Start

```bash
pip install inkan stable-baselines3 gymnasium
python train/train_kan_ppo.py
```

This trains a 114-parameter KAN policy on CartPole and saves the model. The policy achieves a perfect 500/500 score.

## Train Any Environment

Each config in `configs/` specifies the architecture, grid size, and hyperparameters used to produce the results above.

```bash
# CartPole (PPO, 100K steps, ~1 min)
python train/train_kan_ppo.py

# Pendulum (SAC, 200K steps, ~5 min)
python train/train_pendulum.py

# Acrobot (PPO, 500K steps, ~10 min)
python train/train_acrobot.py

# MountainCar (PPO, 1M steps, ~20 min)
python train/train_mountaincar.py
```

MuJoCo environments use SAC with an MLP critic (discarded after training):

```bash
pip install gymnasium[mujoco]
# See configs/ for architecture and hyperparameters
```

## How It Works

A standard MLP policy has fixed activations (ReLU, Tanh) shared across all neurons. A KAN policy places a learnable B-spline activation on each edge:

```
MLP:  y = ReLU(W @ x + b)           # fixed activation, learned weights
KAN:  y = sum_i spline_i(x_i)       # learned activation per edge
```

Each spline has `grid_size + 3` cubic B-spline coefficients plus a residual weight. A KAN layer with 4 inputs and 2 outputs has 12 edges, each learning its own nonlinear function. The entire policy is small enough to visualize and interpret.

## Repository Structure

```
configs/          Environment configs (architecture, hyperparameters, results)
train/            Training scripts
models/           Saved model checkpoints
website/          Interactive web demo (policymanifolds.com)
deployment/       Terraform infrastructure
```

## Requirements

```
inkan==0.5.2
stable-baselines3>=2.7
gymnasium>=1.1
torch>=2.8
```

## Citation

If you use this work, please cite:

```bibtex
@software{policymanifolds2026,
  author = {Mysore, Naveen},
  title = {Policy Manifolds: Interpretable RL with Minimal KAN Policies},
  url = {https://policymanifolds.com},
  year = {2026}
}
```

## Related Work

- [InKAN](https://pypi.org/project/inkan/) - B-spline KAN implementation used in this project
- [Exact Spline Merging](https://arxiv.org/abs/2609.16579) - Distributed aggregation of spline models

## Author

[Naveen Mysore](https://naveenmn.github.io/) - University of California, Santa Barbara
