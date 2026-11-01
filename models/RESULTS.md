# KAN vs MLP Architecture Comparison

For the current six width-2 demo checkpoints, exact architectures, and selection caveats, see [the model inventory](README.md). Their scores below are selection-validation means, not held-out test results. Older comparison rows describe historical experiments, not necessarily the checkpoint loaded by the app.

## Solved Environments

| Environment | Obs | Act | Target | Smallest KAN | KAN Arch | Grid | KAN Params | KAN Score | Smallest MLP | MLP Params | MLP Score | KAN Advantage |
|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|---|
| InvertedPendulum-v5 | 4 | Cont(1) | 950 | **[4→1]** | 1L | 1 | **22** | 1000.0 | — | — | — | Tiniest possible |
| Pendulum-v1 (SAC) | 3 | Cont(1) | -200 | **[3→1→1]** | 2L | 2 | **26** | -184.8 | None solved | 8,963 | -295.1 | **∞** |
| InvertedDoublePendulum-v5 | 9 | Cont(1) | 9100 | **[9→1]** | 1L | 1 | **47** | 9359.9 | — | — | — | Tiniest possible |
| Acrobot-v1 | 6 | Disc(3) | -100 | **[6→2]** | 1L | 2 | **156** | -84.1 | [6→2→2] | 52 | -88.5 | MLP competitive |
| CartPole-v1 | 4 | Disc(2) | 500 | **[4→2→2]** | 2L | 5 | **225** | 500.0 | [4→16→16] | 755 | 500.0 | **3.4x fewer** |
| MountainCar-v0 | 2 | Disc(3) | -110 | **[2→8→8]** | 2L | 5 | **1,476** | -100.6 | None solved | 8,964 | -192.5 | **∞** |
| LunarLander-v3 | 8 | Disc(4) | 200 | **[8→8→8]** | 2L | 5 | **2,349** | 268.7 | [8→64→64] | 9,797 | 259.8 | **4.2x fewer** |

## Remaining targets and single-seed results (best result within training budget)

| Environment | Obs | Act | Target | Best KAN | KAN Params | KAN Best | Best MLP | MLP Params | MLP Best | KAN vs MLP |
|---|---:|---|---:|---|---:|---:|---|---:|---:|---|
| BipedalWalker-v3 | 24 | Cont(4) | 300 | g2 2L-16 | 7,769 | 277.5 | 2x2 | 131 | 200.2 | KAN closer to target |
| Pendulum-v1 (PPO) | 3 | Cont(1) | -200 | g5 2L-32 | 20,227 | -180.1 | 64x64 | 8,963 | -384.6 | KAN 2x better |
| HalfCheetah-v5 | 17 | Cont(6) | 4800 | [17→1] g1 | 97 | 4426.1 | — | — | — | 92% of target |
| Walker2d-v5 | 17 | Cont(6) | 4000 | [17→2] g5 | 324 | 4,204.46 | — | — | — | One training seed passes; 1/3 at this grid |
| Hopper-v5 | 11 | Cont(3) | 3800 | [11→2] g2 | 141 | 3,428.24 | — | — | — | Below target |
| Reacher-v5 | 10 | Cont(2) | -3.75 | [10→2] g1 | 106 | -5.75 | — | — | — | Below target |
| Swimmer-v5 | 8 | Cont(2) | 360 | [8→2] g5 | 150 | 116.32 | — | — | — | Below target |
| MountainCarCont-v0 | 2 | Cont(1) | 90 | All failed | — | 0.0 | All failed | — | 0.0 | Both fail |
| Ant-v5 | 105 | Cont(8) | 6000 | [105→2] g2 | 1,284 | 2,959.28 | — | — | — | Below target |
| Humanoid-v5 | 348 | Cont(17) | 6000 | [348→2] g1 | 3,531 | 5,534.37 | — | — | — | Below target |

## KAN Edge & Parameter Counts

Each KAN edge learns a B-spline activation with `grid_size + 3` coefficients plus 1 residual weight.

| Model | Architecture | Grid | KAN Edges | Spline Params | Base Params | Total Actor Params |
|---|---|---:|---:|---:|---:|---:|
| InvertedPendulum-v5 | 4→1 | 1 | 4 | 16 | 4 | 22 |
| Pendulum-v1 (SAC) | 3→1→1 | 2 | 4 | 20 | 4 | 26 |
| InvertedDoublePendulum-v5 | 9→1 | 1 | 9 | 36 | 9 | 47 |
| Acrobot-v1 | 6→2 | 2 | 12 | 60 | 12 | 156 |
| CartPole-v1 | 4→2→2 | 5 | 12 | 96 | 12 | 225 |
| MountainCar-v0 | 2→8→8 | 5 | 80 | 640 | 80 | 1,476 |
| LunarLander-v3 | 8→8→8 | 5 | 128 | 1,024 | 128 | 2,349 |

## Interpretation

These are project-reported experiments with differing settings. A failed tested MLP configuration does not show that MLPs cannot solve an environment. Counts in historical tables may include both actor and critic; use [the selected model inventory](README.md) and exported actor tensors for current deployment parameter counts.

The width-2 MuJoCo demos have one KAN layer plus a linear action head. Walker2d's best seed passes, but only one of three seeds at that grid passes; the other five environments remain below their targets. No claim of global minimality or general KAN superiority follows from these results.
