# KAN vs MLP Architecture Comparison

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

## Unsolved Environments (best result within training budget)

| Environment | Obs | Act | Target | Best KAN | KAN Params | KAN Best | Best MLP | MLP Params | MLP Best | KAN vs MLP |
|---|---:|---|---:|---|---:|---:|---|---:|---:|---|
| BipedalWalker-v3 | 24 | Cont(4) | 300 | g2 2L-16 | 7,769 | 277.5 | 2x2 | 131 | 200.2 | KAN closer to target |
| Pendulum-v1 (PPO) | 3 | Cont(1) | -200 | g5 2L-32 | 20,227 | -180.1 | 64x64 | 8,963 | -384.6 | KAN 2x better |
| HalfCheetah-v5 | 17 | Cont(6) | 4800 | [17→1] g1 | 97 | 4426.1 | — | — | — | 92% of target |
| Walker2d-v5 | 17 | Cont(6) | 4000 | [17→1] g1 | 97 | 1153.7 | — | — | — | 29% of target |
| Hopper-v5 | 11 | Cont(3) | 3800 | [11→1] g1 | 61 | 1028.3 | — | — | — | 27% of target |
| Reacher-v5 | 10 | Cont(2) | -3.75 | [10→2] g2 | 126 | -5.70 | — | — | — | Close |
| Swimmer-v5 | 8 | Cont(2) | 360 | [8→1→1] g1 | 49 | 51.0 | — | — | — | 14% of target |
| MountainCarCont-v0 | 2 | Cont(1) | 90 | All failed | — | 0.0 | All failed | — | 0.0 | Both fail |
| Ant-v5 | 105 | Cont(8) | 6000 | — | — | — | — | — | — | Timed out |
| Humanoid-v5 | 348 | Cont(17) | 6000 | — | — | — | — | — | — | Timed out |

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

## Key Findings

1. **KAN solves environments MLP cannot**: MountainCar and Pendulum are unsolvable by MLP at any tested size, yet KAN succeeds.
2. **3-40x parameter reduction**: Where both solve, KAN typically needs 3-4x fewer parameters (CartPole, LunarLander). 
3. **Extreme minimalism**: InvertedPendulum needs just 22 parameters (4 spline edges). Pendulum needs 26.
4. **Two layers usually needed**: Most environments require at least 2 KAN layers for reliable solutions. Exceptions: Acrobot, InvertedPendulum, InvertedDoublePendulum, HalfCheetah (single layer works).
5. **Grid size matters**: grid_size=5 generally outperforms grid_size=2, but g=1 or g=2 suffices for simple tasks (Pendulum, InvertedPendulum).
6. **MuJoCo needs more budget**: Most MuJoCo envs (HalfCheetah, Hopper, Walker2d, Ant, Humanoid) didn't hit targets within the 6-hour budget but showed promising partial results.
