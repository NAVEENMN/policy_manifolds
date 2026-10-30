# Policy Manifolds

Edit the edges of a small KAN neural network and see how the agent’s behavior changes.

The app runs saved policies in the browser, with local MuJoCo physics for
InvertedPendulum and browser physics for the Classic Control demos. Drag
spline control points, compare with the original policy, and switch environments
without losing the edits made in the current browser session. MuJoCo and Box2D scenes project real simulator geometry into the canvas. Networks with more than 16 inputs have an input-group selector; vector actions have one meter per actuator.

![Policy Manifolds](docs/images/policy-manifolds.png)

## Run the app

From this repository:

```bash
python3 app.py
```

Open **http://127.0.0.1:7860**. The bundled actor exports let the app run with
Python’s standard library; training dependencies are not needed just to view it.
Use `--port 7861` if another app already uses 7860.

Four Classic Control, LunarLander, and all nine MuJoCo environments have saved actors included (14 selectable environments).
See the [model inventory](models/README.md) for architectures, algorithms, and
evaluation records. Acrobot and MountainCar use the saved PPO checkpoints;
their actor exports load automatically when selected.
The dropdown lists all 18 planned environments; the remaining four are disabled. Saved evaluation badges distinguish target-passing, below-target, and incomplete runs. InvertedDoublePendulum is the newly collected target-passing MuJoCo model; the other new MuJoCo actors remain useful for exploring interventions but are below their targets.

The InvertedPendulum-v5 model (width 1, grid 1, 22 actor parameters) is also
included with its normalization snapshot and collected evaluation records:
1000 return on 100 held-out episodes for each of three training seeds.
See [its model notes](models/InvertedPendulum-v5/README.md). Select it from the
MuJoCo group to inspect its four edges and edit its policy. Live simulation
requires MuJoCo in the Python environment running the server:

```bash
python -m pip install -r requirements-mujoco.txt
python app.py
```

Restart an already running server after updating the source. The network and
curves remain available if MuJoCo is missing, with a connection message and a
Retry button. MuJoCo loads only when this environment is selected. It runs
headlessly; the browser renders the cart, pole, original ghost, and telemetry.
The demo loads Gymnasium's saved XML directly into MuJoCo, avoiding its optional
video-rendering imports. `imageio` is not needed for this browser demo. The
selected environment stays in the URL when you refresh; open
`http://127.0.0.1:7860/#invertedpendulum` to go directly to InvertedPendulum.

Only the selected model loads initially. Loaded models are cached, and only the
active environment plus its original-policy ghost simulate. Edits are temporary
unless exported; they never overwrite the trained checkpoint.

## Install training and validation dependencies

Python 3.11 is recommended. The local code was also checked on Python 3.9.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install 'swig==4.*'
python -m pip install -r requirements.txt
```

`requirements.txt` includes all app, training, validation, Classic Control,
Box2D, and MuJoCo packages. `requriment.txt` is an alias for the same list.
SWIG must be installed before pip builds Box2D. A C++ compiler is also needed
(macOS: Xcode Command Line Tools). The helper `scripts/install_dependencies.sh`
runs the two install steps and `pip check`; set `PYTHON` to your virtualenv Python.
Node.js is needed only for browser-engine tests, not to run the app.

For a smaller installation, `requirements-core.txt` supports Classic Control;
the following files add only the chosen simulator family:

```bash
# Box2D requires a C++ compiler and SWIG.
python -m pip install swig
python -m pip install -r requirements-box2d.txt

python -m pip install -r requirements-mujoco.txt
```

## Train → validate → install

To train independent single-layer policies in parallel and merge their fixed-basis
prediction statistics, see the [policy-merging experiment](docs/policy_merging.md).
It includes nohup workers, per-teacher learning curves, checkpoint preservation,
and paired teacher-versus-merged evaluation.

For the hard MuJoCo tasks, an optional actor combines 1D edges with native
InKAN 2D joint-position/velocity surfaces. See the [2D experiment guide](docs/pairwise_kan.md)
for wider 1D controls, nohup launch commands, saved surface coefficients, and
CLI validation. The browser's surface editor remains future work.

Train only the architecture you want to investigate:

```bash
python train.py --env MountainCar-v0 --arch 4,4 --grid 5 --steps 300000 --seeds 0 1 2 3 4
python train.py --env Acrobot-v1 --arch 2 --grid 5 --steps 200000 --seeds 0 1 2 3 4
python train.py --env Pendulum-v1 --arch 1,1 --grid 2 --steps 100000
```

Every seed is saved under `models/<env>/runs/<timestamp>/seed-<n>/`:

- `best_model.zip` and `last_model.zip`: full SB3 checkpoints.
- `actor.json`: browser-ready KAN actor with preprocessing.
- `learning_curve.csv` and `learning_curve.png`: validation progress.
- `config.json`, `best_validation.json`, `result.json`: settings and evaluations.

The run directory also contains `results.csv` and `results.json` across seeds.
Checkpoints are saved before evaluations and on handled interruptions. Previous
run directories are never silently reused. SAC replay buffers are optional via
`--save-replay` and are excluded from Git.

Evaluate the exported actor on explicit fresh reset seeds:

```bash
python validate.py --env MountainCar-v0 --actor models/MountainCar-v0/runs/RUN/seed-0/actor.json --episodes 100
python validate.py --env Pendulum-v1 --episodes 1000 --seed 100000
```

Replace `RUN` with the timestamp printed by training. Validation writes JSON
and a CSV summary under `results/`. Calling `python validate.py` evaluates every
installed model and reports missing ones.

Install a chosen run for the demo, without retraining:

```bash
python export.py --env MountainCar-v0 --source models/MountainCar-v0/runs/RUN/seed-0/best_model.zip --install
python app.py
```

Installing a model does not assert that the environment is solved. Training can
also promote automatically with `--promote`: every configured seed must pass
the held-out score criterion, and Pendulum must also satisfy the stability
criterion. The deployed seed is chosen by validation return. `--target` makes
any custom criterion explicit; a full search is still needed to compare sizes.

For normalized batch-search runs, use the matching exported actor and checkpoint:

```bash
python export.py --env Acrobot-v1 --source /path/to/best_actor.json --checkpoint /path/to/best_model.zip --install
```

The normalized JSON includes statistics stored outside an SB3 ZIP. Supplying
the matching JSON preserves them. Raw ZIP export supports the default cubic
InKAN grid, SiLU residuals, and unnormalized observations for discrete policies.

## Project layout

```text
app.py                 Local website server
train.py               Single-architecture training, checkpointing, CSVs and curves
validate.py            Reproducible actor evaluation
export.py              Checkpoint export and model installation
policy_manifolds/      Policies, training, evaluation and export implementations
models/<env>/          Selected model.zip, actor.json and metadata.json
web/dist/              Buildless HTML, CSS, JS and generated browser actor copies
web/tests/             Torch/Gymnasium comparisons for browser inference and physics
kan_search/            Parallel architecture/seed search pipeline
scripts/               Batch launchers, setup, checks and source packaging
tests/                 Checkpoint/export/training tests
docs/                  Model and environment details
```

`pendulum_kan.py` and `train_kan_ppo.py` are import-only compatibility shims for
the original checkpoint class names. Importing them does not start training.

## Batch searches

The batch pipeline keeps CartPole and Pendulum as existing references instead
of retraining them by default. It saves every trial, normalization statistics,
selected actors, CSV reports, and learning curves.

```bash
python scripts/search_kan.py --dry-run --output results/classic-plan
python scripts/search_mujoco.py --dry-run --output results/mujoco-plan

# On a training machine; launch in nohup-backed background processes.
bash scripts/setup_search.sh
bash scripts/run_search.sh

bash scripts/setup_mujoco_search.sh
bash scripts/run_mujoco_search.sh
```

Defaults launch seven remaining Classic Control/Box2D searches, or nine MuJoCo
searches. Limit the machines’ workload with `--envs`, `--jobs`, and `--n-envs`.
Use a new output directory for a new campaign, or `--resume` for an existing
campaign. See `kan_search/README.md` and `kan_search/README_MUJOCO.md` for details.

## Checks and packaging

```bash
python -m unittest discover -s tests -v
python -m unittest kan_search.tests.test_pipeline kan_search.tests.test_mujoco -v
python scripts/check_web.py       # Node.js must be on PATH, or set NODE_BINARY
python app.py --export-only
python web/build_offline.py
python scripts/package_project.py
```

The package command writes `artifacts/policy-manifolds.tar.gz` from a deliberate
source allowlist. It includes the selected small checkpoints, web app, tests,
docs and scripts. It excludes virtual environments, `.git` metadata, machine
settings, secrets, generated runs, and replay buffers. The offline bundle is
`web/policy-manifolds.html`. The offline file supports curve inspection for all installed actors; MuJoCo and Box2D simulation and episode comparison need the local Python server.

The selected checkpoints are small enough for ordinary Git (about 25 MB
combined), so Git LFS is not required. `.gitignore` keeps generated training
runs out of commits. No remote repository or credentials are configured.

The browser reset RNG differs from NumPy’s Gymnasium RNG. Numerical comparisons
use the same explicit initial states; nearly tied actions or long chaotic
trajectories can diverge because JavaScript and Torch round differently.
