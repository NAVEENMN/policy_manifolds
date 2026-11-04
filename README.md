# Policy Manifolds

Edit the edges of a small KAN policy and see how an RL agent's behavior changes.
This checkout is the production **policy editor**: a network on the left, live
simulation in the center, and editable B-spline curves on the right.

Each KAN edge combines a cubic B-spline and a weighted SiLU residual. Nodes sum
their incoming edge outputs. A fixed linear head produces action logits or
continuous controls. Editing a curve changes the next action without retraining;
the original policy remains available as a comparison.

## Run locally

```bash
python3 app.py
```

Open **http://127.0.0.1:7860**. Python's standard library is sufficient to serve
the editor and run the four browser-based Classic Control demos. To run the
MuJoCo demos, install the simulator in the Python environment running the app:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mujoco.txt
python app.py
```

For all supported simulators, including LunarLander's Box2D physics:

```bash
python -m pip install 'swig==4.*'
python -m pip install -r requirements.txt
python app.py
```

Box2D also requires a C++ compiler. Python 3.11 is recommended. Restart the server
after updating files. Use `--port 7862` to run alongside another app. Dependencies
are loaded lazily; unavailable simulators show a Retry connection message while
curve editing remains available.

## 3D MuJoCo viewer

The nine MuJoCo demos render their existing simulator geometry in a lit 3D view
with shadows and the original-policy ghost. This is a visualization change:
physics, reset states, observations, rewards, action scaling, trained policies,
and curve editing use the existing implementations and saved models.

- Drag to orbit; scroll or pinch to zoom; Shift-drag or right-drag to pan.
- Use **Orbit**, **Side**, or **Top** for preset views; **↺ Camera** resets the view.
- **Follow** keeps the camera centered on the live agent.
- Focus the scene for keyboard controls: arrow keys rotate, `+`/`−` zoom, and
  `Home` resets the camera.

The renderer is bundled JavaScript/WebGL 2 and needs no new packages or CDN.
If WebGL 2 is unavailable, the original 2D view remains available. Classic
Control and Box2D retain their existing viewers. Physics still runs in the
Python server; rendering and editable policy inference run in the browser.

Renderer and physics-isolation checks:

```bash
node tests/test_scene_3d.mjs
python -m unittest discover -s tests -p test_geometry3d.py -v
```

## Current models

Four Classic Control, LunarLander, and nine MuJoCo actors are bundled (14 demos).
Four additional planned environments appear disabled. Only the selected actor
loads initially; loaded actors are cached in that browser session. Only the
active environment and its original-policy comparison run.

Hopper, Walker2d, Humanoid, Ant, Swimmer, and Reacher now use the selected width-2
1D KAN trials. These contain **one KAN layer followed by a linear action head**.
The displayed architecture is the KAN portion; a separate caption shows the
head dimensions and grid size. Actor parameter counts include the deterministic
head and exclude critics and the SAC exploration head.

Walker2d's selected checkpoint scored **4,204.46** on 100 selection-validation
episodes, with **1 of 3 training seeds passing at grid 5**. This is not reliable
solving across seeds. The other five updated actors remain below target.
See the [model inventory](models/README.md) for all six architectures, scores,
trial names, matching normalization, learning curves, and selection rules.

Edits stay in browser memory unless exported. They never overwrite saved model
files. Imported edits must match the checkpoint fingerprint. Switching
environments preserves edits until the page reloads. Networks with many inputs
have an input-group selector so every edge remains accessible.

## Select collected trials for the demo

The provided installer selects the highest `validation_mean_return` among
complete 1D trials with all required artifacts. It never selects on held-out
test scores. Back up existing selected models before promoting another collection:

```bash
python tools/install_demo_trials.py --results /path/to/results
python app.py --export-only
python app.py
```

Expected source layout:

```text
results/<environment>/trials/<trial>/
  best_model.zip
  best_actor.json
  best_normalization.pkl
  best_validation.json
  result.json
  learning_curve.csv
```

The selector saves the chosen checkpoint, actor, normalization, and records under
`models/<environment>/`, and updates the browser exports under `web_dist/`.
The six selected full checkpoints are supplied in the local update package;
ZIP and pickle files remain Git-ignored. Git contains browser actors with embedded
normalization, simulator metadata, CSVs, and provenance. Those JSON exports are
sufficient for browser playback and NumPy inference.

The source snapshot has 48 trial folders, 46 complete exports and two incomplete
runs. The [inventory CSV](models/width2_trials.csv) records every entry, including
why incomplete runs were excluded. `best_validation.json` records the 20-episode
checkpoint-tuning score; `result.json` records the separate 100-episode
selection-validation score shown by the website.

## Inference and validation

The deterministic inference pipeline is frozen observation normalization → fixed
scaling → KAN → linear head → tanh and saved action bounds (SAC), or argmax (PPO).
To evaluate a selected actor on explicit reset seeds without loading a training
checkpoint:

```bash
python - <<'PY'
from policy_manifolds.evaluation import load_actor, evaluate
actor = load_actor('models/Walker2d-v5/actor.json', 'Walker2d-v5')
result = evaluate(actor, 'Walker2d-v5', episodes=20, seed=100000)
print(result)
PY
```

This runs a new evaluation; it does not overwrite the saved selection score.
Small numerical differences between JavaScript, NumPy and Torch can accumulate
in long trajectories, so engine comparisons use shared observation states.

Check all bundled actors and edit effects with Python and Node.js:

```bash
python tests/check_demo_models.py
```

For checkpoint tensor, normalization, simulator XML, and native inference parity
checks, retain the local selected ZIP/pickle files and install the validation
libraries:

```bash
python -m pip install torch==2.8.0 inkan==0.5.2 stable-baselines3==2.7.1
python tests/check_demo_models.py --native --output validation.json
```

## Training scope and scientific interpretation

The `train/` folder contains historical experiment scripts. This web checkout
does **not** include the full `kan_search` campaign runner that produced the six
new trials, so it is not a complete training reproduction package. Training
those campaigns requires the original training workspace and configurations.
The selection and inference commands above operate on the collected artifacts
and are included here. The separate 2D actor and critic experiments are not part
of this policy-only UI.

A small policy meeting a task target does not establish superiority over MLPs
or simple controllers. Such comparisons need matched training budgets, multiple
training seeds, and performance-versus-size measurements. Editable curves are
useful interventions, not a complete mechanistic explanation. “Policy Manifolds”
is the project name, not a demonstrated claim about manifold geometry. Scores
are project-reported; independent reproduction is not documented.

## Layout

```text
app.py                      Local server
policy_manifolds/           Inference, export helpers and simulator adapters
web_dist/                   HTML, CSS, JavaScript and bundled browser actors
models/                     Selected actors, metadata, CSVs and local checkpoints
tools/install_demo_trials.py Selection from the collected trial folders
tests/check_demo_models.py   Artifact, inference and edit checks
train/                      Historical training scripts
requirements-mujoco.txt     MuJoCo demo dependencies
requirements.txt            All app simulator dependencies
```
