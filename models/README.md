# Selected demo models

The production policy editor includes 14 saved actors. Six MuJoCo demos now use the collected width-2 1D KAN trials. They have **one KAN layer followed by a linear action head**, not two KAN layers. The graph shows editable KAN edges; the action head stays fixed.

| Environment | KAN architecture | Linear head | Grid | Actor parameters | Validation return (100 episodes) | Target | Passing training seeds at selected grid |
|---|---|---|---:|---:|---:|---:|---|
| Hopper-v5 | 11 → 2 | 2 → 3 | 2 | 141 | 3,428.24 | 3800 | 0/3 |
| Walker2d-v5 | 17 → 2 | 2 → 6 | 5 | 324 | 4,204.46 | 4000 | 1/3 |
| Humanoid-v5 | 348 → 2 | 2 → 17 | 1 | 3,531 | 5,534.37 | 6000 | 0/3 |
| Ant-v5 | 105 → 2 | 2 → 8 | 2 | 1,284 | 2,959.28 | 6000 | 0/3 |
| Swimmer-v5 | 8 → 2 | 2 → 2 | 5 | 150 | 116.32 | 360 | 0/3 |
| Reacher-v5 | 10 → 2 | 2 → 2 | 1 | 106 | -5.75 | -3.75 | 0/3 |

Scores are saved selection-validation means from `result.json`, not new independent evaluations or held-out test scores. The search uses 20 tuning episodes to choose a checkpoint within a run, then 100 selection-validation episodes to compare completed trials. This update selects by those 100-episode results, never by held-out test scores. Walker2d meets the target for one seed only; reliable solving across seeds has not been established. The other five selected policies remain below target.

The local collection contains **48 trial folders: 46 complete exports and 2 incomplete trials**. Humanoid grid 2 / seed 0 and Ant grid 5 / seed 1 lack `best_actor.json` and are excluded. The earlier count of 53 does not match this collection.

The [trial inventory](width2_trials.csv) includes all 48 entries. [Selection provenance](width2_selection.json) records trial names, checkpoint hashes, source artifact hashes, and exact scores. Each selected environment folder holds `actor.json`, `metadata.json`, `environment.json`, `result.json`, `best_validation.json`, and `learning_curve.csv`. Local `model.zip` and `normalization.pkl` are included in the update package but remain Git-ignored. The browser actor embeds its own matching frozen observation statistics, so playback does not need these binary files.

Reacher uses the best 1D width-2 trial in this collection (−5.7468, grid 1). The separate 2D pair-feature experiment is not installed in this univariate curve editor.

To repeat the selection from the original result collection:

```bash
python tools/install_demo_trials.py --results /path/to/results
python app.py --export-only
python app.py
```

The selector copies the winning artifacts into this checkout; it does not modify the source trials. Existing selected model files in the checkout are replaced. Keep previous versions in Git or a backup before promoting another collection.

The browser loads only the selected environment. Its action computation is frozen normalization → fixed observation scaling → KAN → linear head → tanh → action bounds. Editing changes only the browser copy. The original checkpoint is retained for comparison.
