# Operations and troubleshooting

## Local workflow

From the repository root, install the editable package, run `python scripts/run_demo.py`, then start the API and dashboard using the commands in [README.md](../README.md). Generated files are placed in `data/synthetic`, `data/processed`, and `artifacts`.

Training requires one feature row per patient, multiple mood-state classes, and at least three examples of each target class in the training partition for calibration. Too-small/imbalanced simulated datasets can fail these checks; use the default 300-patient demo rather than treating 30 as a guaranteed trainable size.

## Docker workflow

`docker compose up --build` runs the trainer to completion before serving. The image excludes the legacy archive, cached bytecode, local generated CSVs, and joblib artifacts. The trainer creates fresh demo artifacts in bind-mounted directories.

Inspect startup failures with `docker compose logs trainer api dashboard`. Port conflicts require stopping the conflicting service or changing the **host-side** port mapping. `docker compose down` stops containers but leaves the bind-mounted data and model files on disk.

The Compose trainer overwrites demo models on each startup. For custom trained artifacts, use a separately configured serving deployment that does not run this demo trainer. Localhost binding is intentional; authenticated public hosting is outside this prototype's scope.

## Common errors

| Symptom | Action |
|---|---|
| `ModuleNotFoundError: neurofusion` | Activate the intended environment and install `python -m pip install -e ".[dev]"` |
| API returns 503 | Generate data and train; verify both model files in the serving directory |
| API returns 422 | Check the observation schema, unique dates, and single-patient requirement |
| Calibration class-count error | Increase patient count or supply adequately balanced training targets |
| Dashboard upload fails | Use the example schema, finite numeric values, and consistent demographics |
| Predictions unchanged after retraining | Restart the API; the dashboard reloads when artifact modification times change |

No authentication, request auditing, clinical validation, or patient-data storage system is implemented. The sample interfaces are intended for local synthetic research workflows.
