# Contributing

Use a separate Python 3.10+ environment and install `python -m pip install -e ".[dev]"`. Run `python -m pytest`, `ruff check .`, and the documented synthetic demo before opening a PR.

Bug reports should contain package versions, the exact command/configuration, expected/actual behavior, and a small synthetic reproducer. Do not upload private patient records, secrets, or model artifacts from untrusted sources.

Keep evaluation changes patient-aware. Preprocessing must be fitted inside training/calibration folds. Add regression tests when changing schema validation, splitting, inference, or serving behavior. Clearly distinguish software correctness from clinical validation and synthetic associations from prospective forecasting.

Generated data, models, caches, and build outputs do not belong in source commits. Credit real collaborators accurately. New clinical claims require independent evidence; do not infer them from simulator performance.
