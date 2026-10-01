from __future__ import annotations

import argparse
from pathlib import Path

from neurofusion.config import load_config
from neurofusion.data.synthetic import generate_longitudinal_data
from neurofusion.data.validation import validate_longitudinal_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--patients", type=int, default=None)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    patients = args.patients if args.patients is not None else config["data"]["patients"]
    days = args.days if args.days is not None else config["data"]["days"]
    seed = args.seed if args.seed is not None else config["seed"]
    df = generate_longitudinal_data(patients, days, seed)
    validate_longitudinal_frame(df)
    out = Path(args.output or config["data"]["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows for {df.patient_id.nunique():,} patients to {out}")


if __name__ == "__main__":
    main()
