"""Rebuild deterministic synthetic demo data/models and print an example prediction."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from neurofusion.inference import NeuroFusionPredictor

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", type=int, default=300)
    parser.add_argument("--days", type=int, default=21)
    args = parser.parse_args()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/generate_data.py"),
            "--patients",
            str(args.patients),
            "--days",
            str(args.days),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run([sys.executable, str(ROOT / "scripts/train.py")], cwd=ROOT, check=True)
    example = json.loads((ROOT / "examples/prediction_request.json").read_text())
    print(
        json.dumps(
            NeuroFusionPredictor(ROOT / "artifacts").predict(example["observations"]), indent=2
        )
    )


if __name__ == "__main__":
    main()
