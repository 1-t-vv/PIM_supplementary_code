#!/usr/bin/env python3
"""Evaluate the packaged PIM checkpoint on the controlled Flat test split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PIM_ROOT = PACKAGE_ROOT / "pim"
HERE = Path(__file__).resolve().parent
CONFIG_TEMPLATE = HERE / "model_config.json"
TEST_STORE = HERE / "test_data" / "test.store"
CACHE_DIR = HERE / "cache" / "fine_test"
CHECKPOINT = HERE / "checkpoint" / "best_validate_error_model.pth"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PIM on the 100-scene controlled Flat rollout split."
    )
    parser.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=HERE / "reproduced_results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate packaged paths and print the resolved evaluation inputs.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers non-negative")

    required = (
        PIM_ROOT / "evaluate.py",
        CONFIG_TEMPLATE,
        TEST_STORE / "manifest.json",
        CACHE_DIR,
        CHECKPOINT,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing packaged PIM input(s): " + ", ".join(missing))

    results_dir = args.results_dir.resolve()
    with CONFIG_TEMPLATE.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["test_npz"] = str(TEST_STORE)
    config["idx_cache_dir_test"] = str(CACHE_DIR)
    config["save_dir"] = str(CHECKPOINT.parent)
    config["results_dir"] = str(results_dir)

    if args.dry_run:
        print(json.dumps({
            "pim_evaluator": str(PIM_ROOT / "evaluate.py"),
            "test_store": str(TEST_STORE),
            "knn_cache": str(CACHE_DIR),
            "checkpoint": str(CHECKPOINT),
            "results_dir": str(results_dir),
            "num_samples": 100,
        }, indent=2))
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pim-rollout-eval-") as temporary:
        runtime_config = Path(temporary) / "model_config.json"
        with runtime_config.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
            handle.write("\n")

        command = [
            sys.executable,
            str(PIM_ROOT / "evaluate.py"),
            "--scene", "flat",
            "--config", str(runtime_config),
            "--ckpt", str(CHECKPOINT),
            "--test-data", str(TEST_STORE),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--results-dir", str(results_dir),
        ]
        print(" ".join(command), flush=True)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = args.gpus
        environment["PYTHONUNBUFFERED"] = "1"
        subprocess.run(command, cwd=PIM_ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
