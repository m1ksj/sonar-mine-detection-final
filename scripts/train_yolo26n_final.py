from argparse import ArgumentParser
import json
from pathlib import Path
import csv
import sys
import time

import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sonar_mine_detection.training.yolo26n_args import (  # noqa: E402
    build_yolo26n_train_args,
)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_plan_row(path, job_index):
    with open(path, "r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        if int(row["job_index"]) == job_index:
            return row

    raise ValueError(f"Unknown final job index: {job_index}")


def load_hparams(path):
    return load_yaml(path)


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument(
        "--plan",
        default="reports/tables/final_training_plan.csv",
    )
    parser.add_argument(
        "--hparams",
        default="configs/yolo26n_best.yaml",
    )
    parser.add_argument("--job-index", type=int, required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    row = read_plan_row(args.plan, args.job_index)

    if row["model"] != "yolo26n":
        raise ValueError("This script only runs YOLO26n final jobs.")

    hparams = load_hparams(args.hparams)

    run_name = (
        f"final_yolo26n_"
        f"{row['augmentation']}_"
        f"seed{row['seed']}"
    )

    train_args = build_yolo26n_train_args(
        data_yaml=config["paths"]["yolo26n_data_yaml"],
        output_dir=(
            Path(config["paths"]["experiments_dir"]).resolve()
            / "final"
        ),
        run_name=run_name,
        image_size=config["final_training"]["image_size"],
        epochs=config["final_training"]["epochs"],
        batch_size=int(hparams["batch_size"]),
        learning_rate=float(hparams["learning_rate"]),
        patience=int(hparams["patience"]),
        seed=int(row["seed"]),
        augmentation_key=row["augmentation"],
        augmentation_config="configs/augmentations.yaml",
        optimizer=hparams["optimizer"],
    )

    start_time = time.time()

    model = YOLO(config["tuning"]["model"])
    results = model.train(**train_args)

    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"

    model = YOLO(best_weights)

    test_results = model.val(
        data=config["paths"]["yolo26n_data_yaml"],
        split="test",
        plots=False,
        project=str(
            Path(config["paths"]["experiments_dir"]).resolve()
            / "final_test"
        ),
        name=run_name,
        exist_ok=True,
    )

    parameter_count = ""

    if hasattr(model, "model"):
        parameter_count = sum(
            parameter.numel()
            for parameter in model.model.parameters()
        )

    metrics_path = save_dir / "test_metrics.json"
    metrics_values = {
        "test_precision": float(test_results.box.mp),
        "test_recall": float(test_results.box.mr),
        "test_map50": float(test_results.box.map50),
        "test_map50_95": float(test_results.box.map),
    }
    metrics_path.write_text(
        json.dumps(metrics_values, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "job_index": args.job_index,
        "model": "yolo26n",
        "augmentation": row["augmentation"],
        "seed": int(row["seed"]),
        "run_name": run_name,
        "runtime_seconds": time.time() - start_time,
        "best_weights": str(best_weights),
        "best_weights_mb": best_weights.stat().st_size / 1_000_000,
        "parameter_count": parameter_count,
        "train_save_dir": str(save_dir),
        "test_save_dir": str(getattr(test_results, "save_dir", "")),
        "test_metrics_path": str(metrics_path),
        "optimizer": hparams["optimizer"],
    }

    metadata_path = save_dir / "final_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
