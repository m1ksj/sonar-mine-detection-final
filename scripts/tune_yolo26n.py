from argparse import ArgumentParser
import csv
from pathlib import Path
import sys

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
            return {
                "job_index": job_index,
                "fold": int(row["fold"]),
                "lr": float(row["learning_rate"]),
                "batch": int(row["batch_size"]),
                "patience": int(row["patience"]),
            }

    raise ValueError(f"Unknown tuning job index: {job_index}")


def read_manual_args(args):
    required_values = [args.fold, args.lr, args.batch, args.patience]

    if any(value is None for value in required_values):
        raise ValueError("Use --job-index or provide fold/lr/batch/patience.")

    return {
        "job_index": "manual",
        "fold": args.fold,
        "lr": args.lr,
        "batch": args.batch,
        "patience": args.patience,
        "confidence": args.confidence,
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument(
        "--plan",
        default="reports/tables/yolo26n_tuning_plan.csv",
    )
    parser.add_argument("--job-index", type=int, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    config = load_yaml(args.config)

    if args.job_index is None:
        settings = read_manual_args(args)
    else:
        settings = read_plan_row(args.plan, args.job_index)

    run_name = (
        f"job{settings['job_index']}_"
        f"fold{settings['fold']}_"
        f"lr{settings['lr']}_"
        f"batch{settings['batch']}_"
        f"patience{settings['patience']}"
    )

    fold_dir = (
        Path(config["paths"]["yolo26n_cv_dir"])
        / f"fold_{settings['fold']}"
    )
    fold_data_yaml = fold_dir / "data.yaml"

    train_args = build_yolo26n_train_args(
        data_yaml=fold_data_yaml,
        output_dir=(
            Path(config["paths"]["experiments_dir"]).resolve()
            / "tuning"
        ),
        run_name=run_name,
        image_size=config["tuning"]["image_size"],
        epochs=config["tuning"]["epochs"],
        batch_size=settings["batch"],
        learning_rate=settings["lr"],
        patience=settings["patience"],
        seed=config["cv"]["seed"] + settings["fold"],
        augmentation_key="yolo26n_yolov4_style",
        augmentation_config="configs/augmentations.yaml",
    )

    model = YOLO(config["tuning"]["model"])
    results = model.train(**train_args)

    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"

    if best_weights.exists():
        model = YOLO(best_weights)

    model.val(
        data=str(fold_data_yaml),
        conf=settings["confidence"],
        plots=False,
        project=str(
            Path(config["paths"]["experiments_dir"]).resolve()
            / "tuning_val"
        ),
        name=run_name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
