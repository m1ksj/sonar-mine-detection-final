from argparse import ArgumentParser
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


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)

    run_name = (
        f"fold{args.fold}_"
        f"lr{args.lr}_"
        f"batch{args.batch}_"
        f"patience{args.patience}"
    )

    train_args = build_yolo26n_train_args(
        data_yaml=config["paths"]["yolo26n_data_yaml"],
        output_dir=Path(config["paths"]["experiments_dir"]) / "tuning",
        run_name=run_name,
        image_size=config["tuning"]["image_size"],
        epochs=config["tuning"]["epochs"],
        batch_size=args.batch,
        learning_rate=args.lr,
        patience=args.patience,
        seed=config["cv"]["seed"] + args.fold,
        augmentation_key="yolo26n_yolov4_style",
        augmentation_config="configs/augmentations.yaml",
    )

    model = YOLO(config["tuning"]["model"])
    model.train(**train_args)


if __name__ == "__main__":
    main()
