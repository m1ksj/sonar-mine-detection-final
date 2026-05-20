from argparse import ArgumentParser
import json
from pathlib import Path
import csv
import subprocess
import time

import yaml


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


def cfg_for_augmentation(config, augmentation):
    cfg_dir = Path(config["yolov4"]["output_dir"])

    if augmentation == "yolov4_default":
        return cfg_dir / "yolov4_default.cfg"

    if augmentation == "yolov4_no_aug":
        return cfg_dir / "yolov4_no_aug.cfg"

    raise ValueError(f"Unknown YOLOv4 augmentation: {augmentation}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument(
        "--plan",
        default="reports/tables/final_training_plan.csv",
    )
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument("--darknet-bin", required=True)
    parser.add_argument("--pretrained", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    row = read_plan_row(args.plan, args.job_index)

    if row["model"] != "yolov4":
        raise ValueError("This script only runs YOLOv4 final jobs.")

    cfg_path = cfg_for_augmentation(config, row["augmentation"])
    data_path = Path(config["paths"]["darknet_data_dir"]) / "obj.data"

    command = [
        args.darknet_bin,
        "detector",
        "train",
        str(data_path),
        str(cfg_path),
        args.pretrained,
        "-dont_show",
        "-map",
    ]

    run_name = (
        f"final_yolov4_"
        f"{row['augmentation']}_"
        f"seed{row['seed']}"
    )

    start_time = time.time()
    subprocess.run(command, check=True)

    metadata_dir = (
        Path(config["paths"]["experiments_dir"])
        / "final_metadata"
    )
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "job_index": args.job_index,
        "model": "yolov4",
        "augmentation": row["augmentation"],
        "seed": int(row["seed"]),
        "run_name": run_name,
        "runtime_seconds": time.time() - start_time,
        "cfg_path": str(cfg_path),
        "data_path": str(data_path),
    }

    metadata_path = metadata_dir / f"{run_name}.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
