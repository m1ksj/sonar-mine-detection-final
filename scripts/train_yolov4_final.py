from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import re
import subprocess
import time

import yaml


MAP50_PATTERN = re.compile(
    r"mean average precision \(mAP@0\.50\) = ([0-9.]+)"
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


def cfg_for_augmentation(config, augmentation):
    cfg_dir = Path(config["yolov4"]["output_dir"])

    if augmentation == "yolov4_default":
        return cfg_dir / "yolov4_default.cfg"

    if augmentation == "yolov4_no_aug":
        return cfg_dir / "yolov4_no_aug.cfg"

    raise ValueError(f"Unknown YOLOv4 augmentation: {augmentation}")


def read_darknet_data(path):
    values = {}

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()

    return values


def write_darknet_data(path, values):
    lines = [
        f"classes = {values['classes']}",
        f"train = {values['train']}",
        f"valid = {values['valid']}",
        f"names = {values['names']}",
        f"backup = {values['backup']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_run_name(row):
    return (
        f"final_yolov4_"
        f"{row['augmentation']}_"
        f"seed{row['seed']}"
    )


def prepare_run_data(base_data_path, test_list_path, run_dir):
    base_values = read_darknet_data(base_data_path)
    backup_dir = run_dir / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    train_data_path = run_dir / "obj_train.data"
    test_data_path = run_dir / "obj_test.data"

    train_values = base_values.copy()
    train_values["backup"] = backup_dir.resolve().as_posix()
    write_darknet_data(train_data_path, train_values)

    test_values = train_values.copy()
    test_values["valid"] = Path(test_list_path).resolve().as_posix()
    write_darknet_data(test_data_path, test_values)

    return train_data_path, test_data_path, backup_dir


def parse_map50(text):
    match = MAP50_PATTERN.search(text)
    return "" if match is None else match.group(1)


def run_test_map(darknet_bin, data_path, cfg_path, weights_path, output_path):
    if not weights_path.exists():
        return ""

    command = [
        darknet_bin,
        "detector",
        "map",
        str(data_path),
        str(cfg_path),
        str(weights_path),
    ]

    result = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
    )
    text = result.stdout + "\n" + result.stderr
    output_path.write_text(text, encoding="utf-8")
    return parse_map50(text)


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
    darknet_dir = Path(config["paths"]["darknet_data_dir"])
    base_data_path = darknet_dir / "obj.data"
    test_list_path = darknet_dir / "test.txt"
    run_name = make_run_name(row)
    run_dir = Path(config["paths"]["experiments_dir"]) / "final" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_data_path, test_data_path, backup_dir = prepare_run_data(
        base_data_path=base_data_path,
        test_list_path=test_list_path,
        run_dir=run_dir,
    )

    command = [
        args.darknet_bin,
        "detector",
        "train",
        str(train_data_path),
        str(cfg_path),
        args.pretrained,
        "-dont_show",
        "-map",
        "-seed",
        str(row["seed"]),
    ]

    start_time = time.time()
    subprocess.run(command, check=True)

    weights_prefix = Path(cfg_path).stem
    best_weights = backup_dir / f"{weights_prefix}_best.weights"
    last_weights = backup_dir / f"{weights_prefix}_last.weights"
    test_metrics_path = run_dir / "test_metrics.txt"
    test_map50 = run_test_map(
        darknet_bin=args.darknet_bin,
        data_path=test_data_path,
        cfg_path=cfg_path,
        weights_path=best_weights,
        output_path=test_metrics_path,
    )

    metadata = {
        "job_index": args.job_index,
        "model": "yolov4",
        "augmentation": row["augmentation"],
        "seed": int(row["seed"]),
        "run_name": run_name,
        "runtime_seconds": time.time() - start_time,
        "cfg_path": str(cfg_path),
        "data_path": str(train_data_path),
        "test_data_path": str(test_data_path),
        "backup_dir": str(backup_dir),
        "best_weights": str(best_weights),
        "last_weights": str(last_weights),
        "best_weights_mb": (
            best_weights.stat().st_size / 1_000_000
            if best_weights.exists()
            else ""
        ),
        "parameter_count": "",
        "test_map50": test_map50,
        "test_metrics_path": str(test_metrics_path),
    }

    metadata_path = run_dir / "final_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
