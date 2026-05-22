from argparse import ArgumentParser
from pathlib import Path
import csv
import json
import re
import subprocess
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sonar_mine_detection.evaluation.darknet_metrics import (  # noqa: E402
    parse_darknet_map_text,
)


IOU_THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]

MAP_PATTERN = re.compile(
    r"mean average precision \(mAP@(?P<iou>[0-9.]+)\) = "
    r"(?P<map>[0-9.]+)"
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


def read_darknet_data(path):
    values = {}

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()

    return values


def write_darknet_data(path, values):
    lines = [
        f"{key} = {value}"
        for key, value in values.items()
    ]
    Path(path).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def prepare_test_data_file(base_data_file, test_list_file, output_file):
    values = read_darknet_data(base_data_file)
    values["valid"] = Path(test_list_file).resolve().as_posix()
    write_darknet_data(output_file, values)
    return output_file


def run_name(row):
    return (
        f"final_{row['model']}_"
        f"{row['augmentation']}_"
        f"seed{row['seed']}"
    )


def parse_map_value(text):
    match = MAP_PATTERN.search(text)

    if match is None:
        raise ValueError("Darknet mAP line not found.")

    return float(match.group("map"))


def run_darknet_map(darknet_bin, data_file, cfg_file, weights_file, iou):
    command = [
        darknet_bin,
        "detector",
        "map",
        data_file,
        cfg_file,
        weights_file,
        "-iou_thresh",
        f"{iou:.2f}",
        "-dont_show",
    ]

    result = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
    )

    return result.stdout + "\n" + result.stderr


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument(
        "--plan",
        default="reports/tables/final_training_plan.csv",
    )
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument(
        "--darknet-bin",
        default="external/yolov4/darknet/darknet",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    row = read_plan_row(args.plan, args.job_index)

    if row["model"] != "yolov4":
        raise ValueError("This script only evaluates YOLOv4 final jobs.")

    name = run_name(row)
    run_dir = Path(config["paths"]["experiments_dir"]) / "final" / name
    metadata = json.loads(
        (run_dir / "final_metadata.json").read_text(encoding="utf-8")
    )

    cfg_file = Path("configs/yolov4") / f"{row['augmentation']}.cfg"
    base_data_file = Path("data/processed/darknet/obj.data")
    test_list_file = Path("data/processed/darknet/test.txt")
    data_file = prepare_test_data_file(
        base_data_file=base_data_file,
        test_list_file=test_list_file,
        output_file=run_dir / "obj_test_iou.data",
    )
    weights_file = metadata["best_weights"]
    sweep_dir = run_dir / "iou_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    aggregate_rows = []
    class_rows_by_iou = []

    for iou in IOU_THRESHOLDS:
        text = run_darknet_map(
            darknet_bin=args.darknet_bin,
            data_file=str(data_file),
            cfg_file=str(cfg_file),
            weights_file=weights_file,
            iou=iou,
        )

        output_path = sweep_dir / f"iou_{iou:.2f}.txt"
        output_path.write_text(text, encoding="utf-8")

        aggregate, class_rows = parse_darknet_map_text(text)
        map_value = parse_map_value(text)

        aggregate_rows.append(
            {
                "iou_threshold": iou,
                "map": map_value,
                "precision": aggregate["test_precision"],
                "recall": aggregate["test_recall"],
                "f1": aggregate["test_f1"],
            }
        )

        for class_row in class_rows:
            class_rows_by_iou.append(
                {
                    "iou_threshold": iou,
                    "class_id": class_row["class_id"],
                    "class_name": class_row["class_name"],
                    "ap": class_row["ap50"],
                    "tp": class_row["tp"],
                    "fp": class_row["fp"],
                }
            )

    map_values = [row["map"] for row in aggregate_rows]
    summary = {
        "test_map50_95": sum(map_values) / len(map_values),
        "iou_thresholds": IOU_THRESHOLDS,
        "maps": map_values,
    }

    (run_dir / "test_iou_sweep_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    write_csv(
        run_dir / "test_iou_sweep.csv",
        aggregate_rows,
        ["iou_threshold", "map", "precision", "recall", "f1"],
    )
    write_csv(
        run_dir / "test_iou_sweep_class_results.csv",
        class_rows_by_iou,
        ["iou_threshold", "class_id", "class_name", "ap", "tp", "fp"],
    )

    metadata["test_iou_sweep_summary_path"] = str(
        run_dir / "test_iou_sweep_summary.json"
    )
    metadata["test_map50_95"] = summary["test_map50_95"]
    (run_dir / "final_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
