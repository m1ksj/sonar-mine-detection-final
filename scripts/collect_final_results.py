from argparse import ArgumentParser
from pathlib import Path
import csv
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sonar_mine_detection.evaluation.darknet_metrics import (  # noqa: E402
    parse_darknet_metrics_file,
)


VAL_METRICS = {
    "val_precision": "metrics/precision(B)",
    "val_recall": "metrics/recall(B)",
    "val_map50": "metrics/mAP50(B)",
    "val_map50_95": "metrics/mAP50-95(B)",
}

OUTPUT_COLUMNS = [
    "job_index",
    "model",
    "augmentation",
    "seed",
    "run_name",
    "runtime_seconds",
    "best_weights",
    "best_weights_mb",
    "parameter_count",
    "val_precision",
    "val_recall",
    "val_map50",
    "val_map50_95",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_map50",
    "test_map50_95",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8") as file:
        return [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(file)
        ]


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def run_name(row):
    return (
        f"final_{row['model']}_"
        f"{row['augmentation']}_"
        f"seed{row['seed']}"
    )


def metric_values(row, mapping):
    return {
        output_name: row[source_name]
        for output_name, source_name in mapping.items()
    }


def f1_score(precision, recall):
    precision = float(precision)
    recall = float(recall)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def best_validation_metrics(path):
    rows = [
        row for row in read_csv(path)
        if row.get("metrics/mAP50-95(B)", "") != ""
    ]

    if not rows:
        raise ValueError(f"No validation metrics found in {path}")

    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    return metric_values(best, VAL_METRICS)


def read_yolo26n_test_metrics(path):
    values = read_json(path)

    if "test_f1" not in values:
        values["test_f1"] = f1_score(
            values["test_precision"],
            values["test_recall"],
        )

    return {
        "test_precision": values["test_precision"],
        "test_recall": values["test_recall"],
        "test_f1": values["test_f1"],
        "test_map50": values["test_map50"],
        "test_map50_95": values["test_map50_95"],
    }


def read_yolov4_test_metrics(run_dir):
    aggregate, _ = parse_darknet_metrics_file(
        run_dir / "test_metrics.txt"
    )
    sweep_path = run_dir / "test_iou_sweep_summary.json"

    if sweep_path.exists():
        sweep = read_json(sweep_path)
        aggregate["test_map50_95"] = sweep["test_map50_95"]

    return aggregate


def collect(plan_path, experiments_dir):
    output_rows = []
    experiments_dir = Path(experiments_dir)

    for row in read_csv(plan_path):
        name = run_name(row)
        run_dir = experiments_dir / "final" / name
        metadata = read_json(run_dir / "final_metadata.json")

        output = {
            "job_index": row["job_index"],
            "model": row["model"],
            "augmentation": row["augmentation"],
            "seed": row["seed"],
            "run_name": name,
            "runtime_seconds": metadata["runtime_seconds"],
            "best_weights": metadata["best_weights"],
            "best_weights_mb": metadata["best_weights_mb"],
            "parameter_count": metadata.get("parameter_count", ""),
        }

        if row["model"] == "yolo26n":
            output.update(best_validation_metrics(run_dir / "results.csv"))
            test_metrics_path = run_dir / "test_metrics.json"
            output.update(read_yolo26n_test_metrics(test_metrics_path))
        else:
            output.update({key: "" for key in VAL_METRICS})
            output.update(read_yolov4_test_metrics(run_dir))

        output_rows.append(output)

    return output_rows


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--plan",
        default="reports/tables/final_training_plan.csv",
    )
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument(
        "--output",
        default="reports/tables/final_run_results.csv",
    )
    args = parser.parse_args()

    rows = collect(args.plan, args.experiments_dir)
    write_csv(args.output, rows)
    print(f"Final result rows written: {len(rows)}")


if __name__ == "__main__":
    main()
