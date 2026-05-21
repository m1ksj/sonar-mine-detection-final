from argparse import ArgumentParser
from pathlib import Path
import csv
import json


METRIC_COLUMNS = {
    "test_precision": "metrics/precision(B)",
    "test_recall": "metrics/recall(B)",
    "test_map50": "metrics/mAP50(B)",
    "test_map50_95": "metrics/mAP50-95(B)",
    "test_fitness": "fitness",
}


def read_csv(path):
    with open(path, "r", encoding="utf-8") as file:
        rows = []

        for row in csv.DictReader(file):
            rows.append(
                {
                    key.strip(): value.strip()
                    for key, value in row.items()
                }
            )

    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_last_metrics(path):
    if not path.exists():
        return {}

    rows = read_csv(path)

    if not rows:
        return {}

    return rows[-1]


def run_name(row):
    return (
        f"final_{row['model']}_"
        f"{row['augmentation']}_"
        f"seed{row['seed']}"
    )


def metadata_path(row, experiments_dir):
    experiments_dir = Path(experiments_dir)
    name = run_name(row)
    return experiments_dir / "final" / name / "final_metadata.json"


def test_results_path(row, experiments_dir):
    name = run_name(row)

    if row["model"] != "yolo26n":
        return None

    return Path(experiments_dir) / "final_test" / name / "results.csv"


def collect_yolo26n_metrics(row, experiments_dir):
    metrics = read_last_metrics(
        test_results_path(row, experiments_dir) or Path("missing")
    )
    return {
        output_name: metrics.get(metric_name, "")
        for output_name, metric_name in METRIC_COLUMNS.items()
    }


def collect_yolov4_metrics(meta):
    return {
        "test_precision": "",
        "test_recall": "",
        "test_map50": meta.get("test_map50", ""),
        "test_map50_95": "",
        "test_fitness": "",
    }


def collect(plan_path, experiments_dir):
    plan_rows = read_csv(plan_path)
    output_rows = []

    for row in plan_rows:
        name = run_name(row)
        meta = read_json(metadata_path(row, experiments_dir))

        output_row = {
            "job_index": row["job_index"],
            "model": row["model"],
            "augmentation": row["augmentation"],
            "seed": row["seed"],
            "run_name": name,
            "status": "done" if meta else "missing",
            "runtime_seconds": meta.get("runtime_seconds", ""),
            "best_weights": meta.get("best_weights", ""),
            "best_weights_mb": meta.get("best_weights_mb", ""),
            "parameter_count": meta.get("parameter_count", ""),
        }

        if row["model"] == "yolo26n":
            output_row.update(collect_yolo26n_metrics(row, experiments_dir))
        else:
            output_row.update(collect_yolov4_metrics(meta))

        output_rows.append(output_row)

    return output_rows


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--plan",
        default="reports/tables/final_training_plan.csv",
    )
    parser.add_argument(
        "--experiments-dir",
        default="experiments",
    )
    parser.add_argument(
        "--output",
        default="reports/tables/final_run_results.csv",
    )
    args = parser.parse_args()

    rows = collect(
        plan_path=args.plan,
        experiments_dir=args.experiments_dir,
    )
    write_csv(args.output, rows)

    print(f"Final result rows written: {len(rows)}")


if __name__ == "__main__":
    main()
