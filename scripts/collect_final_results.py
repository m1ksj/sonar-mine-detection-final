from argparse import ArgumentParser
from pathlib import Path
import csv
import json


VAL_METRICS = {
    "val_precision": "metrics/precision(B)",
    "val_recall": "metrics/recall(B)",
    "val_map50": "metrics/mAP50(B)",
    "val_map50_95": "metrics/mAP50-95(B)",
}

TEST_METRICS = {
    "test_precision": "metrics/precision(B)",
    "test_recall": "metrics/recall(B)",
    "test_map50": "metrics/mAP50(B)",
    "test_map50_95": "metrics/mAP50-95(B)",
}


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
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    if not path.exists():
        return {}

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
        output_name: row.get(source_name, "")
        for output_name, source_name in mapping.items()
    }


def best_validation_metrics(path):
    if not path.exists():
        return {key: "" for key in VAL_METRICS}

    rows = [
        row for row in read_csv(path)
        if row.get("metrics/mAP50-95(B)", "") != ""
    ]

    if not rows:
        return {key: "" for key in VAL_METRICS}

    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    return metric_values(best, VAL_METRICS)


def last_test_metrics(path):
    if not path.exists():
        return {key: "" for key in TEST_METRICS}

    rows = read_csv(path)

    if not rows:
        return {key: "" for key in TEST_METRICS}

    return metric_values(rows[-1], TEST_METRICS)


def collect(plan_path, experiments_dir):
    plan_rows = read_csv(plan_path)
    experiments_dir = Path(experiments_dir)
    output_rows = []

    for row in plan_rows:
        name = run_name(row)
        run_dir = experiments_dir / "final" / name
        metadata = read_json(run_dir / "final_metadata.json")

        output = {
            "job_index": row["job_index"],
            "model": row["model"],
            "augmentation": row["augmentation"],
            "seed": row["seed"],
            "run_name": name,
            "status": "done" if metadata else "missing",
            "runtime_seconds": metadata.get("runtime_seconds", ""),
            "best_weights": metadata.get("best_weights", ""),
            "best_weights_mb": metadata.get("best_weights_mb", ""),
            "parameter_count": metadata.get("parameter_count", ""),
        }

        if row["model"] == "yolo26n":
            output.update(best_validation_metrics(run_dir / "results.csv"))
            output.update(
                last_test_metrics(
                    experiments_dir
                    / "final_test"
                    / name
                    / "results.csv"
                )
            )
        else:
            output.update({key: "" for key in VAL_METRICS})
            output.update(
                {
                    "test_precision": "",
                    "test_recall": "",
                    "test_map50": metadata.get("test_map50", ""),
                    "test_map50_95": "",
                }
            )

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
