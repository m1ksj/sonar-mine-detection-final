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


def best_validation_metrics(path):
    rows = [
        row for row in read_csv(path)
        if row.get("metrics/mAP50-95(B)", "") != ""
    ]

    if not rows:
        raise ValueError(f"No validation metrics found in {path}")

    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    return metric_values(best, VAL_METRICS)


def read_test_metrics_json(path):
    values = read_json(path)

    return {
        "test_precision": values["test_precision"],
        "test_recall": values["test_recall"],
        "test_map50": values["test_map50"],
        "test_map50_95": values["test_map50_95"],
    }


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
            "parameter_count": metadata["parameter_count"],
        }

        if row["model"] == "yolo26n":
            output.update(best_validation_metrics(run_dir / "results.csv"))
            test_metrics_path = run_dir / "test_metrics.json"
            output.update(read_test_metrics_json(test_metrics_path))
        else:
            output.update({key: "" for key in VAL_METRICS})
            output.update(
                {
                    "test_precision": "",
                    "test_recall": "",
                    "test_map50": metadata["test_map50"],
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
