from argparse import ArgumentParser
from pathlib import Path
import csv
import json

import yaml
from ultralytics import YOLO


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


def save_test_metrics(metrics, output_path):
    values = {
        "test_precision": float(metrics.box.mp),
        "test_recall": float(metrics.box.mr),
        "test_map50": float(metrics.box.map50),
        "test_map50_95": float(metrics.box.map),
    }

    output_path.write_text(
        json.dumps(values, indent=2),
        encoding="utf-8",
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument(
        "--plan",
        default="reports/tables/final_training_plan.csv",
    )
    parser.add_argument("--job-index", type=int, required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    row = read_plan_row(args.plan, args.job_index)

    if row["model"] != "yolo26n":
        raise ValueError("This script only evaluates YOLO26n final jobs.")

    run_name = f"final_yolo26n_{row['augmentation']}_seed{row['seed']}"
    run_dir = Path(config["paths"]["experiments_dir"]) / "final" / run_name
    metadata_path = run_dir / "final_metadata.json"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = YOLO(metadata["best_weights"])

    metrics = model.val(
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

    metrics_path = run_dir / "test_metrics.json"
    save_test_metrics(metrics, metrics_path)

    metadata["test_metrics_path"] = str(metrics_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
