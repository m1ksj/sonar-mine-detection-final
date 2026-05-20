from argparse import ArgumentParser
from itertools import product
from pathlib import Path
import csv

import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def build_plan(config):
    rows = []
    job_index = 0

    folds = range(config["cv"]["folds"])
    learning_rates = config["tuning"]["learning_rates"]
    batch_sizes = config["tuning"]["batch_sizes"]
    patience_values = config["tuning"]["patience_values"]
    confidence_values = config["tuning"]["confidence_thresholds"]

    grid = product(
        folds,
        learning_rates,
        batch_sizes,
        patience_values,
        confidence_values,
    )

    for fold, lr, batch, patience, confidence in grid:
        rows.append(
            {
                "job_index": job_index,
                "fold": fold,
                "learning_rate": lr,
                "batch_size": batch,
                "patience": patience,
                "confidence": confidence,
            }
        )
        job_index += 1

    return rows


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument(
        "--output",
        default="reports/tables/yolo26n_tuning_plan.csv",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    rows = build_plan(config)
    write_csv(args.output, rows)

    print(f"YOLO26n tuning jobs written: {len(rows)}")


if __name__ == "__main__":
    main()
