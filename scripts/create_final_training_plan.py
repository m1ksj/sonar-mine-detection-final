from argparse import ArgumentParser
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

    seeds = config["final_training"]["seeds"]
    runs = config["final_training"]["runs"]

    for run in runs:
        for seed in seeds:
            rows.append(
                {
                    "job_index": job_index,
                    "model": run["model"],
                    "augmentation": run["augmentation"],
                    "seed": seed,
                }
            )
            job_index += 1

    return rows


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument(
        "--output",
        default="reports/tables/final_training_plan.csv",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    rows = build_plan(config)
    write_csv(args.output, rows)

    print(f"Final training jobs written: {len(rows)}")


if __name__ == "__main__":
    main()
