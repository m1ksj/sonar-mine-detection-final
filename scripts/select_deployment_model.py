from argparse import ArgumentParser
from pathlib import Path
import csv
import json

import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_csv(path):
    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument(
        "--results",
        default="reports/tables/final_run_results.csv",
    )
    parser.add_argument(
        "--output",
        default="models/final/model_selection.json",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    deployment = config["deployment"]
    rows = read_csv(args.results)

    candidates = [
        row for row in rows
        if row["status"] == "done"
        and row["model"] == deployment["selection_model"]
        and row["augmentation"] == deployment["selection_augmentation"]
        and row[deployment["selection_metric"]] != ""
    ]

    if not candidates:
        raise ValueError("No deployment candidate found.")

    selected = max(
        candidates,
        key=lambda row: float(row[deployment["selection_metric"]]),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "selection_metric": deployment["selection_metric"],
                "selected": selected,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Selected deployment model: {selected['run_name']}")


if __name__ == "__main__":
    main()
