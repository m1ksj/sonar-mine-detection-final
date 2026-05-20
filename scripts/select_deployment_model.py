from argparse import ArgumentParser
from pathlib import Path
import csv
import json

import yaml


VAL_MAP_COLUMN = "metrics/mAP50-95(B)"


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_csv(path):
    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_last_row(path):
    if not path.exists():
        return None

    rows = read_csv(path)

    if not rows:
        return None

    return rows[-1]


def run_name(row):
    return (
        f"final_{row['model']}_"
        f"{row['augmentation']}_"
        f"seed{row['seed']}"
    )


def candidate_rows(config, plan_rows):
    model = config["deployment"]["selection_model"]
    augmentation = config["deployment"]["selection_augmentation"]

    return [
        row
        for row in plan_rows
        if row["model"] == model and row["augmentation"] == augmentation
    ]


def collect_candidates(config, plan_path):
    plan_rows = read_csv(plan_path)
    experiments_dir = Path(config["paths"]["experiments_dir"])
    candidates = []

    for row in candidate_rows(config, plan_rows):
        name = run_name(row)
        train_dir = experiments_dir / "final" / name
        results_path = train_dir / "results.csv"
        metrics = read_last_row(results_path)

        if metrics is None or metrics.get(VAL_MAP_COLUMN, "") == "":
            continue

        best_weights = train_dir / "weights" / "best.pt"

        candidates.append(
            {
                "job_index": int(row["job_index"]),
                "model": row["model"],
                "augmentation": row["augmentation"],
                "seed": int(row["seed"]),
                "run_name": name,
                "val_map50_95": float(metrics[VAL_MAP_COLUMN]),
                "best_weights": str(best_weights),
            }
        )

    return candidates


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument(
        "--plan",
        default="reports/tables/final_training_plan.csv",
    )
    parser.add_argument(
        "--output",
        default="models/final/model_selection.json",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    candidates = collect_candidates(config, args.plan)

    if not candidates:
        print("No completed deployment candidates found.")
        return

    selected = max(candidates, key=lambda row: row["val_map50_95"])

    output = {
        "selection_rule": "highest validation mAP50-95",
        "selected": selected,
        "candidates": candidates,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(f"Selected deployment model: {selected['run_name']}")


if __name__ == "__main__":
    main()
