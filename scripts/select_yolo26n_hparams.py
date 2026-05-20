from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import csv

import yaml


GROUP_KEYS = [
    "learning_rate",
    "batch_size",
    "patience",
    "confidence",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return sum(values) / len(values)


def group_results(rows):
    groups = defaultdict(list)

    for row in rows:
        if row["status"] != "done" or row["map50_95"] == "":
            continue

        key = tuple(row[name] for name in GROUP_KEYS)
        groups[key].append(float(row["map50_95"]))

    summary_rows = []

    for key, values in groups.items():
        row = dict(zip(GROUP_KEYS, key))
        row["n_folds"] = len(values)
        row["mean_map50_95"] = mean(values)
        summary_rows.append(row)

    summary_rows.sort(
        key=lambda row: float(row["mean_map50_95"]),
        reverse=True,
    )

    return summary_rows


def write_best_config(path, best_row):
    config = {
        "learning_rate": float(best_row["learning_rate"]),
        "batch_size": int(best_row["batch_size"]),
        "patience": int(best_row["patience"]),
        "confidence": float(best_row["confidence"]),
        "mean_map50_95": float(best_row["mean_map50_95"]),
        "n_folds": int(best_row["n_folds"]),
    }

    Path(path).write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="reports/tables/yolo26n_tuning_results.csv",
    )
    parser.add_argument(
        "--summary",
        default="reports/tables/yolo26n_hparam_summary.csv",
    )
    parser.add_argument(
        "--best-config",
        default="configs/yolo26n_best.yaml",
    )
    args = parser.parse_args()

    rows = read_csv(args.input)
    summary_rows = group_results(rows)
    write_csv(args.summary, summary_rows)

    if not summary_rows:
        print("No completed tuning results found.")
        return

    write_best_config(args.best_config, summary_rows[0])
    print(f"Best YOLO26n setup written to: {args.best_config}")


if __name__ == "__main__":
    main()
