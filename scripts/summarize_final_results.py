from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import csv
import math


METRICS = [
    "test_precision",
    "test_recall",
    "test_map50",
    "test_map50_95",
    "runtime_seconds",
    "best_weights_mb",
    "parameter_count",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def numeric_values(rows, metric):
    return [
        float(row[metric])
        for row in rows
        if row.get(metric, "") != ""
    ]


def mean(values):
    return sum(values) / len(values)


def std(values):
    if len(values) < 2:
        return 0.0

    value_mean = mean(values)
    variance = sum((value - value_mean) ** 2 for value in values)
    return math.sqrt(variance / (len(values) - 1))


def summarize(rows):
    groups = defaultdict(list)

    for row in rows:
        groups[(row["model"], row["augmentation"])].append(row)

    summary = []

    for (model, augmentation), group_rows in sorted(groups.items()):
        output = {
            "model": model,
            "augmentation": augmentation,
            "n_seeds": len(group_rows),
        }

        for metric in METRICS:
            values = numeric_values(group_rows, metric)
            output[f"{metric}_mean"] = mean(values) if values else ""
            output[f"{metric}_std"] = std(values) if values else ""

        summary.append(output)

    return summary


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="reports/tables/final_run_results.csv",
    )
    parser.add_argument(
        "--output",
        default="reports/tables/final_seed_summary.csv",
    )
    args = parser.parse_args()

    rows = read_csv(args.input)
    summary = summarize(rows)
    write_csv(args.output, summary)
    print(f"Final seed summary rows written: {len(summary)}")


if __name__ == "__main__":
    main()
