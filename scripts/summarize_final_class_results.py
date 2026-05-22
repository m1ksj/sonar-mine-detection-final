from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import csv
import statistics


GROUP_COLUMNS = [
    "model",
    "augmentation",
    "class_id",
    "class_name",
]

METRIC_COLUMNS = [
    "ap50",
    "tp",
    "fp",
    "fn",
    "gt_count",
    "class_precision",
    "class_recall",
    "class_f1",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8") as file:
        return [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(file)
        ]


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = GROUP_COLUMNS + ["n_seeds"]

    for metric in METRIC_COLUMNS:
        fieldnames.append(f"{metric}_mean")
        fieldnames.append(f"{metric}_std")

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_key(row):
    return tuple(row[column] for column in GROUP_COLUMNS)


def mean(values):
    return statistics.mean(values)


def std(values):
    if len(values) < 2:
        return 0.0

    return statistics.stdev(values)


def summarize(rows):
    groups = defaultdict(list)

    for row in rows:
        groups[group_key(row)].append(row)

    summary_rows = []

    for key, group_rows in sorted(groups.items()):
        output = {
            column: value
            for column, value in zip(GROUP_COLUMNS, key)
        }
        output["n_seeds"] = len(group_rows)

        for metric in METRIC_COLUMNS:
            values = [
                float(row[metric])
                for row in group_rows
                if row[metric] != ""
            ]

            if values:
                output[f"{metric}_mean"] = mean(values)
                output[f"{metric}_std"] = std(values)
            else:
                output[f"{metric}_mean"] = ""
                output[f"{metric}_std"] = ""

        summary_rows.append(output)

    return summary_rows


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="reports/tables/final_class_results.csv",
    )
    parser.add_argument(
        "--output",
        default="reports/tables/final_class_summary.csv",
    )
    args = parser.parse_args()

    rows = summarize(read_csv(args.input))
    write_csv(args.output, rows)
    print(f"Final class summary rows written: {len(rows)}")


if __name__ == "__main__":
    main()
