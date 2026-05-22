from argparse import ArgumentParser
from pathlib import Path
import csv

from sonar_mine_detection.evaluation.darknet_metrics import (
    count_yolo_labels,
    parse_darknet_metrics_file,
)


OUTPUT_COLUMNS = [
    "job_index",
    "model",
    "augmentation",
    "seed",
    "run_name",
    "split",
    "class_id",
    "class_name",
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

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run_name(row):
    return (
        f"final_{row['model']}_"
        f"{row['augmentation']}_"
        f"seed{row['seed']}"
    )


def base_row(row, name):
    return {
        "job_index": row["job_index"],
        "model": row["model"],
        "augmentation": row["augmentation"],
        "seed": row["seed"],
        "run_name": name,
        "split": "test",
    }


def collect_yolov4_class_rows(row, run_dir, label_counts):
    _, class_rows = parse_darknet_metrics_file(
        run_dir / "test_metrics.txt",
        label_counts=label_counts,
    )

    output_rows = []

    for class_row in class_rows:
        output = base_row(row, run_name(row))
        output.update(class_row)
        output_rows.append(output)

    return output_rows


def collect_yolo26n_class_rows(row, run_dir):
    path = run_dir / "test_class_metrics.csv"
    output_rows = []

    for class_row in read_csv(path):
        output = base_row(row, run_name(row))
        output.update(class_row)
        output_rows.append(output)

    return output_rows


def collect(plan_path, experiments_dir, darknet_label_dir):
    rows = []
    experiments_dir = Path(experiments_dir)
    label_counts = count_yolo_labels(darknet_label_dir)

    for row in read_csv(plan_path):
        name = run_name(row)
        run_dir = experiments_dir / "final" / name

        if row["model"] == "yolov4":
            rows.extend(collect_yolov4_class_rows(
                row,
                run_dir,
                label_counts,
            ))
        else:
            rows.extend(collect_yolo26n_class_rows(row, run_dir))

    return rows


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--plan",
        default="reports/tables/final_training_plan.csv",
    )
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument(
        "--darknet-label-dir",
        default="data/processed/darknet/images/test",
    )
    parser.add_argument(
        "--output",
        default="reports/tables/final_class_results.csv",
    )
    args = parser.parse_args()

    rows = collect(
        plan_path=args.plan,
        experiments_dir=args.experiments_dir,
        darknet_label_dir=args.darknet_label_dir,
    )
    write_csv(args.output, rows)
    print(f"Final class result rows written: {len(rows)}")


if __name__ == "__main__":
    main()
