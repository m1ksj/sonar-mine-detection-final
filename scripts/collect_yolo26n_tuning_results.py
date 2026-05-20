from argparse import ArgumentParser
from pathlib import Path
import csv


METRIC_COLUMNS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
    "fitness": "fitness",
}


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


def make_run_name(row):
    return (
        f"job{row['job_index']}_"
        f"fold{row['fold']}_"
        f"lr{row['learning_rate']}_"
        f"batch{row['batch_size']}_"
        f"patience{row['patience']}"
    )


def read_last_result(results_path):
    if not results_path.exists():
        return None

    rows = read_csv(results_path)

    if not rows:
        return None

    return rows[-1]


def collect_results(plan_path, results_dir):
    plan_rows = read_csv(plan_path)
    results_dir = Path(results_dir)
    output_rows = []

    for plan_row in plan_rows:
        run_name = make_run_name(plan_row)
        results_path = results_dir / run_name / "results.csv"
        result_row = read_last_result(results_path)

        output_row = {
            "job_index": plan_row["job_index"],
            "fold": plan_row["fold"],
            "learning_rate": plan_row["learning_rate"],
            "batch_size": plan_row["batch_size"],
            "patience": plan_row["patience"],
            "run_name": run_name,
            "status": "missing" if result_row is None else "done",
            "results_path": str(results_path),
        }

        for output_name, metric_name in METRIC_COLUMNS.items():
            output_row[output_name] = (
                "" if result_row is None else result_row.get(metric_name, "")
            )

        output_rows.append(output_row)

    return output_rows


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--plan",
        default="reports/tables/yolo26n_tuning_plan.csv",
    )
    parser.add_argument(
        "--results-dir",
        default="experiments/tuning_val",
    )
    parser.add_argument(
        "--output",
        default="reports/tables/yolo26n_tuning_results.csv",
    )
    args = parser.parse_args()

    rows = collect_results(
        plan_path=args.plan,
        results_dir=args.results_dir,
    )
    write_csv(args.output, rows)

    print(f"YOLO26n tuning results written: {len(rows)}")


if __name__ == "__main__":
    main()
