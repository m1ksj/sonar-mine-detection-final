from argparse import ArgumentParser
import csv
import subprocess
import sys


def read_plan_row(path, job_index):
    with open(path, "r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        if int(row["job_index"]) == job_index:
            return row

    raise ValueError(f"Unknown final job index: {job_index}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument(
        "--plan",
        default="reports/tables/final_training_plan.csv",
    )
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument("--darknet-bin", required=True)
    parser.add_argument("--pretrained", required=True)
    args = parser.parse_args()

    row = read_plan_row(args.plan, args.job_index)

    if row["model"] == "yolo26n":
        command = [
            sys.executable,
            "scripts/train_yolo26n_final.py",
            "--job-index",
            str(args.job_index),
            "--plan",
            args.plan,
            "--config",
            args.config,
        ]
    elif row["model"] == "yolov4":
        command = [
            sys.executable,
            "scripts/train_yolov4_final.py",
            "--job-index",
            str(args.job_index),
            "--plan",
            args.plan,
            "--config",
            args.config,
            "--darknet-bin",
            args.darknet_bin,
            "--pretrained",
            args.pretrained,
        ]
    else:
        raise ValueError(f"Unknown model: {row['model']}")

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
