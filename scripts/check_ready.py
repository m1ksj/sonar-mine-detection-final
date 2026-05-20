from pathlib import Path
import csv


def count_csv_rows(path):
    with open(path, "r", encoding="utf-8") as file:
        return len(list(csv.DictReader(file)))


def require_file(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)


def require_count(name, actual, expected):
    if actual != expected:
        raise ValueError(f"{name}: expected {expected}, got {actual}")


def main():
    required_files = [
        "data/metadata/dataset_manifest.csv",
        "data/splits/train.csv",
        "data/splits/val.csv",
        "data/splits/test.csv",
        "data/processed/yolo26n/data.yaml",
        "data/processed/yolo26n_cv/fold_0/data.yaml",
        "data/processed/darknet/obj.data",
        "configs/yolov4/yolov4_default.cfg",
        "configs/yolov4/yolov4_no_aug.cfg",
        "reports/tables/yolo26n_tuning_plan.csv",
        "reports/tables/final_training_plan.csv",
    ]

    for path in required_files:
        require_file(path)

    require_count(
        "dataset_manifest",
        count_csv_rows("data/metadata/dataset_manifest.csv"),
        1170,
    )
    require_count(
        "train split",
        count_csv_rows("data/splits/train.csv"),
        819,
    )
    require_count(
        "validation split",
        count_csv_rows("data/splits/val.csv"),
        175,
    )
    require_count(
        "test split",
        count_csv_rows("data/splits/test.csv"),
        176,
    )
    require_count(
        "YOLO26n tuning plan",
        count_csv_rows("reports/tables/yolo26n_tuning_plan.csv"),
        60,
    )
    require_count(
        "final training plan",
        count_csv_rows("reports/tables/final_training_plan.csv"),
        12,
    )

    print("Project preflight check passed.")


if __name__ == "__main__":
    main()
