import csv
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split


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


def write_summary(path, split_rows):
    counts = {}

    for row in split_rows:
        split_name = row["split"]
        counts[split_name] = counts.get(split_name, 0) + 1

    rows = [
        {"split": split_name, "n_images": n_images}
        for split_name, n_images in sorted(counts.items())
    ]

    write_csv(path, rows)


def make_train_val_test_split(
    manifest_path,
    output_dir,
    seed,
    train_size,
    val_size,
    test_size,
    stratify_by,
):
    rows = read_csv(manifest_path)
    labels = [row[stratify_by] for row in rows]

    train_rows, temp_rows = train_test_split(
        rows,
        train_size=train_size,
        random_state=seed,
        stratify=labels,
    )

    temp_labels = [row[stratify_by] for row in temp_rows]
    relative_val_size = val_size / (val_size + test_size)

    val_rows, test_rows = train_test_split(
        temp_rows,
        train_size=relative_val_size,
        random_state=seed,
        stratify=temp_labels,
    )

    for row in train_rows:
        row["split"] = "train"
    for row in val_rows:
        row["split"] = "val"
    for row in test_rows:
        row["split"] = "test"

    output_dir = Path(output_dir)

    write_csv(output_dir / "train.csv", train_rows)
    write_csv(output_dir / "val.csv", val_rows)
    write_csv(output_dir / "test.csv", test_rows)
    write_summary(
        output_dir / "split_summary.csv",
        train_rows + val_rows + test_rows,
    )

    return train_rows, val_rows, test_rows


def make_cv_folds(source_csv_path, output_dir, n_splits, seed, stratify_by):
    rows = read_csv(source_csv_path)
    labels = [row[stratify_by] for row in rows]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    summary_rows = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(rows, labels)):
        train_rows = [rows[index].copy() for index in train_idx]
        val_rows = [rows[index].copy() for index in val_idx]

        for row in train_rows:
            row["cv_fold"] = fold
            row["cv_split"] = "train"

        for row in val_rows:
            row["cv_fold"] = fold
            row["cv_split"] = "val"

        write_csv(output_dir / f"fold_{fold}_train.csv", train_rows)
        write_csv(output_dir / f"fold_{fold}_val.csv", val_rows)

        summary_rows.append(
            {
                "fold": fold,
                "train_images": len(train_rows),
                "val_images": len(val_rows),
            }
        )

    write_csv(output_dir / "cv_summary.csv", summary_rows)

    return summary_rows
