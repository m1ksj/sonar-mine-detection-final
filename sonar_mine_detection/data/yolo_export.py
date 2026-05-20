from pathlib import Path
from shutil import copyfile


CLASS_NAMES = {0: "MILCO", 1: "NOMBO"}


def read_split_csv(path):
    import csv

    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def copy_split_files(rows, output_root, split_name):
    output_root = Path(output_root)
    image_dir = output_root / "images" / split_name
    label_dir = output_root / "labels" / split_name

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    copied_images = []

    for row in rows:
        image_path = Path(row["image_path"])
        label_path = Path(row["label_path"])

        target_image = image_dir / image_path.name
        target_label = label_dir / label_path.name

        copyfile(image_path, target_image)

        if label_path.exists():
            copyfile(label_path, target_label)
        else:
            target_label.write_text("", encoding="utf-8")

        copied_images.append(target_image.resolve())

    return copied_images


def write_yolo26n_yaml(output_root):
    output_root = Path(output_root)
    data_yaml = output_root / "data.yaml"

    lines = [
        f"path: {output_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]

    for class_id, class_name in CLASS_NAMES.items():
        lines.append(f"  {class_id}: {class_name}")

    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_darknet_files(output_root, split_images):
    output_root = Path(output_root)

    names_path = output_root / "obj.names"
    data_path = output_root / "obj.data"

    names_path.write_text("MILCO\nNOMBO\n", encoding="utf-8")

    for split_name, image_paths in split_images.items():
        file_name = "valid.txt" if split_name == "val" else f"{split_name}.txt"
        list_path = output_root / file_name
        lines = [path.as_posix() for path in image_paths]
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    data_lines = [
        "classes = 2",
        f"train = {(output_root / 'train.txt').resolve().as_posix()}",
        f"valid = {(output_root / 'valid.txt').resolve().as_posix()}",
        f"names = {names_path.resolve().as_posix()}",
        f"backup = {(output_root / 'backup').resolve().as_posix()}",
    ]

    data_path.write_text("\n".join(data_lines) + "\n", encoding="utf-8")


def export_yolo_datasets(splits_dir, yolo26n_dir, darknet_dir):
    splits_dir = Path(splits_dir)
    yolo26n_dir = Path(yolo26n_dir)
    darknet_dir = Path(darknet_dir)

    split_images = {}

    for split_name in ["train", "val", "test"]:
        rows = read_split_csv(splits_dir / f"{split_name}.csv")

        copy_split_files(rows, yolo26n_dir, split_name)
        split_images[split_name] = copy_split_files(
            rows,
            darknet_dir,
            split_name,
        )

    write_yolo26n_yaml(yolo26n_dir)
    write_darknet_files(darknet_dir, split_images)

    return {
        "yolo26n_dir": str(yolo26n_dir),
        "darknet_dir": str(darknet_dir),
    }


def export_yolo26n_cv_folds(splits_dir, output_root):
    splits_dir = Path(splits_dir)
    output_root = Path(output_root)

    cv_dir = splits_dir / "cv"
    fold_files = sorted(cv_dir.glob("fold_*_train.csv"))

    for train_file in fold_files:
        fold_name = train_file.stem.replace("_train", "")
        fold_dir = output_root / fold_name

        val_file = cv_dir / f"{fold_name}_val.csv"

        train_rows = read_split_csv(train_file)
        val_rows = read_split_csv(val_file)

        copy_split_files(train_rows, fold_dir, "train")
        copy_split_files(val_rows, fold_dir, "val")

        data_yaml = fold_dir / "data.yaml"
        lines = [
            f"path: {fold_dir.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: MILCO",
            "  1: NOMBO",
        ]
        data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return len(fold_files)
