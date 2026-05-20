import csv
from pathlib import Path

CLASS_NAMES = {0: "MILCO", 1: "NOMBO"}


def read_label_classes(label_path):
    label_path = Path(label_path)

    if not label_path.exists():
        return []

    class_ids = []

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid label format: {label_path}")

        class_id = int(parts[0])
        box_values = [float(value) for value in parts[1:]]

        if class_id not in CLASS_NAMES:
            raise ValueError(f"Invalid class id in: {label_path}")

        if not all(0 <= value <= 1 for value in box_values):
            raise ValueError(f"Invalid YOLO box in: {label_path}")

        class_ids.append(class_id)

    return class_ids


def image_category(class_ids):
    class_ids = set(class_ids)

    if 0 in class_ids and 1 in class_ids:
        return "mixed"
    if 0 in class_ids:
        return "milco_only"
    if 1 in class_ids:
        return "nombo_only"

    return "empty"


def build_manifest(raw_dir, years, output_path):
    from PIL import Image

    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for year in years:
        image_dir = raw_dir / str(year)

        for image_path in sorted(image_dir.glob("*.jpg")):
            label_path = image_path.with_suffix(".txt")
            class_ids = read_label_classes(label_path)

            with Image.open(image_path) as image:
                width, height = image.size

            rows.append(
                {
                    "image_id": image_path.stem,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "year": year,
                    "width": width,
                    "height": height,
                    "image_category": image_category(class_ids),
                    "n_objects": len(class_ids),
                }
            )

    if rows:
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return rows
