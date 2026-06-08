from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import csv
import json

import yaml
from PIL import Image
from ultralytics import YOLO


CLASS_NAMES = {
    0: "MILCO",
    1: "NOMBO",
}

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_plan_row(path, job_index):
    with open(path, "r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        if int(row["job_index"]) == job_index:
            return row

    raise ValueError(f"Unknown final job index: {job_index}")


def save_test_metrics(metrics, output_path):
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)

    values = {
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1_score(precision, recall),
        "test_map50": float(metrics.box.map50),
        "test_map50_95": float(metrics.box.map),
    }

    output_path.write_text(
        json.dumps(values, indent=2),
        encoding="utf-8",
    )


def f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def yolo_to_xyxy(label, image_width, image_height):
    class_id, x_center, y_center, width, height = label

    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height

    return {
        "class_id": int(class_id),
        "x1": x_center - width / 2,
        "y1": y_center - height / 2,
        "x2": x_center + width / 2,
        "y2": y_center + height / 2,
    }


def read_labels(path, image_width, image_height):
    labels = []

    if not path.exists():
        return labels

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            values = [float(value) for value in line.split()]
            labels.append(yolo_to_xyxy(values, image_width, image_height))

    return labels


def box_iou(box_a, box_b):
    x_left = max(box_a["x1"], box_b["x1"])
    y_top = max(box_a["y1"], box_b["y1"])
    x_right = min(box_a["x2"], box_b["x2"])
    y_bottom = min(box_a["y2"], box_b["y2"])

    intersection_width = max(0.0, x_right - x_left)
    intersection_height = max(0.0, y_bottom - y_top)
    intersection = intersection_width * intersection_height

    area_a = max(0.0, box_a["x2"] - box_a["x1"]) * max(
        0.0,
        box_a["y2"] - box_a["y1"],
    )
    area_b = max(0.0, box_b["x2"] - box_b["x1"]) * max(
        0.0,
        box_b["y2"] - box_b["y1"],
    )

    union = area_a + area_b - intersection

    if union == 0:
        return 0.0

    return intersection / union


def predictions_from_result(result):
    predictions = []

    for box in result.boxes:
        confidence = float(box.conf.item())

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        xyxy = box.xyxy[0].tolist()

        predictions.append(
            {
                "class_id": int(box.cls.item()),
                "confidence": confidence,
                "x1": float(xyxy[0]),
                "y1": float(xyxy[1]),
                "x2": float(xyxy[2]),
                "y2": float(xyxy[3]),
            }
        )

    return sorted(
        predictions,
        key=lambda prediction: prediction["confidence"],
        reverse=True,
    )


def match_predictions(predictions, labels):
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "gt_count": 0})
    matched_labels = set()

    for label in labels:
        stats[label["class_id"]]["gt_count"] += 1

    for prediction in predictions:
        class_id = prediction["class_id"]
        best_iou = 0.0
        best_label_index = None

        for label_index, label in enumerate(labels):
            if label_index in matched_labels:
                continue

            if label["class_id"] != class_id:
                continue

            iou = box_iou(prediction, label)

            if iou > best_iou:
                best_iou = iou
                best_label_index = label_index

        if best_iou >= IOU_THRESHOLD:
            stats[class_id]["tp"] += 1
            matched_labels.add(best_label_index)
        else:
            stats[class_id]["fp"] += 1

    for label_index, label in enumerate(labels):
        if label_index not in matched_labels:
            stats[label["class_id"]]["fn"] += 1

    return stats


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_rows_from_stats(stats, ap50_by_class):
    rows = []

    for class_id in sorted(CLASS_NAMES):
        values = stats[class_id]
        tp = values["tp"]
        fp = values["fp"]
        fn = values["fn"]
        gt_count = values["gt_count"]

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / gt_count if gt_count else 0.0

        rows.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "ap50": ap50_by_class.get(class_id, ""),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "gt_count": gt_count,
                "class_precision": precision,
                "class_recall": recall,
                "class_f1": f1_score(precision, recall),
            }
        )

    return rows


def ap50_lookup(metrics):
    lookup = {}

    if not hasattr(metrics.box, "ap50"):
        return lookup

    for index, class_id in enumerate(metrics.box.ap_class_index):
        lookup[int(class_id)] = float(metrics.box.ap50[index])

    return lookup


def evaluate_predictions(model, image_dir, label_dir):
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "gt_count": 0})
    prediction_rows = []
    ground_truth_rows = []

    for image_path in sorted(Path(image_dir).glob("*.jpg")):
        image = Image.open(image_path)
        image_width, image_height = image.size

        label_path = Path(label_dir) / f"{image_path.stem}.txt"
        labels = read_labels(label_path, image_width, image_height)
        result = model.predict(
            source=str(image_path),
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )[0]
        predictions = predictions_from_result(result)
        image_stats = match_predictions(predictions, labels)

        for class_id, values in image_stats.items():
            for key, value in values.items():
                stats[class_id][key] += value

        for prediction in predictions:
            prediction_rows.append(
                {
                    "image_name": image_path.name,
                    "class_id": prediction["class_id"],
                    "class_name": CLASS_NAMES[prediction["class_id"]],
                    "confidence": prediction["confidence"],
                    "x1": prediction["x1"],
                    "y1": prediction["y1"],
                    "x2": prediction["x2"],
                    "y2": prediction["y2"],
                }
            )

        for label in labels:
            ground_truth_rows.append(
                {
                    "image_name": image_path.name,
                    "class_id": label["class_id"],
                    "class_name": CLASS_NAMES[label["class_id"]],
                    "x1": label["x1"],
                    "y1": label["y1"],
                    "x2": label["x2"],
                    "y2": label["y2"],
                }
            )

    return stats, prediction_rows, ground_truth_rows


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument(
        "--plan",
        default="reports/tables/final_training_plan.csv",
    )
    parser.add_argument("--job-index", type=int, required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    row = read_plan_row(args.plan, args.job_index)

    if row["model"] != "yolo26n":
        raise ValueError("This script only evaluates YOLO26n final jobs.")

    run_name = f"final_yolo26n_{row['augmentation']}_seed{row['seed']}"
    run_dir = Path(config["paths"]["experiments_dir"]) / "final" / run_name
    metadata_path = run_dir / "final_metadata.json"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = YOLO(metadata["best_weights"])

    metrics = model.val(
        data=config["paths"]["yolo26n_data_yaml"],
        split="test",
        plots=False,
        project=str(
            Path(config["paths"]["experiments_dir"]).resolve()
            / "final_test"
        ),
        name=run_name,
        exist_ok=True,
    )

    metrics_path = run_dir / "test_metrics.json"
    save_test_metrics(metrics, metrics_path)

    data_root = Path("data/processed/yolo26n")
    stats, prediction_rows, ground_truth_rows = evaluate_predictions(
        model=model,
        image_dir=data_root / "images" / "test",
        label_dir=data_root / "labels" / "test",
    )

    class_rows = metric_rows_from_stats(stats, ap50_lookup(metrics))
    write_csv(
        run_dir / "test_class_metrics.csv",
        class_rows,
        [
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
        ],
    )
    write_csv(
        run_dir / "test_predictions.csv",
        prediction_rows,
        [
            "image_name",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
        ],
    )
    write_csv(
        run_dir / "test_ground_truth.csv",
        ground_truth_rows,
        [
            "image_name",
            "class_id",
            "class_name",
            "x1",
            "y1",
            "x2",
            "y2",
        ],
    )

    metadata["test_metrics_path"] = str(metrics_path)
    metadata["test_class_metrics_path"] = str(
        run_dir / "test_class_metrics.csv"
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
