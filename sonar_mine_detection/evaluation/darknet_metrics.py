from collections import Counter
from pathlib import Path
import re


CLASS_PATTERN = re.compile(
    r"class_id = (?P<class_id>\d+), "
    r"name = (?P<class_name>[^,]+), "
    r"ap = (?P<ap>[0-9.]+)%\s+"
    r"\(TP = (?P<tp>\d+), FP = (?P<fp>\d+)\)"
)

SUMMARY_PATTERN = re.compile(
    r"for conf_thresh = (?P<conf>[0-9.]+), "
    r"precision = (?P<precision>[0-9.]+), "
    r"recall = (?P<recall>[0-9.]+), "
    r"F1-score = (?P<f1>[0-9.]+)"
)

MAP_PATTERN = re.compile(
    r"mean average precision \(mAP@(?P<iou>[0-9.]+)\) = "
    r"(?P<map>[0-9.]+)"
)


def parse_darknet_map_text(text):
    class_rows = []

    for match in CLASS_PATTERN.finditer(text):
        class_rows.append(
            {
                "class_id": int(match.group("class_id")),
                "class_name": match.group("class_name").strip(),
                "ap50": float(match.group("ap")) / 100.0,
                "tp": int(match.group("tp")),
                "fp": int(match.group("fp")),
            }
        )

    summary_match = SUMMARY_PATTERN.search(text)
    map_match = MAP_PATTERN.search(text)

    if summary_match is None:
        raise ValueError("Darknet precision/recall/F1 line not found.")

    if map_match is None:
        raise ValueError("Darknet mAP line not found.")

    aggregate = {
        "test_precision": float(summary_match.group("precision")),
        "test_recall": float(summary_match.group("recall")),
        "test_f1": float(summary_match.group("f1")),
        "test_map50": float(map_match.group("map")),
        "test_map50_95": "",
    }

    return aggregate, class_rows


def count_yolo_labels(label_dir):
    counts = Counter()

    for path in Path(label_dir).glob("*.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                class_id = int(line.split()[0])
                counts[class_id] += 1

    return counts


def add_ground_truth_counts(class_rows, label_counts):
    output_rows = []

    for row in class_rows:
        output = dict(row)
        class_id = output["class_id"]
        gt_count = label_counts[class_id]
        tp = output["tp"]
        fp = output["fp"]
        fn = gt_count - tp

        output["gt_count"] = gt_count
        output["fn"] = fn
        output["class_precision"] = tp / (tp + fp) if tp + fp else 0.0
        output["class_recall"] = tp / gt_count if gt_count else 0.0

        precision = output["class_precision"]
        recall = output["class_recall"]

        if precision + recall:
            output["class_f1"] = 2 * precision * recall / (precision + recall)
        else:
            output["class_f1"] = 0.0

        output_rows.append(output)

    return output_rows


def parse_darknet_metrics_file(path, label_counts=None):
    text = Path(path).read_text(encoding="utf-8")
    aggregate, class_rows = parse_darknet_map_text(text)

    if label_counts is not None:
        class_rows = add_ground_truth_counts(class_rows, label_counts)

    return aggregate, class_rows
