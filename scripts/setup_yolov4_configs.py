from argparse import ArgumentParser
from pathlib import Path
import re

import requests
import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def download_text(url):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.text


def set_cfg_value(lines, key, value):
    pattern = re.compile(rf"^{re.escape(key)}\s*=")
    updated = []

    for line in lines:
        stripped = line.strip()

        if not stripped.startswith("#") and pattern.match(stripped):
            updated.append(f"{key}={value}")
        else:
            updated.append(line)

    return updated


def set_filters_before_yolo(lines, filters):
    updated = lines[:]

    for index, line in enumerate(updated):
        if line.strip() != "[yolo]":
            continue

        cursor = index - 1

        while cursor >= 0:
            if updated[cursor].strip().startswith("filters"):
                updated[cursor] = f"filters={filters}"
                break

            cursor -= 1

    return updated


def build_cfg(base_text, augmentation):
    lines = base_text.splitlines()

    fixed_values = {
        "width": 640,
        "height": 640,
        "batch": 64,
        "subdivisions": 16,
        "max_batches": 6000,
        "steps": "4800,5400",
        "classes": 2,
    }

    for key, value in fixed_values.items():
        lines = set_cfg_value(lines, key, value)

    aug_keys = [
        "mosaic",
        "jitter",
        "saturation",
        "exposure",
        "hue",
        "random",
    ]

    for key in aug_keys:
        lines = set_cfg_value(lines, key, augmentation[key])

    lines = set_filters_before_yolo(lines, filters=21)

    return "\n".join(lines) + "\n"


def write_cfgs(project_config, augmentation_config):
    project = load_yaml(project_config)
    augmentations = load_yaml(augmentation_config)

    output_dir = Path(project["yolov4"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = download_text(project["yolov4"]["base_cfg_url"])
    (output_dir / "yolov4_base.cfg").write_text(
        base_cfg,
        encoding="utf-8",
    )

    cfgs = {
        "yolov4_default.cfg": augmentations["yolov4_default"],
        "yolov4_no_aug.cfg": augmentations["yolov4_no_aug"],
    }

    for file_name, augmentation in cfgs.items():
        cfg_text = build_cfg(base_cfg, augmentation)
        (output_dir / file_name).write_text(cfg_text, encoding="utf-8")


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--aug-config", default="configs/augmentations.yaml")
    args = parser.parse_args()

    write_cfgs(
        project_config=args.config,
        augmentation_config=args.aug_config,
    )

    print("YOLOv4 cfg files written.")


if __name__ == "__main__":
    main()
