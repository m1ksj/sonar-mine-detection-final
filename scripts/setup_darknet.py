from argparse import ArgumentParser
from pathlib import Path
import subprocess

import requests
import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def download_file(url, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        return

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                file.write(chunk)


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args()

    yolov4 = load_yaml(args.config)["yolov4"]
    external_dir = Path(yolov4["external_dir"])
    darknet_dir = external_dir / "darknet"

    download_file(
        yolov4["pretrained_url"],
        external_dir / "yolov4.conv.137",
    )

    if not darknet_dir.exists():
        subprocess.run(
            ["git", "clone", yolov4["darknet_repo"], str(darknet_dir)],
            check=True,
        )

    print("Darknet source and YOLOv4 pretrained weights are ready.")


if __name__ == "__main__":
    main()
