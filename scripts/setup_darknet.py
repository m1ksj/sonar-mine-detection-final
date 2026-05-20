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
        return output_path

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()

        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    return output_path


def clone_darknet(repo_url, output_dir):
    output_dir = Path(output_dir)

    if output_dir.exists():
        return output_dir

    subprocess.run(
        ["git", "clone", repo_url, str(output_dir)],
        check=True,
    )

    return output_dir


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--clone", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    yolov4_config = config["yolov4"]

    external_dir = Path(yolov4_config["external_dir"])
    pretrained_path = external_dir / "yolov4.conv.137"

    download_file(
        url=yolov4_config["pretrained_url"],
        output_path=pretrained_path,
    )

    print(f"YOLOv4 pretrained weights: {pretrained_path}")

    if args.clone:
        darknet_dir = external_dir / "darknet"
        clone_darknet(yolov4_config["darknet_repo"], darknet_dir)
        print(f"Darknet repository: {darknet_dir}")


if __name__ == "__main__":
    main()
