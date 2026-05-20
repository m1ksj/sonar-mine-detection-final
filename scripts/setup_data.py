from argparse import ArgumentParser
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def make_data_dirs(config: dict) -> None:
    data_config = config["data"]
    path_config = config["paths"]

    dirs = [
        data_config["raw_dir"],
        data_config["processed_dir"],
        data_config["splits_dir"],
        data_config["metadata_dir"],
        path_config["darknet_data_dir"],
        Path(path_config["yolo26n_data_yaml"]).parent,
    ]

    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    make_data_dirs(config)

    print("Data directories created.")


if __name__ == "__main__":
    main()
