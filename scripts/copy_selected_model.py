from argparse import ArgumentParser
from pathlib import Path
import json
import shutil

import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument(
        "--selection",
        default="models/final/model_selection.json",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)

    with open(args.selection, "r", encoding="utf-8") as file:
        selection = json.load(file)

    source = Path(selection["selected"]["best_weights"])
    target = Path(config["deployment"]["local_model_path"])
    target.parent.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(source, target)
    print(f"Copied selected model to: {target}")


if __name__ == "__main__":
    main()
