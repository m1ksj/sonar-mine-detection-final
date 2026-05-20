from argparse import ArgumentParser
from pathlib import Path
import csv

import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def export_settings(config, output_path):
    rows = []

    for setting, values in config.items():
        framework = values.get("framework", "")

        for parameter, value in values.items():
            if parameter == "framework":
                continue

            rows.append(
                {
                    "setting": setting,
                    "framework": framework,
                    "parameter": parameter,
                    "value": value,
                }
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["setting", "framework", "parameter", "value"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/augmentations.yaml")
    parser.add_argument(
        "--output",
        default="reports/tables/augmentation_settings.csv",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    rows = export_settings(config, args.output)

    print(f"Augmentation settings exported: {len(rows)} rows")


if __name__ == "__main__":
    main()
