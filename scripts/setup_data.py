from argparse import ArgumentParser
from pathlib import Path
from shutil import copyfile
from zipfile import ZipFile
import hashlib
import sys

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def download_file(url, output_path):
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()

        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def get_figshare_files(config):
    url = config["data"]["figshare_files_url"]
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def prepare_archives(config, source_zip=None):
    raw_dir = Path(config["data"]["raw_dir"])
    archives_dir = raw_dir / "archives"
    archives_dir.mkdir(parents=True, exist_ok=True)

    if source_zip:
        output_path = archives_dir / Path(source_zip).name
        copyfile(source_zip, output_path)
        return [output_path]

    wanted_files = set(config["data"]["download_files"])
    figshare_files = get_figshare_files(config)

    archive_paths = []

    for file_info in figshare_files:
        name = file_info["name"]

        if name not in wanted_files:
            continue

        output_path = archives_dir / name
        archive_paths.append(output_path)

        if not output_path.exists():
            print(f"Downloading {name}")
            download_file(file_info["download_url"], output_path)

    return archive_paths


def extract_archives(archive_paths, raw_dir, force=False):
    raw_dir = Path(raw_dir)
    marker = raw_dir / ".extracted"

    if marker.exists() and not force:
        return

    for archive_path in archive_paths:
        print(f"Extracting {archive_path.name}")

        with ZipFile(archive_path, "r") as archive:
            archive.extractall(raw_dir)

    marker.write_text("done\n", encoding="utf-8")


def setup_data(config, source_zip=None, force=False):
    from sonar_mine_detection.data.audit import build_manifest
    from sonar_mine_detection.data.split import (
        make_cv_folds,
        make_train_val_test_split,
    )

    archive_paths = prepare_archives(config, source_zip=source_zip)
    extract_archives(archive_paths, config["data"]["raw_dir"], force=force)

    metadata_dir = Path(config["data"]["metadata_dir"])
    manifest_path = metadata_dir / "dataset_manifest.csv"

    manifest = build_manifest(
        raw_dir=config["data"]["raw_dir"],
        years=config["data"]["years"],
        output_path=manifest_path,
    )

    train_rows, val_rows, test_rows = make_train_val_test_split(
        manifest_path=manifest_path,
        output_dir=config["data"]["splits_dir"],
        seed=config["split"]["seed"],
        train_size=config["split"]["train"],
        val_size=config["split"]["val"],
        test_size=config["split"]["test"],
        stratify_by=config["split"]["stratify_by"],
    )

    cv_summary = make_cv_folds(
        source_csv_path=Path(config["data"]["splits_dir"]) / "train.csv",
        output_dir=Path(config["data"]["splits_dir"]) / "cv",
        n_splits=config["cv"]["folds"],
        seed=config["cv"]["seed"],
        stratify_by=config["split"]["stratify_by"],
    )

    print(f"Images found: {len(manifest)}")
    print(f"Manifest written to: {manifest_path}")
    print(
        "Split sizes: "
        f"train={len(train_rows)}, "
        f"val={len(val_rows)}, "
        f"test={len(test_rows)}"
    )
    print(f"CV folds written: {len(cv_summary)}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--source-zip", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_data(config, source_zip=args.source_zip, force=args.force)


if __name__ == "__main__":
    main()
