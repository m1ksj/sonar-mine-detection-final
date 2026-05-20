from argparse import ArgumentParser
from pathlib import Path
import hashlib

import requests


def sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def read_json(path):
    import json

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def download_file(url, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()

        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--artifact",
        default="models/final/model_artifact.json",
    )
    args = parser.parse_args()

    artifact = read_json(args.artifact)
    artifact_url = artifact.get("artifact_url", "")
    local_path = Path(artifact["local_path"])
    expected_hash = artifact.get("sha256", "")

    if not artifact_url:
        raise ValueError(
            "No artifact_url set in models/final/model_artifact.json."
        )

    if not local_path.exists():
        download_file(artifact_url, local_path)

    if expected_hash and sha256(local_path) != expected_hash:
        raise ValueError("Downloaded model checksum does not match.")

    print(f"Model ready: {local_path}")


if __name__ == "__main__":
    main()
