# Sonar Mine Detection

Real-world side-scan sonar object detection project for Applied Machine Learning.

The project compares YOLOv4 and YOLO26n for MILCO/NOMBO detection. The final repository is designed around scripted data setup, scripted training entry points, small result tables and local API deployment.


## Setup

If Pipenv is not installed, install it with:

    py -m pip install --user pipenv

Then install the project dependencies with:

    py -m pipenv install --dev

Install the pre-commit hooks:

    py -m pipenv run pre-commit install

Run the tests:

    py -m pipenv run python -m unittest discover tests

## Minimal deployment workflow

Prepare the dataset:

    py -m pipenv run python scripts/setup_data.py --config configs/project.yaml

Download the selected deployment model:

    py -m pipenv run python scripts/setup_model.py --config configs/project.yaml

Start the local API:

    py -m pipenv run python scripts/run_api.py

## Training workflow

The final training setup is controlled by:

    configs/project.yaml
    configs/augmentations.yaml

The final comparison uses YOLOv4 and YOLO26n with and without online augmentation across three seeds.

YOLO26n tuning uses cross-validation. YOLOv4 does not use cross-validation and is only trained in the final seed-based comparison.

## Repository policy

Raw data, processed data, checkpoints, training runs, large logs and model weights are not committed to Git.

Tracked files should be limited to source code, configs, small result tables, final figures and lightweight metadata.
