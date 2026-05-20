# Sonar Mine Detection

Real-world side-scan sonar object detection project for Applied Machine Learning.

The project compares YOLOv4 and YOLO26n for MILCO/NOMBO detection. The final repository is designed around scripted data setup, scripted training entry points, small result tables and local API deployment.


## Setup

If Pipenv is not installed, install it with:

    py -m pip install --user pipenv

Then install the project dependencies with:

    py -m pipenv install --dev

## Minimal workflow

1. Install dependencies with Pipenv.
2. Prepare the dataset with scripts/setup_data.py.
3. Download the selected deployment model with scripts/setup_model.py.
4. Start the local API with scripts/run_api.py.

## Repository policy

Raw data, processed data, checkpoints, training runs, large logs and model weights are not committed to Git.

Tracked files should be limited to source code, configs, small result tables, final figures and lightweight metadata.
