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

## Command modes

From normal PowerShell, prefix project commands with:

    py -m pipenv run

Example:

    py -m pipenv run python scripts/setup_data.py --config configs/project.yaml
    py -m pipenv run pre-commit run --all-files

Inside the Pipenv shell, do not use py -m pipenv run.

Example:

    python scripts/setup_data.py --config configs/project.yaml
    pre-commit run --all-files

## Minimal deployment workflow

Prepare the dataset:

    py -m pipenv run python scripts/setup_data.py --config configs/project.yaml

Download the selected deployment model:

    py -m pipenv run python scripts/setup_model.py --config configs/project.yaml

Start the local API:

    py -m pipenv run python scripts/run_api.py

## Data setup

The data setup script downloads the public Figshare year archives, extracts them, builds a dataset manifest, creates the fixed train/validation/test split and creates 5 cross-validation folds from the training split for YOLO26n tuning.

YOLOv4 does not use cross-validation. It is only trained in the final seed-based comparison.

## Training workflow

The final training setup is controlled by:

    configs/project.yaml
    configs/augmentations.yaml

The final comparison uses YOLOv4 and YOLO26n with and without online augmentation across three seeds.

YOLO26n tuning jobs are listed in:

    reports/tables/yolo26n_tuning_plan.csv

A single tuning job can be started with:

    python scripts/tune_yolo26n.py --job-index 0

On SLURM, the array job is:

    sbatch jobs/tune_yolo26n_array.sbatch

Before submitting jobs, create the SLURM output directory:

    mkdir -p experiments/slurm

The tuning array runs at most 10 jobs at the same time.

Final training jobs are listed in:

    reports/tables/final_training_plan.csv

A single final job can be checked with:

    python scripts/train_final.py --job-index 6 --dry-run

On SLURM, the final array job is:

    sbatch jobs/train_final_array.sbatch

YOLO26n tuning uses cross-validation. YOLOv4 does not use cross-validation and is only trained in the final seed-based comparison.

## Repository policy

Raw data, processed data, checkpoints, training runs, large logs and model weights are not committed to Git.

Tracked files should be limited to source code, configs, small result tables, final figures and lightweight metadata.

## Hábrók notes

On Hábrók, load Python 3.11 before installing dependencies:

    module purge
    module load Python/3.11.5-GCCcore-13.2.0
    python --version

If PyTorch fails with missing CUDA libraries, reinstall the CUDA 12.1 wheels inside the Pipenv environment:

    pipenv run pip uninstall -y torch torchvision torchaudio nvidia-cublas-cu13 nvidia-cuda-runtime-cu13 nvidia-cudnn-cu13 nvidia-cuda-nvrtc-cu13
    pipenv run pip install --no-cache-dir --force-reinstall torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

Before submitting SLURM jobs, create the output directory:

    mkdir -p experiments/slurm

Do not pull, merge or edit code in the Hábrók clone while array jobs are running.

## API

Start the local API:

    python scripts/run_api.py

Open the FastAPI documentation:

    http://127.0.0.1:8000/docs

Send one sonar image:

    curl -X POST http://127.0.0.1:8000/predict -F "file=@path/to/image.jpg"

The /predict endpoint accepts a JPEG or PNG image and returns detected MILCO/NOMBO objects with class names, confidence scores and bounding boxes. The selected deployment model must first be available through:

    python scripts/setup_model.py

