# Sonar Mine Detection

Real-world side-scan sonar object detection project for Applied Machine Learning.
The project compares YOLOv4 and YOLO26n for MILCO/NOMBO detection using scripted data setup, fixed splits, cross-validation for YOLO26n tuning and final seed-based model comparison.

## Local setup

Install Pipenv if needed:

    py -m pip install --user pipenv

Install dependencies and run checks:

    py -m pipenv install --dev
    py -m pipenv run pre-commit install
    py -m pipenv run python -m unittest discover tests
    py -m pipenv run pre-commit run --all-files

## Data setup

The data setup script downloads the public Figshare year archives, extracts them, builds a manifest, creates fixed train/validation/test splits and creates 5 cross-validation folds from the training split.

    py -m pipenv run python scripts/setup_data.py --config configs/project.yaml

The Darknet export intentionally stores labels next to images because Darknet expects:

    images/train/example.jpg
    images/train/example.txt

## Experiment design

YOLO26n tuning uses 5-fold cross-validation over:

    learning_rates: [0.001, 0.005, 0.01]
    batch_sizes: [8, 16]
    optimizers: [SGD, AdamW]
    patience: 50

This gives 60 tuning jobs. The best setup is selected by mean validation mAP50-95 across folds.

Final training compares YOLOv4 and YOLO26n with and without augmentation across three seeds.

## Habrok fresh-clone workflow

From Habrok login node:

    git clone <repo-url> sonar-mine-detection-final
    cd sonar-mine-detection-final

Create scratch-backed runtime directories:

    mkdir -p /scratch/$USER/sonar-mine-detection-final/data
    mkdir -p /scratch/$USER/sonar-mine-detection-final/experiments
    mkdir -p /scratch/$USER/sonar-mine-detection-final/external
    ln -sfn /scratch/$USER/sonar-mine-detection-final/data data
    ln -sfn /scratch/$USER/sonar-mine-detection-final/experiments experiments
    ln -sfn /scratch/$USER/sonar-mine-detection-final/external external

Load Python and install dependencies:

    module purge
    module load Python/3.11.5-GCCcore-13.2.0
    export PATH="$HOME/.local/bin:$PATH"
    python -m pip install --user pipenv
    pipenv install --dev

If needed, install CUDA 12.1 PyTorch wheels inside Pipenv:

    pipenv run pip install --no-cache-dir --force-reinstall torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

Prepare data and plan tables:

    pipenv run python scripts/setup_data.py --config configs/project.yaml
    pipenv run python scripts/setup_yolov4_configs.py --config configs/project.yaml
    pipenv run python scripts/create_yolo26n_tuning_plan.py
    pipenv run python scripts/create_final_training_plan.py

Prepare Darknet for YOLOv4:

    pipenv run python scripts/setup_darknet.py --clone
    bash scripts/build_darknet_habrok.sh

Run YOLO26n tuning:

    sbatch jobs/tune_yolo26n_array.sbatch

After tuning finishes:

    pipenv run python scripts/collect_yolo26n_tuning_results.py
    pipenv run python scripts/select_yolo26n_hparams.py

Run final training:

    pipenv run python scripts/train_final.py --job-index 0 --dry-run --darknet-bin external/yolov4/darknet/darknet --pretrained external/yolov4/yolov4.conv.137
    pipenv run python scripts/train_final.py --job-index 6 --dry-run
    sbatch jobs/train_final_array.sbatch

After final training finishes:

    pipenv run python scripts/collect_final_results.py
    pipenv run python scripts/select_deployment_model.py

## Deployment

The API expects a selected YOLO26n model at:

    models/final/yolo26n_selected.pt

Start the local API:

    py -m pipenv run python scripts/run_api.py

Open:

    http://127.0.0.1:8000/docs

Send one sonar image:

    curl -X POST http://127.0.0.1:8000/predict -F "file=@path/to/image.jpg"

## Repository policy

Raw data, processed data, checkpoints, training runs, large logs and model weights are not committed to Git.
Tracked files are limited to source code, configs, final result tables, figures and lightweight metadata.
