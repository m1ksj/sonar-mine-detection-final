# Sonar Mine Detection

Reproducible side-scan sonar object detection project comparing YOLOv4 and YOLO26n for MILCO/NOMBO detection.

## First-time setup

This repository contains the source code, configs, final result tables and the selected deployment model. It does not contain the dataset or generated experiment folders.

The setup is Docker-based. This avoids local Python/Pipenv version issues and should work the same on Windows, macOS and Linux, as long as Docker is installed.

### Requirements

- Git
- Docker with Docker Compose

### Setup and run

Clone the repository, build the Docker image, prepare the dataset, run the tests and start the API plus Streamlit demo:

```bash
git clone -b dev https://github.com/m1ksj/sonar-mine-detection-final.git
cd sonar-mine-detection-final

docker compose build
docker compose run --rm api python scripts/setup_data.py --config configs/project.yaml
docker compose run --rm api python -m unittest discover tests
docker compose run --rm api pre-commit run --all-files
docker compose up
```

After `docker compose up`, open:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8501
```

The FastAPI endpoint is the actual deployment interface. The Streamlit app is only a visual frontend for the same model.

To stop the services, press `Ctrl + C` and then run:

```bash
docker compose down
```

### Prediction endpoint

The selected model is included in the repository and must be available at:

```text
models/final/yolo26n_selected.pt
```

The `/predict` endpoint accepts one JPEG or PNG image as `multipart/form-data` with field name `file`. It returns JSON with the image name and a list of MILCO/NOMBO detections containing class name, confidence and pixel-space `x1`, `y1`, `x2`, `y2` bounding boxes.

Minimal request on Windows PowerShell:

```powershell
$img = Get-ChildItem data/processed/yolo26n/images/test -Filter *.jpg | Select-Object -First 1
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@$($img.FullName)"
```

Minimal request on macOS / Linux:

```bash
img=$(find data/processed/yolo26n/images/test -name "*.jpg" | head -n 1)
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@${img}"
```

### Scope of the Docker setup

The Docker setup is intended for local reproducibility of the dataset setup, tests, FastAPI deployment endpoint and Streamlit demo.

The Habrok workflow is separate and is only needed to reproduce the GPU training experiments and final training runs.

## Full Habrok reproducibility run

Connect from a local terminal. Replace `<your-s-number>` with your own Habrok/RUG student account:

```bash
ssh <your-s-number>@login1.hb.hpc.rug.nl
```

All following commands are executed on the Habrok login node.

Start from a clean clone:

```bash
cd $HOME
rm -rf sonar-mine-detection-final
rm -rf /scratch/$USER/sonar-mine-detection-final

git clone https://github.com/m1ksj/sonar-mine-detection-final.git sonar-mine-detection-final
cd sonar-mine-detection-final
git checkout dev
git pull --ff-only origin dev
```

Create scratch-backed runtime folders:

```bash
mkdir -p /scratch/$USER/sonar-mine-detection-final/data
mkdir -p /scratch/$USER/sonar-mine-detection-final/experiments
mkdir -p /scratch/$USER/sonar-mine-detection-final/external

ln -sfn /scratch/$USER/sonar-mine-detection-final/data data
ln -sfn /scratch/$USER/sonar-mine-detection-final/experiments experiments
ln -sfn /scratch/$USER/sonar-mine-detection-final/external external
```

Install the Python environment:

```bash
module purge
module load Python/3.11.5-GCCcore-13.2.0
export PATH="$HOME/.local/bin:$PATH"

python -m pip install --user pipenv
pipenv install --dev
```

Install CUDA 12.1 PyTorch wheels:

```bash
pipenv run pip install --no-cache-dir --force-reinstall \
  torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Prepare data, YOLOv4 configs and job plans:

```bash
pipenv run python scripts/setup_data.py --config configs/project.yaml
pipenv run python scripts/setup_yolov4_configs.py --config configs/project.yaml
pipenv run python scripts/create_yolo26n_tuning_plan.py
pipenv run python scripts/create_final_training_plan.py
```

Prepare Darknet:

```bash
pipenv run python scripts/setup_darknet.py
bash scripts/build_darknet_habrok.sh
```

Darknet is built with:

```text
GPU=1
OPENCV=1
CUDNN=0
CUDNN_HALF=0
ARCH=compute_70,compute_80
```

Reason:
- OPENCV=1 is required because `yolov4_default.cfg` uses mosaic=1.
- CUDNN=0 and CUDNN_HALF=0 are used for stable Habrok execution.
- compute_70 supports V100 nodes and compute_80 supports A100 nodes.

Check Darknet on a GPU node:

```bash
sbatch jobs/test_darknet.sbatch
squeue --me
```

Run YOLO26n tuning:

```bash
sbatch jobs/tune_yolo26n_array.sbatch
```

Monitor tuning:

```bash
squeue --me
```

Collect tuning results:

```bash
pipenv run python scripts/collect_yolo26n_tuning_results.py
pipenv run python scripts/select_yolo26n_hparams.py
```

Run final training:

```bash
sbatch jobs/train_final_array.sbatch
```

After final training has finished, run the final test evaluations:

```bash
sbatch jobs/evaluate_yolo26n_final_test.sbatch
sbatch jobs/evaluate_yolov4_iou_sweep.sbatch
```

Monitor final jobs:

```bash
squeue --me
```

Collect final results, seed summaries and the selected API model:

```bash
pipenv run python scripts/collect_final_results.py
pipenv run python scripts/collect_final_class_results.py
pipenv run python scripts/summarize_final_results.py
pipenv run python scripts/summarize_final_class_results.py
pipenv run python scripts/select_deployment_model.py
pipenv run python scripts/copy_selected_model.py
```

YOLO26n learning curves are stored in each run folder as `results.csv`. YOLOv4 weights and logs are stored under the corresponding `experiments/final/final_yolov4_...` folders.

## Copy Habrok artifacts back to the local repo
### Windows PowerShell

Run this from local Windows PowerShell, not from the SSH session. Replace `<your-s-number>` with the Habrok account that ran the experiments.

```powershell
$HabrokUser = "<your-s-number>"
$Remote = "$HabrokUser@login1.hb.hpc.rug.nl"
$RemoteProject = "~/sonar-mine-detection-final"

New-Item -ItemType Directory -Force reports/tables | Out-Null
New-Item -ItemType Directory -Force configs | Out-Null
New-Item -ItemType Directory -Force models/final | Out-Null

scp "${Remote}:${RemoteProject}/reports/tables/yolo26n_tuning_results.csv" reports/tables/
scp "${Remote}:${RemoteProject}/reports/tables/yolo26n_hparam_summary.csv" reports/tables/
scp "${Remote}:${RemoteProject}/reports/tables/final_run_results.csv" reports/tables/
scp "${Remote}:${RemoteProject}/reports/tables/final_seed_summary.csv" reports/tables/
scp "${Remote}:${RemoteProject}/reports/tables/final_class_results.csv" reports/tables/
scp "${Remote}:${RemoteProject}/reports/tables/final_class_summary.csv" reports/tables/
scp "${Remote}:${RemoteProject}/configs/yolo26n_best.yaml" configs/
scp "${Remote}:${RemoteProject}/models/final/model_selection.json" models/final/
scp "${Remote}:${RemoteProject}/models/final/yolo26n_selected.pt" models/final/
```

Optional integrity check for the selected model:

```powershell
Get-FileHash models/final/yolo26n_selected.pt -Algorithm SHA256
ssh $Remote "cd $RemoteProject && sha256sum models/final/yolo26n_selected.pt"
```

### macOS/Linux terminal

Run this from your local machine, not from the SSH session. Replace `<your-s-number>` with the Habrok account that ran the experiments.

```bash
HABROK_USER="<your-s-number>"
REMOTE="${HABROK_USER}@login1.hb.hpc.rug.nl"
REMOTE_PROJECT="~/sonar-mine-detection-final"

mkdir -p reports/tables
mkdir -p configs
mkdir -p models/final

scp "${REMOTE}:${REMOTE_PROJECT}/reports/tables/yolo26n_tuning_results.csv" reports/tables/
scp "${REMOTE}:${REMOTE_PROJECT}/reports/tables/yolo26n_hparam_summary.csv" reports/tables/
scp "${REMOTE}:${REMOTE_PROJECT}/reports/tables/final_run_results.csv" reports/tables/
scp "${REMOTE}:${REMOTE_PROJECT}/reports/tables/final_seed_summary.csv" reports/tables/
scp "${REMOTE}:${REMOTE_PROJECT}/reports/tables/final_class_results.csv" reports/tables/
scp "${REMOTE}:${REMOTE_PROJECT}/reports/tables/final_class_summary.csv" reports/tables/
scp "${REMOTE}:${REMOTE_PROJECT}/configs/yolo26n_best.yaml" configs/
scp "${REMOTE}:${REMOTE_PROJECT}/models/final/model_selection.json" models/final/
scp "${REMOTE}:${REMOTE_PROJECT}/models/final/yolo26n_selected.pt" models/final/
```

Optional integrity check:

```bash
sha256sum models/final/yolo26n_selected.pt
ssh "$REMOTE" "cd $REMOTE_PROJECT && sha256sum models/final/yolo26n_selected.pt"
```

## Optional analysis sources for presentation figures

These files are not committed. They are only needed to regenerate presentation figures locally after the final Habrok runs.

```powershell
$HabrokUser = "<your-s-number>"
$Remote = "$HabrokUser@login1.hb.hpc.rug.nl"
$RemoteProject = "~/sonar-mine-detection-final"

New-Item -ItemType Directory -Force experiments/analysis_sources/yolo26n_final_results_csv | Out-Null
New-Item -ItemType Directory -Force experiments/analysis_sources/yolov4_slurm_logs | Out-Null

scp "${Remote}:${RemoteProject}/experiments/final/final_yolo26n_yolo26n_yolov4_style_seed117/results.csv" experiments/analysis_sources/yolo26n_final_results_csv/yolo26n_yolov4_style_seed117_results.csv
scp "${Remote}:${RemoteProject}/experiments/final/final_yolo26n_yolo26n_yolov4_style_seed221/results.csv" experiments/analysis_sources/yolo26n_final_results_csv/yolo26n_yolov4_style_seed221_results.csv
scp "${Remote}:${RemoteProject}/experiments/final/final_yolo26n_yolo26n_yolov4_style_seed333/results.csv" experiments/analysis_sources/yolo26n_final_results_csv/yolo26n_yolov4_style_seed333_results.csv
scp "${Remote}:${RemoteProject}/experiments/final/final_yolo26n_yolo26n_no_aug_seed117/results.csv" experiments/analysis_sources/yolo26n_final_results_csv/yolo26n_no_aug_seed117_results.csv
scp "${Remote}:${RemoteProject}/experiments/final/final_yolo26n_yolo26n_no_aug_seed221/results.csv" experiments/analysis_sources/yolo26n_final_results_csv/yolo26n_no_aug_seed221_results.csv
scp "${Remote}:${RemoteProject}/experiments/final/final_yolo26n_yolo26n_no_aug_seed333/results.csv" experiments/analysis_sources/yolo26n_final_results_csv/yolo26n_no_aug_seed333_results.csv
scp "${Remote}:${RemoteProject}/experiments/slurm/final_*.out" experiments/analysis_sources/yolov4_slurm_logs/
scp "${Remote}:${RemoteProject}/experiments/slurm/final_*.err" experiments/analysis_sources/yolov4_slurm_logs/
scp "${Remote}:${RemoteProject}/reports/tables/final_training_plan.csv" experiments/analysis_sources/final_training_plan.csv
```


YOLO26n `results.csv` files provide true train/validation loss curves and validation mAP curves. YOLOv4 Darknet logs provide training average loss and validation mAP50, but not a clean validation-loss curve.



## Workflow in this repository

All changes should go through a feature branch and Pull Request. Direct pushes to `dev` and `main` should be avoided.

Start from the latest `dev` branch:

```bash
git checkout dev
git pull --ff-only origin dev
```

Create a feature branch:

```bash
git checkout -b docs/example
```

Before committing, run the tests and pre-commit checks:

```bash
docker compose run --rm api python -m unittest discover tests
docker compose run --rm api pre-commit run --all-files
```

Commit and push the branch:

```bash
git status
git add <changed-files>
git commit -m "Example message"
git push -u origin docs/example
```

Then open a Pull Request on GitHub with:

```text
base: dev
compare: your feature branch
```

Do not commit raw data, training runs, experiment folders, intermediate checkpoints or temporary analysis files. The selected deployment artifact `models/final/yolo26n_selected.pt` is intentionally included.

## Final project design

Data:
- Public Figshare side-scan sonar dataset, year archives 2010, 2015, 2017, 2018 and 2021.
- Fixed train/validation/test split with seed 117.
- Split stratified by image category: `empty`, `milco_only`, `nombo_only`, `mixed`.
- Five cross-validation folds created only from the training split.

YOLO26n tuning:
- learning rates: 0.001, 0.005, 0.01
- batch sizes: 8, 16
- optimizers: SGD, AdamW
- patience fixed at 100
- tuning metric: mean validation mAP50-95 across five folds

Final comparison:
- YOLOv4 default augmentation
- YOLOv4 no augmentation
- YOLO26n YOLOv4-style augmentation
- YOLO26n no augmentation
- seeds: 117, 221, 333
- final reporting: held-out test metrics over seeds; presentation tables report mean +/- SEM
- YOLO26n final training uses the selected CV hyperparameters with patience=100

Deployment selection:
- select only among YOLO26n YOLOv4-style final runs
- select by validation mAP50-95, not test mAP50-95
- keep the test set only for final reporting

Main final artifacts:

```text
configs/yolo26n_best.yaml
models/final/model_selection.json
models/final/yolo26n_selected.pt
reports/tables/yolo26n_tuning_results.csv
reports/tables/yolo26n_hparam_summary.csv
reports/tables/final_run_results.csv
reports/tables/final_seed_summary.csv
reports/tables/final_class_results.csv
reports/tables/final_class_summary.csv
```
