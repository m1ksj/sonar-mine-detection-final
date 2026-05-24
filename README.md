# Sonar Mine Detection

Reproducible side-scan sonar object detection project comparing YOLOv4 and YOLO26n for MILCO/NOMBO detection.

## Quick local use

This repository contains the source code, configs, final result tables and the selected deployment model. It does not contain the dataset or generated experiment folders.

Install the environment:

```powershell
py -m pip install --user pipenv
py -m pipenv install --dev
```

Prepare the public Figshare dataset locally:

```powershell
py -m pipenv run python scripts/setup_data.py --config configs/project.yaml
```

Run tests and checks:

```powershell
py -m pipenv run python -m unittest discover tests
py -m pipenv run pre-commit run --all-files
```

Start the FastAPI deployment endpoint:

```powershell
py -m pipenv run python scripts/run_api.py
```

The selected model must exist at:

```text
models/final/yolo26n_selected.pt
```

Minimal terminal request:

```powershell
$img = Get-ChildItem data/processed/yolo26n/images/test -Filter *.jpg | Select-Object -First 1
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@$($img.FullName)"
```

The `/predict` endpoint accepts one JPEG or PNG image as `multipart/form-data` with field name `file`. It returns JSON with the image name and a list of MILCO/NOMBO detections containing class name, confidence and pixel-space `x1`, `y1`, `x2`, `y2` bounding boxes.

Open the automatically generated FastAPI documentation only when needed:

```text
http://127.0.0.1:8000/docs
```

Optional Streamlit demo, with the FastAPI backend already running in another terminal:

```powershell
py -m pipenv run python scripts/run_streamlit.py
```

Then open:

```text
http://localhost:8501
```

The Streamlit demo is only a visual frontend. The actual deployment interface is the FastAPI endpoint.

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
- final reporting: held-out test metrics as mean +/- standard deviation over seeds
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
git pull origin dev
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
pipenv run pip install --no-cache-dir --force-reinstall `
  torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 `
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
tail -80 $(ls -t experiments/slurm/darknet_test_*.out | head -1)
tail -80 $(ls -t experiments/slurm/darknet_test_*.err | head -1)
```

Run YOLO26n tuning:

```bash
sbatch jobs/tune_yolo26n_array.sbatch
```

Monitor tuning:

```bash
squeue --me
tail -80 $(ls -t experiments/slurm/yolo26n_tune_*.out | head -1)
tail -80 $(ls -t experiments/slurm/yolo26n_tune_*.err | head -1)
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
tail -80 $(ls -t experiments/slurm/final_*.out | head -1)
tail -80 $(ls -t experiments/slurm/final_*.err | head -1)
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

## Optional analysis sources for presentation figures

These files are not committed. They are only needed if you want to regenerate presentation figures locally after the final Habrok runs.

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

Use these files only for local figure generation. YOLO26n `results.csv` files provide true train/validation loss curves and validation mAP curves. YOLOv4 Darknet logs provide training average loss and validation mAP50, but not a clean validation-loss curve.

## Git policy

Do not commit:
- data/
- experiments/
- external/
- runs/
- SLURM logs
- intermediate model weights
- generated result tables before the final run

Commit:
- source code
- configs
- final small result tables
- final figures
- `models/final/model_selection.json`
- `models/final/yolo26n_selected.pt` as the selected deployment artifact

