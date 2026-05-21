# Sonar Mine Detection

Reproducible side-scan sonar object detection project comparing YOLOv4 and YOLO26n for MILCO/NOMBO detection.

## Experiment design

Data:
- Public Figshare sonar dataset, year archives 2010, 2015, 2017, 2018 and 2021.
- Fixed train/validation/test split with seed 117.
- Five cross-validation folds created from the training split.

YOLO26n tuning:
- learning rates: 0.001, 0.005, 0.01
- batch sizes: 8, 16
- optimizers: SGD, AdamW
- patience fixed at 50
- tuning metric: mean validation mAP50-95 across five folds

Final comparison:
- YOLOv4 default augmentation
- YOLOv4 no augmentation
- YOLO26n YOLOv4-style augmentation
- YOLO26n no augmentation
- seeds: 117, 221, 333
- final reporting: held-out test metrics as mean +/- standard deviation over seeds

Deployment selection:
- select only among YOLO26n YOLOv4-style final runs
- select by validation mAP50-95, not test mAP50-95
- keep the test set only for final reporting

## Local checks

```powershell
py -m pip install --user pipenv
py -m pipenv install --dev
py -m pipenv run python -m unittest discover tests
py -m pipenv run pre-commit run --all-files
```

## Habrok access

Connect from a local terminal:

```bash
ssh <s-number>@login1.hb.hpc.rug.nl
```

For this project account:

```bash
ssh s5626595@login1.hb.hpc.rug.nl
```

All following commands are executed on the Habrok login node.

## Fresh Habrok run

Start from a clean clone:

```bash
cd $HOME
rm -rf sonar-mine-detection-final
rm -rf /scratch/$USER/sonar-mine-detection-final

git clone https://github.com/m1ksj/sonar-mine-detection sonar-mine-detection-final
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

The Darknet export writes labels next to images because Darknet expects:

```text
images/train/example.jpg
images/train/example.txt
```

Prepare Darknet:

```bash
pipenv run python scripts/setup_darknet.py
bash scripts/build_darknet_habrok.sh
```

The Darknet build is intentionally:

```text
GPU=1
OPENCV=1
CUDNN=0
CUDNN_HALF=0
ARCH=compute_70,compute_80
```

Reason:
- OPENCV=1 is required because yolov4_default.cfg uses mosaic=1.
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

Monitor final training:

```bash
squeue --me
tail -80 $(ls -t experiments/slurm/final_*.out | head -1)
tail -80 $(ls -t experiments/slurm/final_*.err | head -1)
```

Collect final results, seed summaries and select the API model:

```bash
pipenv run python scripts/collect_final_results.py
pipenv run python scripts/summarize_final_results.py
pipenv run python scripts/select_deployment_model.py
pipenv run python scripts/copy_selected_model.py
```

Main result tables:

```text
reports/tables/yolo26n_tuning_results.csv
reports/tables/yolo26n_hparam_summary.csv
reports/tables/final_run_results.csv
reports/tables/final_seed_summary.csv
```

YOLO26n learning curves are stored in each run folder as results.csv.
YOLOv4 weights and logs are stored under each run-specific experiments/final/final_yolov4_... folder.

## Copy result tables back to the local repo

Run this from local Windows PowerShell, not from the SSH session:

```powershell
scp s5626595@login1.hb.hpc.rug.nl:~/sonar-mine-detection-final/reports/tables/yolo26n_tuning_results.csv reports/tables/
scp s5626595@login1.hb.hpc.rug.nl:~/sonar-mine-detection-final/reports/tables/yolo26n_hparam_summary.csv reports/tables/
scp s5626595@login1.hb.hpc.rug.nl:~/sonar-mine-detection-final/reports/tables/final_run_results.csv reports/tables/
scp s5626595@login1.hb.hpc.rug.nl:~/sonar-mine-detection-final/reports/tables/final_seed_summary.csv reports/tables/
scp s5626595@login1.hb.hpc.rug.nl:~/sonar-mine-detection-final/configs/yolo26n_best.yaml configs/
```

Do not commit model weights.

## API

Start locally after models/final/yolo26n_selected.pt exists:

```powershell
py -m pipenv run python scripts/run_api.py
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Git policy

Do not commit:
- data/
- experiments/
- external/
- runs/
- model weights
- SLURM logs
- generated result tables before the final run

Commit only source code, configs, final small report tables and figures.