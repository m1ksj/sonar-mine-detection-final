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

# After final training has finished, run the model-specific test evaluations:
sbatch jobs/evaluate_yolo26n_final_test.sbatch
sbatch jobs/evaluate_yolov4_iou_sweep.sbatch
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
pipenv run python scripts/collect_final_class_results.py
pipenv run python scripts/summarize_final_results.py
pipenv run python scripts/summarize_final_class_results.py
pipenv run python scripts/select_deployment_model.py
pipenv run python scripts/copy_selected_model.py
```

Main result tables:

```text
reports/tables/yolo26n_tuning_results.csv
reports/tables/yolo26n_hparam_summary.csv
reports/tables/final_run_results.csv
reports/tables/final_seed_summary.csv
reports/tables/final_class_results.csv
reports/tables/final_class_summary.csv
models/final/model_selection.json
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
scp s5626595@login1.hb.hpc.rug.nl:~/sonar-mine-detection-final/reports/tables/final_class_results.csv reports/tables/
scp s5626595@login1.hb.hpc.rug.nl:~/sonar-mine-detection-final/reports/tables/final_class_summary.csv reports/tables/
scp s5626595@login1.hb.hpc.rug.nl:~/sonar-mine-detection-final/configs/yolo26n_best.yaml configs/
scp s5626595@login1.hb.hpc.rug.nl:~/sonar-mine-detection-final/models/final/model_selection.json models/final/
```

The selected PyTorch weight file is intentionally not committed because *.pt files are ignored. To run the local API or Streamlit demo, copy it locally as well:

```powershell
scp s5626595@login1.hb.hpc.rug.nl:~/sonar-mine-detection-final/models/final/yolo26n_selected.pt models/final/
```

Commit only the small CSV/YAML/JSON artifacts. Do not commit model weights.

## Copy optional analysis sources for presentation figures

These files are not committed. They are only needed if you want to regenerate presentation figures locally after the final Habrok runs.

Run this from local Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force experiments\analysis_sources\yolo26n_final_results_csv
New-Item -ItemType Directory -Force experiments\analysis_sources\yolov4_slurm_logs

scp 's5626595@login1.hb.hpc.rug.nl:~/sonar-mine-detection-final/experiments/final/final_yolo26n_*/results.csv' experiments/analysis_sources/yolo26n_final_results_csv/

scp 's5626595@login1.hb.hpc.rug.nl:~/sonar-mine-detection-final/experiments/slurm/final_28990301_*.out' experiments/analysis_sources/yolov4_slurm_logs/
scp 's5626595@login1.hb.hpc.rug.nl:~/sonar-mine-detection-final/experiments/slurm/final_28990301_*.err' experiments/analysis_sources/yolov4_slurm_logs/

scp s5626595@login1.hb.hpc.rug.nl:~/sonar-mine-detection-final/reports/tables/final_training_plan.csv experiments/analysis_sources/final_training_plan.csv
```

These analysis sources stay inside `experiments/`, which is ignored by Git. They should not be committed.

Use these files only for local figure generation:

- YOLO26n `results.csv` files provide true train/validation loss curves and validation mAP curves.
- YOLOv4 Darknet logs provide training average loss and validation mAP50, but not a clean validation-loss curve.
- Therefore, YOLOv4 overfitting should be shown as training average loss plus validation mAP50, not as train-vs-validation loss.

## API

The deployed model is served through a local FastAPI endpoint. The API requires the selected YOLO26n weight file at:

```text
models/final/yolo26n_selected.pt
```

This file is not tracked by Git because model weights are ignored. If it is missing, copy it from Habrok or place the selected model weight at that path.

Start the API locally:

```powershell
python scripts/run_api.py
```

Open the automatically generated API documentation:

```text
http://127.0.0.1:8000/docs
```

Example request from Windows PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@data/processed/yolo26n/images/test/0002_2015.jpg"
```

The response is a JSON object containing the input image name and a list of detections with class ID, class name, confidence and bounding-box coordinates in pixel xyxy format.

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

## Streamlit demo UI

The project also includes an optional Streamlit demo interface. The API remains the actual deployment interface; Streamlit is only a visual frontend for demonstration.

Start the FastAPI backend in one terminal:

```bash
python scripts/run_api.py
```

Start the Streamlit demo in a second terminal:

```bash
python scripts/run_streamlit.py
```

Then open:

```text
http://localhost:8501
```

The demo can either select a local test image from data/processed/yolo26n/images/test or accept a manual image upload. If the corresponding YOLO label file is available, the demo overlays ground-truth boxes in green and model predictions in red.

