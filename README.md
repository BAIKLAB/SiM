# SiM

SiM is a task-vector merging method that uses task-specific subspace information to route samples and apply task-adaptive merged parameters at inference time.

## Setup

Create the conda environment and activate it.
```bash
conda env create -f environment.yml
conda activate sim

## Run
```bash
python -m src.sim \
  --model ViT-B-32 \
  --data-location /path/to/datasets \
  --model-ckpt-dir /path/to/checkpoints/original \
  --openclip-cachedir /path/to/.cache/open_clip \
  --num-tasks 8 \
  --merging-method emr \
  --n-samples 32 \
  --k 0.1

## Acknowledgement
Our implementation references the code below, thanks to them.
Task Arithmetic: https://github.com/mlfoundations/task_vectors
EMR-Merging: https://github.com/harveyhuang18/EMR_Merging
TALL Masks: https://github.com/nik-dim/tall_masks
Task Singular Vectors: https://github.com/AntoAndGar/task_singular_vectors
