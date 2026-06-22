# SiM

SiM is an inference-time task-vector merging framework that leverages task-specific subspace information to route samples and apply task-adaptive merged parameters.

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate sim
```

## Run

```bash
python -m src.sim \
  --model ViT-B-32 \
  --data-location /path/to/datasets \
  --model-ckpt-dir /path/to/checkpoints \
  --num-tasks 8 \
  --merging-method <base_merging_method> \
  --n-samples 32 \
  --k 0.1
```
Replace `<base_merging_method>` with one of `emr`, `tm_ta`, `tm_ties`, or `tsv`.

## Acknowledgements

This implementation is based on or adapted from the following repositories. We thank the authors for releasing their code.

- Task Arithmetic: https://github.com/mlfoundations/task_vectors
- EMR-Merging: https://github.com/harveyhuang18/EMR_Merging
- TALL Masks: https://github.com/nik-dim/tall_masks
- Task Singular Vectors: https://github.com/AntoAndGar/task_singular_vectors
