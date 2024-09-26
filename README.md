<!-- TITLE -->
# **Hierarchical-Prompting-for-Diffusion-Classifiers**
Code release for "Hierarchical-Prompting-for-Diffusion-Classifiers" (ACCV 2024)

## Installation
Create a conda environment with the following command:
```bash
conda env create -f environment.yml
```

## Train

```bash
python train.py --dataset aircraft --n_trials 1 --to_keep 1 --n_samples 25 --loss l1 --prompt_path prompts/aircraft_name.csv
```

When applied to different datasets, the `set up prompts` part in the `train.py` needs to be modified accordingly, refer to the files in `/prompts/xxx.csv`.

## Test

```bash
python test_hierarchical.py --dataset aircraft --n_trials 1  --to_keep 5 1 --n_samples 5 25  --prompt_path prompts/aircraft_name.csv
```

## Citation
