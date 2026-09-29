# Texas A&M YOLO Dataset Preparation Pipeline

This is a small, CPU-only dataset analysis and training-manifest pipeline for Texas A&M Grace. It reads YOLO `.txt` labels and directory listings; it never opens image pixels, uses ML inference, or requires a third-party Python package.

## Architecture

```text
GitHub Repository
       ↓
git clone / git pull on Grace
       ↓
SLURM Job
       ↓
Python Dataset Analysis
       ↓
Dataset stored in /scratch
       ↓
Analysis + Balanced Manifest
```

Important design decisions:

> Original images and labels are never changed.

> Training balancing is performed by generating a training manifest. Repeated paths in `balanced_train.txt` are intentional; no files are duplicated.

> Validation and test datasets are not balanced.

## Repository layout

`scripts/analyze_and_balance.py` performs validation, statistics, and training-manifest generation. `scripts/pull_dataset.sh` handles Git and archive downloads. `jobs/` contains Grace SLURM wrappers, and `sample_data/` creates a tiny local fixture.

## Local quick start

```bash
python3 sample_data/create_sample_dataset.py
python3 scripts/analyze_and_balance.py \
  --dataset-root sample_data/generated_dataset \
  --config configs/example_config.toml \
  --output-dir sample_output
python3 -m unittest discover -s tests -v
```

The analyzer writes `analysis_summary.txt`, `dataset_summary.csv`, `category_summary.csv`, `class_summary.csv`, `size_summary.csv`, `validation_report.txt`, `balanced_train.txt`, and `dataset_summary.json`. `size_summary.csv` reports normalized bounding-box width, height, and area statistics by split and class; it does not require opening image pixels.

## Grace usage

```bash
ssh <netid>@grace.hprc.tamu.edu
git clone <repo>
cd dataset_pipeline
git pull
```

Run an analysis job:

```bash
sbatch jobs/analyze_dataset.sbatch \
  /scratch/user/$USER/DFire/dataset/data \
  configs/example_config.toml \
  /scratch/user/$USER/DFire_Analysis
```

Check and manage it:

```bash
squeue -u $USER
tail -f <slurm-output-file>
scancel JOB_ID
du -sh /scratch/user/$USER/DFire
```

Pull a dataset with the reusable wrapper:

```bash
sbatch jobs/pull_dataset.sbatch \
  git https://github.com/example/dataset.git \
  /scratch/user/$USER/datasets/example
```

For direct use, the equivalent script syntax is:

```bash
bash scripts/pull_dataset.sh \
  --type git \
  --source https://github.com/example/dataset.git \
  --destination /scratch/user/$USER/datasets/example
```

The pull script supports `git`, `zip`, `tar`, and `tar.gz`, uses `curl -L` for HTTP(S) archives, refuses to replace an existing destination unless `--force` is supplied, and prints the final `du -sh` size. It does not bypass authentication for protected datasets.

The SLURM analysis job accepts exactly three positional arguments: dataset root, config path, and output directory. Its Python script path is resolved relative to the repository, so `sbatch` does not need to be launched from the repository root.

## Configuration and another dataset

Start with `configs/example_config.toml`. The class IDs are configuration data, not hardcoded Python logic. To add Dataset #2 or #3:

1. Pull it into `/scratch`.
2. Find its image and label directories.
3. Determine its class IDs and names.
4. Create a TOML config with the class map and, if needed, the `paths` values.
5. Run the same analysis job.
6. Review the reports and manifest.

The balancing strategy can be `none`, `downsample`, or `oversample_manifest` (the default). The seed makes selection deterministic. Only `train` is balanced; `val` and `test` are reported as stored.

## Assumptions to verify on Grace

- `python3` is Python 3.11 or newer so `tomllib` is available.
- `curl`, `git`, `tar`, `unzip`, and `du` are available in the default environment.
- The account can read the dataset under `/scratch/user/$USER` and write the requested output directory.
- The local SLURM site accepts the short CPU-only resource request in the example job files.
