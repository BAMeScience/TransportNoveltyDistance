# MP-20 data helpers

This repository relies on the MP-20 split popularized by DiffCSP, MatterGen, CrystalFormer, and related works. Two helper scripts keep the workflow simple:

## Usage

```bash
# 1. Download/stage the DiffCSP MP-20 split
python scripts/download_mp20.py

# 2. Train the encoder once the CSVs are in place
python scripts/train_mp20.py --epochs 10 --checkpoint-path checkpoints/gcn_mp20.pt
```

`download_mp20.py` handles grabbing `train.csv`, `val.csv`, and `test.csv` into `data/mp_20/`. Pass `--force` to overwrite existing files or change the destination with `--output-dir`.

`train_mp20.py` is a CLI wrapper around `train_contrastive_model`; tweak hyperparameters and artifact paths through its flags. The script will exit early with a helpful message if the train/val CSVs are missing, so always run the download step (or supply your own CSV paths) first.

## xtalmet model outputs

The novelty benchmarks in `experiments/model_eval.py` rely on pickled structure lists from the xtalmet dataset. To fetch them:

```bash
python scripts/download_xtalmet_models.py
```

By default the files land in `data/xtalmet_models/`. Use `--force` to overwrite or `--output-dir` to select another folder before running the evaluation script.
This helper depends on `huggingface_hub`, so run `pip install huggingface_hub` if the module is missing.

## WBM dataset

For the WBM novelty experiment (`experiments/wbm_exp.py`), mirror the public WBM summary and structure data via `matbench_discovery` (install with `pip install matbench-discovery`):

```bash
python scripts/download_wbm_data.py
```

The script creates `data/wbm/` containing the summary CSV, initial atoms archive, and (unless you pass `--skip-step-json`) per-step relaxed structure dumps used by `experiments/wbm_exp.py`. The relaxed-structure export iterates over the full 250k-structure archive and can take a while; use `--step-limit N` for smaller debugging extracts. Keep this folder out of git; the experiment script reads from it automatically.
