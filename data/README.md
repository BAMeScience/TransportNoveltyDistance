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
