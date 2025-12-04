# Transport Novelty Distance
<img width="2348" height="1625" alt="TND_workflow_overview_v4" src="https://github.com/user-attachments/assets/ef82a791-95c2-4534-add0-12f0d9048c58" />


Reusable tooling for training equivariant GNN encoders on inorganic crystal datasets and benchmarking generative models with the OT-based novelty metric introduced in the accompanying research.

The repository now ships as an installable Python package (`matscinovelty`) backed by a modern `pyproject.toml`. All code that was previously embedded in loose scripts is exposed as importable modules under `matscinovelty/`, while the exploratory notebooks and scripts (e.g., `toy.py`, `model_eval.py`) continue to work as examples that depend on the package.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .  # editable install for local development
```

The package depends on PyTorch, PyTorch Geometric, Pymatgen, POT (`ot`), pandas, numpy, and matplotlib. You may need to follow the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) if wheels are not available for your CUDA/CPU combination.

> **Note:** SchNet-based models (e.g., `SchNetEncoder` or `--model schnet` in `scripts/train_mp20.py`) require the optional `torch-cluster` dependency. Install it via the PyG instructions above, e.g.:
> ```bash
> pip install torch-cluster -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html
> ```

## Data requirements

Most scripts expect:

- `data/mp_20/train.csv` and `data/mp_20/val.csv`: CSV files with a `cif` column containing serialized crystal structures (the MP20 dataset in our experiments).
- Optional WBM data cached under `data/wbm/` (see the WBM section below).
- Generative model outputs stored in `data_models/*.csv` using the Wyckoff-format columns consumed by `load_structures_from_json_column`.

Make sure these files are present in the working directory before running the example scripts.

### Fetch the MP-20 split automatically

1. Download and stage the DiffCSP MP-20 CSVs:

    ```bash
    python scripts/download_mp20.py
    ```

    This pulls `train.csv`, `val.csv`, and `test.csv` into `data/mp_20/`. Use `--force` to overwrite existing files or change the download destination with `--output-dir`.

2. Launch training (after confirming the CSVs exist) with:

    ```bash
    python scripts/train_mp20.py --epochs 10 --checkpoint-path checkpoints/gcn_mp20.pt
    ```

    The training script is a thin wrapper around `train_contrastive_model`, so you can adjust hyperparameters and file paths through its CLI flags. Pass `--accelerate` (after `pip install accelerate` or `pip install -e .[train]`) to launch via Hugging Face Accelerate for multi-GPU/distributed runs.

3. (Optional) Split the MP-20 CSVs into oxide/non-oxide subsets:

    ```bash
    python scripts/split_mp20_oxides.py --inputs data/mp_20/train.csv data/mp_20/val.csv
    ```

    The script uses pymatgen's `BVAnalyzer` to classify each structure and writes `_oxides` / `_non_oxides` CSVs alongside the originals.

### Download xtalmet model outputs (MatterGen, DiffCSP, …)

Some experiments (e.g., `experiments/model_eval.py`) compare against published generative outputs packaged as pickled `Structure` lists on Hugging Face. Grab them via:

```bash
python scripts/download_xtalmet_models.py
```

This stores the files under `data/xtalmet_models/`. Use `--force` to re-download or `--output-dir` if you prefer a different cache directory.
The helper requires `huggingface_hub`; install it via `pip install huggingface_hub` if it's not already available in your environment.

### Cache the WBM dataset for `experiments/wbm_exp.py`

With `matbench-discovery` (install via `pip install matbench-discovery`) in your environment, mirror the WBM summary and structure files locally:

```bash
python scripts/download_wbm_data.py
```

The script writes the summary CSV, initial atoms archive, and (by default) the relaxed structure dumps for steps 1–5 into `data/wbm/`, which is where `experiments/wbm_exp.py` looks for them. Generating the per-step JSON files requires iterating over ~250k structures and can take a while; pass `--skip-step-json` to skip that step or `--step-limit 1000` to export a smaller slice for quick tests.

## Package layout

- `matscinovelty.gcn`: Equivariant GNN encoder (`EquivariantCrystalGCN`), SchNet/CGCNN-style embedding backbones (`SchNetEncoder`, `CGCNNEncoder`), InfoNCE loss, validation utilities, and a helper `train_contrastive_model`.
- `matscinovelty.utils`: Structure loading helpers, PyG graph conversion, augmentation/perturbation routines, and novelty/coverage statistics.
- `matscinovelty.wasserstein_novelty`: `OTNoveltyScorer`, which computes quality/memorization losses via optimal transport in the learned feature space.

## Quick start

Train an equivariant encoder:

```python
from pathlib import Path
from matscinovelty.gcn import train_contrastive_model

DATA_DIR = Path("data/mp_20")

train_contrastive_model(
    str(DATA_DIR / "train.csv"),
    val_csv=str(DATA_DIR / "val.csv"),
    epochs=10,
    checkpoint_path="checkpoints/equivariant.pt",
    plot_path="imgs/validation_curve.png",
)
```

Score a generative model:

```python
from pathlib import Path
import pandas as pd
from matscinovelty import (
    EquivariantCrystalGCN,
    OTNoveltyScorer,
    load_structures_from_json_column,
    read_structure_from_csv,
)

DATA_DIR = Path("data/mp_20")
train_structs = read_structure_from_csv(DATA_DIR / "train.csv")
gen_structs = load_structures_from_json_column(pd.read_csv("data_models/mattergen.csv"))

model = EquivariantCrystalGCN(hidden_dim=128)
model.load_state_dict(torch.load("checkpoints/gcn_fine.pt", map_location="cpu"))

scorer = OTNoveltyScorer(train_structs, gnn_model=model, device="cpu", tau=0.36, memorization_weight=10.0)
total, quality, memorization = scorer.compute_novelty(gen_structs)
print(total, quality, memorization)

# Prefer a lightweight encoder for embeddings only?
from matscinovelty import SchNetEncoder, CGCNNEncoder

schnet = SchNetEncoder(embedding_dim=256)   # SchNet backbone
cgnn = CGCNNEncoder(hidden_dim=128)         # CGCNN-style backbone
```

## Experiments

Reusable experiment drivers now live under `experiments/`:

- `experiments/model_eval.py`: Full comparison against multiple Wyckoff-based generative models.
- `experiments/mattergen_eval.py`: Focused evaluation for MatterGen variants.
- `experiments/toy.py`: Perturbation sweeps showing how the novelty score reacts to synthetic corruptions.
- `experiments/wbm_exp.py`: Tracks novelty across WBM optimization stages.

Run them from the project root (e.g., `python experiments/model_eval.py`). Each script automatically resolves dataset/checkpoint paths relative to the repository, so you can also `cd experiments/` and invoke them there if you prefer.

## Development tips

- Format/lint locally with `pip install -e .[dev] && ruff check .`.
- Keep large pretrained checkpoints (e.g., `gcn_fine.pt`) out of version control unless needed.
- When extending the package, prefer adding new modules under `src/matscinovelty/` and re-export relevant APIs in `src/matscinovelty/__init__.py`.
- To smoke-test the training loop without heavy datasets, activate your virtualenv and run `pytest tests/test_smoke_train.py`.
