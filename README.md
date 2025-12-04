# 🚀 Transport Novelty Distance (TND)

**The distributional metric for crystal generative models.**

TND evaluates **novelty** and **quality** simultaneously by combining contrastive GNN embeddings with Optimal Transport (OT). It solves the "validity vs. uniqueness" trade-off common in generative materials science.

### Why TND?

> ❌ **Memorization:** If your model copies training data, TND penalizes it.
>
> ❌ **Hallucination:** If your model generates chemically invalid nonsense, TND penalizes it.
>
> ✅ **Generalization:** If your model creates *new*, *stable* crystals, TND rewards it.

---

### 🧠 How it Works

Based on Optimal Transport theory, TND finds a minimum-cost matching between the distribution of generated structures and the training set within a chemically aware feature space.

<div align="center">
<img width="100%" alt="TND_workflow_overview" src="https://github.com/user-attachments/assets/ef82a791-95c2-4534-add0-12f0d9048c58" />
</div>

The repository ships as an installable Python package (`matscinovelty`). All core logic is exposed as importable modules, while exploratory notebooks and drivers remain as scripts.

---

## 🔧 Installation

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate

# Install package in editable mode
pip install --upgrade pip
pip install -e .
```

---

## ⚡ Quick Start

### 1. Score a Generative Model
Calculate the novelty score for a generated dataset against a training baseline.

```python
from pathlib import Path
import pandas as pd
import torch
from matscinovelty import (
    EquivariantCrystalGCN,
    OTNoveltyScorer,
    load_structures_from_json_column,
    read_structure_from_csv,
)

# 1. Load Data
DATA_DIR = Path("data/mp_20")
train_structs = read_structure_from_csv(DATA_DIR / "train.csv")
gen_structs = load_structures_from_json_column(pd.read_csv("data_models/mattergen.csv"))

# 2. Load the Pre-trained Metric Model
model = EquivariantCrystalGCN(hidden_dim=32)
model.load_state_dict(torch.load("checkpoints/gcn_fine.pt"))

# 3. Compute TND
# 'tau' controls the transport regularization
scorer = OTNoveltyScorer(
    train_structs, 
    gnn_model=model, 
    device="cpu", 
    tau=0.36, 
    memorization_weight=10.0
)

total, quality, memorization = scorer.compute_novelty(gen_structs)
print(f"Total TND: {total:.4f} | Quality: {quality:.4f} | Memorization: {memorization:.4f}")
```

### 2. Train the Encoder
Train your own equivariant encoder if you aren't using the provided checkpoints.

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

---

## 🧪 Experiments

Reusable experiment drivers are located in the `experiments/` directory. Each script automatically resolves paths relative to the project root.

| Script | Description |
| :--- | :--- |
| `experiments/model_eval.py` | Full comparison against multiple Wyckoff-based generative models. |
| `experiments/mattergen_eval.py` | Focused evaluation for MatterGen variants. |
| `experiments/toy.py` | Perturbation sweeps showing how TND reacts to synthetic corruptions. |
| `experiments/wbm_exp.py` | Tracks novelty across WBM optimization stages. |

**Run an experiment:**
```bash
python experiments/model_eval.py
```

### 📂 Data
Scripts for downloading MP20, WBM, and generated model outputs can be found in the `downloads/` subfolder.

## 📜 Citation
TBD
