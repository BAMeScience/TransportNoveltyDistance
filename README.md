# 🚀 Transport Novelty Distance (TNovD)

**The distributional metric for crystal generative models.**

TNovD evaluates **novelty** and **quality** simultaneously by combining contrastive GNN embeddings with Optimal Transport (OT). It solves the "validity vs. uniqueness" trade-off common in generative materials science.

<img width="2348" height="1625" alt="TNovD_workflow_overview" src="https://github.com/user-attachments/assets/c4da9b60-187c-4a22-9f5a-44900d2402be" />

### Why TNovD?

> ❌ **Memorization:** If your model copies training data, TNovD penalizes it.
>
> ❌ **Hallucination:** If your model generates chemically invalid nonsense, TNovD penalizes it.
>
> ✅ **Generalization:** If your model creates *new*, *reasonable* crystals, TNovD rewards it.

---

### 🧠 How it Works

Based on Optimal Transport theory, TNovD finds a minimum-cost matching between the distribution of generated structures and the training set within a chemically and structurally aware feature space.
We aim at detecting memorization and quality, combined in one **distributional** score. 



The repository ships as an installable Python package (`TNovD`). All core logic is exposed as importable modules, the papers experiments can be reproduced via the experiments folder.

---

## 🔧 Installation

```bash
# Navigate to the respective project folder, then clone this repository
git clone https://github.com/BAMeScience/MatSciNovelty.git

# Create python environment
cd MatSciNovelty
python -m venv .venv
source .venv/bin/activate

# Install package in editable mode
pip install --upgrade pip
pip install -e .
```

---

## ⚡ Quick Start

### 1. Score a Generative Model
Calculate the novelty score for a generated dataset (here: MatterGen) against a training baseline (here: MP20).

```python
import pandas as pd
import torch
from pathlib import Path

# 1. New Imports from TNovD package
from TNovD import (
    EquivariantCrystalGCN,
    TransportNoveltyDistance,
    read_structure_from_csv,
    load_structures_from_json_column
)

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load Data
DATA_DIR = Path("data/mp_20")
train_structs = read_structure_from_csv(DATA_DIR / "train.csv")

# Assuming 'mattergen.csv' has a JSON column for structures
gen_structs = load_structures_from_json_column(pd.read_csv("data_models/mattergen.csv"))

# 3. Load the Pre-trained Metric Model
model = EquivariantCrystalGCN(hidden_dim=32, num_rbf=128).to(device)
model.load_state_dict(torch.load("checkpoints/gcn_mp20_final.pt", map_location=device))

# 4. Compute TNovD
# Class renamed: OTNoveltyScorer -> TransportNoveltyDistance
scorer = TransportNoveltyDistance(
    train_structures=train_structs,
    gnn_model=model,
    device=device  # Good practice to pass device explicitly
)

# Method renamed: compute_novelty -> compute_TNovD
total, quality, memorization = scorer.compute_TNovD(gen_structs)

print(f"Total TNovD: {total:.4f} | Quality: {quality:.4f} | Memorization: {memorization:.4f}")
```

### 2. Train the Encoder
Train your own equivariant encoder if you aren't using the provided checkpoints (for instance if you want different positives or negatives, or try a different architecture). The feature space dimensions can be adapted by changing hidden_dim argument.

```python
from TNovD import EquivariantCrystalGCN
from TNovD.gcn import train_contrastive_model

train_contrastive_model(
    "data/mp_20/train.csv",
    val_csv="data/mp_20/val.csv",
    checkpoint_path="checkpoints/equivariant.pt",
    plot_path="imgs/validation_curve.png",
    epochs=10,
    # Define the model construction inline
    model_builder=lambda: EquivariantCrystalGCN(hidden_dim=32, num_rbf=32, n_layers=3)
)
```


## 🧪 Experiments

Reusable experiment drivers are located in the `experiments/` directory. Each script automatically resolves paths relative to the project root.

| Script | Description |
| :--- | :--- |
| `experiments/model_eval.py` | Comparison of common material generative models in TNovD. |
| `experiments/toy.py` | Perturbation sweeps showing how TNovD reacts to synthetic corruptions. |
| `experiments/wbm_exp.py` | TNovD for WBM dataset. |

**Run an experiment:**
```bash
python experiments/model_eval.py
```

### 📂 Data
Scripts for downloading MP20, WBM, and generated model outputs can be found in the `scripts/` subfolder, marked with the prefix "download". 

### 🤝 Contributing

If you have any issues with the code, or have new ideas on how we can improve this package, please create a GitHub Issue. PRs are also very welcome! The code is still researchy, so feedback is appreciated.

## 📜 Citation
TBD
