# 🚀 Transport Novelty Distance (TNovD)

**The distributional metric for crystal generative models.**

TNovD evaluates **novelty** and **quality** simultaneously by combining contrastive GNN embeddings with Optimal Transport (OT). It solves the "quality vs. novelty" trade-off common in generative materials science.

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
git clone https://github.com/BAMeScience/TransportNoveltyDistance.git

# Create python environment
cd TransportNoveltyDistance
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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR =  PROJECT_ROOT / "data" / "mp_20"
DATA_XTALMET = PROJECT_ROOT / "data" / "xtalmet_models"

# Load the structures used to train the generative model
train_structs = read_structure_from_csv(DATA_DIR / "train.csv")

# Load the generated structures 
with open(DATA_XTALMET / "mattergen.pkl", "rb") as f:
    gen_structs = pickle.load(f)

# 3. Load the Pre-trained Metric Model
model = EquivariantCrystalGCN(hidden_dim=32, num_rbf=128).to(device)
model.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints/gcn_mp20_final.pt", map_location=device))

# 4. Compute TNovD
# Class renamed: OTNoveltyScorer -> TransportNoveltyDistance
scorer = TransportNoveltyDistance(
    train_structures=train_structs,
    gnn_model=model,
    device=device  
)
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
    model_builder=lambda: EquivariantCrystalGCN(hidden_dim=32, n_layers=3)
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

### 📜 Citation


> If you find the paper or the code useful, please cite:

*Transport Novelty Distance: A Distributional Metric for Evaluating Material Generative Models*  
Hagemann, Müller, George, Benner  
arXiv: [2512.09514](https://arxiv.org/abs/2512.09514)

```bibtex
@article{hagemann2025transportnoveltydistancedistributional,
  title   = {Transport Novelty Distance: A Distributional Metric for Evaluating Material Generative Models},
  author  = {Hagemann, Paul and Müller, Simon and George, Janine and Benner, Philipp},
  year    = {2025},
  journal = {arXiv preprint arXiv:2512.09514}
}

