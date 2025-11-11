import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure

from matscinovelty import (
    EquivariantCrystalGCN,
    OTNoveltyScorer,
    read_structure_from_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WBM_DIR = PROJECT_ROOT / "wbm_data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
IMGS_DIR = PROJECT_ROOT / "imgs"
IMGS_DIR.mkdir(exist_ok=True)

# ===========================================================
# 1️⃣ Load Data
# ===========================================================
print("Loading WBM data...")
str_train = read_structure_from_csv(PROJECT_ROOT / "train.csv")

# --- Load stable WBM structures ---
summary = pd.read_csv(WBM_DIR / "wbm-summary.txt", sep="\t", header=None)
stable = summary[summary[5] < 0]


def load_structures_for_step(step):
    path = WBM_DIR / f"wbm-structures-step-{step}.json"
    with open(path) as fh:
        data = json.load(fh)
    mask = np.array([
        stable[7].iloc[i].split("_")[1] == str(step) for i in range(len(stable))
    ])
    subset = stable.iloc[np.where(mask)[0]]
    structs = [Structure.from_dict(data[e]["opt"]) for e in subset[7].values]
    print(f"Loaded {len(structs)} structures for WBM step {step}")
    return structs


wbm_steps = [load_structures_for_step(i) for i in range(1, 6)]

# ===========================================================
# 2️⃣ Initialize Scorer
# ===========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading pretrained GCN model...")
model = EquivariantCrystalGCN(hidden_dim=128).to(device)
checkpoint_path = CHECKPOINTS_DIR / "gcn_fine.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
print("Loaded weights from gcn_fine.pt ✅")

scorer = OTNoveltyScorer(
    train_structures=str_train,
    gnn_model=model,
    tau=None,  # auto-estimate τ
    tau_quantile=0.05,
    memorization_weight=10.0,
    device=device,
)
print(f"Estimated τ = {scorer.tau:.4f}")

# ===========================================================
# 3️⃣ Evaluate WBM Steps
# ===========================================================
scores_total, scores_quality, scores_mem = [], [], []
max_len = min(len(s) for s in wbm_steps)  # or set manually, e.g. max_len = 1000

for step_idx, step_structs in enumerate(wbm_steps, start=1):
    print(f"\n▶ Evaluating WBM step {step_idx}")

    # ✅ sample same number of materials each step
    if len(step_structs) > max_len:
        step_subset = random.sample(step_structs, max_len)
    else:
        step_subset = step_structs

    total, qual, mem = scorer.compute_novelty(step_subset)
    print(
        f"Step {step_idx}: Total={total:.4f} | Quality={qual:.4f} | Memorization={mem:.4f}"
    )

    scores_total.append(total)
    scores_quality.append(qual)
    scores_mem.append(mem)
# ===========================================================
# 4️⃣ Plot Results
# ===========================================================
steps = range(1, len(scores_total) + 1)
plt.figure(figsize=(8, 5))
plt.plot(steps, scores_total, marker="o", label="Total Loss", linewidth=2)
plt.plot(steps, scores_quality, marker="s", label="Quality Component", linestyle="--")
plt.plot(steps, scores_mem, marker="^", label="Memorization Component", linestyle=":")
plt.xlabel("WBM Step")
plt.ylabel("Novelty Loss")
plt.title("Novelty Loss Components vs. WBM Step")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(IMGS_DIR / "novelty_wbm_fine_components.png", dpi=300)
plt.show()
