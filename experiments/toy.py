import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.exceptions import ComplexWarning

from matscinovelty import (
    EquivariantCrystalGCN,
    OTNoveltyScorer,
    augment,
    perturb_structures_corrupt,
    perturb_structures_gaussian,
    read_structure_from_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_MP20 = PROJECT_ROOT / "data" / "mp_20"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
IMGS_DIR = PROJECT_ROOT / "imgs"
IMGS_DIR.mkdir(exist_ok=True)

warnings.simplefilter("ignore", ComplexWarning)  # Suppress ComplexWarning specifically
# ===========================================================
# 1️⃣ Setup
# ===========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load structures ---
train_structs = read_structure_from_csv(DATA_MP20 / "train.csv")
val_structs = read_structure_from_csv(DATA_MP20 / "val.csv")
del val_structs[232]  # remove broken entry if needed

# --- Load pretrained model ---
print("Loading pretrained GCN model...")
model = EquivariantCrystalGCN(hidden_dim=128).to(device)
checkpoint_path = CHECKPOINTS_DIR / "gcn_fine.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
print("Loaded weights from gcn_fine.pt ✅")

# --- Initialize scorer ---
scorer = OTNoveltyScorer(
    train_structures=train_structs,
    gnn_model=model,  # directly pass model
    tau=None,  #  pass tau directly or leave None if auto-estimate
    tau_quantile=0.05,  # 1% quantile threshold
    memorization_weight=10.0,
    device=device,
)

print(f"\n✅ Estimated τ = {scorer.tau:.4f}\n")


# ===========================================================
# 2️⃣ Vacancy experiment
# ===========================================================
vacancies = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
scores = []

for vac in vacancies:
    pert = perturb_structures_corrupt(val_structs, vacancy_prob=vac, swap_prob=0.0)
    score, *_ = scorer.compute_novelty(pert)
    print(f"vacancy={vac:.2f} -> novelty loss={score:.4f}")
    scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(vacancies, scores, marker="o", color="C0")
plt.xlabel("Percentage of dropped atoms")
plt.ylabel("Novelty loss")
plt.title("Novelty vs. Structure Corruption (Vacancies)")
plt.grid(True)
plt.tight_layout()
plt.savefig(IMGS_DIR / "novelty_vs_vacancies.png", dpi=300)
plt.show()


# ===========================================================
# 3️⃣ Swap experiment
# ===========================================================
swaps = np.linspace(0, 1, 11)
scores = []

for swa in swaps:
    pert = perturb_structures_corrupt(val_structs, vacancy_prob=0.0, swap_prob=swa)
    score, *_ = scorer.compute_novelty(pert)
    print(f"swap={swa:.2f}: {score:.4f}")
    scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(swaps, scores, marker="o", color="C1")
plt.xlabel("Swap probability")
plt.ylabel("Novelty loss")
plt.title("Novelty vs. Atom Swapping")
plt.grid(True)
plt.tight_layout()
plt.savefig(IMGS_DIR / "novelty_vs_swaps.png", dpi=300)
plt.show()


# ===========================================================
# 4️⃣ Gaussian noise experiment
# ===========================================================
sigmas = np.linspace(0, 1, 11)
scores = []

for sig in sigmas:
    pert = perturb_structures_gaussian(val_structs, sigma=sig, teleport_prob=0.0)
    score, *_ = scorer.compute_novelty(pert)
    print(f"sigma={sig:.2f}: {score:.4f}")
    scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(sigmas, scores, marker="o", color="C2")
plt.xlabel("Gaussian noise σ")
plt.ylabel("Novelty loss")
plt.title("Novelty vs. Gaussian Noise")
plt.grid(True)
plt.tight_layout()
plt.savefig(IMGS_DIR / "novelty_vs_sigma.png", dpi=300)
plt.show()


# ===========================================================
# 5️⃣ Teleport experiment
# ===========================================================
teles = np.linspace(0, 1, 11)
scores = []

for tele in teles:
    pert = perturb_structures_gaussian(val_structs, teleport_prob=tele, sigma=0.0)
    score, *_ = scorer.compute_novelty(pert)
    print(f"teleport={tele:.2f}: {score:.4f}")
    scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(teles, scores, marker="o", color="C3")
plt.xlabel("Teleport probability")
plt.ylabel("Novelty loss")
plt.title("Novelty vs. Atom Teleportation")
plt.grid(True)
plt.tight_layout()
plt.savefig(IMGS_DIR / "novelty_vs_teleport.png", dpi=300)
plt.show()


# ===========================================================
# 6️⃣ Shared samples / memorization experiment
# ===========================================================
fractions = np.linspace(0, 1, 11)
scores = []

for frac in fractions:
    num_shared = int(frac * len(val_structs))
    if num_shared > 0:
        rep_idx = np.random.choice(len(train_structs), size=num_shared, replace=True)
        s_aug = [augment(train_structs[i]) for i in rep_idx]
        mixed = s_aug + val_structs[num_shared:]
    else:
        mixed = val_structs

    score, *_ = scorer.compute_novelty(mixed)
    print(f"{frac * 100:.0f}% shared: {score:.4f}")
    scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(fractions * 100, scores, marker="o", color="crimson")
plt.xlabel("Percentage of Shared (Memorized) Samples")
plt.ylabel("Novelty loss")
plt.title("Novelty vs. Dataset Memorization")
plt.grid(True)
plt.tight_layout()
plt.savefig(IMGS_DIR / "novelty_vs_shared.png", dpi=300)
plt.show()
