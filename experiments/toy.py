import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.exceptions import ComplexWarning

from matscinovelty import (
    EquivariantCrystalGCN,
    TransportNoveltyDistance,
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
checkpoint_path = CHECKPOINTS_DIR / "egnn_invariant_mp20.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
print("Loaded pretrained weights.✅")

# --- Initialize scorer ---
scorer = TransportNoveltyDistance(
    train_structures=train_structs,
    gnn_model=model,  # directly pass model
    device=device,
)



############################################################
# 1️⃣ Gaussian Perturbation Experiment
############################################################
sigmas = np.linspace(0, 0.2, 9)
scores = []

print("\n=== Gaussian Noise Experiment ===")
for sigma in sigmas:
    pert = perturb_structures_gaussian(val_structs, sigma=sigma, teleport_prob=0.0)
    score, *_ = scorer.compute_novelty(pert)
    print(f"sigma={sigma:.3f} -> TND={score:.4f}")
    scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(sigmas, scores, marker="o")
plt.xlabel("Gaussian σ")
plt.ylabel("Novelty loss")
plt.title("Novelty vs Gaussian Noise")
plt.grid(True)
plt.tight_layout()
plt.savefig(IMGS_DIR / "toy_gaussian.png", dpi=300)
plt.close()


############################################################
# 2️⃣ Lattice Deformation Experiment
############################################################
strains = np.linspace(0, 0.4, 9)
scores = []

print("\n=== Lattice Deformation Experiment ===")
for eps in strains:
    pert = [random_lattice_deformation(s, max_strain=eps) for s in val_structs]
    score, *_ = scorer.compute_novelty(pert)
    print(f"strain={eps:.3f} -> TND={score:.4f}")
    scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(strains, scores, marker="o")
plt.xlabel("Max Lattice Strain")
plt.ylabel("Novelty loss")
plt.title("Novelty vs Lattice Deformation")
plt.grid(True)
plt.tight_layout()
plt.savefig(IMGS_DIR / "toy_lattice.png", dpi=300)
plt.close()


############################################################
# 3️⃣ Supercell Substitution Experiment (2×2×2)
############################################################
probs = np.linspace(0, 0.2, 9)
scores = []

print("\n=== Supercell Substitution Experiment ===")
for p in probs:
    pert = [
        supercell_with_random_substitutions(
            s, scale_matrix=(2, 2, 2), p_change=p
        )
        for s in val_structs
    ]
    score, *_ = scorer.compute_novelty(pert)
    print(f"p_change={p:.3f} -> TND={score:.4f}")
    scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(probs, scores, marker="o")
plt.xlabel("Substitution Probability (supercell 2×2×2)")
plt.ylabel("Novelty loss")
plt.title("Novelty vs Supercell Substitution")
plt.grid(True)
plt.tight_layout()
plt.savefig(IMGS_DIR / "toy_supercell_substitution.png", dpi=300)
plt.close()


############################################################
# 4️⃣ Data Leakage / Shared Samples Experiment
#    using augment_supercell
############################################################
shared_fracs = np.linspace(0, 0.2, 9)
scores = []

print("\n=== Data Leakage (Shared Samples via augment_supercell) ===")
for f in shared_fracs:
    n_shared = int(f * len(val_structs))

    if n_shared > 0:
        rep_idx = np.random.choice(len(train_structs), size=n_shared, replace=True)
        leaked = [augment_supercell(train_structs[i]) for i in rep_idx]
        mixed = leaked + val_structs[n_shared:]
    else:
        mixed = val_structs

    score, *_ = scorer.compute_novelty(mixed)
    print(f"shared={f:.3f} -> TND ={score:.4f}")
    scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(shared_fracs, scores, marker="o", color="crimson")
plt.xlabel("Fraction of Training Samples Reinserted")
plt.ylabel("Novelty loss")
plt.title("Novelty vs Data Leakage (augment_supercell)")
plt.grid(True)
plt.tight_layout()
plt.savefig(IMGS_DIR / "toy_data_leakage.png", dpi=300)
plt.close()

print("\nAll plots saved to imgs/. Done! 🎉")