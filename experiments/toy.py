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
    augment_supercell,
    random_lattice_deformation,
    supercell_with_random_substitutions,
    supercell_with_substitutions_list,
    random_group_substitution,
    random_group_substitution,
    random_supercell,
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
checkpoint_path = CHECKPOINTS_DIR / "gcn_mp20.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
print("Loaded pretrained weights.✅")

# --- Initialize scorer ---
scorer = TransportNoveltyDistance(
    train_structures=train_structs,
    gnn_model=model,  # directly pass model
    device=device,
)

sigmas = np.linspace(0, 0.15, 10)
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
# 2️⃣ Random Binary Lattice Strain Experiment
############################################################
strains = np.linspace(0, 0.6, 10)
scores = []

print("\n=== Random Binary Lattice Strain Experiment ===")
for eps in strains:
    pert = [random_lattice_deformation(s, max_strain=eps) for s in val_structs]
    score, *_ = scorer.compute_novelty(pert)
    print(f"strain={eps:.3f} -> TND={score:.4f}")
    scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(strains, scores, marker="o")
plt.xlabel("Binary Lattice Strain Magnitude")
plt.ylabel("Novelty loss")
plt.title("Novelty vs Binary Lattice Strain")
plt.grid(True)
plt.tight_layout()
plt.savefig(IMGS_DIR / "toy_binary_lattice.png", dpi=300)
plt.close()


############################################################
# 3️⃣ Random Supercell Experiment (probabilistic)
############################################################
probs = np.linspace(0, 0.5, 10)
scores = []

print("\n=== Random Supercell Experiment ===")
for p in probs:
    pert = [random_supercell(s, p=p) for s in val_structs]
    score, *_ = scorer.compute_novelty(pert)
    print(f"p_supercell={p:.3f} -> TND={score:.4f}")
    scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(probs, scores, marker="o")
plt.xlabel("Supercell Probability")
plt.ylabel("Novelty loss")
plt.title("Novelty vs Random Supercell Probability")
plt.grid(True)
plt.tight_layout()
plt.savefig(IMGS_DIR / "toy_supercell_prob.png", dpi=300)
plt.close()


############################################################
# 4️⃣ Random Same-Group Substitution Experiment
############################################################

allowed_elements = set()
for s in train_structs:
    for site in s.sites:
        allowed_elements.add(str(site.specie))
probs = np.linspace(0, 0.5, 10)
scores = []

print("\n=== Random Same-Group Substitution Experiment ===")
for p in probs:
    pert = [random_group_substitution(s, allowed_elements, p=p) for s in val_structs]
    score, *_ = scorer.compute_novelty(pert)
    print(f"p_group_sub={p:.3f} -> TND={score:.4f}")
    scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(probs, scores, marker="o")
plt.xlabel("Same-Group Substitution Probability")
plt.ylabel("Novelty loss")
plt.title("Novelty vs Group Substitution")
plt.grid(True)
plt.tight_layout()
plt.savefig(IMGS_DIR / "toy_group_substitution.png", dpi=300)
plt.close()


############################################################
# 5️⃣ Data Leakage / Shared Samples Experiment
############################################################
shared_fracs = np.linspace(0, 1.0, 10)
scores = []

print("\n=== Data Leakage (Shared Samples) ===")
for f in shared_fracs:
    n_shared = int(f * len(val_structs))

    if n_shared > 0:
        rep_idx = np.random.choice(len(train_structs), size=n_shared, replace=True)
        leaked = [random_supercell(train_structs[i], p=1.0) for i in rep_idx]
        mixed = leaked + val_structs[n_shared:]
    else:
        mixed = val_structs

    score, *_ = scorer.compute_novelty(mixed)
    print(f"shared={f:.3f} -> TND={score:.4f}")
    scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(shared_fracs, scores, marker="o", color="crimson")
plt.xlabel("Fraction of Training Samples Reinserted")
plt.ylabel("Novelty loss")
plt.title("Novelty vs Data Leakage")
plt.grid(True)
plt.tight_layout()
plt.savefig(IMGS_DIR / "toy_data_leakage.png", dpi=300)
plt.close()


print("\n✅ All updated perturbation plots saved to imgs/. Done! 🎉")
