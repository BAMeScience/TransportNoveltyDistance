import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.exceptions import ComplexWarning
import pickle

from TNovD import (
    EquivariantCrystalGCN,
    TransportNoveltyDistance,
    augment,
    random_lattice_deformation,
    random_group_substitution,
    random_group_substitution,
    random_supercell,
    random_substitution,
    perturb_structures_gaussian,
    read_structure_from_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_MP20 = PROJECT_ROOT / "data" / "mp_20"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
IMGS_DIR = PROJECT_ROOT / "imgs"
IMGS_DIR.mkdir(exist_ok=True)
PICKLE_DIR = PROJECT_ROOT/"pkl"
PICKLE_DIR.mkdir(exist_ok=True)
FIGSIZE = (10, 7)
LINE_KW = {"color": "black", "marker": "o", "linewidth": 2.5}
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

warnings.simplefilter("ignore", ComplexWarning)  # Suppress ComplexWarning specifically
# ===========================================================
# 1️⃣ Setup
# ===========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load structures ---
train_structs = read_structure_from_csv(DATA_MP20 / "train.csv")
val_structs = read_structure_from_csv(DATA_MP20 / "val.csv")
test_structs = read_structure_from_csv(DATA_MP20 / "test.csv")

# --- Load pretrained model ---
print("Loading pretrained GCN model...")
model = EquivariantCrystalGCN(hidden_dim=32, num_rbf = 128).to(device)
checkpoint_path = CHECKPOINTS_DIR / "gcn_mp20_hidden32.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
print("Loaded pretrained weights.✅")

# --- Initialize scorer ---
scorer = TransportNoveltyDistance(
    train_structures=train_structs,
    gnn_model=model,  # directly pass model
    device=device,
    calibration_sample_size=len(val_structs),
    calibration_structures=val_structs,
)
toy_values = {}

sigmas = np.linspace(0, 0.2, 10)
scores = []

print("\n=== Gaussian Noise Experiment ===")
for sigma in sigmas:
    pert = perturb_structures_gaussian(test_structs, sigma=sigma)
    score, *_ = scorer.compute_TNovD(pert)
    print(f"sigma={sigma:.3f} -> TNovD={score:.4f}")
    scores.append(score)

with open(PICKLE_DIR/"gauss.pkl", "wb") as f:
    pickle.dump((sigmas,scores), f)
toy_values["gaussian"] = (sigmas, scores)

plt.figure(figsize=FIGSIZE)
plt.plot(sigmas, scores, **LINE_KW)
plt.xlabel("Gaussian σ")
plt.ylabel("Transport Novelty Distance")
plt.title("TNovD vs Gaussian Noise")
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
    pert = [random_lattice_deformation(s, max_strain=eps) for s in test_structs]
    score, *_ = scorer.compute_TNovD(pert)
    print(f"strain={eps:.3f} -> TNovD={score:.4f}")
    scores.append(score)


with open(PICKLE_DIR/"lattice.pkl", "wb") as f:
    pickle.dump((strains,scores), f)
toy_values["lattice"] = (strains, scores)

plt.figure(figsize=FIGSIZE)
plt.plot(strains, scores, **LINE_KW)
plt.xlabel("Binary Lattice Strain Magnitude")
plt.ylabel("Transport Novelty Distance")
plt.title("TNovD vs Binary Lattice Strain")
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
    pert = [random_supercell(s, p=p) for s in test_structs]
    score, *_ = scorer.compute_TNovD(pert)
    print(f"p_supercell={p:.3f} -> TNovD={score:.4f}")
    scores.append(score)


with open(PICKLE_DIR/"supercell.pkl", "wb") as f:
    pickle.dump((probs,scores), f)
toy_values["supercell"] = (probs, scores)

plt.figure(figsize=FIGSIZE)
plt.plot(probs, scores, **LINE_KW)
plt.xlabel("Supercell Probability")
plt.ylabel("Transport Novelty Distance")
plt.ylim(0.0, 1.0)
plt.title("TNovD vs Random Supercell Probability")
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
group_scores = {"total": [], "quality": [], "mem": []}

print("\n=== Random Same-Group Substitution Experiment ===")
for p in probs:
    pert = [random_group_substitution(s, allowed_elements, p=p) for s in test_structs]
    score, quality, mem = scorer.compute_TNovD(pert)
    print(f"p_group_sub={p:.3f} -> TNovD={score:.4f}")
    group_scores["total"].append(score)
    group_scores["quality"].append(quality)
    group_scores["mem"].append(mem)


with open(PICKLE_DIR/"group_sub.pkl", "wb") as f:
    pickle.dump((probs, group_scores), f)

probs = np.linspace(0, 0.5, 10)
random_scores = {"total": [], "quality": [], "mem": []}

print("\n=== Random Substitution Experiment ===")
for p in probs:
    pert = [random_substitution(s, allowed_elements, p=p) for s in test_structs]
    score, quality, mem = scorer.compute_TNovD(pert)
    print(f"p_random_sub={p:.3f} -> TNovD={score:.4f}")
    random_scores["total"].append(score)
    random_scores["quality"].append(quality)
    random_scores["mem"].append(mem)


with open(PICKLE_DIR/"random_sub.pkl", "wb") as f:
    pickle.dump((probs, random_scores), f)
with open(PICKLE_DIR/"substitutions.pkl", "wb") as f:
    pickle.dump({"p": probs, "group": group_scores, "random": random_scores}, f)
toy_values["group_substitution"] = (probs, group_scores)
toy_values["random_substitution"] = (probs, random_scores)

plt.figure(figsize=FIGSIZE)
plt.plot(probs, group_scores["total"], color="blue", marker="o", linewidth=2.5, label="Same-group")
plt.plot(probs, random_scores["total"], color="red", marker="o", linewidth=2.5, label="Random")
plt.xlabel("Substitution Probability")
plt.ylabel("Transport Novelty Distance")
plt.title("TNovD vs Random Substitution Probability")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(IMGS_DIR / "toy_substitutions.png", dpi=300)
plt.savefig(IMGS_DIR / "toy_substitutions.pdf")
plt.close()




############################################################
# 5️⃣ Data Leakage / Shared Samples Experiment
############################################################
shared_fracs = np.linspace(0, 1.0, 10)
scores = []

print("\n=== Data Leakage (Shared Samples) ===")
for f in shared_fracs:
    n_shared = int(f * len(test_structs))

    if n_shared > 0:
        rep_idx = np.random.choice(len(train_structs), size=n_shared, replace=True)
        leaked = [augment(train_structs[i]) for i in rep_idx]
        mixed = leaked + test_structs[n_shared:]
    else:
        mixed = test_structs

    score, *_ = scorer.compute_TNovD(mixed)
    print(f"shared={f:.3f} -> TNovD={score:.4f}")
    scores.append(score)


with open(PICKLE_DIR/"replace.pkl", "wb") as f:
    pickle.dump((shared_fracs,scores), f)
toy_values["data_leakage"] = (shared_fracs, scores)

plt.figure(figsize=FIGSIZE)
plt.plot(shared_fracs, scores, **LINE_KW)
plt.xlabel("Fraction of Training Samples Reinserted")
plt.ylabel("Transport Novelty Distance")
plt.title("TNovD vs Data Leakage")
plt.grid(True)
plt.tight_layout()
plt.savefig(IMGS_DIR / "toy_data_leakage.png", dpi=300)
plt.close()

with open(PICKLE_DIR/"toy_values_hidden32.pkl", "wb") as f:
    pickle.dump(toy_values, f)

print("\n✅ All updated perturbation plots saved to imgs/. Done! 🎉")
