import pandas as pd
import torch
from pymatgen.core import Structure
import warnings
from matminer.featurizers.site import SOAP
import numpy as np
from numpy.exceptions import ComplexWarning
from pymatgen.core import Structure, Lattice, PeriodicSite
import ot
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import math
import hashlib
import pickle
from was_utils import *
from utils_new import *
from gcn import *
from gcn_coarse import *

warnings.simplefilter("ignore", ComplexWarning) # Suppress ComplexWarning specifically

# --- 0. Load data and model ---
#str_train_ori = read_structure_from_csv('train.csv')[:1000]
#str_val = read_csv('val.csv')[:1000]
# remove Structure only consisting of Xenon -> coarste Embedding does not work as expected
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Featurizing with GCN (fine)...")
train_structs = read_structure_from_csv("train.csv")[:1000]
val_structs   = read_structure_from_csv("val.csv")[:1000]
del val_structs[232]   # remove broken entry
val_structs = drop_val_duplicates_structural(train_structs, val_structs)

model = EquivariantCrystalGCN(hidden_dim=128, num_rbf=32).to(device)
model.load_state_dict(torch.load("gcn_fine.pt", map_location=device))

train_feat_fine = featurize_fine(train_structs, model, device=device).view(len(train_structs), -1)
val_feat_fine   = featurize_fine(val_structs,   model, device=device).view(len(val_structs),   -1)

print("Shapes:", train_feat_fine.shape, val_feat_fine.shape)

# -----------------------------
# 3. Estimate τ
# -----------------------------
tau = estimate_tau_ot(train_feat_fine, split_ratio=0.5, quantile=0.01)
print(f"Estimated τ: {tau:.4f}")

# -----------------------------
# 4. Sanity noise experiment
# -----------------------------
vacancies = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
scores = []

for vac in vacancies:
    pert = perturb_structures_corrupt(val_structs, vacancy_prob=vac, swap_prob=0.)
    pert_feat_fine = featurize_fine(pert, model, device=device).view(len(pert), -1)

    P, C = get_ot_plan(train_feat_fine, pert_feat_fine)
    score, _, _, _  = get_novelty_loss_only_fine(P, C, tau=tau)
    print(f"vacancy={vac:.2f} -> novelty loss={score:.4f}")
    scores.append(score)


# -----------------------------
# 5. Plot
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(vacancies, scores, marker='o', color='C0')
plt.xlabel("Percentage of dropped atoms")
plt.ylabel("Novelty loss (fine only)")
plt.title("Novelty vs. structure corruption (fine-only OT)")
plt.grid(True)
plt.tight_layout()
plt.savefig("imgs/novelty_fineonly_vs_vac.png")
plt.show()

# 1️⃣  Swaps Experiment
# ===========================================================
swaps = np.linspace(0, 1, 11)
scores = []

for swa in swaps:
    perturbed = perturb_structures_corrupt(val_structs, vacancy_prob=0., swap_prob=swa)
    pf = featurize_fine(perturbed, model, device=device).view(len(perturbed), -1)

    P, C = get_ot_plan(train_feat_fine, pf)
    score, _, _, _ = get_novelty_loss_only_fine(P, C, tau=tau)
    print(f"swap={swa:.2f}: {score:.4f}")
    scores.append(score)


plt.figure(figsize=(8,5))
plt.plot(swaps, scores, marker='o', color='C0')
plt.xlabel('Swap probability')
plt.ylabel('Novelty loss (fine only)')
plt.title('Novelty vs. Atom Swapping (fine-only OT)')
plt.grid(True)
plt.tight_layout()
plt.savefig('imgs/novelty_fineonly_vs_swap.png')
plt.show()

# ===========================================================
# 2️⃣  Gaussian Noise Experiment
# ===========================================================
sigmas = np.linspace(0, 1, 11)
scores = []

for sig in sigmas:
    perturbed = perturb_structures_gaussian(val_structs, sigma=sig, teleport_prob=0.)
    pf = featurize_fine(perturbed, model, device=device).view(len(perturbed), -1)

    P, C = get_ot_plan(train_feat_fine, pf)
    score, _ , _, _= get_novelty_loss_only_fine(P, C, tau=tau)
    print(f"sigma={sig:.2f}: {score:.4f}")
    scores.append(score)


plt.figure(figsize=(8,5))
plt.plot(sigmas, scores, marker='o', color='C1')
plt.xlabel('Gaussian noise σ')
plt.ylabel('Novelty loss (fine only)')
plt.title('Novelty vs. Gaussian Noise (fine-only OT)')
plt.grid(True)
plt.tight_layout()
plt.savefig('imgs/novelty_fineonly_vs_sigma.png')
plt.show()

# ===========================================================
# 3️⃣  Teleport Experiment
# ===========================================================
teles = np.linspace(0, 1, 11)
scores = []

for tele in teles:
    perturbed = perturb_structures_gaussian(val_structs, teleport_prob=tele, sigma=0.)
    pf = featurize_fine(perturbed, model, device=device).view(len(perturbed), -1)

    P, C = get_ot_plan(train_feat_fine, pf)
    score, _, _,_ = get_novelty_loss_only_fine(P, C, tau=tau)
    print(f"teleport={tele:.2f}: {score:.4f}")
    scores.append(score)


plt.figure(figsize=(8,5))
plt.plot(teles, scores, marker='o', color='C2')
plt.xlabel('Teleport probability')
plt.ylabel('Novelty loss (fine only)')
plt.title('Novelty vs. Atom Teleportation (fine-only OT)')
plt.grid(True)
plt.tight_layout()
plt.savefig('imgs/novelty_fineonly_vs_tele.png')
plt.show() 

# ===========================================================
# 4️⃣  Shared Samples / Memorization Experiment
# ===========================================================
fractions = np.linspace(0, 1, 11)
scores = []

A_fine = train_feat_fine
B0_fine = val_feat_fine
N_B = B0_fine.shape[0]

for frac in fractions:
    num_shared = int(frac * N_B)
    B_fine = B0_fine.clone()

    if num_shared > 0:
        rep_idx = np.random.choice(len(train_structs), size=num_shared, replace=True)
        s_aug_list = [augment(train_structs[i]) for i in rep_idx]
        bf_aug = featurize_fine(s_aug_list, model, device=device).view(num_shared, -1)
        B_fine[:num_shared] = bf_aug  # overwrite first rows with memorized copies


    P, C = get_ot_plan(A_fine, B_fine)
    score, _, _, _ = get_novelty_loss_only_fine(P, C, tau=tau)
    print(f"{frac*100:.0f}% shared: {score:.4f}")
    scores.append(score)



plt.figure(figsize=(8,5))
plt.plot(fractions*100, scores, marker='o', color='crimson')
plt.xlabel('Percentage of Shared (Memorized) Samples')
plt.ylabel('Novelty loss (fine only)')
plt.title('Novelty vs. Dataset Memorization (fine-only OT)')
plt.grid(True)
plt.tight_layout()
plt.savefig('imgs/novelty_fineonly_vs_shared_samples.png')
plt.show()