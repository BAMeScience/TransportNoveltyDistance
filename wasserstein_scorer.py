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
from was_utils import *
from utils_new import *
from gcn import *

warnings.simplefilter("ignore", ComplexWarning) # Suppress ComplexWarning specifically


# --- 0. Load data and model ---
str_val = read_csv('val.csv')[:1000]
device  = "cuda" if torch.cuda.is_available() else "cpu"

model = CrystalGCN(hidden_dim=128, out_dim=256).to(device)
model.load_state_dict(torch.load('gcn.pt', map_location=device))

# --- 2. Featurize ---
print("Featurizing coarsely...")
val_feat_coarse = featurize_coarse(str_val)

print("Featurizing with GCN (fine)...")
val_feat_fine = featurize_fine(str_val, model, device=device)

# --- 3. Force 2D shapes ---
val_feat_coarse = val_feat_coarse.view(len(val_feat_coarse), -1)
val_feat_fine   = val_feat_fine.view(len(val_feat_fine), -1)
print(val_feat_coarse.shape, val_feat_fine.shape)

# 4. filter to get out duplicated

labels = cluster_split(val_feat_coarse.numpy(), n_clusters=2)
idx_X = np.where(labels == 0)[0]
idx_Y = np.where(labels == 1)[0]

Xc, Yc = val_feat_coarse[idx_X], val_feat_coarse[idx_Y]
Xf, Yf = val_feat_fine[idx_X],   val_feat_fine[idx_Y]

# Save the structures too for perturbation
structs_X = [str_val[i] for i in idx_X]
structs_Y = [str_val[i] for i in idx_Y]

# --- 5. ▶ Standardize based on X halves ---
mu_c, sd_c = fit_standardizer(Xc)
mu_f, sd_f = fit_standardizer(Xf)
Xc, Yc = apply_standardizer(Xc, mu_c, sd_c), apply_standardizer(Yc, mu_c, sd_c)
Xf, Yf = apply_standardizer(Xf, mu_f, sd_f), apply_standardizer(Yf, mu_f, sd_f)

# --- 6. Delta calibration ---
P        = get_ot_plan(Xf, Yf)[0]
C_coarse = get_scaled_distance_matrix(Xc, Yc)
C_fine   = get_scaled_distance_matrix(Xf, Yf)
delta    = choose_delta(P, C_coarse, C_fine, tau=0.01)
print(f"Delta calculated: {delta:.4f}")




# --- 7. Sanity: noise experiments ---
vacancies = [0.,0.01,0.05,0.2,0.5,0.8]
scores, sanity = [], []

for vac in vacancies:
    pert = perturb_structures_corrupt(structs_Y, vacancy_prob=vac, swap_prob=0.)

    pc = featurize_coarse(pert).view(len(pert), -1)
    pf = featurize_fine(pert, model, device=device).view(len(pert), -1)

    # ▶ standardize with same mu, std
    pc = apply_standardizer(pc, mu_c, sd_c)
    pf = apply_standardizer(pf, mu_f, sd_f)

    ot_plan, _ = get_ot_plan(Xf, pf)
    score = get_novelty_score(ot_plan, Xc, pc, Xf, pf, delta)
    print(f"vacancy={vac}: {score:.4f}")
    scores.append(score)


# Plotting
plt.figure(figsize=(8, 5))
plt.plot(vacancies, scores, marker='o')
plt.xlabel('Percentage of dropped atoms')
plt.ylabel('Novelty Score')
plt.title('Novelty Score vs. Drop Level')
plt.grid(True)
plt.tight_layout()
plt.savefig('novelty_score_vs_vac.png')
plt.show()
swaps = [0., 0.01, 0.05, 0.2, 0.5, 0.8]
scores = []

for swa in swaps:
    perturbed = perturb_structures_corrupt(structs_Y, vacancy_prob=0., swap_prob=swa)

    # ▶ Reuse the same SOAP object
    pc = featurize_coarse(perturbed).view(len(perturbed), -1)
    pf = featurize_fine(perturbed, model, device=device).view(len(perturbed), -1)

    # ▶ Apply the same standardizers fitted on Xc/Xf
    pc = apply_standardizer(pc, mu_c, sd_c)
    pf = apply_standardizer(pf, mu_f, sd_f)

    ot_plan, _ = get_ot_plan(Xf, pf)
    score = get_novelty_score(ot_plan, Xc, pc, Xf, pf, delta)
    print(f"swap={swa}: {score:.4f}")
    scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(swaps, scores, marker='o')
plt.xlabel('Swap probability')
plt.ylabel('Novelty Score')
plt.title('Novelty Score vs. Swapped')
plt.grid(True)
plt.tight_layout()
plt.savefig('novelty_score_vs_swap.png')
plt.show()

sigmas = [0., 0.01, 0.05, 0.2, 0.5, 0.8]
scores = []

for sig in sigmas:
    perturbed = perturb_structures_gaussian(structs_Y, sigma = sig, teleport_prob=0.)

    # ▶ Reuse the same SOAP object
    pc = featurize_coarse(perturbed).view(len(perturbed), -1)
    pf = featurize_fine(perturbed, model, device=device).view(len(perturbed), -1)

    # ▶ Apply the same standardizers fitted on Xc/Xf
    pc = apply_standardizer(pc, mu_c, sd_c)
    pf = apply_standardizer(pf, mu_f, sd_f)

    ot_plan, _ = get_ot_plan(Xf, pf)
    score = get_novelty_score(ot_plan, Xc, pc, Xf, pf, delta)
    print(f"sigma={sig}: {score:.4f}")
    scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(sigmas, scores, marker='o')
plt.xlabel('Gaussian noise sigma')
plt.ylabel('Novelty Score')
plt.title('Novelty Score vs. Gaussian Noise')
plt.grid(True)
plt.tight_layout()
plt.savefig('novelty_score_vs_sigma.png')
plt.show()

teles = [0., 0.01, 0.05, 0.2, 0.5, 0.8]
scores = []

for tele in teles:
    perturbed = perturb_structures_gaussian(structs_Y, teleport_prob=tele, sigma = 0.)

    # ▶ Reuse the same SOAP object
    pc = featurize_coarse(perturbed).view(len(perturbed), -1)
    pf = featurize_fine(perturbed, model, device=device).view(len(perturbed), -1)

    # ▶ Apply the same standardizers fitted on Xc/Xf
    pc = apply_standardizer(pc, mu_c, sd_c)
    pf = apply_standardizer(pf, mu_f, sd_f)

    ot_plan, _ = get_ot_plan(Xf, pf)
    score = get_novelty_score(ot_plan, Xc, pc, Xf, pf, delta)
    print(f"teleport={tele}: {score:.4f}")
    scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(teles, scores, marker='o')
plt.xlabel('Teleport probability')
plt.ylabel('Novelty Score')
plt.title('Novelty Score vs. Teleported')
plt.grid(True)
plt.tight_layout()
plt.savefig('novelty_score_vs_tele.png')
plt.show()



# --- 8. Shared Samples Test ---
print("\n--- Running Experiment 2: Shared Samples Test ---")

A_coarse, A_fine = Xc, Xf             # standardized reference half
B0_coarse, B0_fine = Yc, Yf           # standardized test half
N_B = B0_coarse.shape[0]

fractions = np.linspace(0, 1, 11)
scores = []

for frac in fractions:
    num_shared = int(frac * N_B)

    # start from original (standardized) B
    B_coarse = B0_coarse.clone()
    B_fine   = B0_fine.clone()

    if num_shared > 0:
        # choose which A samples to "memorize" into B (rotated/translated copies)
        rep_idx = np.random.choice(len(structs_X), size=num_shared, replace=True)
        s_aug_list = [augment(structs_X[i]) for i in rep_idx]

        # re-featurize augmented structures
        bc_aug = featurize_coarse(s_aug_list).view(num_shared, -1)
        bf_aug = featurize_fine(s_aug_list, model, device=device).view(num_shared, -1)

        # apply the SAME standardizers fitted on X
        bc_aug = apply_standardizer(bc_aug, mu_c, sd_c)
        bf_aug = apply_standardizer(bf_aug, mu_f, sd_f)

        # overwrite first num_shared rows in B with these (standardized) copies
        B_coarse[:num_shared] = bc_aug
        B_fine[:num_shared]   = bf_aug

    # OT is computed in fine space (as in your pipeline)
    P, _ = get_ot_plan(A_fine, B_fine)
    score = get_novelty_score(P, A_coarse, B_coarse, A_fine, B_fine, delta)
    scores.append(float(score))

    print(f"Score for {frac*100:.0f}% equivariant-shared samples: {scores[-1]:.4f}")

# plot as before
plt.figure(figsize=(8, 5))
plt.plot(fractions * 100, scores, marker='o', color='crimson')
plt.xlabel('Percentage of Shared (Memorized) Samples')
plt.ylabel('Novelty Score')
plt.title('Novelty Score vs. Dataset Memorization (equivariant copies)')
plt.grid(True)
plt.tight_layout()
plt.savefig('novelty_score_vs_shared_samples.png')
plt.show()