import pandas as pd
import torch
from pymatgen.core import Structure
import warnings
from matminer.featurizers.site import SOAP
import numpy as np
from numpy.exceptions import ComplexWarning
import json, bz2
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.core import Structure, Lattice, PeriodicSite
import ot
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import math
import hashlib
import random
from was_utils import *
from utils_new import *
from gcn import *

warnings.simplefilter("ignore", ComplexWarning) # Suppress ComplexWarning specifically

# --- 0. Load data and model ---

# load training data
str_train = read_structure_from_csv('train.csv')
#str_train = remove_structural_duplicates(str_train)
#str_train = random.sample(str_train,1000)

# load all wbm structures and filter to only include stable ones
load_summary_txt = pd.read_csv("wbm_data/wbm-summary.txt", sep="\t", header = None)
stable_structures = load_summary_txt[load_summary_txt[5]<0]

stable_structures_step_1 = stable_structures.iloc[np.where(np.array([stable_structures[7].iloc[i].split("_")[1] for i in range(len(stable_structures))]) == '1')[0]]
stable_structures_step_2 = stable_structures.iloc[np.where(np.array([stable_structures[7].iloc[i].split("_")[1] for i in range(len(stable_structures))]) == '2')[0]]
stable_structures_step_3 = stable_structures.iloc[np.where(np.array([stable_structures[7].iloc[i].split("_")[1] for i in range(len(stable_structures))]) == '3')[0]]
stable_structures_step_4 = stable_structures.iloc[np.where(np.array([stable_structures[7].iloc[i].split("_")[1] for i in range(len(stable_structures))]) == '4')[0]]
stable_structures_step_5 = stable_structures.iloc[np.where(np.array([stable_structures[7].iloc[i].split("_")[1] for i in range(len(stable_structures))]) == '5')[0]]

# First round of substitution
with open("wbm_data/wbm-structures-step-1.json") as fh:
     data = json.load(fh)

structures_1 = [Structure.from_dict(data[entry]["opt"]) for entry in stable_structures_step_1[7].values]
#structures_1 = random.sample(structures_1,2000)

print("Loaded wbm step 1")

# Second round of substitution
with open("wbm_data/wbm-structures-step-2.json") as fh:
     data = json.load(fh)

structures_2 = [Structure.from_dict(data[entry]["opt"]) for entry in stable_structures_step_2[7].values]
#structures_2 = random.sample(structures_2,2000)

print("Loaded wbm step 2")

# Third round of substitution
with open("wbm_data/wbm-structures-step-3.json") as fh:
     data = json.load(fh)

structures_3 = [Structure.from_dict(data[entry]["opt"]) for entry in stable_structures_step_3[7].values]
#structures_3 = random.sample(structures_3,2000)

print("Loaded wbm step 3")

# Fourth round of substitution
with open("wbm_data/wbm-structures-step-4.json") as fh:
     data = json.load(fh)

structures_4 = [Structure.from_dict(data[entry]["opt"]) for entry in stable_structures_step_4[7].values]
#structures_4 = random.sample(structures_4,2000)

print("Loaded wbm step 4")

# Fifth round of substitution
with open("wbm_data/wbm-structures-step-5.json") as fh:
    data = json.load(fh)

structures_5 = [Structure.from_dict(data[entry]["opt"]) for entry in stable_structures_step_5[7].values]
#structures_5 = random.sample(structures_5,2000)

print("Loaded wbm step 5")
list_with_wbm_struct = [structures_1,structures_2,structures_3,structures_4,structures_5]

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. load fine model ---
model = EquivariantCrystalGCN(hidden_dim=128, num_rbf=32).to(device)
model.load_state_dict(torch.load("gcn_fine.pt", map_location=device))

# --- 2. featurize training set (fine only) ---
print("Featurizing with GCN (fine)...")
train_feat_fine = featurize_fine(str_train, model, device=device)
train_feat_fine = train_feat_fine.view(len(train_feat_fine), -1)

# --- 3. estimate τ automatically from train features ---
tau = estimate_tau_ot(train_feat_fine, quantile=0.05)
print(f"Estimated τ = {tau:.4f}")

# --- 4. evaluate novelty across WBM steps ---
scores_total = []
scores_quality = []
scores_memorization = []

for k, wbm_step in enumerate(list_with_wbm_struct, start=1):
    print(f"\n▶ Step {k}")
    pf = featurize_fine(wbm_step, model, device=device).view(len(wbm_step), -1)

    # OT plan between full train and current step
    P, C = get_ot_plan(train_feat_fine, pf)

    # fine-only asymmetric loss
    loss, _, qual_comp, mem_comp = get_novelty_loss_only_fine(P, C, tau=tau)

    print(f"  Total={loss.item():.4f} | Quality={qual_comp.item():.4f} | Memorization={mem_comp.item():.4f}")

    scores_total.append(loss.item())
    scores_quality.append(qual_comp.item())
    scores_memorization.append(mem_comp.item())

# --- plotting ---
plt.figure(figsize=(8, 5))
steps = range(1, len(scores_total) + 1)

plt.plot(steps, scores_total, marker='o', label='Total Loss', linewidth=2)
plt.plot(steps, scores_quality, marker='s', label='Quality Component', linestyle='--')
plt.plot(steps, scores_memorization, marker='^', label='Memorization Component', linestyle=':')

plt.xlabel('WBM Step')
plt.ylabel('Novelty Loss')
plt.title('Novelty Loss Components vs. WBM Step')
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig('imgs/novelty_wbm_fine_components.png')
plt.show()