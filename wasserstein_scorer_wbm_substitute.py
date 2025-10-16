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
str_train = read_structure_from_csv('/home/smuelle4/digital_materials_design/MatSciNovelty/simonpaul/train.csv')
#str_train = random.sample(str_train,1000)

# load all wbm structures and filter to only include stable ones
load_summary_txt = pd.read_csv("/home/smuelle4/digital_materials_design/matbench-discovery/data/wbm/raw/wbm-summary.txt", sep="\t", header = None)
stable_structures = load_summary_txt[load_summary_txt[5]<0]

stable_structures_step_1 = stable_structures.iloc[np.where(np.array([stable_structures[7].iloc[i].split("_")[1] for i in range(len(stable_structures))]) == '1')[0]]
stable_structures_step_2 = stable_structures.iloc[np.where(np.array([stable_structures[7].iloc[i].split("_")[1] for i in range(len(stable_structures))]) == '2')[0]]
stable_structures_step_3 = stable_structures.iloc[np.where(np.array([stable_structures[7].iloc[i].split("_")[1] for i in range(len(stable_structures))]) == '3')[0]]
stable_structures_step_4 = stable_structures.iloc[np.where(np.array([stable_structures[7].iloc[i].split("_")[1] for i in range(len(stable_structures))]) == '4')[0]]
stable_structures_step_5 = stable_structures.iloc[np.where(np.array([stable_structures[7].iloc[i].split("_")[1] for i in range(len(stable_structures))]) == '5')[0]]

unstable_structures = load_summary_txt[load_summary_txt[5]>0]

unstable_structures_step_1 = unstable_structures.iloc[np.where(np.array([unstable_structures[7].iloc[i].split("_")[1] for i in range(len(unstable_structures))]) == '1')[0]]
unstable_structures_step_2 = unstable_structures.iloc[np.where(np.array([unstable_structures[7].iloc[i].split("_")[1] for i in range(len(unstable_structures))]) == '2')[0]]
unstable_structures_step_3 = unstable_structures.iloc[np.where(np.array([unstable_structures[7].iloc[i].split("_")[1] for i in range(len(unstable_structures))]) == '3')[0]]
unstable_structures_step_4 = unstable_structures.iloc[np.where(np.array([unstable_structures[7].iloc[i].split("_")[1] for i in range(len(unstable_structures))]) == '4')[0]]
unstable_structures_step_5 = unstable_structures.iloc[np.where(np.array([unstable_structures[7].iloc[i].split("_")[1] for i in range(len(unstable_structures))]) == '5')[0]]


# First round of substitution
with bz2.open("/home/smuelle4/digital_materials_design/matbench-discovery/data/wbm/raw/wbm-structures-step-1.json.bz2") as fh:
  data = json.loads(fh.read().decode('utf-8'))

stable_structures_1 = [Structure.from_dict(data[entry]["opt"]) for entry in stable_structures_step_1[7].values]
#structures_1 = random.sample(structures_1,2000)
unstable_structures_1 = [Structure.from_dict(data[entry]["opt"]) for entry in unstable_structures_step_1[7].values]

print("Loaded wbm step 1")

# Second round of substitution
with bz2.open("/home/smuelle4/digital_materials_design/matbench-discovery/data/wbm/raw/wbm-structures-step-2.json.bz2") as fh:
  data = json.loads(fh.read().decode('utf-8'))

stable_structures_2 = [Structure.from_dict(data[entry]["opt"]) for entry in stable_structures_step_2[7].values]
#structures_2 = random.sample(structures_2,2000)
unstable_structures_2 = [Structure.from_dict(data[entry]["opt"]) for entry in unstable_structures_step_2[7].values]

print("Loaded wbm step 2")

# Third round of substitution
with bz2.open("/home/smuelle4/digital_materials_design/matbench-discovery/data/wbm/raw/wbm-structures-step-3.json.bz2") as fh:
  data = json.loads(fh.read().decode('utf-8'))

stable_structures_3 = [Structure.from_dict(data[entry]["opt"]) for entry in stable_structures_step_3[7].values]
#structures_3 = random.sample(structures_3,2000)
unstable_structures_3 = [Structure.from_dict(data[entry]["opt"]) for entry in unstable_structures_step_3[7].values]

print("Loaded wbm step 3")

# Fourth round of substitution
with bz2.open("/home/smuelle4/digital_materials_design/matbench-discovery/data/wbm/raw/wbm-structures-step-4.json.bz2") as fh:
  data = json.loads(fh.read().decode('utf-8'))

stable_structures_4 = [Structure.from_dict(data[entry]["opt"]) for entry in stable_structures_step_4[7].values]
#structures_4 = random.sample(structures_4,2000)
unstable_structures_4 = [Structure.from_dict(data[entry]["opt"]) for entry in unstable_structures_step_4[7].values]

print("Loaded wbm step 4")

# Fifth round of substitution
with bz2.open("/home/smuelle4/digital_materials_design/matbench-discovery/data/wbm/raw/wbm-structures-step-5.json.bz2") as fh:
  data = json.loads(fh.read().decode('utf-8'))

stable_structures_5 = [Structure.from_dict(data[entry]["opt"]) for entry in stable_structures_step_5[7].values]
#structures_5 = random.sample(structures_5,2000)
unstable_structures_5 = [Structure.from_dict(data[entry]["opt"]) for entry in unstable_structures_step_5[7].values]

print("Loaded wbm step 5")

device  = "cuda" if torch.cuda.is_available() else "cpu"

model = CrystalGCN(hidden_dim=128).to(device)
model.load_state_dict(torch.load('gcn.pt', map_location=device))

# --- 2. Featurize ---
print("Featurizing coarsely...")
train_feat_coarse = featurize_coarse(str_train)
# Create a mask
mask = torch.ones(train_feat_coarse.size(0), dtype=torch.bool)
mask[np.unique(np.where(train_feat_coarse.isnan())[0])] = False
# Apply mask
train_feat_coarse = train_feat_coarse[mask]

print("Featurizing with GCN (fine)...")
train_feat_fine = featurize_fine(str_train, model, device=device)
# Apply mask
train_feat_fine = train_feat_fine[mask]

print(train_feat_coarse.shape, train_feat_fine.shape)

# --- 3. Force 2D shapes ---
train_feat_coarse = train_feat_coarse.view(len(train_feat_coarse), -1)
train_feat_fine   = train_feat_fine.view(len(train_feat_fine), -1)

# 4. filter to get out duplicated
labels = cluster_split(train_feat_coarse.numpy(), n_clusters=2)
idx_X = np.where(labels == 0)[0]
idx_Y = np.where(labels == 1)[0]

Xc, Yc = train_feat_coarse[idx_X], train_feat_coarse[idx_Y]
Xf, Yf = train_feat_fine[idx_X],   train_feat_fine[idx_Y]

# Save the structures too for perturbation
structs_X = [str_train[i] for i in idx_X]
structs_Y = [str_train[i] for i in idx_Y]

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

# --- 7. wbm - experiments ---
to_keep = list(range(2000, 0, -200))
to_substitute = list(range(0, 2000, 200))

stable_struct_1 =random.sample(stable_structures_1, 2000)
unstable_struct_1 =random.sample(unstable_structures_1, 2000)

scores = []
k = 1

for i in range(len(to_substitute)):

    fin_struct_set = stable_struct_1[0:to_keep[i]] + unstable_struct_1[0:to_substitute[i]]

    pc = featurize_coarse(fin_struct_set).view(len(fin_struct_set), -1)
    pf = featurize_fine(fin_struct_set, model, device=device).view(len(fin_struct_set), -1)

    # Create a mask
    mask = torch.ones(pc.size(0), dtype=torch.bool)
    mask[np.unique(np.where(pc.isnan())[0])] = False
    # Apply mask
    pc = pc[mask]
    pf = pf[mask]

    # ▶ standardize with same mu, std
    pc = apply_standardizer(pc, mu_c, sd_c)
    pf = apply_standardizer(pf, mu_f, sd_f)

    ot_plan, _ = get_ot_plan(Xf, pf)
    score = get_novelty_score(ot_plan, Xc, pc, Xf, pf, delta)
    print(f"step {k}: {score:.4f}")
    scores.append(score)
    k += 1

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(range(1,len(to_substitute)+1), scores, marker='o')
plt.xlabel('Percentage of dropped atoms')
plt.ylabel('Novelty Score')
plt.title('Novelty Score vs. Drop Level')
plt.grid(True)
plt.tight_layout()
plt.savefig('imgs/novelty_wbm_substitute_step1.png')
plt.show()


scores = []
k = 1

for i in range(len(to_substitute)):

    stable_struct_2 =random.sample(stable_structures_2,to_keep[i])
    unstable_struct_2 =random.sample(unstable_structures_2,to_substitute[i])

    fin_struct_set = stable_struct_2+unstable_struct_2

    pc = featurize_coarse(fin_struct_set).view(len(fin_struct_set), -1)
    pf = featurize_fine(fin_struct_set, model, device=device).view(len(fin_struct_set), -1)

    # Create a mask
    mask = torch.ones(pc.size(0), dtype=torch.bool)
    mask[np.unique(np.where(pc.isnan())[0])] = False
    # Apply mask
    pc = pc[mask]
    pf = pf[mask]

    # ▶ standardize with same mu, std
    pc = apply_standardizer(pc, mu_c, sd_c)
    pf = apply_standardizer(pf, mu_f, sd_f)

    ot_plan, _ = get_ot_plan(Xf, pf)
    score = get_novelty_score(ot_plan, Xc, pc, Xf, pf, delta)
    print(f"step {k}: {score:.4f}")
    scores.append(score)
    k += 1

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(range(1,len(to_substitute)+1), scores, marker='o')
plt.xlabel('Percentage of dropped atoms')
plt.ylabel('Novelty Score')
plt.title('Novelty Score vs. Drop Level')
plt.grid(True)
plt.tight_layout()
plt.savefig('imgs/novelty_wbm_substitute_step2.png')
plt.show()



scores = []
k = 1

for i in range(len(to_substitute)):

    stable_struct_3 =random.sample(stable_structures_3,to_keep[i])
    unstable_struct_3 =random.sample(unstable_structures_3,to_substitute[i])

    fin_struct_set = stable_struct_3+unstable_struct_3

    pc = featurize_coarse(fin_struct_set).view(len(fin_struct_set), -1)
    pf = featurize_fine(fin_struct_set, model, device=device).view(len(fin_struct_set), -1)

    # Create a mask
    mask = torch.ones(pc.size(0), dtype=torch.bool)
    mask[np.unique(np.where(pc.isnan())[0])] = False
    # Apply mask
    pc = pc[mask]
    pf = pf[mask]

    # ▶ standardize with same mu, std
    pc = apply_standardizer(pc, mu_c, sd_c)
    pf = apply_standardizer(pf, mu_f, sd_f)

    ot_plan, _ = get_ot_plan(Xf, pf)
    score = get_novelty_score(ot_plan, Xc, pc, Xf, pf, delta)
    print(f"step {k}: {score:.4f}")
    scores.append(score)
    k += 1

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(range(1,len(to_substitute)+1), scores, marker='o')
plt.xlabel('Percentage of dropped atoms')
plt.ylabel('Novelty Score')
plt.title('Novelty Score vs. Drop Level')
plt.grid(True)
plt.tight_layout()
plt.savefig('imgs/novelty_wbm_substitute_step3.png')
plt.show()



scores = []
k = 1

for i in range(len(to_substitute)):

    stable_struct_4 =random.sample(stable_structures_4,to_keep[i])
    unstable_struct_4 =random.sample(unstable_structures_4,to_substitute[i])

    fin_struct_set = stable_struct_4+unstable_struct_4

    pc = featurize_coarse(fin_struct_set).view(len(fin_struct_set), -1)
    pf = featurize_fine(fin_struct_set, model, device=device).view(len(fin_struct_set), -1)

    # Create a mask
    mask = torch.ones(pc.size(0), dtype=torch.bool)
    mask[np.unique(np.where(pc.isnan())[0])] = False
    # Apply mask
    pc = pc[mask]
    pf = pf[mask]

    # ▶ standardize with same mu, std
    pc = apply_standardizer(pc, mu_c, sd_c)
    pf = apply_standardizer(pf, mu_f, sd_f)

    ot_plan, _ = get_ot_plan(Xf, pf)
    score = get_novelty_score(ot_plan, Xc, pc, Xf, pf, delta)
    print(f"step {k}: {score:.4f}")
    scores.append(score)
    k += 1

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(range(1,len(to_substitute)+1), scores, marker='o')
plt.xlabel('Percentage of dropped atoms')
plt.ylabel('Novelty Score')
plt.title('Novelty Score vs. Drop Level')
plt.grid(True)
plt.tight_layout()
plt.savefig('imgs/novelty_wbm_substitute_step4.png')
plt.show()



scores = []
k = 1

for i in range(len(to_substitute)):

    stable_struct_5 =random.sample(stable_structures_5,to_keep[i])
    unstable_struct_5 =random.sample(unstable_structures_5,to_substitute[i])

    fin_struct_set = stable_struct_5+unstable_struct_5

    pc = featurize_coarse(fin_struct_set).view(len(fin_struct_set), -1)
    pf = featurize_fine(fin_struct_set, model, device=device).view(len(fin_struct_set), -1)

    # Create a mask
    mask = torch.ones(pc.size(0), dtype=torch.bool)
    mask[np.unique(np.where(pc.isnan())[0])] = False
    # Apply mask
    pc = pc[mask]
    pf = pf[mask]

    # ▶ standardize with same mu, std
    pc = apply_standardizer(pc, mu_c, sd_c)
    pf = apply_standardizer(pf, mu_f, sd_f)

    ot_plan, _ = get_ot_plan(Xf, pf)
    score = get_novelty_score(ot_plan, Xc, pc, Xf, pf, delta)
    print(f"step {k}: {score:.4f}")
    scores.append(score)
    k += 1

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(range(1,len(to_substitute)+1), scores, marker='o')
plt.xlabel('Percentage of dropped atoms')
plt.ylabel('Novelty Score')
plt.title('Novelty Score vs. Drop Level')
plt.grid(True)
plt.tight_layout()
plt.savefig('imgs/novelty_wbm_substitute_step5.png')
plt.show()