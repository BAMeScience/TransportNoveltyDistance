import pandas as pd
import torch
from numpy.exceptions import ComplexWarning

import matplotlib.pyplot as plt
from was_utils import *
from utils_new import *
from gcn import *


# --- 0. Load data and model ---

# load training data
str_train = read_structure_from_csv('train.csv')

str_val = read_structure_from_csv('val.csv')


struc_mattergen = pd.read_csv("data_models/mattergen.csv")

struc_mattergen = load_structures_from_json_column(struc_mattergen)

struc_diffcsp= pd.read_csv("data_models/diffcsp.csv")
struc_diffcsp = load_structures_from_json_column(struc_diffcsp)


struc_diffcspplus = pd.read_csv("data_models/diffcsp++.csv")
struc_diffcspplus  = load_structures_from_json_column(struc_diffcspplus)




struc_symmcd = pd.read_csv("data_models/symmcd.csv")
struc_symmcd= load_structures_from_json_column(struc_symmcd)
print('lengths of structure sets', len(str_train), len(str_val), len(struc_mattergen), 
len(struc_diffcsp), len(struc_diffcspplus), len(struc_symmcd))


device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. load fine model ---

model = EquivariantCrystalGCN(hidden_dim=128, num_rbf=32).to(device)
model.load_state_dict(torch.load("gcn_fine.pt", map_location=device))

# --- 2. featurize training set (fine only) ---
print("Featurizing with GCN (fine)...")
train_feat_fine = featurize_fine(str_train, model, device=device)
train_feat_fine = train_feat_fine.view(len(train_feat_fine), -1)

# --- 3. estimate τ automatically from train features ---
#tau = estimate_tau_ot(train_feat_fine, quantile=0.05)
tau  = 0.36 
print(f"Estimated τ = {tau:.4f}")



# --- structure sets ---
structure_list = [str_val,
    struc_mattergen,
    struc_diffcsp,
    struc_diffcspplus,
    struc_undiffcspplus,
    struc_crystalformer,
    struc_symmcd
]

# --- corresponding model names ---
model_names = ["val",
    "MatterGen",
    "DiffCSP",
    "DiffCSP++",
    "Un-DiffCSP++",
    "CrystalFormer",
    "SymmCD"
]
scores_total, scores_quality, scores_memorization = [], [], []

for name, struct_set in zip(model_names, structure_list):
    print(f"\n▶ Evaluating {name}")
    pf = featurize_fine(struct_set, model, device=device).view(len(struct_set), -1)

    # OT plan between training set and generated set
    P, C = get_ot_plan(train_feat_fine, pf)

    # compute asymmetric novelty loss
    loss, _, qual_comp, mem_comp = get_novelty_loss_only_fine(P, C, tau=tau)

    print(f"  {name}: Total={loss.item():.4f} | Quality={qual_comp.item():.4f} | Memorization={mem_comp.item():.4f}")

    scores_total.append(loss.item())
    scores_quality.append(qual_comp.item())
    scores_memorization.append(mem_comp.item())


# ✅ 1️⃣ Total Loss
plt.figure(figsize=(7, 5))
plt.bar(model_names, scores_total, color='steelblue')
plt.title("Total Novelty Loss", fontsize=14, fontweight='bold')
plt.ylabel("Loss")
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("imgs/novelty_comparison_total.png", dpi=300)
plt.show()

# ✅ 2️⃣ Quality Component
plt.figure(figsize=(7, 5))
plt.bar(model_names, scores_quality, color='seagreen')
plt.title("Quality Component", fontsize=14, fontweight='bold')
plt.ylabel("Loss")
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("imgs/novelty_comparison_quality.png", dpi=300)
plt.show()

# ✅ 3️⃣ Memorization Component
plt.figure(figsize=(7, 5))
plt.bar(model_names, scores_memorization, color='crimson')
plt.title("Memorization Component", fontsize=14, fontweight='bold')
plt.ylabel("Loss")
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("imgs/novelty_comparison_memorization.png", dpi=300)
plt.show()