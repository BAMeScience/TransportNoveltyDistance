import pandas as pd
import torch
import matplotlib.pyplot as plt
from was_utils import *
from utils_new import *
from gcn import CrystalGCN
from wasserstein_novelty import OTNoveltyScorer   # your new class
from utils import * 

# ===========================================================
# 1️⃣ Load Data
# ===========================================================
print("Loading structures...")
str_train = read_structure_from_csv('train.csv')
str_val   = read_structure_from_csv('val.csv')

def load_generated_model(path):
    df = pd.read_csv(path)
    return load_structures_from_json_column(df)

# all generative models
struc_mattergen      = load_generated_model("data_models/mattergen.csv")
struc_diffcsp        = load_generated_model("data_models/diffcsp.csv")
struc_diffcspplus    = load_generated_model("data_models/diffcsp++.csv")
struc_undiffcspplus  = load_generated_model("data_models/undiffcsp++44.csv")
struc_crystalformer  = load_generated_model("data_models/crystalformer.csv")
struc_symmcd         = load_generated_model("data_models/symmcd.csv")

structure_list = [
    str_val,
    struc_mattergen,
    struc_diffcsp,
    struc_diffcspplus,
    struc_undiffcspplus,
    struc_crystalformer,
    struc_symmcd
]

model_names = [
    "Validation",
    "MatterGen",
    "DiffCSP",
    "DiffCSP++",
    "Un-DiffCSP++",
    "CrystalFormer",
    "SymmCD"
]

# ===========================================================
# 2️⃣ Initialize Scorer
# ===========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

scorer = OTNoveltyScorer(
    train_structures=str_train,
    gnn_model=CrystalGCN(hidden_dim=128).to(device),
    tau=0.36,                    # fixed τ (your calibrated value)
    memorization_weight=10.0,
    device=device
)

# ===========================================================
# 3️⃣ Evaluate All Models
# ===========================================================
scores_total, scores_quality, scores_mem = [], [], []

for name, structs in zip(model_names, structure_list):
    print(f"\n▶ Evaluating {name}")
    total, qual, mem = scorer.compute_novelty(structs)
    print(f"  {name}: Total={total:.4f} | Quality={qual:.4f} | Memorization={mem:.4f}")
    scores_total.append(total)
    scores_quality.append(qual)
    scores_mem.append(mem)

# ===========================================================
# 4️⃣ Plot Results
# ===========================================================
plt.rcParams.update({'font.size': 12})

# 1️⃣ Total Novelty
plt.figure(figsize=(8,5))
plt.bar(model_names, scores_total, color='steelblue')
plt.title("Total Novelty Loss", fontsize=14, fontweight='bold')
plt.ylabel("Loss")
plt.xticks(rotation=30, ha='right')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("imgs/novelty_comparison_total.png", dpi=300)
plt.show()

# 2️⃣ Quality Component
plt.figure(figsize=(8,5))
plt.bar(model_names, scores_quality, color='seagreen')
plt.title("Quality Component", fontsize=14, fontweight='bold')
plt.ylabel("Loss")
plt.xticks(rotation=30, ha='right')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("imgs/novelty_comparison_quality.png", dpi=300)
plt.show()

# 3️⃣ Memorization Component
plt.figure(figsize=(8,5))
plt.bar(model_names, scores_mem, color='crimson')
plt.title("Memorization Component", fontsize=14, fontweight='bold')
plt.ylabel("Loss")
plt.xticks(rotation=30, ha='right')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("imgs/novelty_comparison_memorization.png", dpi=300)
plt.show()