import pandas as pd
import torch
import matplotlib.pyplot as plt
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
struc_crystalformer  = load_generated_model("data_models/crystalformer.csv")
struc_symmcd         = load_generated_model("data_models/symmcd.csv")
struc_wyckoffdiffcspplus = load_generated_model("data_models/wyckoff_diffcsp++.csv")

structure_list = [
    str_val,
    struc_mattergen,
    struc_diffcsp,
    struc_diffcspplus,
    struc_crystalformer,
    struc_symmcd,
    struc_wyckoffdiffcspplus
]

model_names = [
    "Validation",
    "MatterGen",
    "DiffCSP",
    "DiffCSP++",
    "CFormer",
    "SymmCD", 
    "WyckoffDiffCSP++"

]

# ===========================================================
# 2️⃣ Initialize Scorer
# ===========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading pretrained GCN model...")
model = EquivariantCrystalGCN(hidden_dim=128).to(device)
model.load_state_dict(torch.load("gcn_fine.pt", map_location=device))
print("Loaded weights from gcn_fine.pt ✅")


scorer = OTNoveltyScorer(
    train_structures=str_train,
    gnn_model=model,
    tau=0.36,                    # fixed τ (your calibrated value)
    memorization_weight=50.0,
    device=device
)

# ===========================================================
# 3️⃣ Evaluate All Models
# ===========================================================
scores_total, scores_quality, scores_mem = [], [], []
scores_novelty, scores_coverage = [], []

for name, structs in zip(model_names, structure_list):
    print(f"\n▶ Evaluating {name}")
    total, qual, mem = scorer.compute_novelty(structs)
    print(f"  {name}: Total={total:.4f} | Quality={qual:.4f} | Memorization={mem:.4f}")
    scores_total.append(total)
    scores_quality.append(qual)
    scores_mem.append(mem)

    # --- Compute novelty & coverage ---
    gen_feats = scorer.featurizer(structs).to(scorer.device)
    nov = novelty_score(gen_feats, scorer.train_feats, threshold=0.1)
    cov = coverage_score(scorer.train_feats, gen_feats, threshold=0.1)
    print(f"  {name}: Novelty={nov:.3f} | Coverage={cov:.3f}")
    scores_novelty.append(nov)
    scores_coverage.append(cov)

# ===========================================================
# 4️⃣ Plot Results
# ===========================================================
plt.rcParams.update({'font.size': 12})

# 1️⃣ Total Novelty
plt.figure(figsize=(12,5))
plt.bar(model_names, scores_total, color='steelblue')
plt.title("Total Novelty Loss", fontsize=14, fontweight='bold')
plt.ylabel("Loss")
plt.xticks(rotation=30, ha='right')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("imgs/novelty_comparison_total.png", dpi=300)
plt.show()

# 2️⃣ Quality Component
plt.figure(figsize=(12,5))
plt.bar(model_names, scores_quality, color='seagreen')
plt.title("Quality Component", fontsize=14, fontweight='bold')
plt.ylabel("Loss")
plt.xticks(rotation=30, ha='right')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("imgs/novelty_comparison_quality.png", dpi=300)
plt.show()

# 3️⃣ Memorization Component
plt.figure(figsize=(12,5))
plt.bar(model_names, scores_mem, color='crimson')
plt.title("Memorization Component", fontsize=14, fontweight='bold')
plt.ylabel("Loss")
plt.xticks(rotation=30, ha='right')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("imgs/novelty_comparison_memorization.png", dpi=300)
plt.show()

# 4️⃣ Novelty vs Coverage (added metric visualization)
plt.figure(figsize=(12,5))
x = np.arange(len(model_names))
plt.bar(x - 0.2, scores_novelty, width=0.4, label='Novelty', color='royalblue')
plt.bar(x + 0.2, scores_coverage, width=0.4, label='Coverage', color='orange')
plt.xticks(x, model_names, rotation=30, ha='right')
plt.ylabel("Score")
plt.title("GNN Feature Space Novelty & Coverage", fontsize=14, fontweight='bold')
plt.grid(True, axis='y', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("imgs/novelty_coverage_comparison.png", dpi=300)
plt.show()