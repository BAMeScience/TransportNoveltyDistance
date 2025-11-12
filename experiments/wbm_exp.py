import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from pymatgen.core import Structure

from matscinovelty import (
    EquivariantCrystalGCN,
    OTNoveltyScorer,
    read_structure_from_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_MP20 = PROJECT_ROOT / "data" / "mp_20"
WBM_DIR = PROJECT_ROOT / "data" / "wbm"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
IMGS_DIR = PROJECT_ROOT / "imgs"
IMGS_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate WBM progression using the novelty metric."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINTS_DIR / "gcn_fine.pt",
        help="Path to the encoder checkpoint (default: checkpoints/gcn_fine.pt).",
    )
    return parser.parse_args()


args = parse_args()

# ===========================================================
# 1️⃣ Load Data
# ===========================================================
print("Loading WBM data...")
str_train = read_structure_from_csv(DATA_MP20 / "train.csv")

# --- Load stable WBM structures ---
summary = pd.read_csv(WBM_DIR / "wbm-summary.csv")
stable = summary[summary["e_form_per_atom_wbm"] < 0].copy()
stable["step"] = (
    pd.to_numeric(stable["material_id"].str.split("-").str[1], errors="coerce")
    .fillna(0)
    .astype(int)
)


def load_structures_for_step(step):
    path = WBM_DIR / f"wbm-structures-step-{step}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_wbm_data.py` to "
            "cache the per-step relaxed structures."
        )
    with open(path) as fh:
        data = json.load(fh)
    subset = stable[stable["step"] == step]

    structs = []
    for mid in subset["material_id"]:
        entry = data.get(str(mid))
        if entry is None:
            continue
        struct_dict = (
            entry["opt"] if isinstance(entry, dict) and "opt" in entry else entry
        )
        structs.append(Structure.from_dict(struct_dict))

    print(f"Loaded {len(structs)} structures for WBM step {step}")
    return structs


wbm_steps = [load_structures_for_step(i) for i in range(1, 6)]

# ===========================================================
# 2️⃣ Initialize Scorer
# ===========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading pretrained GCN model...")
checkpoint_path = args.checkpoint
if not checkpoint_path.exists():
    raise SystemExit(f"Checkpoint not found at {checkpoint_path}.")

model = EquivariantCrystalGCN(hidden_dim=128).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
print(f"Loaded weights from {checkpoint_path.name} ✅")

scorer = OTNoveltyScorer(
    train_structures=str_train,
    gnn_model=model,
    tau=None,  # auto-estimate τ
    tau_quantile=0.05,
    memorization_weight=10.0,
    device=device,
)
print(f"Estimated τ = {scorer.tau:.4f}")

# ===========================================================
# 3️⃣ Evaluate WBM Steps
# ===========================================================
scores_total, scores_quality, scores_mem = [], [], []
max_len = min(len(s) for s in wbm_steps)  # or set manually, e.g. max_len = 1000

for step_idx, step_structs in enumerate(wbm_steps, start=1):
    print(f"\n▶ Evaluating WBM step {step_idx}")

    # ✅ sample same number of materials each step
    if len(step_structs) > max_len:
        step_subset = random.sample(step_structs, max_len)
    else:
        step_subset = step_structs

    total, qual, mem = scorer.compute_novelty(step_subset)
    print(
        f"Step {step_idx}: Total={total:.4f} | Quality={qual:.4f} | Memorization={mem:.4f}"
    )

    scores_total.append(total)
    scores_quality.append(qual)
    scores_mem.append(mem)
# ===========================================================
# 4️⃣ Plot Results
# ===========================================================
steps = range(1, len(scores_total) + 1)
plt.figure(figsize=(8, 5))
plt.plot(steps, scores_total, marker="o", label="Total Loss", linewidth=2)
plt.plot(steps, scores_quality, marker="s", label="Quality Component", linestyle="--")
plt.plot(steps, scores_mem, marker="^", label="Memorization Component", linestyle=":")
plt.xlabel("WBM Step")
plt.ylabel("Novelty Loss")
plt.title("Novelty Loss Components vs. WBM Step")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(IMGS_DIR / "novelty_wbm_fine_components.png", dpi=300)
plt.show()
