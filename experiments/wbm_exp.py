import argparse
import json
import random
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from pymatgen.core import Structure

from TNovD import (
    EquivariantCrystalGCN,
    TransportNoveltyDistance,
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
        "--model",
        choices=("equivariant"),
        default="equivariant",
        help="Trained backbone architecture.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINTS_DIR / "gcn_mp20_final.pt",
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
STABILITY_COLUMN = "e_form_per_atom_mp2020_corrected"
# MP2020-corrected energies incorporate the Materials Project correction scheme
# (anion redox, GGA/GGA+U alignment, etc.) to align DFT formation energies with
# experimental thermochemistry. Using the corrected column ensures stability
# decisions match the canonical MP phase diagram rather than raw (uncorrected)
# energies, which can be biased for transition-metal oxides and similar chemistries.
if STABILITY_COLUMN not in summary.columns:
    raise KeyError(
        f"Column '{STABILITY_COLUMN}' missing from WBM summary. "
        "Please rerun scripts/download_wbm_data.py to refresh the dataset."
    )
summary["step"] = (
    pd.to_numeric(summary["material_id"].str.split("-").str[1], errors="coerce")
    .fillna(0)
    .astype(int)
)
summary_stable = summary[summary[STABILITY_COLUMN] < 0].copy()


def load_structures_for_step(step: int, summary_df: pd.DataFrame, stable: bool = False):
    suffix = "-stable" if stable else ""
    path = WBM_DIR / f"wbm-structures-step-{step}{suffix}.json"
    if not path.exists():
        if stable:
            print(f"⚠ Missing stable file for step {step}; skipping.")
            return []
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_wbm_data.py` "
            "to cache the per-step relaxed structures."
        )
    with open(path) as fh:
        data = json.load(fh)
    subset = summary_df[summary_df["step"] == step]

    structs = []
    if subset.empty and step == 0:
        for entry in data.values():
            struct_dict = (
                entry["opt"] if isinstance(entry, dict) and "opt" in entry else entry
            )
            structs.append(Structure.from_dict(struct_dict))
    else:
        for mid in subset["material_id"]:
            entry = data.get(str(mid))
            if entry is None:
                continue
            struct_dict = (
                entry["opt"] if isinstance(entry, dict) and "opt" in entry else entry
            )
            structs.append(Structure.from_dict(struct_dict))

    print(
        f"Loaded {len(structs)} {'stable ' if stable else ''}structures for WBM step {step}"
    )
    return structs


def build_step_structures(stable: bool) -> list[list[Structure]]:
    df = summary_stable if stable else summary
    baseline = load_structures_for_step(0, df, stable=stable)
    if not baseline:
        baseline = str_train
    wbm_sets = [load_structures_for_step(i, df, stable=stable) for i in range(1, 6)]
    return [baseline] + wbm_sets


# ===========================================================
# 2️⃣ Initialize Scorer
# ===========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = args.checkpoint
if not checkpoint_path.exists():
    raise SystemExit(f"Checkpoint not found at {checkpoint_path}.")

print("Loading pretrained GCN model...")
model = EquivariantCrystalGCN(hidden_dim=32).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
print("Loaded weights from gnn ✅")

scorer = TransportNoveltyDistance(
    train_structures=str_train,
    gnn_model=model,
    tau=None,  # auto-estimate τ
    memorization_weight=None,
    device=device,
)
print(f"Estimated τ = {scorer.tau:.4f}")

# ===========================================================
# 3️⃣ Evaluate WBM Steps
# ===========================================================


# helper functions for evaluation/plotting
def compute_scores(step_structs: list[list[Structure]], label: str):
    scores_total, scores_quality, scores_mem = [], [], []
    wbm_only = [s for s in step_structs[1:] if len(s) > 0]
    max_len = min((len(s) for s in wbm_only), default=len(step_structs[0]))

    for step_idx, step_struct_list in enumerate(step_structs):
        label_str = (
            "0 (WBM initial / MP-20)"
            if step_idx == 0
            else f"{step_idx} (WBM step {step_idx})"
        )
        print(f"\n▶ Evaluating {label_str} [{label}]")

        if len(step_struct_list) == 0:
            print(f"⚠ Skipping step {step_idx} for {label}: no structures available.")
            scores_total.append(float("nan"))
            scores_quality.append(float("nan"))
            scores_mem.append(float("nan"))
            continue

        cap = max_len if step_idx > 0 and max_len > 0 else len(step_struct_list)
        step_subset = (
            random.sample(step_struct_list, cap)
            if len(step_struct_list) > cap
            else step_struct_list
        )

        total, qual, mem = scorer.compute_TNovD(step_subset)
        print(
            f"Step {step_idx}: Total={total:.4f} | Quality={qual:.4f} | Memorization={mem:.4f}"
        )

        scores_total.append(total)
        scores_quality.append(qual)
        scores_mem.append(mem)

    return list(range(len(step_structs))), scores_total, scores_quality, scores_mem


def plot_scores(step_indices, totals, qualities, mems, suffix, note):
    plt.rcParams.update({"font.size": 14})
    plt.figure(figsize=(8, 5))
    plt.plot(step_indices, totals, marker="o", label="Total Loss", linewidth=2)
    plt.plot(
        step_indices, qualities, marker="s", label="Quality Component", linestyle="--"
    )
    plt.plot(
        step_indices, mems, marker="^", label="Memorization Component", linestyle=":"
    )
    plt.xlabel("Step (0 = MP-20 baseline)")
    plt.ylabel("Novelty Loss")
    plt.title(f"Novelty Loss Components vs. WBM Step{note}")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.xticks(step_indices, [str(i) for i in step_indices])
    base_name = f"novelty_wbm_fine_components{suffix}"
    plt.savefig(IMGS_DIR / f"{base_name}_autotuned.png", dpi=300)
    plt.savefig(IMGS_DIR / f"{base_name}_autotuned.pdf")
    plt.show()


all_steps = build_step_structures(stable=False)
stable_steps = build_step_structures(stable=True)

indices_all, totals_all, quals_all, mems_all = compute_scores(all_steps, "all")
plot_scores(indices_all, totals_all, quals_all, mems_all, "", "")

indices_stable, totals_stable, quals_stable, mems_stable = compute_scores(
    stable_steps, "stable"
)
plot_scores(
    indices_stable, totals_stable, quals_stable, mems_stable, "_stable", " (Stable)"
)
