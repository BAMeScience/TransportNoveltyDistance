import argparse
import io
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from TNovD import (
    EquivariantCrystalGCN,
    TransportNoveltyDistance,
    coverage_score,
    novelty_score,
    read_structure_from_csv,
)
from TNovD.utils import relax_structures

import xtalmet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_MP20 = PROJECT_ROOT / "data" / "mp_20"
DATA_XTALMET = PROJECT_ROOT / "data" / "xtalmet_models"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
IMGS_DIR = PROJECT_ROOT / "imgs"
IMGS_DIR.mkdir(exist_ok=True)

eval_relax = True
RELAX_MODEL = "small"
RELAX_STEPS = 50
RELAX_FMAX = 0.03
RELAX_SUFFIX = "_mace-small_steps50_fmax0p03.pkl"
torch.storage._load_from_bytes = lambda b: torch.load(
    io.BytesIO(b), map_location="cpu", weights_only=False
)
PLOT_FONT_SIZE = 18
PLOT_TITLE_SIZE = 22
PLOT_LABEL_SIZE = 20
PLOT_TICK_SIZE = 17
PLOT_LEGEND_SIZE = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generative models with OT novelty.")
    parser.add_argument(
            "--checkpoint",
            type=Path,
            default=CHECKPOINTS_DIR / "gcn_mp20_hidden32.pt",
            help="Path to the encoder checkpoint (default: checkpoints/egnn_invariant_mp20.pt).")
    return parser.parse_args()



# ===========================================================
# 1️⃣ Load Data
# ===========================================================
print("Loading structures...")
str_train = read_structure_from_csv(DATA_MP20 / "train.csv")
str_val = read_structure_from_csv(DATA_MP20 / "val.csv")
str_test = read_structure_from_csv(DATA_MP20 / "test.csv")


def load_generated_model(path: Path):
    """load generated models."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} structures from {path}.")
    return data


# all generative models (downloaded via scripts/download_xtalmet_models.py)
model_files = [
    ("MatterGen", "mattergen.pkl"),
    ("DiffCSP", "diffcsp.pkl"),
    ("DiffCSP++", "diffcsppp.pkl"),
    ("CDVAE", "cdvae.pkl"),
    ("ADiT", "adit.pkl"),
    ("Chemeleon", "chemeleon.pkl"),
]

model_names = ["Test"]
structure_list = [str_test]
loaded_models = {}

for model_name, filename in model_files:
    model_names.append(model_name)
    structs = load_generated_model(DATA_XTALMET / filename)
    loaded_models[model_name] = structs
    structure_list.append(structs)

if eval_relax:
    for model_name, filename in model_files:
        relaxed_path = DATA_XTALMET / f"{Path(filename).stem}{RELAX_SUFFIX}"
        if not relaxed_path.exists() or relaxed_path.stat().st_size == 0:
            print(
                f"Missing relaxed {model_name}; relaxing for {RELAX_STEPS} steps "
                f"with fmax={RELAX_FMAX}."
            )
            relaxed_structs = relax_structures(
                loaded_models[model_name],
                mace_model=RELAX_MODEL,
                device="cuda" if torch.cuda.is_available() else "cpu",
                steps=RELAX_STEPS,
                fmax=RELAX_FMAX,
            )
            with open(relaxed_path, "wb") as f:
                pickle.dump(relaxed_structs, f)
            print(f"Saved {len(relaxed_structs)} structures to {relaxed_path}.")
        else:
            relaxed_structs = load_generated_model(relaxed_path)

        model_names.append(f"{model_name} relaxed")
        structure_list.append(relaxed_structs)

# ===========================================================
# 2️⃣ Initialize Scorer
# ===========================================================
args = parse_args()
checkpoint_path = args.checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"
# --- Load pretrained model ---
print("Loading pretrained GCN model...")
model = EquivariantCrystalGCN(hidden_dim=32).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
print("Loaded weights from gcn_fine.pt ✅")


scorer = TransportNoveltyDistance(
    train_structures=str_train,
    gnn_model=model,
    device=device,
    calibration_sample_size=len(str_val),
    calibration_structures=str_val,
)
# ===========================================================
# 3️⃣ Evaluate All Models
# ===========================================================
scores_total, scores_quality, scores_mem = [], [], []
scores_novelty, scores_coverage = [], []

for name, structs in zip(model_names, structure_list):
    print(f"\n▶ Evaluating {name}")
    total, qual, mem = scorer.compute_TNovD(structs)
    print(f"  {name}: Total={total:.4f} | Quality={qual:.4f} | Memorization={mem:.4f}")
    scores_total.append(total)
    scores_quality.append(qual)
    scores_mem.append(mem)

    # --- Compute novelty & coverage ---
    gen_feats = scorer.featurizer(structs)
    nov = novelty_score(gen_feats, scorer.train_feats, threshold=0.1)
    cov = coverage_score(scorer.train_feats, gen_feats, threshold=0.1)
    print(f"  {name}: Novelty={nov:.3f} | Coverage={cov:.3f}")
    scores_novelty.append(nov)
    scores_coverage.append(cov)

# ===========================================================
#  Plot Results
# ===========================================================
plt.rcParams.update(
    {
        "font.size": PLOT_FONT_SIZE,
        "axes.titlesize": PLOT_TITLE_SIZE,
        "axes.labelsize": PLOT_LABEL_SIZE,
        "xtick.labelsize": PLOT_TICK_SIZE,
        "ytick.labelsize": PLOT_TICK_SIZE,
        "legend.fontsize": PLOT_LEGEND_SIZE,
    }
)
raw_idx = [i for i, name in enumerate(model_names) if not name.endswith(" relaxed")]
test_idx = [i for i, name in enumerate(model_names) if name == "Test"]
relaxed_idx = test_idx + [
    i for i, name in enumerate(model_names) if name.endswith(" relaxed")
]

raw_names = [model_names[i] for i in raw_idx]
raw_quality = [scores_quality[i] for i in raw_idx]
raw_mem = [scores_mem[i] for i in raw_idx]
fig_width = max(12, 0.9 * len(raw_names))

plt.figure(figsize=(fig_width, 6))
plt.bar(raw_names, raw_quality, label="Quality", color="lightblue", alpha=0.8)
plt.bar(
    raw_names,
    raw_mem,
    bottom=raw_quality,
    label="Memorization",
    color="red",
    alpha=0.6,
)
plt.title("Unrelaxed Transport Novelty Distance", fontweight="bold")
plt.ylabel("TNovD")
plt.ylim(top=max(np.array(raw_quality) + np.array(raw_mem)) * 1.15)
plt.xticks(rotation=30, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), borderaxespad=0.0)
plt.tight_layout()
plt.savefig(IMGS_DIR / "novelty_comparison_raw_components_cgc.png", dpi=300)
plt.show()

if relaxed_idx:
    relaxed_names = [model_names[i].replace(" relaxed", "") for i in relaxed_idx]
    relaxed_quality = [scores_quality[i] for i in relaxed_idx]
    relaxed_mem = [scores_mem[i] for i in relaxed_idx]
    fig_width = max(12, 0.9 * len(relaxed_names))

    plt.figure(figsize=(fig_width, 6))
    plt.bar(relaxed_names, relaxed_quality, label="Quality", color="lightblue", alpha=0.8)
    plt.bar(
        relaxed_names,
        relaxed_mem,
        bottom=relaxed_quality,
        label="Memorization",
        color="red",
        alpha=0.6,
    )
    plt.title("Relaxed Transport Novelty Distance", fontweight="bold")
    plt.ylabel("TNovD")
    plt.ylim(top=max(np.array(relaxed_quality) + np.array(relaxed_mem)) * 1.15)
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig(IMGS_DIR / "novelty_comparison_relaxed_components_cgc.png", dpi=300)
    plt.show()
