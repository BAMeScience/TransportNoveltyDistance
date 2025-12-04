import argparse
import pickle
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from TNovD import (
    EquivariantCrystalGCN,
    TransportNoveltyDistance,
    canonicalize_structure,
    coverage_score,
    novelty_score,
    read_structure_from_csv,
)

try:  # Ensure xtalmet package is present for pickle deserialization.
    import xtalmet  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "The xtalmet package is required to deserialize the downloaded pickles. "
        "Install it (or provide an equivalent shim on PYTHONPATH) before running this script."
    ) from exc



PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_MP20 = PROJECT_ROOT / "data" / "mp_20"
DATA_MODELS = PROJECT_ROOT / "data_models"
DATA_XTALMET = PROJECT_ROOT / "data" / "xtalmet_models"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
IMGS_DIR = PROJECT_ROOT / "imgs"
IMGS_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generative models with OT novelty."
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
        help="Path to the encoder checkpoint (default: checkpoints/egnn_invariant_mp20.pt).",
    )
    return parser.parse_args()


# ===========================================================
# 1️⃣ Load Data
# ===========================================================
print("Loading structures...")
str_train = read_structure_from_csv(DATA_MP20 / "train.csv")
str_val = read_structure_from_csv(DATA_MP20 / "val.csv")


def load_generated_model(path: Path, canonicalize: bool = True):
    """Load a list of pymatgen Structures from a pickle file, optionally canonicalizing each."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/download_xtalmet_models.py or point the "
            "script at a folder containing the xtalmet pickles."
        )

    with open(path, "rb") as f:
        data = pickle.load(f)

    if canonicalize:
        data = [canonicalize_structure(s) for s in data]

    print(f"Loaded {len(data)} structures from {path}.")
    return data


# all generative models (downloaded via scripts/download_xtalmet_models.py)
struc_mattergen = load_generated_model(DATA_XTALMET / "mattergen.pkl")
struc_diffcsp = load_generated_model(DATA_XTALMET / "diffcsp.pkl")
struc_diffcspplus = load_generated_model(DATA_XTALMET / "diffcsppp.pkl")
struc_cdvae = load_generated_model(DATA_XTALMET / "cdvae.pkl")
struc_adit = load_generated_model(DATA_XTALMET / "adit.pkl")
struc_chemeleon = load_generated_model(DATA_XTALMET / "chemeleon.pkl")

structure_list = [
    str_val,
    struc_mattergen,
    struc_diffcsp,
    struc_diffcspplus,
    struc_cdvae,
    struc_adit,
    struc_chemeleon,
]

model_names = [
    "Validation",
    "MatterGen",
    "DiffCSP",
    "DiffCSP++",
    "CdVAE",
    "Adit",
    "Chemeleon",
]

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
    gen_feats = scorer.featurizer(structs).to(scorer.device)
    nov = novelty_score(gen_feats, scorer.train_feats, threshold=0.1)
    cov = coverage_score(scorer.train_feats, gen_feats, threshold=0.1)
    print(f"  {name}: Novelty={nov:.3f} | Coverage={cov:.3f}")
    scores_novelty.append(nov)
    scores_coverage.append(cov)

# ===========================================================
# 4️⃣ Plot Results
# ===========================================================
plt.rcParams.update({"font.size": 12})

# 1️⃣ Total Novelty
plt.figure(figsize=(12, 5))
plt.bar(model_names, scores_total, color="steelblue")
plt.title("Total Novelty Loss", fontsize=14, fontweight="bold")
plt.ylabel("Loss")
plt.xticks(rotation=30, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(IMGS_DIR / "novelty_comparison_total_cgc.png", dpi=300)
plt.show()

# 2️⃣ Quality Component
plt.figure(figsize=(12, 5))
plt.bar(model_names, scores_quality, color="seagreen")
plt.title("Quality Component", fontsize=14, fontweight="bold")
plt.ylabel("Loss")
plt.xticks(rotation=30, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(IMGS_DIR / "novelty_comparison_quality_cgc.png", dpi=300)
plt.show()

# 3️⃣ Memorization Component
plt.figure(figsize=(12, 5))
plt.bar(model_names, scores_mem, color="crimson")
plt.title("Memorization Component", fontsize=14, fontweight="bold")
plt.ylabel("Loss")
plt.xticks(rotation=30, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(IMGS_DIR / "novelty_comparison_memorization_cgc.png", dpi=300)
plt.show()

# 4️⃣ Novelty vs Coverage (added metric visualization)
plt.figure(figsize=(12, 5))
x = np.arange(len(model_names))
plt.bar(x - 0.2, scores_novelty, width=0.4, label="Novelty", color="royalblue")
plt.bar(x + 0.2, scores_coverage, width=0.4, label="Coverage", color="orange")
plt.xticks(x, model_names, rotation=30, ha="right")
plt.ylabel("Score")
plt.title("GNN Feature Space Novelty & Coverage", fontsize=14, fontweight="bold")
plt.grid(True, axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(IMGS_DIR / "novelty_coverage_comparison_cgc.png", dpi=300)
plt.show()
