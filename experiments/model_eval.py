import argparse
import pickle
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from matscinovelty import (
    EquivariantCrystalGCN,
    OTNoveltyScorer,
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


def _extract_state_dict(raw_checkpoint: dict | OrderedDict) -> OrderedDict:
    """Return the model state dict no matter how the checkpoint was stored."""
    if isinstance(raw_checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in raw_checkpoint and isinstance(raw_checkpoint[key], OrderedDict):
                return raw_checkpoint[key]
    if isinstance(raw_checkpoint, OrderedDict):
        return raw_checkpoint
    if isinstance(raw_checkpoint, dict):
        return OrderedDict(raw_checkpoint)
    raise TypeError("Unrecognized checkpoint format when extracting state dict.")


def _strip_prefix(state_dict: OrderedDict, prefix: str) -> OrderedDict:
    if not state_dict:
        return state_dict
    if all(key.startswith(prefix) for key in state_dict.keys()):
        return OrderedDict((k[len(prefix) :], v) for k, v in state_dict.items())
    return state_dict


def _prepare_state_dict(state_dict: OrderedDict) -> OrderedDict:
    for prefix in ("module.", "model."):
        state_dict = _strip_prefix(state_dict, prefix)
    return state_dict


def _infer_gcn_config(state_dict: OrderedDict) -> tuple[int, int, int]:
    """Infer hidden_dim, num_rbf, and n_layers from a checkpoint."""
    try:
        hidden_dim = state_dict["emb.weight"].shape[1]
    except KeyError as exc:
        raise KeyError(
            "Checkpoint missing 'emb.weight'; cannot infer hidden_dim."
        ) from exc

    edge_key = "layers.0.edge_mlp.0.weight"
    if edge_key not in state_dict:
        raise KeyError(f"Checkpoint missing '{edge_key}'; cannot infer num_rbf.")
    num_rbf = state_dict[edge_key].shape[1] - (2 * hidden_dim + 1)
    if num_rbf < 0:
        raise ValueError("Inferred num_rbf was negative; checkpoint/config mismatch.")

    layer_ids = set()
    for key in state_dict.keys():
        if key.startswith("layers."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_ids.add(int(parts[1]))
    n_layers = max(layer_ids) + 1 if layer_ids else 1
    return hidden_dim, num_rbf, n_layers


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
        "--checkpoint",
        type=Path,
        default=CHECKPOINTS_DIR / "gcn_fine.pt",
        help="Path to the encoder checkpoint (default: checkpoints/gcn_fine.pt).",
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

if not checkpoint_path.exists():
    raise SystemExit(f"Checkpoint not found at {checkpoint_path}.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading pretrained GCN model...")

raw_checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = _prepare_state_dict(_extract_state_dict(raw_checkpoint))
hidden_dim, num_rbf, n_layers = _infer_gcn_config(state_dict)
print(
    f"Checkpoint config detected -> hidden_dim={hidden_dim}, num_rbf={num_rbf}, n_layers={n_layers}"
)
model = EquivariantCrystalGCN(
    hidden_dim=hidden_dim, num_rbf=num_rbf, n_layers=n_layers
).to(device)
model.load_state_dict(state_dict)
print("Loaded weights from gcn_fine.pt ✅")


scorer = OTNoveltyScorer(
    train_structures=str_train,
    gnn_model=model,
    tau=0.36,  # fixed τ (your calibrated value)
    memorization_weight=50.0,
    device=device,
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
plt.rcParams.update({"font.size": 12})

# 1️⃣ Total Novelty
plt.figure(figsize=(12, 5))
plt.bar(model_names, scores_total, color="steelblue")
plt.title("Total Novelty Loss", fontsize=14, fontweight="bold")
plt.ylabel("Loss")
plt.xticks(rotation=30, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(IMGS_DIR / "novelty_comparison_total.png", dpi=300)
plt.show()

# 2️⃣ Quality Component
plt.figure(figsize=(12, 5))
plt.bar(model_names, scores_quality, color="seagreen")
plt.title("Quality Component", fontsize=14, fontweight="bold")
plt.ylabel("Loss")
plt.xticks(rotation=30, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(IMGS_DIR / "novelty_comparison_quality.png", dpi=300)
plt.show()

# 3️⃣ Memorization Component
plt.figure(figsize=(12, 5))
plt.bar(model_names, scores_mem, color="crimson")
plt.title("Memorization Component", fontsize=14, fontweight="bold")
plt.ylabel("Loss")
plt.xticks(rotation=30, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(IMGS_DIR / "novelty_comparison_memorization.png", dpi=300)
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
plt.savefig(IMGS_DIR / "novelty_coverage_comparison.png", dpi=300)
plt.show()
