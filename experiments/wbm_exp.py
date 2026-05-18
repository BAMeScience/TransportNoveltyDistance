import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from pymatgen.core import Structure

from TNovD.gcn import EquivariantCrystalGCN
from TNovD.TransportNoveltyDistance import TransportNoveltyDistance
from TNovD.utils import read_structure_from_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_MP20 = PROJECT_ROOT / "data" / "mp_20"
WBM_DIR = PROJECT_ROOT / "data" / "wbm"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
IMGS_DIR = PROJECT_ROOT / "imgs"
CHECKPOINT_PATH = CHECKPOINTS_DIR / "gcn_mp20_hidden32.pt"
MP20_SPLITS = ("train.csv", "val.csv", "test.csv")
STABILITY_COL = "e_above_hull_mp2020_corrected_ppd_mp"
STABILITY_THRESHOLD = 0.0
RANDOM_SEED = 0
IMGS_DIR.mkdir(exist_ok=True)


def plot_scores(step_indices, totals, qualities, mems, suffix):
    plt.rcParams.update({"font.size": 14})
    plt.figure(figsize=(9.6, 6))
    plt.bar(step_indices, qualities, label="Quality", color="skyblue")
    plt.bar(step_indices, mems, bottom=qualities, label="Memorization", color="red")
    plt.xlabel("WBM substitution step")
    plt.ylabel("TNovD")
    plt.title("Transport Novelty Distance vs. WBM")
    plt.grid(True, alpha=0.4)
    plt.ylim(top=max(totals) * 1.25)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.xticks(step_indices, [str(i) for i in step_indices])
    base_name = f"novelty_wbm_fine_components{suffix}"
    plt.savefig(IMGS_DIR / f"{base_name}_autotuned.png", dpi=300)
    plt.show()


# ===========================================================
# 1️⃣ Load Data
# ===========================================================
print("Loading MP-20 reference structures...")
mp20_reference = []
for split in MP20_SPLITS:
    split_path = DATA_MP20 / split
    if not split_path.exists():
        raise FileNotFoundError(f"Missing MP-20 split at {split_path}.")
    mp20_reference.extend(read_structure_from_csv(split_path))
print(f"Loaded {len(mp20_reference)} MP-20 reference structures from train/val/test.")

stable_ids_by_step = {step: set() for step in range(1, 6)}
unstable_ids = set()
with open(WBM_DIR / "wbm-summary.csv", newline="") as fh:
    for row in csv.DictReader(fh):
        material_id = row["material_id"]
        step = int(material_id.split("-")[1])
        e_hull = float(row[STABILITY_COL] or "inf")
        if e_hull <= STABILITY_THRESHOLD:
            stable_ids_by_step[step].add(material_id)
        else:
            unstable_ids.add(material_id)

print("Loading WBM structures...")
wbm_steps = {}
calibration_pool = []
for step in range(1, 6):
    step_path = WBM_DIR / f"wbm-structures-step-{step}.json"
    if not step_path.exists():
        raise FileNotFoundError(
            f"{step_path} not found. Run `python scripts/download_wbm_data.py`."
        )
    with open(step_path) as fh:
        data = json.load(fh)
    structs = []
    for material_id, entry in data.items():
        struct_dict = (
            entry["opt"] if isinstance(entry, dict) and "opt" in entry else entry
        )
        if material_id in stable_ids_by_step[step]:
            structs.append(Structure.from_dict(struct_dict))
        elif material_id in unstable_ids:
            calibration_pool.append(struct_dict)
    wbm_steps[step] = structs
    print(f"Loaded {len(structs)} stable structures for WBM step {step}")

print("\n=== Equal-count stable WBM sample: subsampled to minimum step size ===")
rng = random.Random(RANDOM_SEED)
equal_count = min(len(structs) for structs in wbm_steps.values())
if equal_count == 0:
    raise ValueError("Cannot build equal-count WBM sample: at least one step is empty.")
wbm_steps_equal = {}
for step, structs in wbm_steps.items():
    if len(structs) > equal_count:
        wbm_steps_equal[step] = rng.sample(structs, equal_count)
        print(f"Subsampled WBM step {step} to {equal_count} stable structures")
    else:
        wbm_steps_equal[step] = list(structs)
        print(f"Kept WBM step {step} at {len(structs)} stable structures")
calibration_dicts = rng.sample(calibration_pool, equal_count)
calibration_structs = [Structure.from_dict(struct_dict) for struct_dict in calibration_dicts]
print(f"Loaded {len(calibration_structs)} unstable WBM structures for calibration")

# ===========================================================
# 2️⃣ Initialize Scorer
# ===========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
if not CHECKPOINT_PATH.exists():
    raise SystemExit(f"Checkpoint not found at {CHECKPOINT_PATH}.")

print("Loading pretrained GCN model...")
model = EquivariantCrystalGCN(hidden_dim=32, num_rbf=128).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
print(f"Loaded weights from {CHECKPOINT_PATH}")

scorer = TransportNoveltyDistance(
    train_structures=mp20_reference,
    gnn_model=model,
    tau=None,  # auto-estimate τ
    memorization_weight=None,
    device=device,
    calibration_sample_size=equal_count,
    calibration_structures=calibration_structs,
)
print(f"Estimated τ = {scorer.tau:.4f}")

# ===========================================================
# 3️⃣ Evaluate WBM Steps
# ===========================================================
print("\n=== Equal-count stable WBM evaluation ===")
step_indices, totals, qualities, mems = [], [], [], []
for step, structs in wbm_steps_equal.items():
    print(f"\n▶ Evaluating WBM step {step} [stable/equal-count {equal_count}, n={len(structs)}]")
    total, qual, mem = scorer.compute_TNovD(structs)
    print(f"Step {step}: Total={total:.4f} | Quality={qual:.4f} | Memorization={mem:.4f}")
    step_indices.append(step)
    totals.append(total)
    qualities.append(qual)
    mems.append(mem)
plot_scores(
    step_indices,
    totals,
    qualities,
    mems,
    "_stable_equal_count",
)
