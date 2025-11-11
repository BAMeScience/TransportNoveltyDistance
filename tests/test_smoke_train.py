from __future__ import annotations

"""
Pytest smoke test for the contrastive training loop.

Uses a couple of tiny NaCl-like structures so the run finishes quickly while
still exercising the end-to-end pipeline (CSV I/O, dataloaders, model, loss,
checkpointing, and plotting).
"""

from pathlib import Path

import pandas as pd
from pymatgen.core import Lattice, Structure

from matscinovelty.gcn import train_contrastive_model


def _build_structures() -> list[Structure]:
    base = Structure.from_spacegroup(
        "Fm-3m",
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    distort = base.copy()
    distort.translate_sites(1, [0.05, 0.05, 0.05], frac_coords=True, to_unit_cell=True)

    structs = [base.copy(), distort.copy()]
    structs.append(distort.copy())
    structs[-1].perturb(distance=0.02)
    structs.append(base.copy())
    structs[-1].perturb(distance=0.01)
    return structs


def _write_csv(structures: list[Structure], path: Path) -> None:
    df = pd.DataFrame({"cif": [s.to(fmt="cif") for s in structures]})
    df.to_csv(path)


def test_smoke_contrastive_training(tmp_path: Path) -> None:
    """Ensures a short training run completes and writes artifacts."""
    structures = _build_structures()
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    checkpoint = tmp_path / "smoke.pt"
    curve = tmp_path / "curve.png"

    _write_csv(structures, train_csv)
    _write_csv(structures, val_csv)

    train_contrastive_model(
        str(train_csv),
        val_csv=str(val_csv),
        epochs=1,
        batch_size=2,
        lr=5e-4,
        tau=0.2,
        checkpoint_path=str(checkpoint),
        plot_path=str(curve),
        hidden_dim=64,
        num_rbf=16,
        n_layers=2,
        device="cpu",
    )

    assert checkpoint.exists(), "Smoke checkpoint missing"
    assert curve.exists(), "Validation curve plot missing"
