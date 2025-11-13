#!/usr/bin/env python
"""Utilities for mirroring the WBM dataset locally.

This script relies on ``matbench-discovery`` to fetch the official CSV/ZIP assets,
then reshapes them into the format consumed by ``experiments/wbm_exp.py``. By
default it downloads three artifacts:

1. ``wbm-summary.csv`` – the high-level metadata table.
2. ``wbm-initial-atoms.extxyz.zip`` – the initial structures as ASE ``extxyz`` files.
3. ``wbm-structures-step-1.json`` … ``wbm-structures-step-5.json`` – relaxed
   structures grouped by substitution step. These are generated on the fly by
   converting every relaxed ``extxyz`` entry into a ``pymatgen.Structure``.

Generating the last set involves ~250k conversions and can take several minutes;
pass ``--skip-step-json`` or ``--step-limit`` for quicker debugging runs.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import ase.io
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / ".matbench_cache"
os.environ.setdefault("MATBENCH_DISCOVERY_CACHE_DIR", str(CACHE_DIR))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

from matbench_discovery.data import DataFiles

DEFAULT_DEST = PROJECT_ROOT / "data" / "wbm"
MP20_DIR = PROJECT_ROOT / "data" / "mp_20"


def _copy_summary(dest_dir: Path) -> pd.DataFrame:
    """Download/cache the WBM summary table and store it as CSV."""

    summary_path = DataFiles.wbm_summary.path
    df = pd.read_csv(summary_path)
    out_path = dest_dir / "wbm-summary.csv"
    df.to_csv(out_path, index=False)
    print(f"✔ Wrote summary to {out_path} (shape={df.shape})")
    return df


def _copy_atoms(dest_dir: Path) -> None:
    """Copy the initial ASE atoms archive as-is for reproducibility."""

    atoms_path = DataFiles.wbm_initial_atoms.path
    out_path = dest_dir / "wbm-initial-atoms.extxyz.zip"
    with open(atoms_path, "rb") as src, open(out_path, "wb") as dst:
        dst.write(src.read())
    print(f"✔ Copied ASE atoms archive to {out_path}")


def _load_mp20_ids(mp20_dir: Path) -> set[str]:
    ids: set[str] = set()
    for split in ("train.csv", "val.csv", "test.csv"):
        path = mp20_dir / split
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "material_id" not in df.columns:
            continue
        ids.update(df["material_id"].dropna().astype(str))
    return ids


def _copy_step_structures(
    dest_dir: Path, summary_df: pd.DataFrame, *, limit: int | None = None
) -> None:
    """Emit one JSON file per WBM substitution step with relaxed structures.

    Parameters
    ----------
    dest_dir
        Output directory (typically ``data/wbm``).
    summary_df
        DataFrame returned by :func:`_copy_summary`; used to determine step membership.
    limit
        Optional maximum number of relaxed structures to export per step. Helpful
        for smoke tests; ``None`` means “use the full dataset”.
    """

    adaptor = AseAtomsAdaptor()
    steps = pd.to_numeric(
        summary_df["material_id"].str.split("-").str[1], errors="coerce"
    )
    step_lookup = dict(zip(summary_df["material_id"], steps))

    writers: dict[int, tuple[Path, any]] = {}
    first_entry: dict[int, bool] = {}
    counts = defaultdict(int)

    for step in range(1, 6):
        dest_file = dest_dir / f"wbm-structures-step-{step}.json"
        fh = open(dest_file, "w")
        fh.write("{")  # stream JSON object manually to avoid holding everything in RAM
        writers[step] = (dest_file, fh)
        first_entry[step] = True

    relaxed_zip = DataFiles.wbm_relaxed_atoms.path
    with ZipFile(relaxed_zip) as zf:
        names = [name for name in zf.namelist() if name.endswith(".extxyz")]
        for name in tqdm(names, desc="Converting relaxed structures"):
            mat_id = Path(name).stem
            step = step_lookup.get(mat_id)
            if step is None or step not in writers:
                continue
            if limit is not None and counts[step] >= limit:
                continue

            with zf.open(name) as fh:
                atoms = ase.io.read(
                    io.TextIOWrapper(fh, encoding="utf-8"), format="extxyz"
                )
            structure = adaptor.get_structure(atoms)

            _, fh_out = writers[step]
            if not first_entry[step]:
                fh_out.write(",")
            first_entry[step] = False
            fh_out.write(f'\n  "{mat_id}": ')
            json.dump({"opt": structure.as_dict()}, fh_out)
            counts[step] += 1

    for step, (dest_file, fh) in writers.items():
        if not first_entry[step]:
            fh.write("\n")
        fh.write("}\n")
        fh.close()
        msg = (
            f"✔ Wrote {counts[step]} relaxed structures for step {step} -> {dest_file}"
        )
        if limit is not None:
            msg += f" (limit={limit})"
        print(msg)

    mp_ids = _load_mp20_ids(MP20_DIR)
    _copy_mp20_baseline(dest_dir, mp_ids, limit=limit)


def _copy_mp20_baseline(
    dest_dir: Path, mp_ids: set[str], limit: int | None = None
) -> None:
    """Emit step-0 JSON by extracting MP-20 structures from the MP entries file."""

    if not mp_ids:
        print("⚠ No MP-20 CSVs found; skipping step-0 export.")
        return

    mp_path = Path(DataFiles.mp_computed_structure_entries.path)
    dest_file = dest_dir / "wbm-structures-step-0.json"
    count = 0

    dest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_file, "w") as out:
        out.write("{")
        first = True
        try:
            with gzip.open(mp_path, "rt") as fh:
                payload = json.load(fh)
        except Exception as exc:
            print(f"⚠ Failed to load MP computed structure entries: {exc}")
            out.write("\n}\n")
            return

        material_map: dict[str, str] = payload.get("material_id", {})
        entries: dict[str, dict] = payload.get("entry", {})

        for key, entry_id in material_map.items():
            entry_id = str(entry_id)
            if entry_id not in mp_ids:
                continue
            entry = entries.get(key) or {}
            structure = entry.get("structure")
            if structure is None:
                continue
            if limit is not None and count >= limit:
                break
            if not first:
                out.write(",")
            first = False
            out.write(f'\n  "{entry_id}": ')
            json.dump(structure, out)
            count += 1
        if not first:
            out.write("\n")
        out.write("}\n")

    print(f"✔ Wrote {count} baseline MP-20 structures -> {dest_file}")


def parse_args() -> argparse.Namespace:
    """Configure CLI options for the download helper."""

    parser = argparse.ArgumentParser(
        description="Download/cache WBM dataset artifacts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DEST,
        help="Directory where WBM files will be stored (default: data/wbm).",
    )
    parser.add_argument(
        "--skip-step-json",
        action="store_true",
        help="Skip generating the per-step relaxed structure JSON dumps (saves time).",
    )
    parser.add_argument(
        "--step-limit",
        type=int,
        default=None,
        help="If set, only export up to this many relaxed structures per step.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the CLI script."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = _copy_summary(args.output_dir)
    _copy_atoms(args.output_dir)
    if args.skip_step_json:
        print("↪ Skipping relaxed structure extraction as requested.")
    else:
        _copy_step_structures(args.output_dir, summary_df, limit=args.step_limit)

    print("✅ WBM data ready.")


if __name__ == "__main__":
    main()
