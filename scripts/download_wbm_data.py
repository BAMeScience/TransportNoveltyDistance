#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from matbench_discovery.data import DataFiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = PROJECT_ROOT / "data" / "wbm"


def _copy_summary(dest_dir: Path) -> pd.DataFrame:
    summary_path = DataFiles.wbm_summary.path
    df = pd.read_csv(summary_path)
    out_path = dest_dir / "wbm-summary.csv"
    df.to_csv(out_path, index=False)
    print(f"✔ Wrote summary to {out_path} (shape={df.shape})")
    return df


def _copy_structures(dest_dir: Path) -> pd.DataFrame:
    struct_path = DataFiles.wbm_initial_structures.path
    df = pd.read_json(struct_path, lines=True)
    out_path = dest_dir / "wbm-initial-structures.jsonl"
    df.to_json(out_path, orient="records", lines=True)
    print(f"✔ Wrote initial structures JSONL to {out_path}")
    return df


def _copy_atoms(dest_dir: Path) -> None:
    atoms_path = DataFiles.wbm_initial_atoms.path
    out_path = dest_dir / "wbm-initial-atoms.json.gz"
    with open(atoms_path, "rb") as src, open(out_path, "wb") as dst:
        dst.write(src.read())
    print(f"✔ Copied ASE atoms archive to {out_path}")


def _copy_step_structures(dest_dir: Path, summary_df: pd.DataFrame) -> None:
    step_col = "wyckoff_spglib"
    struct_col = "wyckoff_spglib_initial_structure"
    if step_col not in summary_df.columns or struct_col not in summary_df.columns:
        print(
            "⚠ Summary file missing expected columns for step filtering; skipping per-step JSON."
        )
        return

    for step in range(1, 6):
        mask = summary_df[step_col].str.endswith(f"_{step}", na=False)
        subset = summary_df.loc[mask, ["material_id", struct_col]]
        if subset.empty:
            continue

        dest_file = dest_dir / f"wbm-structures-step-{step}.json"
        payload = {
            str(row.material_id): json.loads(row[struct_col])
            for row in subset.itertuples(index=False)
        }
        with open(dest_file, "w") as fh:
            json.dump(payload, fh)
        print(f"✔ Extracted {len(payload)} structures for step {step} -> {dest_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/cache WBM dataset artifacts.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DEST,
        help="Directory where WBM files will be stored (default: data/wbm).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = _copy_summary(args.output_dir)
    _copy_structures(args.output_dir)
    _copy_atoms(args.output_dir)
    _copy_step_structures(args.output_dir, summary_df)

    print("✅ WBM data ready.")


if __name__ == "__main__":
    main()
