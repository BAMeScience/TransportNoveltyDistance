#!/usr/bin/env python
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Structure
from tqdm import tqdm

warnings.filterwarnings(
    "ignore",
    message="Issues encountered while parsing CIF",
    category=UserWarning,
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MP20 = PROJECT_ROOT / "data" / "mp_20"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split MP-20 CSV files into oxides and non-oxides using pymatgen's "
            "bond-valence analysis."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[
            DEFAULT_MP20 / "train.csv",
            DEFAULT_MP20 / "val.csv",
            DEFAULT_MP20 / "test.csv",
        ],
        help=(
            "One or more CSV files to split (default: MP-20 train/val in data/mp_20). "
            "Each CSV must contain a 'cif' column."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MP20,
        help="Directory where the splitted CSVs will be written.",
    )
    parser.add_argument(
        "--column",
        default="cif",
        help="Name of the column containing CIF strings (default: 'cif').",
    )
    return parser.parse_args()


def classify_structure(cif: str, analyzer: BVAnalyzer, tol: float = 0.25) -> bool:
    structure = Structure.from_str(cif, fmt="cif")
    contains_oxygen = any(site.specie.symbol == "O" for site in structure)
    if not contains_oxygen:
        return False

    try:
        valences = analyzer.get_valences(structure)
    except Exception:
        # Fall back to simple element check if bond-valence analysis fails.
        return True

    return any(
        site.specie.symbol == "O" and abs(val + 2.0) <= tol
        for site, val in zip(structure, valences)
    )


def split_file(csv_path: Path, column: str, output_dir: Path, analyzer: BVAnalyzer):
    if not csv_path.exists():
        raise SystemExit(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise SystemExit(f"Column '{column}' missing from {csv_path}")

    is_oxide = []
    for cif in tqdm(df[column], desc=f"Classifying {csv_path.name}"):
        try:
            flag = classify_structure(cif, analyzer)
        except Exception as exc:
            print(f"⚠️  Failed to classify entry (defaulting to non-oxide): {exc}")
            flag = False
        is_oxide.append(flag)

    df_oxides = df[is_oxide].reset_index(drop=True)
    df_non = df[~pd.Series(is_oxide)].reset_index(drop=True)

    base = csv_path.stem
    oxide_path = output_dir / f"{base}_oxides.csv"
    non_path = output_dir / f"{base}_non_oxides.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    df_oxides.to_csv(oxide_path, index=False)
    df_non.to_csv(non_path, index=False)

    print(
        f"{csv_path.name}: {len(df_oxides)} oxides / {len(df_non)} non-oxides "
        f"-> {oxide_path.name}, {non_path.name}"
    )


def main() -> None:
    args = parse_args()
    analyzer = BVAnalyzer()

    for path in args.inputs:
        split_file(path, args.column, args.output_dir, analyzer)


if __name__ == "__main__":
    main()
