#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "huggingface_hub is required for this script. "
        "Install it via `pip install huggingface_hub`."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = PROJECT_ROOT / "data" / "xtalmet_models"
REPO_ID = "masahiro-negishi/xtalmet"
REPO_TYPE = "dataset"
ALLOW_PATTERNS = ("**/*.pkl", "**/*.pkl.gz")


def download_pickles(destination: Path, overwrite: bool) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    print(f"⬇ Syncing pickle files from {REPO_ID}")
    snapshot_dir = Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            allow_patterns=list(ALLOW_PATTERNS),
            local_dir_use_symlinks=False,
        )
    )

    found = False
    for src in sorted(snapshot_dir.rglob("*.pkl")):
        dest_file = destination / src.name
        found = True
        if dest_file.exists() and not overwrite:
            print(f"✔ {dest_file} already exists; skipping.")
            continue
        shutil.copy(src, dest_file)
        print(f"✔ Copied {src.name} to {dest_file}")

    for src in sorted(snapshot_dir.rglob("*.pkl.gz")):
        dest_file = destination / src.stem  # remove .gz
        found = True
        if dest_file.exists() and not overwrite:
            print(f"✔ {dest_file} already exists; skipping.")
            continue
        with gzip.open(src, "rb") as fin, open(dest_file, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        print(f"✔ Decompressed {src.name} to {dest_file}")

    if not found:
        raise SystemExit(
            "No pickle artifacts were found in the snapshot. "
            "Check the repository structure or update ALLOW_PATTERNS."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download xtalmet MP-20 model outputs (MatterGen, DiffCSP, etc.)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Directory to store the pickle files (default: {DEFAULT_DEST}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_pickles(args.output_dir, args.force)
    print(f"✅ xtalmet pickles ready under {args.output_dir}")
    print("   You can now run experiments/model_eval.py.")


if __name__ == "__main__":
    main()
