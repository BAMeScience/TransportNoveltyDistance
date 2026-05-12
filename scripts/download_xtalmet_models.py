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
MODEL_PICKLES = (
    "mattergen.pkl",
    "diffcsp.pkl",
    "diffcsppp.pkl",
    "cdvae.pkl",
    "adit.pkl",
    "chemeleon.pkl",
)
REMOTE_MODEL_DIR = Path("mp20") / "model"
REMOTE_PICKLE_PATHS = tuple(REMOTE_MODEL_DIR / name for name in MODEL_PICKLES)
REMOTE_GZ_PICKLE_PATHS = tuple(
    REMOTE_MODEL_DIR / f"{name}.gz" for name in MODEL_PICKLES
)
ALLOW_PATTERNS = tuple(
    str(path) for path in (*REMOTE_PICKLE_PATHS, *REMOTE_GZ_PICKLE_PATHS)
)


def download_pickles(destination: Path, overwrite: bool) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    print(f"⬇ Syncing selected MP-20 model pickle files from {REPO_ID}")
    snapshot_dir = Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            allow_patterns=list(ALLOW_PATTERNS),
            local_dir_use_symlinks=False,
        )
    )

    copied = 0
    for pickle_path, gz_pickle_path in zip(REMOTE_PICKLE_PATHS, REMOTE_GZ_PICKLE_PATHS):
        src = snapshot_dir / gz_pickle_path
        dest_file = destination / pickle_path.name
        if dest_file.exists() and not overwrite:
            print(f"✔ {dest_file} already exists; skipping.")
            continue

        if src.exists():
            with gzip.open(src, "rb") as fin, open(dest_file, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            print(f"✔ Decompressed {src.name} to {dest_file}")
            copied += 1
            continue

        src = snapshot_dir / pickle_path
        if src.exists():
            shutil.copy(src, dest_file)
            print(f"✔ Copied {src.name} to {dest_file}")
            copied += 1
            continue

        raise SystemExit(
            f"Could not find {gz_pickle_path} or {pickle_path} in the snapshot."
        )

    if copied == 0:
        print("✔ All selected pickle files already exist; nothing to download.")

    present = {path.name for path in destination.glob("*.pkl")}
    expected = set(MODEL_PICKLES)
    extras = sorted(present - expected)
    if extras:
        print(
            "ℹ Extra .pkl files remain in the output directory; "
            "remove them manually if you want a clean folder: "
            + ", ".join(extras)
        )

    missing = sorted(expected - present)
    if missing:
        raise SystemExit(
            "The following selected pickle files are still missing: "
            + ", ".join(missing)
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
