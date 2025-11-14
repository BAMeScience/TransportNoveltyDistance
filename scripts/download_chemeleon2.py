#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

BASE_URL = 'https://raw.githubusercontent.com/hspark1212/chemeleon2/refs/heads/main/benchmarks/dng'
DATA_FILES = ('chemeleon2_rl_dng_mp_20.json.gz',)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "chemeleon2"

def download_file(filename: str, destination: Path, overwrite: bool) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        print(f"✔ {destination} already exists; skipping download.")
        return destination

    url = f"{BASE_URL}/{filename}"
    print(f"⬇ Downloading {filename} from {url}")
    try:
        with urlopen(url) as response:
            data = response.read()
    except HTTPError as exc:
        raise SystemExit(f"Failed to download {filename}: HTTP {exc.code}") from exc
    except URLError as exc:
        raise SystemExit(f"Failed to download {filename}: {exc.reason}") from exc

    destination.write_bytes(data)
    print(f"✔ Saved {filename} to {destination}")
    return destination


def ensure_files(
    files: Iterable[str], output_dir: Path, overwrite: bool
) -> dict[str, Path]:
    downloaded: dict[str, Path] = {}
    for name in files:
        downloaded[name] = download_file(name, output_dir / name, overwrite)
    return downloaded

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the 10'000 structures created with Chemeleon2 with Reinforced Learning"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory that will store the downloaded CSV files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing downloads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensure_files(DATA_FILES, args.output_dir, args.force)
    print(f"✅ Chemeleon2 structures ready under {args.output_dir}.")
    print("   Next step: after downloading xtalmet structures, run experiments/model_eval.py")


if __name__ == "__main__":
    main()
