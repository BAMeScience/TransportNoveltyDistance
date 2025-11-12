#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from matscinovelty import CGCNNEncoder, SchNetEncoder
from matscinovelty.gcn import train_contrastive_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "mp_20"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the EquivariantCrystalGCN on the MP-20 CSVs."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_DATA_DIR / "train.csv",
        help=f"Path to the training CSV (default: {DEFAULT_DATA_DIR / 'train.csv'}).",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=DEFAULT_DATA_DIR / "val.csv",
        help=f"Path to the validation CSV (default: {DEFAULT_DATA_DIR / 'val.csv'}).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "gcn_mp20.pt",
        help="Where to save the trained model checkpoint.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Optional path to store the validation curve plot.",
    )
    parser.add_argument(
        "--model",
        choices=("equivariant", "cgc", "schnet"),
        default="equivariant",
        help="Backbone architecture to train.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--tau", type=float, default=0.1, help="InfoNCE temperature.")
    parser.add_argument(
        "--hidden-dim", type=int, default=128, help="Embedding dimension."
    )
    parser.add_argument(
        "--num-rbf", type=int, default=32, help="Number of RBF features per edge."
    )
    parser.add_argument(
        "--n-layers", type=int, default=3, help="Number of EGNN layers to stack."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device string passed to train_contrastive_model (default: auto).",
    )
    parser.add_argument(
        "--accelerate",
        action="store_true",
        help="Use Hugging Face Accelerate to manage devices/distributed training.",
    )
    return parser.parse_args()


def verify_csv(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(
            f"Missing {label} CSV at {path}. Run scripts/download_mp20.py first "
            "or point --train-csv / --val-csv to existing files."
        )


def main() -> None:
    args = parse_args()
    verify_csv(args.train_csv, "train")
    verify_csv(args.val_csv, "val")

    accelerator = None
    if args.accelerate:
        try:
            from accelerate import Accelerator
        except ImportError as exc:  # pragma: no cover - optional dep
            raise SystemExit(
                "Accelerate is not installed. Install it via `pip install accelerate` "
                "or rerun without --accelerate."
            ) from exc
        accelerator = Accelerator()

    def _is_main():
        return accelerator is None or accelerator.is_main_process

    if _is_main():
        print("🚀 Launching training on MP-20 splits.")

    def make_model_builder():
        if args.model == "equivariant":
            return None
        if args.model == "cgc":
            return lambda: CGCNNEncoder(
                hidden_dim=args.hidden_dim,
                num_rbf=args.num_rbf,
                num_layers=args.n_layers,
            )
        if args.model == "schnet":
            return lambda: SchNetEncoder(
                embedding_dim=args.hidden_dim,
                hidden_channels=args.hidden_dim,
                num_gaussians=args.num_rbf,
                num_interactions=args.n_layers,
            )
        raise ValueError(f"Unknown model type: {args.model}")

    model_builder = make_model_builder()

    train_contrastive_model(
        str(args.train_csv),
        val_csv=str(args.val_csv),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        tau=args.tau,
        hidden_dim=args.hidden_dim,
        num_rbf=args.num_rbf,
        n_layers=args.n_layers,
        device=args.device,
        checkpoint_path=str(args.checkpoint_path) if args.checkpoint_path else None,
        plot_path=str(args.plot_path) if args.plot_path else None,
        accelerator=accelerator,
        model_builder=model_builder,
    )
    if _is_main():
        print(f"✅ Training finished. Checkpoint saved to {args.checkpoint_path}.")


if __name__ == "__main__":
    main()
