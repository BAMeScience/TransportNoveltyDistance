from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from torch_geometric.nn import CGConv, global_mean_pool
from torch_geometric.nn.models import SchNet as PyGSchNet
from torch_geometric.typing import WITH_TORCH_CLUSTER

warnings.filterwarnings(
    "ignore",
    message="`torch_geometric.distributed` has been deprecated",
    category=DeprecationWarning,
)

from .utils import (  # noqa: E402
    augment,
    augment_supercell,
    read_structure_from_csv,
    structure_to_graph,
)

if TYPE_CHECKING:  # pragma: no cover
    from accelerate import Accelerator


class BaseCrystalEncoder(nn.Module):
    """Shared utilities for crystal encoders (graph featurization + lattice stats)."""

    def __init__(
        self,
        *,
        cutoff: float = 8.0,
        num_rbf: int = 32,
        lattice_scale_abc: float = 10.0,
        lattice_scale_angles: float = 180.0,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.lattice_scale_abc = lattice_scale_abc
        self.lattice_scale_angles = lattice_scale_angles

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _graph_kwargs(self) -> dict:
        return {
            "cutoff": self.cutoff,
            "num_rbf": self.num_rbf,
        }

    def featurize(self, structures: Sequence) -> torch.Tensor:
        """
        Canonical featurizer shared by all encoders:
        - GNN processes the raw structures using :func:`structure_to_graph`.
        - Lattice descriptors use primitive/Niggli-reduced cells.
        - Outputs concatenated, L2-normalized embeddings.
        """

        self.eval()
        with torch.no_grad():
            graphs = [structure_to_graph(s, **self._graph_kwargs()) for s in structures]
            batch = Batch.from_data_list(graphs).to(self.device)
            z = F.normalize(self(batch), dim=1)

        return z.cpu()

class EGNNLayer(nn.Module):
    """A lightweight EGNN block operating on invariant node features and positions."""

    def __init__(
        self, in_features: int, hidden_features: int, edge_features: int = 0
    ) -> None:
        super().__init__()
        input_dim = in_features * 2 + edge_features
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, in_features),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr=None,
    ):
        row, col = edge_index

        edge_input = torch.cat([x[row], x[col], edge_attr], dim=-1)

        m_ij = self.edge_mlp(edge_input)

        agg = torch.zeros_like(x)
        agg.index_add_(0, row, m_ij)
        x = x + self.node_mlp(torch.cat([x, agg], dim=-1))

        return x


class EquivariantCrystalGCN(BaseCrystalEncoder):
    """EGNN-style encoder that augments graph embeddings with reduced lattice features."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_rbf: int = 128,
        n_layers: int = 3,
        cutoff: float = 5.0,
    ) -> None:
        super().__init__(cutoff=cutoff, num_rbf=num_rbf)
        self.emb = nn.Embedding(100, hidden_dim)
        self.layers = nn.ModuleList(
            [
                EGNNLayer(hidden_dim, hidden_dim, edge_features=num_rbf)
                for _ in range(n_layers)
            ]
        )
        self.lin = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data):
        x = self.emb(data.x).float()
        edge_index, edge_attr = data.edge_index, data.edge_attr
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        x = global_mean_pool(x, data.batch)
        return self.lin(F.relu(x))



class CGCNNEncoder(BaseCrystalEncoder):
    """CGCNN-style encoder using CGConv layers for fast embedding extraction."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_rbf: int = 32,
        num_layers: int = 3,
        cutoff: float = 8.0,
    ) -> None:
        super().__init__(cutoff=cutoff, num_rbf=num_rbf)
        if num_layers < 1:
            raise ValueError("CGCNNEncoder requires at least one convolutional layer.")
        self.emb = nn.Embedding(100, hidden_dim)
        self.convs = nn.ModuleList(
            CGConv(hidden_dim, dim=num_rbf) for _ in range(num_layers)
        )
        self.lin = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.emb(x)

        for conv in self.convs:
            x = F.silu(conv(x, edge_index, edge_attr))

        x = global_mean_pool(x, data.batch)
        return self.lin(x)


class SchNetEncoder(BaseCrystalEncoder):
    """Wrapper around PyG's SchNet for graph-level embeddings."""

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 3,
        num_gaussians: int = 50,
        cutoff: float = 8.0,
        max_neighbors: int = 32,
        readout: str = "mean",
    ) -> None:
        super().__init__(cutoff=cutoff, num_rbf=num_gaussians)
        if not WITH_TORCH_CLUSTER:
            raise ImportError(
                "SchNetEncoder requires the optional 'torch-cluster' dependency. "
                "Install it with pip following the PyTorch Geometric instructions "
                "(https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)."
            )
        self.cutoff = cutoff
        self.embedding_dim = embedding_dim
        self._edge_index = None
        self._edge_weight = None
        self.model = PyGSchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            max_num_neighbors=max_neighbors,
            readout=readout,
            interaction_graph=self._interaction_graph,
        )
        if embedding_dim != hidden_channels:
            self.out_proj = nn.Linear(hidden_channels, embedding_dim)
        else:
            self.out_proj = nn.Identity()

    def _interaction_graph(self, pos, batch):
        if self._edge_index is None or self._edge_weight is None:
            raise RuntimeError(
                "SchNetEncoder interaction_graph was called without cached edges. "
                "Ensure edge_index/edge_weight are provided."
            )
        return self._edge_index, self._edge_weight

    def forward(self, data):
        z = getattr(data, "z", data.x)

        edge_index = getattr(data, "edge_index", None)
        edge_shift = getattr(data, "edge_shift", None)

        if edge_index is None or edge_shift is None or edge_index.numel() == 0:
            raise ValueError(
                "SchNetEncoder expects edge_index and edge_shift on each PyG Data object."
            )

        device = data.pos.device
        self._edge_index = edge_index.to(device)
        row, col = self._edge_index
        rel = data.pos[row] - (data.pos[col] + edge_shift.to(device))
        self._edge_weight = rel.norm(dim=-1)
        out = self.model(z=z, pos=data.pos, batch=data.batch)
        return self.out_proj(out)

    @torch.no_grad()
    def featurize(self, structures: Sequence, device: str | torch.device | None = None):
        device = (
            torch.device(device)
            if device is not None
            else next(self.parameters()).device
        )
        self.eval()
        graphs = [structure_to_graph(s, cutoff=self.cutoff) for s in structures]
        batch = Batch.from_data_list(graphs).to(device)
        embeddings = self(batch)
        return embeddings.detach().cpu()


class GraphContrastiveDataset(Dataset):
    """
    Dataset that yields two augmented PyG graphs per structure, enabling the
    DataLoader workers to perform heavy featurization work in parallel.
    """

    def __init__(self, structures: Sequence, num_rbf: int = 32) -> None:
        self.structures = list(structures)
        self.num_rbf = num_rbf

    def __len__(self) -> int:
        return len(self.structures)

    def __getitem__(self, idx: int):
        structure = self.structures[idx]
        g1 = structure_to_graph(augment_supercell(structure), num_rbf=self.num_rbf)
        g2 = structure_to_graph(augment_supercell(structure), num_rbf=self.num_rbf)
        return g1, g2


def graph_pair_collate(batch):
    graphs1, graphs2 = zip(*batch)
    batch1 = Batch.from_data_list(graphs1)
    batch2 = Batch.from_data_list(graphs2)
    return batch1, batch2


def make_contrastive_dataloader(
    structures: Sequence,
    *,
    batch_size: int,
    num_rbf: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int | None,
    persistent_workers: bool | None,
):
    dataset = GraphContrastiveDataset(structures, num_rbf=num_rbf)
    if num_workers == 0:
        effective_prefetch = None
        effective_persistent = (
            False if persistent_workers is None else persistent_workers
        )
    else:
        effective_prefetch = prefetch_factor
        effective_persistent = (
            persistent_workers if persistent_workers is not None else True
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=graph_pair_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=effective_prefetch,
        persistent_workers=effective_persistent,
    )


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device=None,
    pin_memory: bool = False,
):
    """Evaluate intra-structure consistency and inter-structure separation."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model.eval()
    intra_sims, inter_sims = [], []

    with torch.no_grad():
        for batch1, batch2 in dataloader:
            batch1 = batch1.to(device, non_blocking=pin_memory)
            batch2 = batch2.to(device, non_blocking=pin_memory)

            z1 = F.normalize(model(batch1), dim=1)
            z2 = F.normalize(model(batch2), dim=1)

            intra = (z1 * z2).sum(dim=1)
            intra_sims.extend(intra.cpu().numpy())

            sim_matrix = z1 @ z2.T
            mask = ~torch.eye(len(sim_matrix), dtype=torch.bool, device=device)
            inter = sim_matrix[mask]
            inter_sims.extend(inter.cpu().numpy())

    model.train()
    return float(np.mean(intra_sims)), float(np.mean(inter_sims))


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Standard InfoNCE loss between two augmented batches."""
    B, _ = z1.shape
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    reps = torch.cat([z1, z2], dim=0)
    sim = reps @ reps.T / tau

    mask = torch.eye(2 * B, dtype=torch.bool, device=z1.device)
    sim = sim.masked_fill(mask, -9e15)

    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z1.device)
    return F.cross_entropy(sim, targets)


def train_contrastive_model(
    train_csv: str,
    val_csv: str | None = None,
    *,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    tau: float = 0.1,
    hidden_dim: int = 128,
    num_rbf: int = 32,
    n_layers: int = 3,
    device: str | torch.device | None = None,
    checkpoint_path: str | None = None,
    plot_path: str | None = None,
    accelerator: "Accelerator | None" = None,
    model_builder: Callable[[], nn.Module] | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
    persistent_workers: bool | None = None,
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Contrastively train the EquivariantCrystalGCN on structures stored as CSVs.
    Returns the trained model plus tracked intra/inter validation curves.
    """
    if accelerator is not None:
        torch_device = accelerator.device
    else:
        torch_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    train_structs = read_structure_from_csv(train_csv)
    val_structs = read_structure_from_csv(val_csv) if val_csv else train_structs

    if model_builder is None:
        model = EquivariantCrystalGCN(
            hidden_dim=hidden_dim, num_rbf=num_rbf, n_layers=n_layers
        ).to(torch_device)
    else:
        model = model_builder().to(torch_device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = make_contrastive_dataloader(
        train_structs,
        batch_size=batch_size,
        num_rbf=num_rbf,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    val_loader = make_contrastive_dataloader(
        val_structs,
        batch_size=batch_size,
        num_rbf=num_rbf,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
    if accelerator is not None:
        model, opt = accelerator.prepare(model, opt)

    val_intra, val_inter = [], []

    for epoch in range(epochs):
        ema_loss = 0.0
        for step, (batch1, batch2) in enumerate(train_loader):
            batch1 = batch1.to(torch_device, non_blocking=pin_memory)
            batch2 = batch2.to(torch_device, non_blocking=pin_memory)

            z1 = model(batch1)
            z2 = model(batch2)

            loss = info_nce_loss(z1, z2, tau=tau)
            opt.zero_grad()
            if accelerator is not None:
                accelerator.backward(loss)
            else:
                loss.backward()
            opt.step()

            ema_loss = (loss.item() + ema_loss * step) / (step + 1)

        intra, inter = validate(
            model,
            val_loader,
            device=torch_device,
            pin_memory=pin_memory,
        )
        val_intra.append(intra)
        val_inter.append(inter)

        if accelerator is None or accelerator.is_main_process:
            print(
                f"Epoch {epoch + 1}: loss={ema_loss:.4f} | intra={intra:.3f} | inter={inter:.3f}"
            )

        if checkpoint_path and (accelerator is None or accelerator.is_main_process):
            checkpoint_file = Path(checkpoint_path)
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            target_model = accelerator.unwrap_model(model) if accelerator else model
            torch.save(target_model.state_dict(), checkpoint_file)

    if plot_path and (accelerator is None or accelerator.is_main_process):
        plot_file = Path(plot_path)
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.plot(val_intra, label="Same structure (intra)")
        plt.plot(val_inter, label="Different structures (inter)")
        gap = (np.array(val_intra) - np.array(val_inter)).tolist()
        plt.plot(gap, label="Contrastive gap", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Average cosine similarity")
        plt.title("Representation consistency during training")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300)
        plt.close()

    final_model = accelerator.unwrap_model(model) if accelerator else model
    return final_model, val_intra, val_inter
