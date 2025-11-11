from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import CGConv, global_mean_pool

from .utils import (
    StructureDataset,
    augment,
    read_structure_from_csv,
    structure_to_graph,
)


class EGNNLayer(nn.Module):
    """A lightweight EGNN block operating on invariant node features and positions."""

    def __init__(
        self, in_features: int, hidden_features: int, edge_features: int = 0
    ) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_features * 2 + 1 + edge_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, in_features),
        )
        self.coord_mlp = nn.Sequential(nn.Linear(hidden_features, 1), nn.SiLU())

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr=None,
    ):
        row, col = edge_index
        rij = pos[row] - pos[col]
        dij = (rij**2).sum(dim=-1, keepdim=True)

        if edge_attr is not None:
            edge_input = torch.cat([x[row], x[col], dij, edge_attr], dim=-1)
        else:
            edge_input = torch.cat([x[row], x[col], dij], dim=-1)

        m_ij = self.edge_mlp(edge_input)

        agg = torch.zeros_like(x)
        agg.index_add_(0, row, m_ij)
        x = x + self.node_mlp(torch.cat([x, agg], dim=-1))

        w_ij = self.coord_mlp(m_ij)
        rij_norm = rij / (rij.norm(dim=-1, keepdim=True) + 1e-8)
        trans = rij_norm * w_ij
        delta = torch.zeros_like(pos)
        delta.index_add_(0, row, trans)
        pos = pos + delta

        return x, pos


class EquivariantCrystalGCN(nn.Module):
    """EGNN-style encoder that augments graph embeddings with reduced lattice features."""

    def __init__(
        self, hidden_dim: int = 128, num_rbf: int = 32, n_layers: int = 3
    ) -> None:
        super().__init__()
        self.num_rbf = num_rbf
        self.emb = nn.Embedding(100, hidden_dim)
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, hidden_dim, edge_features=num_rbf)
            for _ in range(n_layers)
        ])
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        self.lattice_scale_abc = 10.0
        self.lattice_scale_angles = 180.0

    def forward(self, data):
        x = self.emb(data.x).float()
        pos = data.pos
        edge_index, edge_attr = data.edge_index, data.edge_attr

        for layer in self.layers:
            x, pos = layer(x, pos, edge_index, edge_attr)

        x = global_mean_pool(x, data.batch)
        return self.lin(F.relu(x))

    def featurize(self, structures: Sequence) -> torch.Tensor:
        """Convert pymatgen.Structure objects into normalized embeddings."""
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            graphs = [structure_to_graph(s, num_rbf=self.num_rbf) for s in structures]
            batch = Batch.from_data_list(graphs).to(device)
            z = F.normalize(self(batch), dim=1)

            lat_feats = []
            for s in structures:
                s_red = s.get_reduced_structure(reduction_algo="niggli")

                a, b, c = s_red.lattice.a, s_red.lattice.b, s_red.lattice.c
                alpha, beta, gamma = (
                    s_red.lattice.alpha,
                    s_red.lattice.beta,
                    s_red.lattice.gamma,
                )

                lat = np.array([a, b, c, alpha, beta, gamma], dtype=np.float32)
                lat /= np.array(
                    [
                        self.lattice_scale_abc,
                        self.lattice_scale_abc,
                        self.lattice_scale_abc,
                        self.lattice_scale_angles,
                        self.lattice_scale_angles,
                        self.lattice_scale_angles,
                    ],
                    dtype=np.float32,
                )
                lat_feats.append(lat)

            lat_feats = torch.tensor(lat_feats, device=device)
            z = torch.cat([z, lat_feats], dim=1)

        return z.cpu()


class CrystalGCN(nn.Module):
    """Simpler CGConv-based encoder without explicit positional updates."""

    def __init__(self, hidden_dim: int = 128, num_rbf: int = 32) -> None:
        super().__init__()
        self.emb = nn.Embedding(100, hidden_dim)
        self.conv1 = CGConv(hidden_dim, dim=num_rbf)
        self.conv2 = CGConv(hidden_dim, dim=num_rbf)
        self.conv3 = CGConv(hidden_dim, dim=num_rbf)
        self.lin = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.emb(x)

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))

        x = global_mean_pool(x, data.batch)
        return self.lin(x)


def validate(
    model: nn.Module,
    dataset: StructureDataset,
    device=None,
    batch_size: int = 128,
    num_rbf: int = 32,
):
    """Evaluate intra-structure consistency and inter-structure separation."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model.eval()
    intra_sims, inter_sims = [], []
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x
    )

    with torch.no_grad():
        for structures in dataloader:
            graphs1 = [
                structure_to_graph(augment(s), num_rbf=num_rbf) for s in structures
            ]
            graphs2 = [
                structure_to_graph(augment(s), num_rbf=num_rbf) for s in structures
            ]
            batch1 = Batch.from_data_list(graphs1).to(device)
            batch2 = Batch.from_data_list(graphs2).to(device)

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
) -> Tuple[EquivariantCrystalGCN, List[float], List[float]]:
    """
    Contrastively train the EquivariantCrystalGCN on structures stored as CSVs.
    Returns the trained model plus tracked intra/inter validation curves.
    """
    torch_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_structs = read_structure_from_csv(train_csv)
    dataset = StructureDataset(train_structs)

    val_structs = read_structure_from_csv(val_csv) if val_csv else train_structs
    val_dataset = StructureDataset(val_structs)

    model = EquivariantCrystalGCN(
        hidden_dim=hidden_dim, num_rbf=num_rbf, n_layers=n_layers
    ).to(torch_device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x
    )

    val_intra, val_inter = [], []

    for epoch in range(epochs):
        ema_loss = 0.0
        for step, structures in enumerate(dataloader):
            graphs1 = [
                structure_to_graph(augment(s), num_rbf=num_rbf) for s in structures
            ]
            graphs2 = [
                structure_to_graph(augment(s), num_rbf=num_rbf) for s in structures
            ]

            batch1 = Batch.from_data_list(graphs1).to(torch_device)
            batch2 = Batch.from_data_list(graphs2).to(torch_device)

            z1 = model(batch1)
            z2 = model(batch2)

            loss = info_nce_loss(z1, z2, tau=tau)
            opt.zero_grad()
            loss.backward()
            opt.step()

            ema_loss = (loss.item() + ema_loss * step) / (step + 1)

        intra, inter = validate(
            model,
            val_dataset,
            device=torch_device,
            batch_size=batch_size,
            num_rbf=num_rbf,
        )
        val_intra.append(intra)
        val_inter.append(inter)

        print(
            f"Epoch {epoch + 1}: loss={ema_loss:.4f} | intra={intra:.3f} | inter={inter:.3f}"
        )

        if checkpoint_path:
            checkpoint_file = Path(checkpoint_path)
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_file)

    if plot_path:
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

    return model, val_intra, val_inter


if __name__ == "__main__":
    train_contrastive_model(
        "train.csv",
        val_csv="val.csv",
        checkpoint_path="gcn_tau1.pt",
        plot_path="imgs/validation_curve.png",
    )
