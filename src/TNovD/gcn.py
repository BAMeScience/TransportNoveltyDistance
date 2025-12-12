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
from torch_geometric.nn import global_mean_pool

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
    """wrapper base class for gnn encoders with functionalities needed for TNovD."""

    def __init__(
        self,
        *,
        cutoff: float = 5.0,
        num_rbf: int = 128,
        gamma = 20.
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.gamma = gamma

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _graph_kwargs(self) -> dict:
        return {
            "cutoff": self.cutoff,
            "num_rbf": self.num_rbf,
            "gamma": self.gamma
        }

    def featurize(self, structures: Sequence, batch_size: int = 64) -> torch.Tensor:
            """
            featurizer for a GNN trained. note that the features are passed in L2 normalized form,
            since they are compatible with the InfoNCE loss.
            If one would use another featurizer like MACE this would likely be bad.
            Params:
            structures: sequence of structures to featurize
            batch_size: batch size for chunk wise features
            """
            self.eval()
            embeddings = []

            # Process in chunks
            for i in range(0, len(structures), batch_size):
                chunk_structs = structures[i : i + batch_size]

                graphs = [structure_to_graph(s, self.cutoff, self.num_rbf, self.gamma) for s in chunk_structs]

                # push to device for gnn processing
                batch = Batch.from_data_list(graphs).to(self.device)

                with torch.no_grad():
                    # passing and normalizing
                    z = self.forward(batch)
                    z = F.normalize(z, dim=1)

                    # back to cpu
                    embeddings.append(z.cpu())

            # chunky catting
            return torch.cat(embeddings, dim=0)

class EGNNLayer(nn.Module):
    """A lightweight invariant GNN block, inspired by the EGNN architecture
       We dropped positional updates to make it invariant.
       Input of the layer not only RBF features, but also the pure distance.
       """

    def __init__(self, in_features: int, hidden_features: int, edge_features: int = 0, cut_off = 5.0) -> None:
        super().__init__()

        # in features = hidden embedding of both nodes+ num_rbf + raw dist
        input_dim = in_features * 2 + edge_features + 1
        self.cutoff = cut_off

        # edge and node MLPs
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
        edge_weight: torch.Tensor, # Raw distance scalar
        edge_attr: torch.Tensor,   # RBF features
    ):
        row, col = edge_index

        d_raw = edge_weight.unsqueeze(-1) / self.cutoff  # normalize with cutoff parameter

        # concatenate everything, inclduding raw distance
        edge_input = torch.cat([x[row], x[col], edge_attr, d_raw], dim=-1)

        m_ij = self.edge_mlp(edge_input)

        agg = torch.zeros_like(x)
        agg.index_add_(0, row, m_ij)
        # update hidden representation, but NOT position
        x = x + self.node_mlp(torch.cat([x, agg], dim=-1))

        return x


class EquivariantCrystalGCN(BaseCrystalEncoder):
    """module now stacking many egnn layers """

    def __init__(self,hidden_dim: int = 128, num_rbf: int = 128, n_layers: int = 3,cutoff: float = 5.0, gamma: float = 20.0) -> None:
        super().__init__(cutoff=cutoff, num_rbf=num_rbf, gamma = gamma)
        # embedding for species/atomic number
        self.emb = nn.Embedding(100, hidden_dim)

        # stack of egnn layers
        self.layers = nn.ModuleList(
            [
                EGNNLayer(hidden_dim, hidden_dim, edge_features=num_rbf, cut_off=cutoff)
                for _ in range(n_layers)
            ]
        )
        # final linear layer
        self.lin = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data):
        # embedding
        x = self.emb(data.x).float()
        # get graph info
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        edge_weight = data.edge_weight

        for layer in self.layers:
            # Pass both!
            x = layer(x, edge_index, edge_weight, edge_attr)

        # mean pooling for representation independent of size of struc
        x = global_mean_pool(x, data.batch)
        return self.lin(F.relu(x))



class GraphContrastiveDataset(Dataset):
    """
    Dataset that yields two augmented PyG graphs per structure, enabling the
    DataLoader workers to perform heavy featurization work in parallel.
    """

    def __init__(self, structures: Sequence, cutoff: float = 5.0,  num_rbf: int = 32, gamma: float = 20.0) -> None:
        self.structures = list(structures)
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.gamma = gamma

    def __len__(self) -> int:
        return len(self.structures)

    def __getitem__(self, idx: int):
        structure = self.structures[idx]
        # augment with supercell plus rotation and translation, both with different seeds
        g1 = structure_to_graph(augment_supercell(structure), cutoff=self.cutoff, num_rbf=self.num_rbf, gamma=self.gamma)
        g2 = structure_to_graph(augment_supercell(structure), cutoff=self.cutoff, num_rbf=self.num_rbf, gamma=self.gamma)
        return g1, g2


def graph_pair_collate(batch):
    graphs1, graphs2 = zip(*batch)
    batch1 = Batch.from_data_list(graphs1)
    batch2 = Batch.from_data_list(graphs2)
    return batch1, batch2


# make data loader passing all the args...
def make_contrastive_dataloader(
    structures: Sequence,
    *,
    batch_size: int,
    cutoff: float,
    num_rbf: int,
    gamma: float,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int | None,
    persistent_workers: bool | None,
):
    dataset = GraphContrastiveDataset(structures, cutoff = cutoff, num_rbf=num_rbf, gamma = gamma)
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

# evaluate cosine similarity
# for our invariant gnn this should be pretty close to 1.
# the inter cosine should go down during training
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device=None,
    pin_memory: bool = False,
):
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
            # diagonal distances
            intra = (z1 * z2).sum(dim=1)
            intra_sims.extend(intra.cpu().numpy())
            # outer product for inter
            sim_matrix = z1 @ z2.T
            # kill diagonal
            mask = ~torch.eye(len(sim_matrix), dtype=torch.bool, device=device)
            inter = sim_matrix[mask]
            inter_sims.extend(inter.cpu().numpy())

    model.train()
    return float(np.mean(intra_sims)), float(np.mean(inter_sims))


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """standard info nce loss for contrastive learning."""
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
    cutoff: float = 5.0,
    num_rbf: int = 32,
    gamma : float = 20.0,
    n_layers: int = 3,
    device: str | torch.device | None = None,
    checkpoint_path: str | None = None,
    plot_path: str | None = None,
    accelerator: "Accelerator | None" = None,
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


    model = EquivariantCrystalGCN(
        hidden_dim=hidden_dim, cutoff= cutoff, num_rbf=num_rbf, n_layers=n_layers, gamma = gamma).to(torch_device)

    # adam is best
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # make train and val data loader
    train_loader = make_contrastive_dataloader(
        train_structs,
        batch_size=batch_size,
        cutoff=cutoff,
        num_rbf=num_rbf,
        gamma = gamma,
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
        cutoff=cutoff,
        gamma = gamma,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
    if accelerator is not None:
        model, opt = accelerator.prepare(model, opt)

    val_intra, val_inter = [], []
    # training loop
    # track cosine similiarty
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
