import torch
from torch_geometric.data import Data
from pymatgen.core import Structure
from pymatgen.analysis.local_env import MinimumDistanceNN
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, CGConv
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
import pandas as pd
import warnings
from pymatgen.core.operations import SymmOp
import matplotlib.pyplot as plt
import torch.nn.functional as F


class EGNNLayer(nn.Module):
    def __init__(self, in_features, hidden_features, edge_features=0):
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
            nn.Linear(hidden_features, in_features)
        )
        # 🔧 NEW: scalar projection for coordinate update
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_features, 1),
            nn.SiLU()
        )

    def forward(self, x, pos, edge_index, edge_attr=None):
        row, col = edge_index  # i <- j
        rij = pos[row] - pos[col]
        dij = (rij ** 2).sum(dim=-1, keepdim=True)

        # message construction
        if edge_attr is not None:
            edge_input = torch.cat([x[row], x[col], dij, edge_attr], dim=-1)
        else:
            edge_input = torch.cat([x[row], x[col], dij], dim=-1)

        m_ij = self.edge_mlp(edge_input)

        # aggregate messages (scalar)
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, m_ij)
        x = self.node_mlp(torch.cat([x, agg], dim=-1))

        # 🧭 Position update (equivariant)
        w_ij = self.coord_mlp(m_ij)             # [num_edges, 1]
        trans = rij * w_ij                      # [num_edges, 3]
        delta = torch.zeros_like(pos)
        delta.index_add_(0, row, trans)
        pos = pos + 0.01 * delta                # scaled shift

        return x, pos

    
class EquivariantCrystalGCN(nn.Module):
    def __init__(self, hidden_dim=128, num_rbf=32, n_layers=3):
        super().__init__()
        self.emb = nn.Embedding(100, hidden_dim)
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, hidden_dim, edge_features=num_rbf)
            for _ in range(n_layers)
        ])
        self.lin = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data):
        x = self.emb(data.x).float()
        pos = data.pos
        edge_index, edge_attr = data.edge_index, data.edge_attr

        for layer in self.layers:
            x, pos = layer(x, pos, edge_index, edge_attr)

        # Invariant pooling
        x = global_mean_pool(x, data.batch)
        return self.lin(F.relu(x))

class StructureDataset(Dataset):
    '''
    Dataset creation for the Tensorflow Dataloader
    Input:
    Structures - List of crystal structures; Output of read_structure_from_csv
    '''
    def __init__(self, structures):
        self.structures = structures

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        return self.structures[idx]


def read_structure_from_csv(filename: str):
    """
    Read a CSV with a 'cif' column and returns a list of structures: list[Structure].
    """
    df = pd.read_csv(filename, index_col=0)
    structures = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, row in df.iterrows():
            structures.append(Structure.from_str(row["cif"], fmt="cif"))
    print(f"Parsed {len(structures)} structures from {filename}.")
    return structures



def structure_to_graph(structure, cutoff=8.0, num_rbf=32):
    N = len(structure)
    z = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)

    # positions (cartesian)
    pos = torch.tensor(structure.cart_coords, dtype=torch.float)

    # build neighbor list
    nns = MinimumDistanceNN(cutoff=cutoff).get_all_nn_info(structure)

    edge_index = []
    edge_attr = []

    centers = torch.linspace(0, cutoff, num_rbf)
    gamma = 10.0

    for i, neighs in enumerate(nns):
        for n in neighs:
            j = n['site_index']
            d = n['weight']
            edge_index.append([i, j])
            rbf = torch.exp(-gamma * (d - centers)**2)
            edge_attr.append(rbf)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr)

    # include positions
    return Data(x=z, pos=pos, edge_index=edge_index, edge_attr=edge_attr)

def augment(structure: Structure) -> Structure:
    """
    Return an augmented copy of a pymatgen Structure by applying a random 3D rotation
    (proper rotation, i.e. det = +1) and a random fractional translation (shift in [0,1)^3).

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        Input structure (assumed to have a lattice / periodic cell).

    Returns
    -------
    pymatgen.core.structure.Structure
        A modified copy of the input structure. The original structure is unchanged.
    """
    s = structure.copy()

    # random rotation
    rand_matrix = np.random.normal(size=(3,3))
    Q, _ = np.linalg.qr(rand_matrix)
    if np.linalg.det(Q) < 0:
        Q[:,0] = -Q[:,0]

    # rotate both lattice and atoms
    op = SymmOp.from_rotation_and_translation(rotation_matrix=Q, translation_vec=[0,0,0])
    s.apply_operation(op, fractional=False)  # apply in cartesian space

    # random fractional translation
    shift = np.random.rand(3)
    s.translate_sites(range(len(s)), shift, frac_coords=True, to_unit_cell=True)

    return s


class CrystalGCN(nn.Module):
    """
    A simple Graph Convolutional Network (GCN) for crystals, implemented in PyTorch Geometric.

    This model represents each atom as an embedding (learned from its atomic number),
    passes messages through several GCNConv layers to capture local bonding/environment,
    and then pools node features into a graph-level embedding for downstream prediction.

    Parameters
    ----------
    hidden_dim : int, default=128
        Dimensionality of the atom/node embeddings and hidden representations.
    out_dim : int, default=128
        Dimensionality of the final graph-level embedding (output of the model).
        Default is same as hidden_dim

    Returns
    -------
    An embedding for every structure in the batch
    """
    def __init__(self, hidden_dim=128, num_rbf=32):
        super().__init__()
        self.emb = nn.Embedding(100, hidden_dim)  # atomic number embedding, up to Z=100

        #self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv1 = CGConv(hidden_dim, dim=num_rbf)
        #self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = CGConv(hidden_dim, dim=num_rbf)
        #self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = CGConv(hidden_dim, dim=num_rbf)

        self.lin = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.emb(x)  # atomic embeddings

        # vanilla GCN layers
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))

        # pool to graph-level (invariant)
        x = global_mean_pool(x, data.batch)

        return self.lin(x)
    
def validate(model, dataset, device='cpu', n_batches=5):
    model.eval()
    intra_sims, inter_sims = [], []
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True, collate_fn=lambda x: x)
        for i, structures in enumerate(dataloader):
            if i >= n_batches:  # only a few batches for speed
                break
            graphs1 = [structure_to_graph(augment(s)) for s in structures]
            graphs2 = [structure_to_graph(augment(s)) for s in structures]
            batch1 = Batch.from_data_list(graphs1).to(device)
            batch2 = Batch.from_data_list(graphs2).to(device)
            #print(torch.mean(batch1.pos - batch2.pos).abs(), F.cosine_similarity(batch1.pos, batch2.pos))

            z1 = F.normalize(model(batch1), dim=1)
            z2 = F.normalize(model(batch2), dim=1)
            #print(torch.mean(z1 - z2).abs(), F.cosine_similarity(z1, z2))
            # intra similarities
            intra = (z1 * z2).sum(dim=1)  # cosine sim for each pair
            intra_sims.extend(intra.cpu().numpy())

            # inter similarities
            sim_matrix = z1 @ z2.T
            mask = ~torch.eye(len(sim_matrix), dtype=torch.bool, device=device)
            inter = sim_matrix[mask]
            inter_sims.extend(inter.cpu().numpy())

    model.train()
    return np.mean(intra_sims), np.mean(inter_sims)


def info_nce_loss(z1, z2, tau=0.1):
    """
    Compute the InfoNCE (contrastive) loss between two batches of embeddings.
    Typically used in self-supervised learning with two augmented "views" of the same data.

    Parameters
    ----------
    z1 : torch.Tensor, shape (B, d)
        Embeddings of the first augmented view, where B = batch size and d = embedding dim.
    z2 : torch.Tensor, shape (B, d)
        Embeddings of the second augmented view, same batch but different augmentation.
    tau : float, default=0.1
        Temperature parameter that controls sharpness of similarity distribution.

    Returns
    -------
    torch.Tensor (scalar)
        The InfoNCE loss, averaged across the batch.
    """
    B, d = z1.shape
    # normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # compute similarity matrix (2B x 2B)
    reps = torch.cat([z1, z2], dim=0)  # (2B, d)
    sim = reps @ reps.T / tau          # cosine similarity scaled

    # mask self-similarity
    mask = torch.eye(2*B, dtype=torch.bool, device=z1.device)
    sim = sim.masked_fill(mask, -9e15)

    # positives: (i, i+B) and (i+B, i)
    targets = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(z1.device)

    # cross-entropy loss
    loss = F.cross_entropy(sim, targets)
    return loss


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":

    # load the model and set input and ouput dimensions
    model = EquivariantCrystalGCN(hidden_dim=128, num_rbf=32).to(device)
    # define the optimizer
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # read structures from the csv
    str_train = read_structure_from_csv('train.csv')
    # create a PyTorch-Dataset for the Dataloader
    dataset = StructureDataset(str_train)
    str_val = read_structure_from_csv('val.csv')
    # create a PyTorch-Dataset for the Dataloader
    val_set = StructureDataset(str_train)
    # Create Dataloader; Batch size set to 16
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, collate_fn=lambda x: x)
    # Define number of epochs
    epochs = 40
    val_intra, val_inter = [], []

    for epoch in range(epochs):
        ema_loss = 0.
        if epoch == 0: 
                intra, inter = validate(model, dataset, device=device)
                print('EPOCH 0 (initial): Intra = %.3f, Inter = %.3f' % (intra, inter))
        for i,structures in enumerate(dataloader):
            # make two augmented views of each structure
            graphs1 = [structure_to_graph(augment(s)) for s in structures]
            graphs2 = [structure_to_graph(augment(s)) for s in structures]

            batch1 = Batch.from_data_list(graphs1).to(device)
            batch2 = Batch.from_data_list(graphs2).to(device)

            z1 = model(batch1)
            z2 = model(batch2)

            loss = info_nce_loss(z1, z2, tau=0.1)

            opt.zero_grad()
            loss.backward()
            opt.step()
            ema_loss = (1/(i+1)) * loss.item() + (i/(i+1)) * ema_loss

        print(ema_loss)
    # --- Validation step ---
        intra, inter = validate(model, dataset, device=device)
        val_intra.append(intra)
        val_inter.append(inter)

        print(f"Epoch {epoch+1}: Loss={ema_loss:.4f} | Intra={intra:.3f} | Inter={inter:.3f}")
        torch.save(model.state_dict(), 'gcn_fine.pt')
    plt.figure(figsize=(6,4))
    plt.plot(val_intra, label='Same structure (intra)')
    plt.plot(val_inter, label='Different structures (inter)')
    plt.plot(np.array(val_intra) - np.array(val_inter), label='Contrastive gap', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Average cosine similarity')
    plt.title('Representation consistency during training')
    plt.legend()
    plt.savefig('imgs/validation_curve.png', dpi=300)








