import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, CGConv
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from utils import * 

class CrystalGCN(nn.Module):
    """
    A simple graph-based crystal encoder for OT novelty scoring.

    Includes a built-in `featurize()` method that converts pymatgen Structures
    into normalized embeddings, with optional lattice scaling appended.
    """
    def __init__(self, hidden_dim=128, num_rbf=32, device="cuda"):
        super().__init__()
        self.emb = nn.Embedding(100, hidden_dim)
        self.conv1 = CGConv(hidden_dim, dim=num_rbf)
        self.conv2 = CGConv(hidden_dim, dim=num_rbf)
        self.conv3 = CGConv(hidden_dim, dim=num_rbf)
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        self.lattice_scale_abc = 10
        self.lattice_scale_angles = 180 
        self.device = device

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.emb(x)
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x = global_mean_pool(x, data.batch)
        return self.lin(x)

    # ---- integrated featurizer ----
    def featurize(self, structures):
        """
        Converts a list of pymatgen.Structure objects to normalized embeddings.
        Includes scaled lattice parameters [a,b,c,α,β,γ] if lattice_scale > 0.
        """
        self.eval()
        with torch.no_grad():
            graphs = [structure_to_graph(s) for s in structures]
            batch = Batch.from_data_list(graphs).to(self.device)
            z = F.normalize(self(batch), dim=1)

            lat_feats = []
            for s in structures:
                a, b, c = s.lattice.a, s.lattice.b, s.lattice.c
                alpha, beta, gamma = s.lattice.alpha, s.lattice.beta, s.lattice.gamma
                lat = np.array([a, b, c, alpha, beta, gamma], dtype=np.float32)
                lat /= np.array([self.lattice_scale_abc, self.lattice_scale_abc, self.lattice_scale_abc,
                                    self.lattice_scale_angles, self.lattice_scale_angles, self.lattice_scale_angles])  # normalize
                lat_feats.append(lat)
            lat_feats = torch.tensor(lat_feats, device=z.device)
            z = torch.cat([z,  lat_feats], dim=1)

        return z.cpu()


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
    model = CrystalGCN(hidden_dim=128, num_rbf=32).to(device)
    # define the optimizer
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # read structures from the csv
    str_train = read_structure_from_csv('train.csv')
    # create a PyTorch-Dataset for the Dataloader
    dataset = StructureDataset(str_train)
    # Create Dataloader; Batch size set to 16
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda x: x)
    # Define number of epochs
    epochs = 40

    for epoch in range(epochs):
        ema_loss = 0.
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
        torch.save(model.state_dict(), 'gcn_fine.pt')








