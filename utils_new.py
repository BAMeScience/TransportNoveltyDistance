import torch
from torch_geometric.data import Batch
from gcn import *
import torch
from pymatgen.core.periodic_table import Element
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import torch
from pymatgen.core import Element
import torch.nn.functional as F 
from pymatgen.analysis.structure_matcher import StructureMatcher


def remove_structural_duplicates(structures, ltol=0.2, stol=0.3, angle_tol=5):
    """
    Remove symmetry-equivalent or translated duplicates from a list of pymatgen Structures.
    Returns a list of unique structures.
    """
    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    unique = []
    for s in structures:
        if not any(matcher.fit(s, u) for u in unique):
            unique.append(s)
    print(f"Removed {len(structures) - len(unique)} duplicates.")
    return unique


def drop_val_duplicates_structural(train_structs, val_structs,
                                   ltol=0.2, stol=0.3, angle_tol=5):
    """
    Remove from val any structure structurally equivalent to one in train.
    """
    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    unique_val = []
    for v in val_structs:
        if not any(matcher.fit(v, t) for t in train_structs):
            unique_val.append(v)
    print(f"Removed {len(val_structs) - len(unique_val)} structural duplicates from val")
    return unique_val


# helper: clustering so that we are sure that the same materials end up in the same split
# for toy experiments

def cluster_split(X_coarse, n_clusters=2, random_state=42):
    X_norm = StandardScaler().fit_transform(X_coarse)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X_norm)
    return labels

def build_simple_embeddings():
    '''
    make generic chemical embeddings for the distance functions.
    Embeddings include row number in the periodic table, as well as atomic radius.
    Embeddings are created for Elements up to Atmic Number 119 (biggest Atom Number so far).
    '''
    embeds = {}
    for Z in range(1, 119):
        el = Element.from_Z(Z)
        vec = [el.row, el.atomic_radius]
        embeds[Z] = np.array(vec, dtype=float)
    return embeds

ELEM_EMBEDS = build_simple_embeddings()

def get_knn(structure: Structure, k: int):
    """
    Returns the k nearest neighbors of each atom in a Structure.

    Parameters
    ----------
    structure : pymatgen.core.Structure
        The crystal structure.
    k : int
        Number of nearest neighbors to return.

    Returns
    -------
    list of list of (int, float)
        For each atom i, a list of (neighbor_index, distance).
    """
    all_nns = []
    for i, site in enumerate(structure):
        # use a large cutoff so we definitely get at least k neighbors
        neighbors = structure.get_neighbors(site, r=10.0)
        # sort by distance and take the k closest
        neighbors = sorted(neighbors, key=lambda nn: nn.nn_distance)[:k]
        all_nns.append([(nn.index, nn.nn_distance) for nn in neighbors])
    return all_nns

def get_knn_old(cart_coords, lattice, k):
    '''
    helper function: KNN function for the coarse features
    for each atom find the nearest neighbors (k many)
    materials considered here are periodical, i.e., we need to look at all the shifts from the lattice
    '''
    n = len(cart_coords)
    atom_nns = []
    shifts = np.array(np.meshgrid([-1,0,1], [-1,0,1], [-1,0,1])).T.reshape(-1,3)
    images = shifts @ lattice  # (27,3)
    # iterate over all the possible atom locations
    for i in range(n):
        dists = []
        idxs = []
        for j in range(n):
            # iterate over all the shifts
            for shift in images:
                if i == j and np.allclose(shift, 0):  # skip exact self
                    continue
                # distance is given by coordinates j - coordinates i, and then
                # taking the shift param closest (i.e., minimum over all shifts)
                diff = cart_coords[j] + shift - cart_coords[i]
                dists.append(np.linalg.norm(diff))
                idxs.append(j)
        # sort and return the idx and distances
        order = np.argsort(dists)[:k]
        atom_nns.append([(idxs[o], dists[o]) for o in order])
    return atom_nns

K = 2
MAX_ATOMS = 20

def featurize_coarse_old(structures, k=K, max_atoms=MAX_ATOMS):
    '''
    Coarse featurizer. It is coarse in the sense that it does not allow for precise reconstruction
    of specific materials. The idea is as follows:
    1) Add the header (aka lattice params)
    2) take each atom, and consider its element embeddings.
    3) add its euclidean distances (k nearest neighbors)
    4) and their chemical distances (to account for structure)
    it relies more on pairwise distance structures as opposed to fine featurizers
    also retains nice invariances
    '''
    feats = []
    for s in structures:

        # add the lattice parameters
        lat = s.lattice
        header = [lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma, len(s)]

        # find for each coordinates the closest n (default = 2) atoms
        knn_all = get_knn(s, k)

        # create empty list to save atom features
        atom_feats = []

        # for each atom (idx) find the k nearest neighbors
        for idx, site in enumerate(s.sites[:max_atoms]):
            Zi = site.specie.Z
            emb_i = ELEM_EMBEDS[Zi]

            r_dists, c_dists = [], []
            for j, dist in knn_all[idx]:
                Zj = s.sites[j].specie.Z
                # elemental embedding
                emb_j = ELEM_EMBEDS[Zj]
                # euclidean distances
                r_dists.append(dist)
                # chemical distances
                c_dists.append(float(np.linalg.norm(emb_i - emb_j)))

            vec = [Zi] + r_dists + c_dists
            atom_feats.append(vec)

        # pad atoms (DANGER ZONE)PPPPPPPPP
        while len(atom_feats) < max_atoms:
            atom_feats.append([0.0] * (1 + 2*k))
        # concatenate stuff in the end
        vec = np.concatenate([header, np.array(atom_feats, dtype=float).ravel()])
        feats.append(torch.tensor(vec, dtype=torch.float32))

    return torch.stack(feats)



def featurize_coarse(structures, model, device="cpu"):
    '''
    fine featurizer: use  trained gcn to predict fine features
    "fine", since the gcn is trained to distinguish between positive (rotations&translations)
    and negative (other samples) examples, and therefore allows reconstruction of materials
    '''
    with torch.no_grad():
        model.eval()
        graphs = [structure_to_graph(s) for s in structures]
        batch  = Batch.from_data_list(graphs).to(device)
        z = model(batch)                  # (n, d)
        z = F.normalize(z, dim = 1)
    return z.cpu()

def featurize_fine(structures, model, device="cpu"):
    """
    Fine featurizer:
    - Computes normalized GNN embedding for each structure.
    - Concatenates lattice parameters (a,b,c,alpha,beta,gamma)
      scaled to physically reasonable magnitudes.

    Notes
    -----
    * a,b,c divided by 10 → puts them roughly in [0,1] for most crystals.
    * angles divided by 180 → normalized to [0,1].
    * keeps absolute physical meaning (no dataset-dependent normalization).
    """
    with torch.no_grad():
        model.eval()
        graphs = [structure_to_graph(s) for s in structures]
        batch  = Batch.from_data_list(graphs).to(device)
        z = model(batch)                 # (n, d)
        z = F.normalize(z, dim=1)

        # --- lattice parameters ---
        latt_feats = torch.tensor(
            [[s.lattice.a / 10.0, s.lattice.b / 10.0, s.lattice.c / 10.0,
              s.lattice.alpha / 180.0, s.lattice.beta / 180.0, s.lattice.gamma / 180.0]
             for s in structures],
            dtype=torch.float32,
            device=z.device
        )

        # concatenate GNN embedding + lattice info
        z = torch.cat([z, latt_feats], dim=1)

    return z.cpu()

def featurize_fine_old(structures, model, device="cpu"):
    '''
    fine featurizer: use  trained gcn to predict fine features
    "fine", since the gcn is trained to distinguish between positive (rotations&translations)
    and negative (other samples) examples, and therefore allows reconstruction of materials
    '''
    with torch.no_grad():
        model.eval()
        graphs = [structure_to_graph(s) for s in structures]
        batch  = Batch.from_data_list(graphs).to(device)
        z = model(batch)                  # (n, d)
        z = F.normalize(z, dim = 1)
    return z.cpu()

# all the features live in different spaces
# standardize accordingly
def fit_standardizer(X):
    mu  = X.mean(0)
    std = X.std(0).clamp_min(1e-4)
    return mu, std

# call standardization
def apply_standardizer(X, mu, std):
    return (X - mu) / std

def apply_standardizer_all_in_one(X):
    # call standardization
    mu, std = fit_standardizer(X)
    # apply standardizer
    return (X - mu) / std
