import json
import warnings

import numpy as np
import pandas as pd
import torch
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.core import  Structure, Element
from pymatgen.core.operations import SymmOp
from torch.utils.data import Dataset
from torch_geometric.data import Data

# helper file for pymatgen utils and deformation experiments.
class StructureDataset(Dataset):
    """
    Simple dataset wrapper for the structures.
    """

    def __init__(self, structures):
        self.structures = structures

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        return self.structures[idx]



def augment(structure: Structure) -> Structure:
    """
    return an augmented copy of pymatgen struc, only including random
    rotations (determinant 1) and random translation.
    does not change the underlying material distances.
    """
    s = structure.copy()

    # random rotation
    rand_matrix = np.random.normal(size=(3, 3))
    Q, _ = np.linalg.qr(rand_matrix)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    # rotate both lattice and atoms
    op = SymmOp.from_rotation_and_translation(
        rotation_matrix=Q, translation_vec=[0, 0, 0]
    )
    s.apply_operation(op, fractional=False)  # apply in cartesian space

    # random fractional translation
    shift = np.random.rand(3)
    s.translate_sites(range(len(s)), shift, frac_coords=True, to_unit_cell=True)

    return s


def read_structure_from_csv(filename: str):
    """
    simple csv reader, given a cif input structure.
    """
    df = pd.read_csv(filename, index_col=0)
    structures = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, row in df.iterrows():
            structures.append(Structure.from_str(row["cif"], fmt="cif"))
    print(f"Parsed {len(structures)} structures from {filename}.")
    return structures


def structure_to_graph(structure, cutoff=5.0, num_rbf=128, gamma = 20.0 ):
    """
    encode the pymatgen structure into a graph for the gnn input.
    parameters: cutoff for neighbor finding, rbf (radial basis functions) for the
    distances as an embedding, gamma for the rbf scaling.
    """
    if len(structure) == 0:
        raise ValueError("Cannot build a graph for an empty structure.")

    z = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)
    pos = torch.tensor(structure.cart_coords, dtype=torch.float)

    # important to set get_all_sites = True, as otherwise cut_off does not do anything.
    mdnn = MinimumDistanceNN(cutoff=cutoff, get_all_sites = True)
    # centers for rbf
    centers = torch.linspace(0, cutoff, num_rbf)

    edge_index, edge_attr, edge_weight, edge_shift = [], [], [], []

    for i in range(len(structure)):
        try:
            neighs = mdnn.get_nn_info(structure, i)
        except ValueError:
            neighs = []

        if not neighs:
            continue

        for nn in neighs:
            j = nn["site_index"]
            d = float(nn["weight"])
            edge_index.append([i, j])
            edge_attr.append(torch.exp(-gamma * (d - centers) ** 2))
            edge_weight.append(d)
            image = np.asarray(nn.get("image", (0, 0, 0)), dtype=float)
            shift_vec = structure.lattice.get_cartesian_coords(image)
            edge_shift.append(torch.tensor(shift_vec, dtype=torch.float))

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        edge_shift = torch.stack(edge_shift)
    else:
        num_edges = 0
        edge_index = torch.zeros((2, num_edges), dtype=torch.long)
        edge_attr = torch.zeros((num_edges, num_rbf), dtype=torch.float)
        edge_weight = torch.zeros((num_edges,), dtype=torch.float)
        edge_shift = torch.zeros((num_edges, 3), dtype=torch.float)

    return Data(
        x=z,
        z=z,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_weight=edge_weight,
        edge_shift=edge_shift,
    )


# load structures from
def load_structures_from_json_column(df, col="structure"):
    structures = []
    for i, val in enumerate(df[col]):
        try:
            s_dict = json.loads(val)
            s = Structure.from_dict(s_dict)
            structures.append(s)
        except Exception as e:
            print(f"Skipping row {i}: {type(e).__name__} - {e}")
    return structures



# two helper functions to compare TNovD to the more classical novelty and coverage
# our TNovD should capture both
# IN FEATURE SPACE
def novelty_score(gen_feats, train_feats, threshold=0.05):
    """
    Fraction of generated samples farther than `threshold` from all training samples.
    """
    D = torch.cdist(gen_feats, train_feats, p=2)
    min_dists = D.min(dim=1).values
    return (min_dists > threshold).float().mean().item()


def coverage_score(train_feats, gen_feats, threshold=0.05):
    """
    Fraction of training samples that have at least one generated sample within `threshold`.
    """
    D = torch.cdist(train_feats, gen_feats, p=2)
    min_dists = D.min(dim=1).values
    return (min_dists <= threshold).float().mean().item()




def perturb_structures_gaussian(
    original_structures, sigma=0.05, rng=None
):
    """
    perturbs structures (fractional space) with gaussian noise of standard deviation sigma
    note that sigma = 0.05 is already quite large! (think 0.05 of gaussian in (-3,3))
    proxy for stability
    """

    perturbed_structures = []

    for original in original_structures:
        new_coords = []

        for site in original.sites:
            u = np.asarray(site.frac_coords, dtype=float)


            # Gaussian noise in fractional space, then wrap to [0,1)
            noise = np.random.normal(0.0, sigma, size=3)
            # mod 1 to stay in frac coordinates
            pert = (u + noise) % 1.0

            new_coords.append(pert)
        # set new structure, species stay fixed, lattice to, but frac coords change
        perturbed = Structure(
            lattice=original.lattice,
            species=[s.species for s in original.sites],
            coords=new_coords,
            coords_are_cartesian=False,
        )
        perturbed_structures.append(perturbed)

    return perturbed_structures



def augment_supercell(structure: Structure) -> Structure:
    """
    augment with rrotation, translation, supercell random scaling of 2 in either direction.
    we do this as in the augment func
    """
    s = structure.copy()

    # 50% chance of supercell
    if np.random.rand() < 0.5:
        # Random supercell choice (e.g. 2x1x1, 1x2x2, etc.)
        scale_options = [
            (2, 1, 1), (1, 2, 1), (1, 1, 2),
            (2, 2, 1), (2, 1, 2), (1, 2, 2),
            (2, 2, 2)
        ]
        scale = scale_options[np.random.randint(len(scale_options))]
        s.make_supercell(scale)

    # like augment
    rand_matrix = np.random.normal(size=(3,3))
    Q, _ = np.linalg.qr(rand_matrix)
    if np.linalg.det(Q) < 0:
        Q[:,0] = -Q[:,0]

    op = SymmOp.from_rotation_and_translation(rotation_matrix=Q, translation_vec=[0,0,0])
    s.apply_operation(op, fractional=False)

    shift = np.random.rand(3)
    s.translate_sites(range(len(s)), shift, frac_coords=True, to_unit_cell=True)

    return s


def random_lattice_deformation(
    s: Structure,
    max_strain: float = 0.1) -> Structure:
    """
    apply a diagonal lattice deformation with intensity max_strain
    """

    A = s.lattice.matrix

    # max_strain this in random direction in each axis
    signs = np.random.choice([-1, 1], size=3)
    scale = 1.0 + signs * max_strain
    # scale lattice diagonally
    F = np.diag(scale)

    new_lat = A @ F

    return Structure(
        lattice=new_lat,
        species=[str(site.specie) for site in s.sites],
        coords=[site.frac_coords for site in s.sites],
        coords_are_cartesian=False
    )



def random_supercell(
    s: Structure,
    p: float = 0.5) -> Structure:
    """
    randomly create a supercell with probability p. same scaling as augment_supercell
    note that this does not include rotations or translations
    """
    scale_options = [
        (2, 1, 1),
        (1, 2, 1),
        (1, 1, 2),
        (2, 2, 1),
        (1, 2, 2),
        (2, 1, 2),
        (2, 2, 2),
    ]

    new_struct = s.copy()

    if np.random.random() < p:
        scale = scale_options[np.random.randint(len(scale_options))]
        new_struct.make_supercell(scale)

    return new_struct


def random_group_substitution(
    s: Structure,
    allowed_elements: set,
    p: float = 0.05) -> Structure:
    """
    random substituion within the same "group", i.e. column of periodic table
    """


    coords = np.array([site.frac_coords for site in s.sites])
    old_species = np.array([str(site.specie) for site in s.sites], dtype=object)
    new_species = old_species.copy()

    mask = np.random.random(len(old_species)) < p

    for idx in np.where(mask)[0]:
        elem = Element(old_species[idx])
        group = elem.group
        candidates = [
            e for e in allowed_elements
            if Element(e).group == elem.group and e != elem.symbol
        ]

        if candidates:
            new_species[idx] = np.random.choice(candidates)

    return Structure(
        lattice=s.lattice,
        species=new_species.tolist(),
        coords=coords,
        coords_are_cartesian=False,
    )

def random_substitution(
    s: Structure,
    allowed_elements: set,
    p: float = 0.05) -> Structure:
    """
    randomly substitute within any "allowed" elements (contained in some material in train dataset)

    """

    coords = np.array([site.frac_coords for site in s.sites])
    old_species = np.array([str(site.specie) for site in s.sites], dtype=object)
    new_species = old_species.copy()

    mask = np.random.random(len(old_species)) < p

    for idx in np.where(mask)[0]:
        elem = Element(old_species[idx])

        candidates = [
            e for e in allowed_elements
            if e != elem.symbol
        ]

        if candidates:
            new_species[idx] = np.random.choice(candidates)

    return Structure(
        lattice=s.lattice,
        species=new_species.tolist(),
        coords=coords,
        coords_are_cartesian=False,
    )
