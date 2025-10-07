import pandas as pd
import ast
import torch 
import json 
from pymatgen.core.operations import SymmOp
from pymatgen.core import Structure
import warnings 
from torch_geometric.data import Data
import numpy as np 
from torch.utils.data import Dataset
from pymatgen.analysis.local_env import MinimumDistanceNN


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
    '''
    Converts an atomic structure into a graph suitable
    for graph neural networks (GNNs).

    Parameters:
    -----------
    structure : pymatgen Structure or similar; Parsed from CIF-file
        Atomic structure containing sites with element types and coordinates.
    cutoff : float, optional
        Maximum distance (in Angstroms) to consider two atoms as neighbors.
        Defaults to 15.0 Å.
    num_rbf : int, optional
        Controls the size of the radial basis function (RBF) encoding.
        Defaults to 32.

    Returns:
    --------
    torch_geometric.data.Data
        A graph object where nodes represent atoms with atomic number features
        and edges represent neighboring atom pairs with RBF-expanded distances as attributes.
    '''
    N = len(structure)

    # node features: atomic numbers
    z = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)

    # build neighbor list
    nns = MinimumDistanceNN(cutoff=cutoff).get_all_nn_info(structure)

    edge_index = []
    edge_attr = []

    # RBF expansion helper
    centers = torch.linspace(0, cutoff, num_rbf)
    gamma = 10.0

    for i, neighs in enumerate(nns):
        for n in neighs:
            j = n['site_index'] # index at which the atom lies in the structure
            d = n['weight']  # distance
            edge_index.append([i, j])

            # radial basis expansion
            rbf = torch.exp(-gamma * (d - centers)**2)
            edge_attr.append(rbf)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr)

    return Data(x=z, edge_index=edge_index, edge_attr=edge_attr)


def read_csv(filename):
    # Read the CSV
    df = pd.read_csv(filename, index_col=0)

    # Convert each CIF string to a pymatgen Structure
    structures = []
    for i, row in df.iterrows():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                structure = Structure.from_str(row['cif'], fmt='cif')
            structures.append(structure)
        except Exception as e:
            print(f'Error processing row {i}: {e}')

    print(f'Parsed {len(structures)} structures from {filename}.')

    return structures

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

def perturb_structures(original_structures, mode="lattice_scale", strength=0.1, rng=None):
    """
    Apply perturbations to a list of pymatgen Structures.

    Args:
        original_structures (list[Structure]): input structures.
        mode (str): type of perturbation. One of:
            - "lattice_scale" : uniform lattice scaling
            - "shear"         : random lattice shear/strain
            - "clash"         : move one atom towards another to create overlap
            - "vacancies"     : remove random sites
        strength (float): magnitude parameter (interpreted per mode).
        rng (np.random.Generator or None): RNG.

    Returns:
        list[Structure]
    """
    if rng is None:
        rng = np.random.default_rng()

    perturbed = []

    for s in original_structures:
        if mode == "lattice_scale":
            # scale lattice uniformly
            new_lat = s.lattice.matrix * strength
            new = Structure(
                lattice=new_lat,
                species=[site.species for site in s.sites],
                coords=[site.frac_coords for site in s.sites],
                coords_are_cartesian=False
            )

        elif mode == "shear":
            # random shear/strain
            A = s.lattice.matrix
            S = np.eye(3) + strength * rng.standard_normal((3, 3))
            new_lat = A @ S
            new = Structure(
                lattice=new_lat,
                species=[site.species for site in s.sites],
                coords=[site.frac_coords for site in s.sites],
                coords_are_cartesian=False
            )

        elif mode == "clash":
            # force atomic clashes
            if len(s) < 2:
                new = s.copy()
            else:
                i, j = 0, 1
                v = (s[j].frac_coords - s[i].frac_coords) % 1.0
                new_coords = []
                for idx, site in enumerate(s.sites):
                    if idx == j:
                        pert = (site.frac_coords - strength * v) % 1.0
                        new_coords.append(pert)
                    else:
                        new_coords.append(site.frac_coords)

                new = Structure(
                    lattice=s.lattice,
                    species=[site.species for site in s.sites],
                    coords=new_coords,
                    coords_are_cartesian=False
                )

        elif mode == "vacancies":
            # remove random sites
            n_remove = max(1, int(strength * len(s)))
            idx_remove = rng.choice(len(s), size=n_remove, replace=False)

            new_coords, new_species = [], []
            for idx, site in enumerate(s.sites):
                if idx not in idx_remove:
                    new_coords.append(site.frac_coords)
                    new_species.append(site.species)

            new = Structure(
                lattice=s.lattice,
                species=new_species,
                coords=new_coords,
                coords_are_cartesian=False
            )

        else:
            raise ValueError(f"Unknown perturbation mode: {mode}")

        perturbed.append(new)

    return perturbed


def perturb_structures_corrupt(
    original_structures,
    vacancy_prob=0.1,
    swap_prob=0.1,
    rng=None
):
    """
    Perturb structures by introducing vacancies and random atom swaps.

    Ensures that at least 2 atoms remain, so neighbor-finding won't crash.
    """
    if rng is None:
        rng = np.random.default_rng()

    perturbed_structures = []
    all_species = list({
        str(sp) for s in original_structures for sp in s.species
    })

    for original in original_structures:
        new_coords = []
        new_species = []

        for site in original.sites:
            # --- vacancy ---
            if rng.random() < vacancy_prob:
                continue

            sp = str(site.specie)

            # --- swap species ---
            if rng.random() < swap_prob:
                sp = rng.choice(all_species)

            new_species.append(sp)
            new_coords.append(site.frac_coords)

        # --- safeguard: keep at least 2 atoms ---
        if len(new_species) < 2:
            # take first two sites from the original
            for site in original.sites[:2]:
                if len(new_species) >= 2:
                    break
                new_species.append(str(site.specie))
                new_coords.append(site.frac_coords)

        perturbed = Structure(
            lattice=original.lattice,
            species=new_species,
            coords=new_coords,
            coords_are_cartesian=False
        )
        perturbed_structures.append(perturbed)

    return perturbed_structures


def perturb_structures_gaussian(
    original_structures,
    sigma=0.05,
    teleport_prob=0.0,
    rng=None
):
    """
    Intentionally unphysical perturbations in fractional space using Gaussian noise,
    with optional teleportation (uniform re-placement) of sites.

    Args:
        original_structures (list[pymatgen.core.Structure]): input structures.
        sigma (float): std dev of Gaussian noise in fractional units.
        teleport_prob (float): per-site probability to ignore local noise and
                               place uniformly in [0,1)^3 (very unphysical).
        rng (np.random.Generator or None): optional RNG for reproducibility.

    Returns:
        list[pymatgen.core.Structure]
    """
    if rng is None:
        rng = np.random.default_rng()

    perturbed_structures = []

    for original in original_structures:
        new_coords = []

        for site in original.sites:
            u = np.asarray(site.frac_coords, dtype=float)

            if teleport_prob > 0 and rng.random() < teleport_prob:
                # Maximal disturbance: uniform position in fractional cell
                pert = rng.random(3)
            else:
                # Gaussian noise in fractional space, then wrap to [0,1)
                noise = rng.normal(0.0, sigma, size=3)
                pert = (u + noise) % 1.0

            new_coords.append(pert)

        perturbed = Structure(
            lattice=original.lattice,
            species=[s.species for s in original.sites],
            coords=new_coords,
            coords_are_cartesian=False
        )
        perturbed_structures.append(perturbed)

    return perturbed_structures

