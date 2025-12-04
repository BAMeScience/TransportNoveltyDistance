import json
import warnings

import numpy as np
import pandas as pd
import spglib
import torch
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.core import Lattice, Structure, Element
from pymatgen.core.operations import SymmOp
from torch.utils.data import Dataset
from torch_geometric.data import Data


class StructureDataset(Dataset):
    """
    Dataset creation for the Tensorflow Dataloader
    Input:
    Structures - List of crystal structures; Output of read_structure_from_csv
    """

    def __init__(self, structures):
        self.structures = structures

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        return self.structures[idx]


def canonicalize_structure(
    struct: Structure, symprec=1e-3, angle_tolerance=5.0
) -> Structure:
    """
    Returns a symmetry-reduced, Niggli-reduced, sorted structure.
    Falls back gracefully if spglib reduction fails.
    """
    # Step 1: Find primitive cell (symmetry reduction)
    lattice = struct.lattice.matrix
    positions = struct.frac_coords
    numbers = [site.specie.number for site in struct.sites]
    prim = spglib.find_primitive((lattice, positions, numbers), symprec=symprec)
    if prim is not None:
        struct = Structure(
            prim[0], [s.specie for s in struct.sites[: len(prim[2])]], prim[1]
        )

    # Step 2: Niggli reduction and sorting
    struct = struct.get_reduced_structure(reduction_algo="niggli")
    struct = struct.get_sorted_structure()
    return struct


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


def structure_to_graph(structure, cutoff=5.0, num_rbf=128):
    """
    Encode a pymatgen Structure as a torch_geometric Data graph. Adds node
    embeddings for atomic number, pairwise edges within `cutoff`, and RBF edge
    attributes, plus Cartesian positions for equivariant models.
    """
    if len(structure) == 0:
        raise ValueError("Cannot build a graph for an empty structure.")

    z = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)
    pos = torch.tensor(structure.cart_coords, dtype=torch.float)

    mdnn = MinimumDistanceNN(cutoff=cutoff, get_all_sites = True)
    centers = torch.linspace(0, cutoff, num_rbf)
    gamma = 20.0

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


def read_csv(filename):
    # Read the CSV
    df = pd.read_csv(filename, index_col=0)

    # Convert each CIF string to a pymatgen Structure
    structures = []
    for i, row in df.iterrows():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                structure = Structure.from_str(row["cif"], fmt="cif")
            structures.append(structure)
        except Exception as e:
            print(f"Error processing row {i}: {e}")

    print(f"Parsed {len(structures)} structures from {filename}.")

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

  
                # Gaussian noise in fractional space, then wrap to [0,1)
            noise = rng.normal(0.0, sigma, size=3)
            pert = (u + noise) % 1.0

            new_coords.append(pert)

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
    Return an augmented copy of a pymatgen Structure by applying:
      - random supercell expansion (50% probability)
      - random 3D rotation
      - random fractional translation
    """
    s = structure.copy()

    # --- (1) 20% chance to make a supercell ---
    if np.random.rand() < 0.5:
        # Random supercell choice (e.g. 2x1x1, 1x2x2, etc.)
        scale_options = [
            (2, 1, 1), (1, 2, 1), (1, 1, 2),
            (2, 2, 1), (2, 1, 2), (1, 2, 2),
            (2, 2, 2)
        ]
        scale = scale_options[np.random.randint(len(scale_options))]
        s.make_supercell(scale)

    # --- (2) Random rotation ---
    rand_matrix = np.random.normal(size=(3,3))
    Q, _ = np.linalg.qr(rand_matrix)
    if np.linalg.det(Q) < 0:
        Q[:,0] = -Q[:,0]

    op = SymmOp.from_rotation_and_translation(rotation_matrix=Q, translation_vec=[0,0,0])
    s.apply_operation(op, fractional=False)

    # --- (3) Random fractional translation ---
    shift = np.random.rand(3)
    s.translate_sites(range(len(s)), shift, frac_coords=True, to_unit_cell=True)

    return s


def random_lattice_deformation(
    s: Structure,
    max_strain: float = 0.1,
    rng=None
) -> Structure:
    """
    Apply a binary diagonal lattice strain:
    50% chance of (1 - max_strain) and 50% chance of (1 + max_strain).
    NO shear.

    Args:
        s (Structure): base structure.
        max_strain (float): strain magnitude (e.g. 0.1 = ±10%).
        rng (np.random.Generator): optional RNG.

    Returns:
        Structure: distorted copy with same fractional coordinates.
    """
    if rng is None:
        rng = np.random.default_rng()

    A = s.lattice.matrix

    # choose expansion or compression for each axis
    signs = rng.choice([-1, 1], size=3)
    scale = 1.0 + signs * max_strain

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
    p: float = 0.5,
    rng=None,
) -> Structure:
    """
    Randomly create a supercell with probability p using predefined scale matrices.

    Args:
        s: pymatgen Structure
        p: probability to apply a supercell (otherwise structure is unchanged)
        rng: numpy random generator

    Returns:
        New Structure (possibly expanded supercell)
    """
    if rng is None:
        rng = np.random.default_rng()

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

    if rng.random() < p:
        scale = rng.choice(scale_options)
        new_struct.make_supercell(scale)

    return new_struct


def random_group_substitution(
    s: Structure,
    allowed_elements: set,
    p: float = 0.05,
    rng=None
) -> Structure:
    """
    Randomly substitute atoms with same-group elements BUT only if they are
    in the model-supported element list.

    Args:
        s: pymatgen Structure
        allowed_elements: set of element symbols supported by the model
        p: substitution probability per atom
        rng: numpy random generator

    Returns:
        Safe substituted Structure
    """
    if rng is None:
        rng = np.random.default_rng()

    coords = np.array([site.frac_coords for site in s.sites])
    old_species = np.array([str(site.specie) for site in s.sites], dtype=object)
    new_species = old_species.copy()

    mask = rng.random(len(old_species)) < p

    for idx in np.where(mask)[0]:
        elem = Element(old_species[idx])
        group = elem.group
        candidates = [
            e for e in allowed_elements
            if Element(e).group == elem.group and e != elem.symbol
        ]

        if candidates:
            new_species[idx] = rng.choice(candidates)

    return Structure(
        lattice=s.lattice,
        species=new_species.tolist(),
        coords=coords,
        coords_are_cartesian=False,
    )

def random_substitution(
    s: Structure,
    allowed_elements: set,
    p: float = 0.05,
    rng=None
) -> Structure:
    """
    Randomly substitute atoms with same-group elements BUT only if they are
    in the model-supported element list.

    Args:
        s: pymatgen Structure
        allowed_elements: set of element symbols supported by the model
        p: substitution probability per atom
        rng: numpy random generator

    Returns:
        Safe substituted Structure
    """
    if rng is None:
        rng = np.random.default_rng()

    coords = np.array([site.frac_coords for site in s.sites])
    old_species = np.array([str(site.specie) for site in s.sites], dtype=object)
    new_species = old_species.copy()

    mask = rng.random(len(old_species)) < p

    for idx in np.where(mask)[0]:
        elem = Element(old_species[idx])
        group = elem.group

        candidates = [
            e for e in allowed_elements
            if e != elem.symbol
        ]

        if candidates:
            new_species[idx] = rng.choice(candidates)

    return Structure(
        lattice=s.lattice,
        species=new_species.tolist(),
        coords=coords,
        coords_are_cartesian=False,
    )
