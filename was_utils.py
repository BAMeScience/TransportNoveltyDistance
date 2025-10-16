import pandas as pd
import torch 
from pymatgen.core import Structure
import warnings
from matminer.featurizers.site import SOAP
import numpy as np
from numpy.exceptions import ComplexWarning
from pymatgen.core import Structure, Lattice, PeriodicSite
import ot 
import ast 
import json 

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



# We can create a helper function for this
def get_scaled_distance_matrix(features1, features2):
    # Get the number of dimensions from the feature vectors
    d = features1.shape[1]
    
    # Calculate the standard pairwise distance matrix
    C = ot.dist(features1, features2, metric='euclidean')
    
    # Scale by the square root of the dimension
    # not necessary as of now since dim the same
    #C= C/math.sqrt(d)
    return C


def get_novelty_loss_only_fine(P, C, tau=0.2, memorization_weight=10.0):
    """Single fine-space OT loss: ⟨P, (C - tau)^2⟩."""
    cost = (C - tau)
    quality_cost = torch.relu(cost)**2  # max(C - tau, 0)^2
    memorization_cost = torch.relu(-cost)**2  # max(tau - C, 0)^2
    mem_comp = torch.sum(memorization_weight * memorization_cost*P)
    qual_comp = torch.sum(quality_cost*P)

    loss = qual_comp + mem_comp
    print(f"Score components: Coarse={qual_comp:.4f}, Fine={mem_comp:.4f}")

    return loss, cost, qual_comp, mem_comp

# --- Now, update your other functions to use it ---

def get_ot_plan(features1, features2, reg=0.01):
    a = torch.ones(features1.shape[0]) / features1.shape[0]
    b = torch.ones(features2.shape[0]) / features2.shape[0]

    # Use the new scaled distance matrix for the transport cost
    C = get_scaled_distance_matrix(features1, features2)
    
    # Optional: Normalize for Sinkhorn stability
    #C_norm = C_scaled / C_scaled.max()
 #   ot_plan = ot.sinkhorn(a, b, C_scaled, reg)
    ot_plan = ot.emd(a, b, C, numItermax=100000)
    return ot_plan, C # We don't need the cost from here anymore

def get_novelty_score(ot_plan, feat_coarse1, feat_coarse2, feat_fine1, feat_fine2, delta):
    A = 1.0
    B = A * delta**2
    epsilon = 1e-4

    # Get scaled distances for both feature spaces
    norms_coarse_scaled = get_scaled_distance_matrix(feat_coarse1, feat_coarse2)
    norms_fine_scaled   = get_scaled_distance_matrix(feat_fine1, feat_fine2)

    # Cost decomposition
    part1 = A * norms_coarse_scaled
    part2 = B / (norms_fine_scaled + epsilon)

    # Weighted by OT plan
    coarse_score = torch.sum(part1 * ot_plan)
    fine_score   = torch.sum(part2 * ot_plan)
    score        = coarse_score + fine_score

    # Debug info
    print(f"Score components: Coarse={coarse_score:.4f}, Fine={fine_score:.4f}")

    return score, coarse_score, fine_score

def choose_delta(P, Cc, Cf, tau=0.2, eps=1e-4):
    """
    Simpler delta calibration.
    Scales fine vs. coarse contributions without percentile tricks.
    """
    qc = torch.sum(P * Cc)                      # ⟨P, C_coarse⟩
    qm = torch.sum(P * (1.0 / (Cf + eps)))      # ⟨P, 1/(C_fine+ε)⟩
    delta = torch.sqrt(tau * (qc + eps) / (qm + eps))
    return float(delta.detach().cpu())

def ds_to_flat(ds):
    X = torch.stack([ds[i] for i in range(len(ds))])          # [N, 4, 64, 64]
    return X.view(X.shape[0], -1)                              # [N, D]

def estimate_tau_ot(fine_features: torch.Tensor,
                    split_ratio: float = 0.5,
                    quantile: float = 0.5,
                    reg: float = None) -> float:
    """
    Estimate τ from an OT plan between two halves of the fine-embedding dataset.
    τ is taken as the quantile of pairwise distances weighted by OT mass.

    Args:
        fine_features : [N, d] tensor of fine GNN embeddings.
        split_ratio   : fraction for first half of the split.
        quantile      : which quantile of the OT-weighted distances to use (0.5 = median).
        reg           : optional entropic regularization for Sinkhorn; None for EMD.

    Returns:
        tau (float)
    """
    n = fine_features.size(0)
    n1 = int(split_ratio * n)
    idx = torch.randperm(n)
    f1, f2 = fine_features[idx[:n1]], fine_features[idx[n1:]]

    # Compute true fine-feature distances
    C = torch.cdist(f1, f2, p=2)

    # Uniform weights
    a = torch.full((f1.size(0),), 1.0 / f1.size(0))
    b = torch.full((f2.size(0),), 1.0 / f2.size(0))

    # Optimal transport plan (no scaling of C!)
    a_np, b_np, C_np = a.numpy(), b.numpy(), C.detach().numpy()
    if reg is None or reg <= 0:
        P_np = ot.emd(a_np, b_np, C_np)
    else:
        P_np = ot.sinkhorn(a_np, b_np, C_np, reg)

    P = torch.from_numpy(P_np).to(C.device)

    # Flatten distances and OT weights
    d_flat = C.flatten()
    w_flat = P.flatten() / P.sum()

    # Compute cumulative distribution of distances weighted by OT mass
    sorted_idx = torch.argsort(d_flat)
    d_sorted = d_flat[sorted_idx]
    w_sorted = w_flat[sorted_idx]
    cumw = torch.cumsum(w_sorted, dim=0)

    # τ = smallest distance where cumulative OT mass reaches chosen quantile
    cutoff_idx = torch.searchsorted(cumw, quantile)
    tau = d_sorted[min(cutoff_idx, len(d_sorted) - 1)].item()

    print(f"τ (quantile={quantile:.3f}) = {tau:.4f}")
    return tau