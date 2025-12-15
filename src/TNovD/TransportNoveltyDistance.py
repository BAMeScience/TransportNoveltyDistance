"""
Optimal-transport-based Transport Novelty Distance (TNovD) implementation. Mainly contains the TransportNoveltyDistance class.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import ot
import torch

from .utils import augment


class TransportNoveltyDistance:
    """
    Optimal Transport Based "distance" that is initialized with a training dataset,
    and a featurizer (gnn based as of now).
    It automatically sets the tau threshold and memorization weight M, based on diffusion memorization theory
    and an equilibrium condition for M.
    We featurize the training set. One can then call using the compute_TNovD the Transport Novelty Distance with
    respect to generated materials.

    """

    def __init__(
        self,
        train_structures: Sequence,
        gnn_model: Optional[torch.nn.Module] = None,
        *,
        tau: Optional[float] = None,
        memorization_weight: Optional[float] = None,
        device: str | torch.device | None = None,
        ot_num_itermax: int = 1_000_000,
    ) -> None:
        """
        Parameters
        ----------
        train_structures
            Training dataset, your model was trained on.
        gnn_model
            GNN model used to featurize structures, needs to have a self.featurize method.
        Note BOTH tau and memorization weight need to be None so they are not overwritten.
        tau
            treshold for transport novelty distance, if not provided, it is auto-calibrated.
        memorization_weight
            Weighting that can be set for M term. If not provided, it is auto-calibrated.
        device
            device (recommended gpu)
        ot_num_itermax
            maximum number of iterations for the OT hungarian solver. It still occasionally throws are warning, but
            the error was of 5e-8 magnitude, so that is fine.
        """
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # define featurize func
        self.model = gnn_model.to(self.device)
        self.featurizer = self.model.featurize

        self.train_structs = list(train_structures)
        print("Featurizing training structures using GNN")
        self.train_feats = self.featurizer(self.train_structs)#.to(self.device)
        print(f"Training embeddings shape: {self.train_feats.shape}")

        self.ot_steps = ot_num_itermax
        if tau is None or memorization_weight is None:
            self.tau, self.m_weight = self.set_tau_and_M()
        else:
            self.tau = tau
            self.m_weight = memorization_weight

        print(
            f"Calibrated/set tau = {self.tau:.4f} with memorization weight = {self.m_weight:.4f}"
        )

    def get_ot_plan(self, X: torch.Tensor, Y: torch.Tensor):
        """
        computes the optimal transport plan between two sets of features X and Y
        returns the plan pi and cost C (euclidean)
        """

        # moving to numpy seems to be quicker.
        a = torch.ones((X.shape[0]))/(X.shape[0])
        b = torch.ones((Y.shape[0]))/(Y.shape[0])
        cost = ot.dist(X.cpu(),Y.cpu(), metric ='euclidean')
        # transport plan with naive ot solver
        # sinkhorn does not seem to work well here
        pi = ot.emd(a,b,cost, numItermax=self.ot_steps)


        return pi, cost

    def set_tau_and_M(self):
        """
        Jsetting tau and M based on diffusion memorization rule and equilibrium condition

        Uses only training_features and returns tau and M.
        """

        # take all pairwise distances
        pairwise = torch.cdist(self.train_feats, self.train_feats)
        # self neighbors not allowed
        pairwise.fill_diagonal_(float("inf"))
        # sort by distance
        pw_dist_sorted, _ = torch.sort(pairwise, dim=1)
        nearest_neighbor1 = pw_dist_sorted[:, 0]  # 1-NN
        nearest_neighbor2 = pw_dist_sorted[:, 1]  # 2-NN
        # theoretical memorization rule
        y_true = ((nearest_neighbor1 ** 2) <= (nearest_neighbor2**2 / 9.0)).float()

        # now find the threshold that minimizes the distance to the true y, which are 0 if false and 1 if true
        # kinda arbitrary lin space of 500 steps
        candidate_taus = torch.linspace(0., nearest_neighbor1.max().item(), steps = 500)
        min_error = float("inf")
        tau_opt = candidate_taus.mean().item()

        for tau_candidate in candidate_taus:
            # check if n1-earest ngh is less than tau

            y_pred = (nearest_neighbor1 <= tau_candidate).float()
            # measure distance to memorization rule
            # tries to reduce a 2-neighbor rule to a 1-neighbor one
            error = torch.abs(y_pred - y_true).sum()/len(y_true)

            if error < min_error:
                min_error = error
                tau_opt = tau_candidate.item()

        # equilbrium condition for M
        # split train set into two halves
        n = len(self.train_structs)
        # seed does not seem to vary much
        perm = torch.randperm(n)

        indices1 = perm[: n // 2]
        indices2 = perm[n // 2 :]

        split1 = [self.train_structs[i] for i in indices1]
        split2 = [self.train_structs[i] for i in indices2]

        feats1 = self.train_feats[indices1]
        feats2 = self.train_feats[indices2]

        pw_cost = ot.dist(feats1, feats2, metric= 'euclidean')

        a = torch.ones(len(split1)) / len(split1)
        b = torch.ones(len(split2)) / len(split2)

        ot_plan = ot.emd(a, b, pw_cost, numItermax=self.ot_steps)
        # compute for the M formula
        mask = pw_cost <= tau_opt
        not_mask = pw_cost > tau_opt
        p_close = torch.sum(ot_plan[mask])
        p_far = torch.sum(ot_plan[not_mask])

        M = p_far / max(p_close, 1e-8)

        return tau_opt, M

    def compute_TNovD(self, gen_structures: Sequence):
        """
        computes the transport novelty distance for a set of generated structure.
        It takes gen_structures as as input, featurizes them using the gnn model.
        """
        gen_feats = self.featurizer(gen_structures)
        ot_plan, eucl_costs = self.get_ot_plan(self.train_feats, gen_feats)

        cost = eucl_costs - self.tau
        # split like in the paper, this is the omega split up
        quality_cost = torch.relu(cost)
        memory_cost = torch.relu(-cost)
        qual_comp = torch.sum(ot_plan* quality_cost)
        mem_comp = torch.sum(ot_plan * memory_cost) * self.m_weight

        TNovD = qual_comp + mem_comp

        print(
            f"Quality={qual_comp:.4f}, Memorization={mem_comp:.4f}, Total={TNovD:.4f}"
        )
        return TNovD.item(), qual_comp.item(), mem_comp.item()

