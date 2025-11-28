"""
Optimal-transport-based novelty scoring utilities.

The routines in this module turn sets of crystal structures into feature
embeddings (via an equivariant GNN or a user-provided featurizer) and run an
optimal-transport alignment between a training set and generated samples. The
alignment cost is then split into a "quality" term, which penalizes generated
structures that drift too far from the training manifold, and a
"memorization" term, which down-weights samples that exactly replicate the
training data.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import ot
import torch

from .gcn import EquivariantCrystalGCN
from .utils import augment


class TransportNoveltyDistance:
    """
    Evaluate generative models with the OT-based novelty metric introduced in the
    paper.

    The scorer embeds all structures, calibrates a novelty threshold ``tau`` if
    needed, and exposes :meth:`compute_novelty` to compute the weighted OT loss
    against new samples.
    """

    def __init__(
        self,
        train_structures: Sequence,
        gnn_model: Optional[torch.nn.Module] = None,
        *,
        featurizer: Optional[Callable[[Sequence], torch.Tensor]] = None,
        tau: Optional[float] = None,
        memorization_weight: Optional[float] = None,
        device: str | torch.device | None = None,
        pretrained_weights: Optional[str] = None,
        ot_num_itermax: int = 1_000_000,
    ) -> None:
        """
        Parameters
        ----------
        train_structures
            Reference dataset used to anchor novelty measurements.
        gnn_model
            Optional pre-initialized encoder whose :py:meth:`featurize` method
            converts structures into embeddings. If omitted, a new
            :class:`EquivariantCrystalGCN` is instantiated.
        featurizer
            Callable override that bypasses any GNN loading logic entirely.
        tau
            Fixed novelty threshold. If ``None`` a data-driven calibration is
            performed (see ``tau_quantile``).
        tau_quantile
            If provided, ``tau`` is set to the transport-distance quantile
            between two random halves of the training set. Otherwise a
            leave-one-out / augmentation split is used.
        memorization_weight
            Relative weighting for the memorization component of the loss. When
            not supplied a principled default is derived during calibration.
        device
            Torch device specifier for embedding computation.
        pretrained_weights
            Path to a checkpoint, loaded only when no ``gnn_model`` or
            ``featurizer`` is provided.
        ot_num_itermax
            Maximum iterations for POT's network simplex solver (``ot.emd``).
            Raise this if you hit ``numItermax`` warnings on large batches.
        """
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if featurizer is not None:
            self.model = None
            self.featurizer = featurizer
        elif gnn_model is not None:
            self.model = gnn_model.to(self.device)
            self.featurizer = self.model.featurize
        else:
            self.model = EquivariantCrystalGCN().to(self.device)
            if pretrained_weights is not None:
                state = torch.load(pretrained_weights, map_location=self.device)
                self.model.load_state_dict(state)
            self.featurizer = self.model.featurize

        self.train_structs = list(train_structures)
        print("Featurizing training structures...")
        self.train_feats = self.featurizer(self.train_structs).to(self.device)
        print(f"Training embeddings shape: {self.train_feats.shape}")

        self.ot_num_itermax = ot_num_itermax

        self.tau, self.m_weight = self._resolve_tau_and_weight(
            tau=tau, memorization_weight=memorization_weight
        )

        print(
            f"Using τ = {self.tau:.4f} with memorization weight = {self.m_weight:.4f}"
        )


    def _resolve_tau_and_weight(
        self,
        *,
        tau: float | None,
        memorization_weight: float | None,
    ) -> tuple[float, float]:
        """Determine τ and the memorization weight given user inputs."""

        if tau is None or memorization_weight is None:
            tau, memorization_weight = self.calibrate_tau_and_M()
        #auto_m_weight = self.compute_M_principled(tau_auto)
        return tau, memorization_weight

    def _get_ot_plan(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Compute the balanced OT transport plan between two embedding sets.

        Parameters
        ----------
        X, Y
            Embeddings for the source (training) and target (generated) sets.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(P, C)`` where ``P`` is the optimal transport plan and ``C`` the
            pairwise distance matrix.
        """
        a = torch.full((X.size(0),), 1.0 / X.size(0))
        b = torch.full((Y.size(0),), 1.0 / Y.size(0))
        C = torch.cdist(X, Y, p=2)
        P = torch.from_numpy(
            ot.emd(
                a.numpy(),
                b.numpy(),
                C.cpu().numpy(),
                numItermax=self.ot_num_itermax,
            )
        ).to(C.device)
        return P, C

    def calibrate_tau_and_M(self):
        """
        Jointly calibrate tau and M using:

        tau:
            uniform threshold minimizing misclassification between
            theoretical 1NN/2NN mask and threshold predictor

        M:
            computed via OT mass ratio between two random splits

        Returns
        -------
        (tau, M)
        """

        # ---------- Step 1: compute 1-NN and 2-NN distances ----------

        pairwise = torch.cdist(self.train_feats, self.train_feats)
        pairwise.fill_diagonal_(float("inf"))

        sorted_dists, _ = torch.sort(pairwise, dim=1)
        d1 = sorted_dists[:, 0]  # 1-NN
        d2 = sorted_dists[:, 1]  # 2-NN

        # Theoretical memorization mask from 1NN/2NN rule
        y_true = ((d1 ** 2) <= (d2**2 / 9.0)).float()

        # ---------- Step 2: grid search tau minimizing classification error ----------

        candidate_taus = torch.unique(d1)
        min_error = float("inf")
        tau_opt = candidate_taus.mean().item()

        for tau_candidate in candidate_taus:
            y_pred = (d1 <= tau_candidate).float()
            error = torch.abs(y_pred - y_true).sum()

            if error < min_error:
                min_error = error
                tau_opt = tau_candidate.item()

        tau = tau_opt

        # ---------- Step 3: compute M via OT split mass ratio ----------

        n = len(self.train_structs)
        idx = torch.randperm(n)
        k = n // 2

        idx1, idx2 = idx[:k], idx[k:]

        S1 = [self.train_structs[i] for i in idx1]
        S2 = [self.train_structs[i] for i in idx2]

        F1 = self.train_feats[idx1]
        F2 = self.train_feats[idx2]
        
        M_cost = torch.cdist(F1, F2, p=2).cpu().numpy()

        a = np.ones(len(S1)) / len(S1)
        b = np.ones(len(S2)) / len(S2)

        gamma = ot.emd(a, b, M_cost, numItermax=self.ot_num_itermax)

        mask_close = M_cost <= tau
        mask_far = M_cost > tau

        p_close = np.sum(gamma[mask_close])
        p_far = np.sum(gamma[mask_far])

        M = p_far / max(p_close, 1e-8)

        return tau, M

    def compute_novelty(self, gen_structures: Sequence):
        """
        Compute the total OT-based novelty loss alongside its quality/memorization parts.

        Parameters
        ----------
        gen_structures
            Iterable of structures produced by a generative model.

        Returns
        -------
        tuple[float, float, float]
            ``(total, quality, memorization)`` scalar losses.
        """
        gen_feats = self.featurizer(gen_structures).to(self.device)
        P, C = self._get_ot_plan(self.train_feats, gen_feats)

        cost = C - self.tau
        quality = torch.relu(cost)
        memory = torch.relu(-cost)
        qual_comp = torch.sum(P * quality)
        mem_comp = torch.sum(P * memory) * self.m_weight
        total = qual_comp + mem_comp

        print(
            f"Quality={qual_comp:.4f}, Memorization={mem_comp:.4f}, Total={total:.4f}"
        )
        return total.item(), qual_comp.item(), mem_comp.item()

