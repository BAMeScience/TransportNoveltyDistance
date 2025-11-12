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

import ot
import torch

from .gcn import EquivariantCrystalGCN
from .utils import augment


class OTNoveltyScorer:
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
        tau_quantile: Optional[float] = None,
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
        self.tau_quantile = tau_quantile

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
            tau=tau, tau_quantile=tau_quantile, memorization_weight=memorization_weight
        )

        print(
            f"Using τ = {self.tau:.4f} with memorization weight = {self.m_weight:.4f}"
        )

    def _estimate_tau_ot(
        self,
        fine_features: torch.Tensor,
        split_ratio: float = 0.5,
        quantile: float = 0.5,
    ):
        """
        Estimate ``tau`` from the OT distance between two subsets of the training
        features.

        Parameters
        ----------
        fine_features
            Embedded training structures.
        split_ratio
            Fraction of samples placed into the first subset (the rest go into
            the second subset).
        quantile
            Percentile of the OT distance distribution to use as the cutoff.

        Returns
        -------
        float
            Empirical OT distance quantile used as the novelty threshold.
        """
        n = fine_features.size(0)
        n1 = int(split_ratio * n)
        idx = torch.randperm(n)
        f1, f2 = fine_features[idx[:n1]], fine_features[idx[n1:]]

        C = torch.cdist(f1, f2, p=2)
        a = torch.full((f1.size(0),), 1.0 / f1.size(0))
        b = torch.full((f2.size(0),), 1.0 / f2.size(0))
        P = torch.from_numpy(
            ot.emd(
                a.numpy(),
                b.numpy(),
                C.cpu().numpy(),
                numItermax=self.ot_num_itermax,
            )
        ).to(C.device)
        d_flat, w_flat = C.flatten(), P.flatten() / P.sum()
        sorted_idx = torch.argsort(d_flat)
        cumw = torch.cumsum(w_flat[sorted_idx], dim=0)
        cutoff = torch.searchsorted(cumw, quantile)
        return d_flat[sorted_idx[min(cutoff, len(cumw) - 1)]].item()

    def _resolve_tau_and_weight(
        self,
        *,
        tau: float | None,
        tau_quantile: float | None,
        memorization_weight: float | None,
    ) -> tuple[float, float]:
        """Determine τ and the memorization weight given user inputs."""

        if tau is not None:
            if memorization_weight is not None:
                return tau, memorization_weight
            _, m_scale = self.calibrate_principled_tau(fixed_tau=tau)
            return tau, m_scale

        if tau_quantile is not None:
            tau_auto = self._estimate_tau_ot(self.train_feats, quantile=tau_quantile)
            if memorization_weight is not None:
                return tau_auto, memorization_weight
            _, m_scale = self.calibrate_principled_tau(fixed_tau=tau_auto)
            return tau_auto, m_scale

        tau_auto, auto_m_weight = self.calibrate_principled_tau(fixed_tau=None)
        return tau_auto, memorization_weight or auto_m_weight

    def calibrate_principled_tau(
        self, fixed_tau: float | None = None
    ) -> tuple[float, float]:
        """
        Separate augmented copies from real samples via a brute-force threshold search.

        Returns
        -------
        (tau, memorization_scale)
            ``tau`` maximizes separation between leave-one-out distances and
            augmented copies; the second value rescales the memorization penalty
            so that both terms lie on comparable magnitudes.

        Raises
        ------
        ValueError
            If fewer than two training structures are available.
        """
        if len(self.train_structs) < 2:
            raise ValueError("Calibration requires at least two training structures.")

        pairwise = torch.cdist(self.train_feats, self.train_feats)
        pairwise.fill_diagonal_(float("inf"))
        d_loocv = pairwise.min(dim=1).values

        augmented_structs = [augment(s) for s in self.train_structs]
        aug_feats = self.featurizer(augmented_structs).to(self.device)
        d_aug = torch.cdist(aug_feats, self.train_feats).min(dim=1).values

        if fixed_tau is None:
            all_distances = torch.cat([d_aug, d_loocv])
            labels = torch.cat([torch.ones_like(d_aug), torch.zeros_like(d_loocv)])

            min_error = float("inf")
            optimal_tau = float((d_aug.mean() + d_loocv.mean()) / 2.0)

            for tau_candidate in torch.unique(all_distances):
                predictions = (all_distances < tau_candidate).float()
                error = torch.abs(predictions - labels).sum()
                if error < min_error:
                    min_error = error
                    optimal_tau = tau_candidate.item()
        else:
            optimal_tau = fixed_tau

        numerator = d_loocv.mean() + 2 * d_loocv.std() - optimal_tau
        denominator = max(
            optimal_tau - d_aug.mean(), torch.tensor(1e-6, device=self.device)
        )
        m_weight = (numerator / denominator).item()

        return optimal_tau, m_weight

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
