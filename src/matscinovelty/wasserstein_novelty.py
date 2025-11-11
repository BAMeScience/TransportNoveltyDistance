from __future__ import annotations

from typing import Callable, Optional, Sequence

import ot
import torch

from .gcn import EquivariantCrystalGCN
from .utils import augment


class OTNoveltyScorer:
    """
    Evaluate generative models with the OT-based novelty metric proposed in the project.
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
    ) -> None:
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

        if tau is None:
            if tau_quantile is not None:
                self.tau = self._estimate_tau_ot(
                    self.train_feats, quantile=tau_quantile
                )
                self.m_weight = (
                    memorization_weight if memorization_weight is not None else 1.0
                )
            else:
                tau, m_scale = self.calibrate_principled_tau()
                self.tau = tau
                self.m_weight = (
                    memorization_weight if memorization_weight is not None else m_scale
                )
        else:
            self.tau = tau
            self.m_weight = (
                memorization_weight if memorization_weight is not None else 1.0
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
        n = fine_features.size(0)
        n1 = int(split_ratio * n)
        idx = torch.randperm(n)
        f1, f2 = fine_features[idx[:n1]], fine_features[idx[n1:]]

        C = torch.cdist(f1, f2, p=2)
        a = torch.full((f1.size(0),), 1.0 / f1.size(0))
        b = torch.full((f2.size(0),), 1.0 / f2.size(0))
        P = torch.from_numpy(ot.emd(a.numpy(), b.numpy(), C.cpu().numpy())).to(C.device)
        d_flat, w_flat = C.flatten(), P.flatten() / P.sum()
        sorted_idx = torch.argsort(d_flat)
        cumw = torch.cumsum(w_flat[sorted_idx], dim=0)
        cutoff = torch.searchsorted(cumw, quantile)
        return d_flat[sorted_idx[min(cutoff, len(cumw) - 1)]].item()

    def calibrate_principled_tau(self) -> tuple[float, float]:
        """
        Separate augmented copies from real samples via a brute-force threshold search.
        """
        if len(self.train_structs) < 2:
            raise ValueError("Calibration requires at least two training structures.")

        pairwise = torch.cdist(self.train_feats, self.train_feats)
        pairwise.fill_diagonal_(float("inf"))
        d_loocv = pairwise.min(dim=1).values

        augmented_structs = [augment(s) for s in self.train_structs]
        aug_feats = self.featurizer(augmented_structs).to(self.device)
        d_aug = torch.cdist(aug_feats, self.train_feats).min(dim=1).values

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

        numerator = d_loocv.mean() + 2 * d_loocv.std() - optimal_tau
        denominator = max(
            optimal_tau - d_aug.mean(), torch.tensor(1e-6, device=self.device)
        )
        m_scale = (numerator / denominator).item()

        return optimal_tau, m_scale

    def _get_ot_plan(self, X: torch.Tensor, Y: torch.Tensor):
        a = torch.full((X.size(0),), 1.0 / X.size(0))
        b = torch.full((Y.size(0),), 1.0 / Y.size(0))
        C = torch.cdist(X, Y, p=2)
        P = torch.from_numpy(ot.emd(a.numpy(), b.numpy(), C.cpu().numpy())).to(C.device)
        return P, C

    def compute_novelty(self, gen_structures: Sequence):
        """
        Compute the total OT-based novelty loss alongside its quality/memorization parts.
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
