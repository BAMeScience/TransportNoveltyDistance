import torch
import ot
import torch.nn.functional as F
from torch_geometric.data import Batch
from gcn import *
from utils import *


class OTNoveltyScorer:
    def __init__(self,
                 train_structures,
                 gnn_model=None,
                 featurizer=None,
                 tau=None,
                 tau_quantile=None,
                 memorization_weight=None,
                 device="cuda"):
        """
        Compute fine-space OT-based novelty loss.

        Args:
            train_structures : list[Structure]
            gnn_model        : CrystalGCN or compatible model with .featurize()
            featurizer       : callable(list[Structure]) -> torch.Tensor
            tau              : fixed τ, or None for auto-estimation
            tau_quantile     : quantile for τ estimation
            memorization_weight : multiplier for memorization term
        """
        self.device = device
        self.m_weight = memorization_weight
        self.tau_quantile = tau_quantile

        # --- Featurizer setup ---
        if featurizer is not None:
            self.featurizer = featurizer
        elif gnn_model is not None:
            self.model = gnn_model.to(device)
            self.featurizer = self.model.featurize
        else:
            self.model = EquivariantCrystalGCN(device=device).to(device)
            print("No GNN provided → using default CrystalGCN.")
            # load pretrained weights 
            weights = torch.load('gcn_fine.pt')
            self.model.load_state_dict(weights)
            self.featurizer = self.model.featurize
        self.train_structs = train_structures
        # --- Precompute reference embeddings ---
        print("Featurizing training structures...")
        self.train_feats = self.featurizer(train_structures).to(device)
        print(f"Training embeddings shape: {self.train_feats.shape}")

        # --- τ ---
        if tau is None:
            tau, m_scale = self.calibrate_principled_tau()  
            self.tau = tau
            self.m_weight = m_scale 
        else:
            self.tau = tau
            self.m_weight = memorization_weight 
        print(f"Using τ = {self.tau:.4f}", f"with memorization weight = {self.m_weight:.4f}")

    # ======================================================
    def _estimate_tau_ot(self, fine_features, split_ratio=0.5, quantile=0.5):
        n = fine_features.size(0)
        n1 = int(split_ratio * n)
        idx = torch.randperm(n)
        f1, f2 = fine_features[idx[:n1]], fine_features[idx[n1:]]

        C = ot.dist(f1, f2, metric = 'euclidean')
        a = torch.full((f1.size(0),), 1.0 / f1.size(0))
        b = torch.full((f2.size(0),), 1.0 / f2.size(0))
        P = torch.from_numpy(ot.emd(a.numpy(), b.numpy(), C.cpu().numpy())).to(C.device)
        d_flat, w_flat = C.flatten(), P.flatten() / P.sum()
        sorted_idx = torch.argsort(d_flat)
        cumw = torch.cumsum(w_flat[sorted_idx], dim=0)
        cutoff = torch.searchsorted(cumw, quantile)
        return d_flat[sorted_idx[min(cutoff, len(cumw) - 1)]].item()
    

    def calibrate_principled_tau(self):
        """
        Estimates a principled τ by finding the optimal decision boundary
        between "trivial copy" and "meaningful" distance distributions,
        calculated from the full training set.
        """
        if len(self.train_structs) < 2:
            raise ValueError("Principled calibration requires at least 2 training structures.")

        # 1. "Meaningful Distance" Distribution (D_loocv)
        #    This is the leave-one-out nearest neighbor distance for the full set.
        pairwise_dists_full = ot.dist(self.train_feats, self.train_feats, metric='euclidean')
        pairwise_dists_full.fill_diagonal_(float('inf'))
        d_loocv = pairwise_dists_full.min(dim=1).values

        # 2. "Trivial Copy" Distribution (D_aug)
        #    Augment the full training set and measure distance back to the full set.
        augmented_structs = [augment(s) for s in self.train_structs]
        aug_feats = self.featurizer(augmented_structs).to(self.device)
        # Note: We find min distance to the *original* full training set.
        d_aug = ot.dist(aug_feats, self.train_feats, metric='euclidean').min(dim=1).values
        
        print(f"Debug Info: d_aug mean={d_aug.mean():.4f}, d_loocv mean={d_loocv.mean():.4f}")

        # 3. Brute-force the optimal decision boundary (τ)
        all_distances = torch.cat([d_aug, d_loocv])
        # Labels: 1 for "trivial copy", 0 for "meaningful distance"
        labels = torch.cat([torch.ones_like(d_aug), torch.zeros_like(d_loocv)])

        min_error = float('inf')
        optimal_tau = (d_aug.mean() + d_loocv.mean()) / 2.0

        for tau_candidate in torch.unique(all_distances):
            # Prediction: 1 if distance is less than tau (i.e., a trivial copy)
            predictions = (all_distances < tau_candidate).float()
            error = torch.abs(predictions - labels).sum()

            if error < min_error:
                min_error = error
                optimal_tau = tau_candidate.item()

        m_scale = (d_loocv.mean()+2*d_loocv.std()-optimal_tau)/(optimal_tau-d_aug.mean())
        
        return optimal_tau, m_scale

    # ======================================================
    def _get_ot_plan(self, X, Y, reg = 0.01):
        a = torch.ones(X.size(0)) / X.size(0)
        b = torch.ones(Y.size(0)) / Y.size(0)
        C = ot.dist(X, Y, metric = 'euclidean')
        P = torch.from_numpy(ot.emd(a.numpy(), b.numpy(), C.cpu().numpy())).to(C.device)
        return P, C


    # ======================================================
    def compute_novelty(self, gen_structures):
        """
        Compute novelty loss for generated structures relative to training set.
        """
        gen_feats = self.featurizer(gen_structures).to(self.device)
        P, C = self._get_ot_plan(self.train_feats, gen_feats)

        cost = C - self.tau
        quality = torch.relu(cost)
        memory = torch.relu(-cost) 
        qual_comp = torch.sum(P * quality)
        mem_comp = torch.sum(P * memory) * self.m_weight
        total = qual_comp + mem_comp

        print(f"Quality={qual_comp:.4f}, Memorization={mem_comp:.4f}, Total={total:.4f}")
        return total.item(), qual_comp.item(), mem_comp.item()
