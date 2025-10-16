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
                 tau_quantile=0.05,
                 memorization_weight=10.0,
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
        self.tau = tau or self._estimate_tau_ot(self.train_feats, quantile=tau_quantile)
        print(f"Using τ = {self.tau:.4f}")

    # ======================================================
    def _estimate_tau_ot(self, fine_features, split_ratio=0.5, quantile=0.5):
        n = fine_features.size(0)
        n1 = int(split_ratio * n)
        idx = torch.randperm(n)
        f1, f2 = fine_features[idx[:n1]], fine_features[idx[n1:]]

        C = torch.cdist(f1, f2)
        a = torch.full((f1.size(0),), 1.0 / f1.size(0))
        b = torch.full((f2.size(0),), 1.0 / f2.size(0))
        P = torch.from_numpy(ot.emd(a.numpy(), b.numpy(), C.cpu().numpy())).to(C.device)
        d_flat, w_flat = C.flatten(), P.flatten() / P.sum()
        sorted_idx = torch.argsort(d_flat)
        cumw = torch.cumsum(w_flat[sorted_idx], dim=0)
        cutoff = torch.searchsorted(cumw, quantile)
        return d_flat[sorted_idx[min(cutoff, len(cumw) - 1)]].item()

    # ======================================================
    def _get_ot_plan(self, X, Y, reg = 0.01):
        a = torch.ones(X.size(0)) / X.size(0)
        b = torch.ones(Y.size(0)) / Y.size(0)
        C = torch.cdist(X, Y, metric = 'euclidean')
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