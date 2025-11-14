import numpy as np
import pytest
import torch

from matscinovelty.wasserstein_novelty import EnergyAwareOTNoveltyScorer


def test_energy_aware_scorer_custom_predictor():
    train_structs = [0.0, 1.0]
    gen_structs = [2.0, 3.0]

    def fake_featurizer(structs):
        return torch.tensor([[float(s)] for s in structs], dtype=torch.float32)

    def fake_energy(structs):
        return np.array([float(s) * 0.1 for s in structs], dtype=float)

    scorer = EnergyAwareOTNoveltyScorer(
        train_structures=train_structs,
        featurizer=fake_featurizer,
        tau=0.0,
        memorization_weight=1.0,
        energy_predictor=fake_energy,
        device="cpu",
    )

    total, qual, mem = scorer.compute_novelty(gen_structs)

    assert mem == pytest.approx(0.0)
    assert qual == pytest.approx(0.15, rel=1e-5)
    assert total == pytest.approx(qual, rel=1e-5)
