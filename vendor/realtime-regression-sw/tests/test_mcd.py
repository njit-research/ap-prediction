"""Tests for the MC-dropout band and its additive residual-variance term."""

import numpy as np
import pytest
import torch

from src._vendor.normalizer import Normalizer
from src.analysis.mcd import mcd_forecast

STEPS = 12


class _NoisyModel(torch.nn.Module):
    """Constant forecast plus a dropout layer so stochastic passes differ."""

    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = torch.full((x.shape[0], STEPS, 1), 2.0)
        return base + 0.1 * self.dropout(torch.ones_like(base))


def _normalizer() -> Normalizer:
    stats = {"ap30": {"mean": 0.0, "std": 1.0}}
    return Normalizer(stats, {"default": "zscore", "methods": {"ap30": "zscore"}})


def _run(noise_variance: float, n_std: float = 1.96):
    torch.manual_seed(0)
    model = _NoisyModel().eval()
    return mcd_forecast(model, torch.zeros(1, 4, 1), _normalizer(), "ap30",
                        num_samples=64, n_std=n_std, noise_variance=noise_variance)


def test_default_band_is_raw_mc_band():
    res = _run(noise_variance=0.0)
    assert res.noise_variance == 0.0
    np.testing.assert_allclose(res.std, res.mc_std)
    np.testing.assert_allclose(res.upper, res.mean + 1.96 * res.mc_std)
    assert res.mc_std.max() > 0  # dropout was really switched on


def test_noise_variance_adds_in_quadrature():
    raw = _run(noise_variance=0.0)
    res = _run(noise_variance=156.888489)
    np.testing.assert_allclose(res.mean, raw.mean)
    np.testing.assert_allclose(res.mc_std, raw.mc_std)
    np.testing.assert_allclose(res.std, np.sqrt(raw.mc_std ** 2 + 156.888489))
    np.testing.assert_allclose(res.upper, res.mean + 1.96 * res.std)
    np.testing.assert_allclose(res.lower, np.clip(res.mean - 1.96 * res.std, 0.0, None))
    assert res.std.min() >= np.sqrt(156.888489)


def test_negative_noise_variance_rejected():
    with pytest.raises(ValueError):
        _run(noise_variance=-1.0)
