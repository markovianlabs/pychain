"""
Tests for pychain.diagnostics.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt

from pychain.diagnostics import (
    burn_in,
    thin,
    autocorrelation,
    effective_sample_size,
    gelman_rubin,
    corner_plot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iid_samples(N: int = 500, D: int = 2, seed: int = 0) -> np.ndarray:
    """IID standard normal samples — zero autocorrelation, ESS ≈ N."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, D))


def _correlated_samples(N: int = 500, D: int = 2, rho: float = 0.9, seed: int = 0) -> np.ndarray:
    """AR(1) samples with correlation rho — high autocorrelation, ESS << N."""
    rng = np.random.default_rng(seed)
    samples = np.zeros((N, D))
    samples[0] = rng.standard_normal(D)
    for i in range(1, N):
        samples[i] = rho * samples[i - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal(D)
    return samples


# ---------------------------------------------------------------------------
# burn_in
# ---------------------------------------------------------------------------

class TestBurnIn:
    def test_removes_correct_number(self):
        s = _iid_samples(N=100)
        result = burn_in(s, n_burn=20)
        assert result.shape == (80, 2)

    def test_zero_burn_in(self):
        s = _iid_samples(N=100)
        np.testing.assert_array_equal(burn_in(s, n_burn=0), s)

    def test_returns_correct_rows(self):
        s = np.arange(20).reshape(10, 2)
        result = burn_in(s, n_burn=3)
        np.testing.assert_array_equal(result, s[3:])

    def test_negative_burn_in_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            burn_in(_iid_samples(), n_burn=-1)

    def test_burn_in_gte_n_raises(self):
        s = _iid_samples(N=10)
        with pytest.raises(ValueError, match="less than"):
            burn_in(s, n_burn=10)

    def test_burn_in_equal_n_raises(self):
        s = _iid_samples(N=10)
        with pytest.raises(ValueError):
            burn_in(s, n_burn=10)


# ---------------------------------------------------------------------------
# thin
# ---------------------------------------------------------------------------

class TestThin:
    def test_step_1_returns_all(self):
        s = _iid_samples(N=100)
        np.testing.assert_array_equal(thin(s, step=1), s)

    def test_step_2_halves_samples(self):
        s = _iid_samples(N=100)
        assert len(thin(s, step=2)) == 50

    def test_step_10_reduces_samples(self):
        s = _iid_samples(N=100)
        assert len(thin(s, step=10)) == 10

    def test_returns_correct_rows(self):
        s = np.arange(20).reshape(10, 2)
        result = thin(s, step=3)
        np.testing.assert_array_equal(result, s[::3])

    def test_step_zero_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            thin(_iid_samples(), step=0)

    def test_step_negative_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            thin(_iid_samples(), step=-2)


# ---------------------------------------------------------------------------
# autocorrelation
# ---------------------------------------------------------------------------

class TestAutocorrelation:
    def test_shape(self):
        s = _iid_samples(N=200, D=3)
        acf = autocorrelation(s, max_lag=50)
        assert acf.shape == (50, 3)

    def test_lag0_is_one(self):
        s = _iid_samples(N=200, D=2)
        acf = autocorrelation(s, max_lag=10)
        np.testing.assert_allclose(acf[0], 1.0, atol=1e-10)

    def test_iid_acf_near_zero_for_large_lags(self):
        # IID samples should have near-zero ACF for lag > 0
        rng = np.random.default_rng(42)
        s = rng.standard_normal((2000, 2))
        acf = autocorrelation(s, max_lag=5)
        # Average ACF at lags 1-4 should be small
        assert np.abs(acf[1:]).mean() < 0.1

    def test_correlated_acf_positive_at_lag1(self):
        s = _correlated_samples(N=500, rho=0.9)
        acf = autocorrelation(s, max_lag=10)
        assert np.all(acf[1] > 0.5)

    def test_default_max_lag(self):
        s = _iid_samples(N=200)
        acf = autocorrelation(s)
        assert acf.shape[0] == 100  # N // 2

    def test_max_lag_too_large_raises(self):
        s = _iid_samples(N=100)
        with pytest.raises(ValueError, match="max_lag"):
            autocorrelation(s, max_lag=100)

    def test_1d_input_handled(self):
        s = np.random.default_rng(0).standard_normal(200)
        acf = autocorrelation(s, max_lag=10)
        assert acf.shape == (10, 1)


# ---------------------------------------------------------------------------
# effective_sample_size
# ---------------------------------------------------------------------------

class TestEffectiveSampleSize:
    def test_shape(self):
        s = _iid_samples(N=500, D=3)
        ess = effective_sample_size(s)
        assert ess.shape == (3,)

    def test_iid_ess_near_n(self):
        # IID samples should have ESS close to N
        rng = np.random.default_rng(0)
        s = rng.standard_normal((1000, 2))
        ess = effective_sample_size(s)
        # ESS should be at least 50% of N for truly IID samples
        assert np.all(ess > 500)

    def test_correlated_ess_less_than_iid(self):
        N = 1000
        iid = _iid_samples(N=N, D=1, seed=0)
        corr = _correlated_samples(N=N, D=1, rho=0.9, seed=0)
        ess_iid = effective_sample_size(iid)
        ess_corr = effective_sample_size(corr)
        assert np.all(ess_iid > ess_corr)

    def test_ess_bounded_between_1_and_n(self):
        s = _iid_samples(N=200, D=4)
        ess = effective_sample_size(s)
        assert np.all(ess >= 1.0)
        assert np.all(ess <= 200.0)

    def test_all_positive(self):
        s = _correlated_samples(N=300, D=2)
        ess = effective_sample_size(s)
        assert np.all(ess > 0)


# ---------------------------------------------------------------------------
# gelman_rubin
# ---------------------------------------------------------------------------

class TestGelmanRubin:
    def test_converged_chains_near_one(self):
        # Multiple IID chains from same distribution → R-hat ≈ 1.0
        rng = np.random.default_rng(0)
        chains = [rng.standard_normal((200, 2)) for _ in range(4)]
        r_hat = gelman_rubin(chains)
        np.testing.assert_allclose(r_hat, 1.0, atol=0.15)

    def test_diverged_chains_above_one(self):
        # Chains from very different distributions → R-hat >> 1.0
        rng = np.random.default_rng(0)
        chains = [
            rng.standard_normal((200, 1)) + offset
            for offset in [0, 10, 20, 30]
        ]
        r_hat = gelman_rubin(chains)
        assert np.all(r_hat > 1.1)

    def test_shape(self):
        rng = np.random.default_rng(0)
        chains = [rng.standard_normal((100, 3)) for _ in range(3)]
        assert gelman_rubin(chains).shape == (3,)

    def test_single_chain_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            gelman_rubin([_iid_samples()])

    def test_mismatched_shapes_raises(self):
        rng = np.random.default_rng(0)
        chains = [rng.standard_normal((100, 2)), rng.standard_normal((200, 2))]
        with pytest.raises(ValueError, match="same shape"):
            gelman_rubin(chains)


# ---------------------------------------------------------------------------
# corner_plot
# ---------------------------------------------------------------------------

class TestCornerPlot:
    def test_returns_figure(self):
        s = _iid_samples(N=200, D=2)
        fig = corner_plot(s)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_param_names(self):
        s = _iid_samples(N=200, D=2)
        fig = corner_plot(s, param_names=["m", "c"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_truths(self):
        s = _iid_samples(N=200, D=2)
        fig = corner_plot(s, truths=[0.0, 0.0])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        s = _iid_samples(N=200, D=2)
        out = tmp_path / "corner.png"
        fig = corner_plot(s, output_path=str(out))
        assert out.exists()
        plt.close(fig)

    def test_single_param(self):
        s = _iid_samples(N=200, D=1)
        fig = corner_plot(s, param_names=["x"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
