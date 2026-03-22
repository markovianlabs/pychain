"""
Tests for pychain.diagnostics.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pychain.diagnostics import (
    burn_in,
    thin,
    autocorrelation,
    effective_sample_size,
    tail_ess,
    gelman_rubin,
    corner_plot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iid_samples(N: int = 500, D: int = 2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, D))


def _correlated_samples(N: int = 500, D: int = 2, rho: float = 0.9, seed: int = 0) -> np.ndarray:
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
        assert burn_in(s, n_burn=20).shape == (80, 2)

    def test_zero_burn_in(self):
        s = _iid_samples(N=100)
        np.testing.assert_array_equal(burn_in(s, n_burn=0), s)

    def test_returns_correct_rows(self):
        s = np.arange(20).reshape(10, 2)
        np.testing.assert_array_equal(burn_in(s, n_burn=3), s[3:])

    def test_negative_burn_in_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            burn_in(_iid_samples(), n_burn=-1)

    def test_burn_in_gte_n_raises(self):
        with pytest.raises(ValueError, match="less than"):
            burn_in(_iid_samples(N=10), n_burn=10)

    def test_burn_in_equal_n_raises(self):
        with pytest.raises(ValueError):
            burn_in(_iid_samples(N=10), n_burn=10)


# ---------------------------------------------------------------------------
# thin
# ---------------------------------------------------------------------------

class TestThin:
    def test_step_1_returns_all(self):
        s = _iid_samples(N=100)
        np.testing.assert_array_equal(thin(s, step=1), s)

    def test_step_2_halves_samples(self):
        assert len(thin(_iid_samples(N=100), step=2)) == 50

    def test_step_10_reduces_samples(self):
        assert len(thin(_iid_samples(N=100), step=10)) == 10

    def test_returns_correct_rows(self):
        s = np.arange(20).reshape(10, 2)
        np.testing.assert_array_equal(thin(s, step=3), s[::3])

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
        acf = autocorrelation(_iid_samples(N=200, D=3), max_lag=50)
        assert acf.shape == (50, 3)

    def test_lag0_is_one(self):
        acf = autocorrelation(_iid_samples(N=200, D=2), max_lag=10)
        np.testing.assert_allclose(acf[0], 1.0, atol=1e-10)

    def test_iid_acf_near_zero_for_large_lags(self):
        rng = np.random.default_rng(42)
        s = rng.standard_normal((2000, 2))
        acf = autocorrelation(s, max_lag=5)
        assert np.abs(acf[1:]).mean() < 0.1

    def test_correlated_acf_positive_at_lag1(self):
        s = _correlated_samples(N=500, rho=0.9)
        acf = autocorrelation(s, max_lag=10)
        assert np.all(acf[1] > 0.5)

    def test_default_max_lag(self):
        acf = autocorrelation(_iid_samples(N=200))
        assert acf.shape[0] == 100

    def test_max_lag_too_large_raises(self):
        with pytest.raises(ValueError, match="max_lag"):
            autocorrelation(_iid_samples(N=100), max_lag=100)

    def test_1d_input_handled(self):
        s = np.random.default_rng(0).standard_normal(200)
        acf = autocorrelation(s, max_lag=10)
        assert acf.shape == (10, 1)


# ---------------------------------------------------------------------------
# effective_sample_size (bulk + tail)
# ---------------------------------------------------------------------------

class TestEffectiveSampleSize:
    def test_shape_bulk(self):
        ess = effective_sample_size(_iid_samples(N=500, D=3))
        assert ess.shape == (3,)

    def test_shape_tail(self):
        ess = effective_sample_size(_iid_samples(N=500, D=3), kind="tail")
        assert ess.shape == (3,)

    def test_iid_bulk_ess_near_n(self):
        rng = np.random.default_rng(0)
        s = rng.standard_normal((1000, 2))
        ess = effective_sample_size(s, kind="bulk")
        assert np.all(ess > 500)

    def test_correlated_bulk_ess_less_than_iid(self):
        iid = _iid_samples(N=1000, D=1, seed=0)
        corr = _correlated_samples(N=1000, D=1, rho=0.9, seed=0)
        assert np.all(effective_sample_size(iid) > effective_sample_size(corr))

    def test_bulk_ess_bounded(self):
        s = _iid_samples(N=200, D=4)
        ess = effective_sample_size(s)
        assert np.all(ess >= 1.0)
        assert np.all(ess <= 200.0)

    def test_tail_ess_bounded(self):
        s = _iid_samples(N=200, D=2)
        ess = effective_sample_size(s, kind="tail")
        assert np.all(ess >= 1.0)
        assert np.all(ess <= 200.0)

    def test_all_positive(self):
        s = _correlated_samples(N=300, D=2)
        assert np.all(effective_sample_size(s) > 0)

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            effective_sample_size(_iid_samples(), kind="bogus")

    def test_deprecated_max_lag_no_exception(self):
        # max_lag is silently ignored (logs a warning, raises no exception)
        result = effective_sample_size(_iid_samples(), max_lag=50)
        assert result.shape == (2,)


# ---------------------------------------------------------------------------
# tail_ess (convenience wrapper)
# ---------------------------------------------------------------------------

class TestTailEss:
    def test_shape(self):
        assert tail_ess(_iid_samples(N=200, D=3)).shape == (3,)

    def test_matches_effective_sample_size_tail(self):
        s = _iid_samples(N=300, D=2, seed=7)
        np.testing.assert_array_equal(tail_ess(s), effective_sample_size(s, kind="tail"))


# ---------------------------------------------------------------------------
# gelman_rubin (Vehtari 2021 rank-normalised split R-hat)
# ---------------------------------------------------------------------------

class TestGelmanRubin:
    def test_converged_chains_near_one(self):
        rng = np.random.default_rng(0)
        chains = [rng.standard_normal((200, 2)) for _ in range(4)]
        r_hat = gelman_rubin(chains)
        np.testing.assert_allclose(r_hat, 1.0, atol=0.2)

    def test_diverged_chains_above_one(self):
        rng = np.random.default_rng(0)
        chains = [
            rng.standard_normal((200, 1)) + offset
            for offset in [0, 10, 20, 30]
        ]
        assert np.all(gelman_rubin(chains) > 1.1)

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

    def test_too_short_chains_raise(self):
        rng = np.random.default_rng(0)
        chains = [rng.standard_normal((6, 2)) for _ in range(2)]  # 6 // 2 = 3 < 4
        with pytest.raises(ValueError, match="at least 8"):
            gelman_rubin(chains)

    def test_threshold_recommendation(self):
        # Converged chains should have R-hat < 1.01 (Vehtari 2021 threshold)
        rng = np.random.default_rng(42)
        chains = [rng.standard_normal((500, 2)) for _ in range(4)]
        r_hat = gelman_rubin(chains)
        assert np.all(r_hat < 1.05)  # generous tolerance for finite samples


# ---------------------------------------------------------------------------
# corner_plot
# ---------------------------------------------------------------------------

class TestCornerPlot:
    def test_returns_figure(self):
        fig = corner_plot(_iid_samples(N=200, D=2))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_param_names(self):
        fig = corner_plot(_iid_samples(N=200, D=2), param_names=["m", "c"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_truths(self):
        fig = corner_plot(_iid_samples(N=200, D=2), truths=[0.0, 0.0])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        out = tmp_path / "corner.png"
        fig = corner_plot(_iid_samples(N=200, D=2), output_path=str(out))
        assert out.exists()
        plt.close(fig)

    def test_single_param(self):
        fig = corner_plot(_iid_samples(N=200, D=1), param_names=["x"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
