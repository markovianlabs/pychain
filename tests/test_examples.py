"""
Integration tests for the LinearFit and QuadraticFit example subclasses.
"""

import pytest
import numpy as np
from pychain.mcmc import ChainResult
from examples.linearfit import LinearFit
from examples.quadraticfit import QuadraticFit


# ---------------------------------------------------------------------------
# LinearFit
# ---------------------------------------------------------------------------

class TestLinearFit:
    @pytest.fixture
    def chain(self):
        return LinearFit(m=5.0, c=25.0, RedStd=15.0)

    def test_returns_chain_result(self, chain):
        assert isinstance(chain.MainChain(), ChainResult)

    def test_accepted_steps_match_target(self, chain):
        result = chain.MainChain()
        assert result.accepted_steps == chain.TargetAcceptedPoints

    def test_acceptance_ratio_reasonable(self, chain):
        result = chain.MainChain()
        assert 0.05 < result.acceptance_ratio < 0.95

    def test_best_params_within_bounds(self, chain):
        result = chain.MainChain()
        assert np.all(result.best_params >= chain.mins)
        assert np.all(result.best_params <= chain.maxs)

    def test_best_chi2_matches_best_params(self, chain):
        result = chain.MainChain()
        assert abs(result.best_chi2 - chain.chisquare(result.best_params)) < 1e-10

    def test_fitting_function_shape(self, chain):
        params = np.array([5.0, 25.0])
        assert chain.FittingFunction(params).shape == chain.X.shape

    def test_chisquare_uses_sigma_not_delta(self, chain):
        # chi-square should use sigma (fixed), so identical params give same chi2
        params = np.array([5.0, 25.0])
        chi1 = chain.chisquare(params)
        chi2 = chain.chisquare(params)
        assert chi1 == chi2

    def test_sigma_is_rms_of_uniform(self, chain):
        # sigma must be RedStd / sqrt(3), not per-point noise draw
        expected_sigma = 15.0 / np.sqrt(3.0)
        assert abs(chain.sigma - expected_sigma) < 1e-10

    def test_chisquare_positive(self, chain):
        assert chain.chisquare(np.array([5.0, 25.0])) >= 0.0

    def test_samples_shape(self, chain):
        result = chain.MainChain()
        assert result.samples.shape == (result.accepted_steps, 2)

    def test_n_adapt_samples_non_negative(self, chain):
        result = chain.MainChain()
        assert result.n_adapt_samples >= 0

    def test_reproducibility(self):
        r1 = LinearFit(m=5.0, c=25.0).MainChain()
        r2 = LinearFit(m=5.0, c=25.0).MainChain()
        assert r1.best_chi2 == r2.best_chi2
        np.testing.assert_array_equal(r1.best_params, r2.best_params)

    def test_write2file(self, tmp_path):
        chain = LinearFit(m=5.0, c=25.0)
        chain.write2file = True
        chain.outputfilename = str(tmp_path / "linear.mcmc")
        chain.TargetAcceptedPoints = 50
        result = chain.MainChain()
        from pathlib import Path
        lines = Path(chain.outputfilename).read_text().strip().split("\n")
        assert len(lines) == result.accepted_steps


# ---------------------------------------------------------------------------
# QuadraticFit
# ---------------------------------------------------------------------------

class TestQuadraticFit:
    @pytest.fixture
    def chain(self):
        return QuadraticFit(a=1.0, b=10.0, c=25.0, RedStd=15.0)

    def test_returns_chain_result(self, chain):
        assert isinstance(chain.MainChain(), ChainResult)

    def test_accepted_steps_match_target(self, chain):
        result = chain.MainChain()
        assert result.accepted_steps == chain.TargetAcceptedPoints

    def test_acceptance_ratio_reasonable(self, chain):
        result = chain.MainChain()
        assert 0.05 < result.acceptance_ratio < 0.95

    def test_best_params_within_bounds(self, chain):
        result = chain.MainChain()
        assert np.all(result.best_params >= chain.mins)
        assert np.all(result.best_params <= chain.maxs)

    def test_best_chi2_matches_best_params(self, chain):
        result = chain.MainChain()
        assert abs(result.best_chi2 - chain.chisquare(result.best_params)) < 1e-10

    def test_number_of_params(self, chain):
        assert chain.NumberOfParams == 3

    def test_fitting_function_shape(self, chain):
        params = np.array([1.0, 10.0, 25.0])
        assert chain.FittingFunction(params).shape == chain.X.shape

    def test_sigma_is_rms_of_uniform(self, chain):
        expected_sigma = 15.0 / np.sqrt(3.0)
        assert abs(chain.sigma - expected_sigma) < 1e-10

    def test_chisquare_positive(self, chain):
        assert chain.chisquare(np.array([1.0, 10.0, 25.0])) >= 0.0

    def test_samples_shape(self, chain):
        result = chain.MainChain()
        assert result.samples.shape == (result.accepted_steps, 3)

    def test_reproducibility(self):
        r1 = QuadraticFit(a=1.0, b=10.0, c=25.0).MainChain()
        r2 = QuadraticFit(a=1.0, b=10.0, c=25.0).MainChain()
        assert r1.best_chi2 == r2.best_chi2
        np.testing.assert_array_equal(r1.best_params, r2.best_params)
