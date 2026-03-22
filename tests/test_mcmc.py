"""
Unit tests for pychain.mcmc — MCMC base class and ChainResult.
"""

import pytest
import numpy as np
from pychain.mcmc import MCMC, ChainResult


# ---------------------------------------------------------------------------
# Minimal concrete subclass used across all unit tests
# ---------------------------------------------------------------------------

class FixedChi2(MCMC):
    """Always returns a fixed chi-square value. Useful for testing MH criterion."""

    def __init__(self, fixed_chi2: float = 1.0, **kwargs):
        defaults = dict(
            NumberOfParams=2,
            Mins=[0.0, 0.0],
            Maxs=[10.0, 10.0],
            SDs=[1.0, 1.0],
            alpha=1.0,
            randomseed=42,
        )
        defaults.update(kwargs)
        super().__init__(**defaults)
        self._fixed_chi2 = fixed_chi2

    def chisquare(self, Params: np.ndarray) -> float:
        return self._fixed_chi2


class ParabolicChi2(MCMC):
    """Chi-square is the squared distance from [5, 5]. Used for integration tests."""

    def __init__(self, **kwargs):
        defaults = dict(
            NumberOfParams=2,
            Mins=[0.0, 0.0],
            Maxs=[10.0, 10.0],
            SDs=[1.0, 1.0],
            alpha=0.5,
            randomseed=42,
            EstimateCovariance=False,
        )
        defaults.update(kwargs)
        super().__init__(**defaults)

    def chisquare(self, Params: np.ndarray) -> float:
        return float(np.sum((Params - 5.0) ** 2))


# ---------------------------------------------------------------------------
# Abstract base class enforcement
# ---------------------------------------------------------------------------

class TestAbstractBase:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError, match="abstract"):
            MCMC(NumberOfParams=2, Mins=[0.0, 0.0], Maxs=[1.0, 1.0], SDs=[1.0, 1.0])

    def test_subclass_without_chisquare_cannot_instantiate(self):
        class Incomplete(MCMC):
            pass

        with pytest.raises(TypeError, match="abstract"):
            Incomplete(NumberOfParams=2, Mins=[0.0, 0.0], Maxs=[1.0, 1.0], SDs=[1.0, 1.0])


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def _make(self, **kwargs) -> FixedChi2:
        return FixedChi2(**kwargs)

    def test_mins_wrong_length(self):
        with pytest.raises(ValueError, match="Mins must have length"):
            self._make(NumberOfParams=2, Mins=[0.0], Maxs=[1.0, 1.0], SDs=[1.0, 1.0])

    def test_maxs_wrong_length(self):
        with pytest.raises(ValueError, match="Maxs must have length"):
            self._make(NumberOfParams=2, Mins=[0.0, 0.0], Maxs=[1.0], SDs=[1.0, 1.0])

    def test_sds_wrong_length(self):
        with pytest.raises(ValueError, match="SDs must have length"):
            self._make(NumberOfParams=2, Mins=[0.0, 0.0], Maxs=[1.0, 1.0], SDs=[1.0])

    def test_maxs_not_greater_than_mins(self):
        with pytest.raises(ValueError, match="strictly greater"):
            self._make(NumberOfParams=2, Mins=[1.0, 0.0], Maxs=[0.0, 1.0], SDs=[1.0, 1.0])

    def test_maxs_equal_to_mins_rejected(self):
        with pytest.raises(ValueError, match="strictly greater"):
            self._make(NumberOfParams=1, Mins=[1.0], Maxs=[1.0], SDs=[1.0])

    def test_negative_sds(self):
        with pytest.raises(ValueError, match="positive"):
            self._make(NumberOfParams=2, Mins=[0.0, 0.0], Maxs=[1.0, 1.0], SDs=[-1.0, 1.0])

    def test_zero_sds(self):
        with pytest.raises(ValueError, match="positive"):
            self._make(NumberOfParams=2, Mins=[0.0, 0.0], Maxs=[1.0, 1.0], SDs=[0.0, 1.0])

    def test_negative_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            self._make(alpha=-0.1)

    def test_zero_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            self._make(alpha=0.0)

    def test_zero_target_accepted_points(self):
        with pytest.raises(ValueError, match="TargetAcceptedPoints"):
            self._make(TargetAcceptedPoints=0)

    def test_negative_target_accepted_points(self):
        with pytest.raises(ValueError, match="TargetAcceptedPoints"):
            self._make(TargetAcceptedPoints=-10)

    def test_covnum_too_small(self):
        with pytest.raises(ValueError, match="CovNum"):
            self._make(CovNum=1)

    def test_negative_goodchi2(self):
        with pytest.raises(ValueError, match="goodchi2"):
            self._make(goodchi2=-5.0)


# ---------------------------------------------------------------------------
# FirstStep
# ---------------------------------------------------------------------------

class TestFirstStep:
    def test_shape(self):
        sampler = FixedChi2(NumberOfParams=3, Mins=[0.0]*3, Maxs=[1.0]*3, SDs=[0.1]*3)
        step = sampler.FirstStep()
        assert step.shape == (3,)

    def test_within_bounds(self):
        sampler = FixedChi2(
            NumberOfParams=2, Mins=[2.0, -5.0], Maxs=[8.0, 5.0], SDs=[1.0, 1.0]
        )
        for _ in range(50):
            step = sampler.FirstStep()
            assert np.all(step >= sampler.mins)
            assert np.all(step <= sampler.maxs)

    def test_returns_ndarray(self):
        sampler = FixedChi2()
        assert isinstance(sampler.FirstStep(), np.ndarray)


# ---------------------------------------------------------------------------
# NextStep
# ---------------------------------------------------------------------------

class TestNextStep:
    def test_within_bounds(self):
        sampler = FixedChi2(
            NumberOfParams=2, Mins=[0.0, 0.0], Maxs=[10.0, 10.0], SDs=[0.5, 0.5],
            randomseed=0,
        )
        current = np.array([5.0, 5.0])
        for _ in range(100):
            proposal = sampler.NextStep(current)
            assert np.all(proposal >= sampler.mins)
            assert np.all(proposal <= sampler.maxs)

    def test_shape_preserved(self):
        sampler = FixedChi2(NumberOfParams=3, Mins=[0.0]*3, Maxs=[1.0]*3, SDs=[0.1]*3)
        current = np.array([0.5, 0.5, 0.5])
        assert sampler.NextStep(current).shape == (3,)

    def test_returns_ndarray(self):
        sampler = FixedChi2()
        current = np.array([5.0, 5.0])
        assert isinstance(sampler.NextStep(current), np.ndarray)


# ---------------------------------------------------------------------------
# MetropolisHastings
# ---------------------------------------------------------------------------

class TestMetropolisHastings:
    def test_better_step_always_accepted(self):
        sampler = FixedChi2(randomseed=0)
        # If new chi2 is much lower, likelihood ratio >> 1 → always accepted
        for _ in range(50):
            assert sampler.MetropolisHastings(Oldchi2=100.0, Newchi2=1.0) is True

    def test_much_worse_step_almost_always_rejected(self):
        sampler = FixedChi2(randomseed=0)
        # exp(-(1000 - 1) / 2) ≈ 0 → essentially always rejected
        results = [sampler.MetropolisHastings(Oldchi2=1.0, Newchi2=1000.0) for _ in range(100)]
        assert sum(results) == 0  # all rejected

    def test_equal_chi2_accepted_about_half(self):
        sampler = FixedChi2(randomseed=42)
        # likelihood_ratio == 1.0, accepted iff uniform < 1.0 → always accepted
        results = [sampler.MetropolisHastings(Oldchi2=5.0, Newchi2=5.0) for _ in range(100)]
        assert all(results)  # ratio=1.0, always >= uniform in [0,1)

    def test_returns_bool(self):
        sampler = FixedChi2()
        result = sampler.MetropolisHastings(1.0, 2.0)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# MainChain
# ---------------------------------------------------------------------------

class TestMainChain:
    def test_returns_chain_result(self):
        sampler = ParabolicChi2(TargetAcceptedPoints=50)
        result = sampler.MainChain()
        assert isinstance(result, ChainResult)

    def test_accepted_steps_matches_target(self):
        target = 100
        sampler = ParabolicChi2(TargetAcceptedPoints=target)
        result = sampler.MainChain()
        assert result.accepted_steps == target

    def test_acceptance_ratio_in_range(self):
        sampler = ParabolicChi2(TargetAcceptedPoints=200)
        result = sampler.MainChain()
        assert 0.0 < result.acceptance_ratio <= 1.0

    def test_best_params_within_bounds(self):
        sampler = ParabolicChi2(TargetAcceptedPoints=200)
        result = sampler.MainChain()
        assert np.all(result.best_params >= sampler.mins)
        assert np.all(result.best_params <= sampler.maxs)

    def test_best_chi2_is_minimum(self):
        sampler = ParabolicChi2(TargetAcceptedPoints=200)
        result = sampler.MainChain()
        # best_chi2 should be consistent with best_params
        assert abs(result.best_chi2 - sampler.chisquare(result.best_params)) < 1e-10

    def test_total_steps_geq_accepted(self):
        sampler = ParabolicChi2(TargetAcceptedPoints=100)
        result = sampler.MainChain()
        assert result.total_steps >= result.accepted_steps

    def test_write2file(self, tmp_path):
        output = tmp_path / "chain.mcmc"
        sampler = ParabolicChi2(
            TargetAcceptedPoints=20,
            write2file=True,
            outputfilename=str(output),
        )
        result = sampler.MainChain()
        assert output.exists()
        lines = output.read_text().strip().split("\n")
        assert len(lines) == result.accepted_steps


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_contains_class_name(self):
        sampler = FixedChi2()
        assert "FixedChi2" in repr(sampler)

    def test_repr_contains_key_params(self):
        sampler = FixedChi2(alpha=0.5, goodchi2=10.0)
        r = repr(sampler)
        assert "alpha=0.5" in r
        assert "goodchi2=10.0" in r
