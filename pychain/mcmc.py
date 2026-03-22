"""
Markov Chain Monte Carlo (MCMC) sampling using the Metropolis-Hastings algorithm.

Copyright: MarkovianLabs
"""

import math
import logging
from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np

from pychain._utils import norm_ppf as _norm_ppf

__author__ = "Irshad Mohammed <creativeishu@gmail.com>"

logger = logging.getLogger(__name__)


class ChainResult(NamedTuple):
    """Results returned by :meth:`MCMC.MainChain`.

    Attributes
    ----------
    acceptance_ratio : float
        Fraction of proposed steps that were accepted.
    best_chi2 : float
        Lowest chi-square value encountered during the chain.
    best_params : np.ndarray
        Parameter vector corresponding to ``best_chi2``.
    total_steps : int
        Total number of proposed steps (accepted + rejected).
    accepted_steps : int
        Total number of accepted steps.
    samples : np.ndarray
        All accepted parameter vectors, shape ``(accepted_steps, NumberOfParams)``.
        **Important**: the first ``n_adapt_samples`` rows were collected while the
        proposal covariance was still adapting and should be treated as burn-in.
        Use ``diagnostics.burn_in(result.samples, result.n_adapt_samples)`` to
        discard them before analysis.
    n_adapt_samples : int
        Number of accepted samples collected during the adaptation phase.
        These should be discarded as additional burn-in — see ``samples`` above.
    """

    acceptance_ratio: float
    best_chi2: float
    best_params: np.ndarray
    total_steps: int
    accepted_steps: int
    samples: np.ndarray
    n_adapt_samples: int


class MCMC(ABC):
    """
    Abstract base class for MCMC sampling using the Metropolis-Hastings algorithm.

    Subclasses must implement :meth:`chisquare`.

    Parameters
    ----------
    TargetAcceptedPoints : int
        Number of accepted samples to collect before stopping. Default: 10000.
    NumberOfParams : int
        Dimensionality of the parameter space.
    Mins : list[float] | None
        Lower bounds for each parameter (uniform prior).
        Must have length equal to ``NumberOfParams``.
    Maxs : list[float] | None
        Upper bounds for each parameter (uniform prior).
        Must be element-wise strictly greater than ``Mins``.
    SDs : list[float] | None
        Initial proposal standard deviations for each parameter (all > 0).
    alpha : float
        Scaling factor applied to the proposal covariance matrix. Must be > 0.
    write2file : bool
        If True, accepted samples are written to ``outputfilename``. Default: False.
    outputfilename : str
        Path of the output file when ``write2file`` is True.
    randomseed : int
        Seed for NumPy's random number generator for reproducibility.
    debug : bool
        If True, logs detailed per-step information at DEBUG level. Default: False.
    EstimateCovariance : bool
        If True, the proposal covariance is updated from the first ``CovNum``
        accepted samples with chi2 < ``goodchi2``. Default: True.
        Samples collected during this adaptation phase should be discarded
        as burn-in (see ``ChainResult.n_adapt_samples``).
    CovNum : int
        Number of good samples used to estimate the proposal covariance. Must be > 1.
    goodchi2 : float
        Chi-square threshold that defines a "good" sample for covariance estimation
        and for the one-time proposal scale-down.

        **How to choose**: for a well-specified model with ``n_data`` observations
        and ``n_params`` free parameters, the chi-square statistic follows a
        chi2(df = n_data - n_params) distribution under the null. A sensible
        threshold is the 95th–99th percentile of that distribution. Use the
        convenience method :meth:`suggested_goodchi2` to compute this
        automatically. Default: 35.0 (suitable for ~23 degrees of freedom).
    """

    def __init__(
        self,
        TargetAcceptedPoints: int = 10000,
        NumberOfParams: int = 2,
        Mins: list[float] | None = None,
        Maxs: list[float] | None = None,
        SDs: list[float] | None = None,
        alpha: float = 1.0,
        write2file: bool = False,
        outputfilename: str = "chain.mcmc",
        randomseed: int = 250192,
        debug: bool = False,
        EstimateCovariance: bool = True,
        CovNum: int = 100,
        goodchi2: float = 35.0,
    ) -> None:
        # --- Resolve mutable defaults ---
        if Mins is None:
            Mins = [0.0, -1.0]
        if Maxs is None:
            Maxs = [2.0, 1.0]
        if SDs is None:
            SDs = [1.0, 1.0]

        # --- Validate scalar parameters ---
        if not isinstance(TargetAcceptedPoints, int) or TargetAcceptedPoints < 1:
            raise ValueError("TargetAcceptedPoints must be a positive integer.")
        if not isinstance(NumberOfParams, int) or NumberOfParams < 1:
            raise ValueError("NumberOfParams must be a positive integer.")
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if CovNum <= 1:
            raise ValueError("CovNum must be greater than 1.")
        if goodchi2 <= 0:
            raise ValueError("goodchi2 must be positive.")

        # --- Validate array parameters ---
        if len(Mins) != NumberOfParams:
            raise ValueError(f"Mins must have length {NumberOfParams}, got {len(Mins)}.")
        if len(Maxs) != NumberOfParams:
            raise ValueError(f"Maxs must have length {NumberOfParams}, got {len(Maxs)}.")
        if len(SDs) != NumberOfParams:
            raise ValueError(f"SDs must have length {NumberOfParams}, got {len(SDs)}.")

        mins_arr = np.array(Mins, dtype=float)
        maxs_arr = np.array(Maxs, dtype=float)
        sds_arr = np.array(SDs, dtype=float)

        if np.any(maxs_arr <= mins_arr):
            raise ValueError(
                "Each element of Maxs must be strictly greater than the "
                "corresponding element of Mins."
            )
        if np.any(sds_arr <= 0):
            raise ValueError("All elements of SDs must be positive.")

        # --- Initialise RNG ---
        np.random.seed(randomseed)

        # --- Store attributes ---
        self.TargetAcceptedPoints = TargetAcceptedPoints
        self.NumberOfParams = NumberOfParams
        self.mins = mins_arr
        self.maxs = maxs_arr
        self.SD = sds_arr
        self.alpha = alpha
        self.CovMat: np.ndarray = 100.0 * self.alpha * np.diag(self.SD ** 2)

        self.write2file = write2file
        self.outputfilename = outputfilename
        self.debug = debug
        self.EstimateCovariance = EstimateCovariance
        self.CovNum = CovNum
        self.goodchi2 = goodchi2

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"NumberOfParams={self.NumberOfParams}, "
            f"TargetAcceptedPoints={self.TargetAcceptedPoints}, "
            f"alpha={self.alpha}, "
            f"goodchi2={self.goodchi2})"
        )

    @staticmethod
    def suggested_goodchi2(
        n_data: int, n_params: int, percentile: float = 0.99
    ) -> float:
        """
        Suggest a ``goodchi2`` threshold from the chi-square distribution.

        For a well-specified model, chi2 ~ chi2(df = n_data - n_params).
        This returns the requested percentile of that distribution using the
        Wilson-Hilferty normal approximation (accurate to < 1% for df >= 3).

        Parameters
        ----------
        n_data : int
            Number of data points.
        n_params : int
            Number of free model parameters.
        percentile : float
            Desired percentile in (0, 1). Default: 0.99.

        Returns
        -------
        float
            Suggested goodchi2 threshold.

        Examples
        --------
        >>> MCMC.suggested_goodchi2(n_data=25, n_params=2, percentile=0.99)
        40.29  # approximate
        """
        if not (0 < percentile < 1):
            raise ValueError("percentile must be in (0, 1).")
        df = max(n_data - n_params, 1)
        # Wilson-Hilferty approximation for chi-square quantile
        z_p = float(_norm_ppf(percentile))
        val = df * (1.0 - 2.0 / (9.0 * df) + z_p * math.sqrt(2.0 / (9.0 * df))) ** 3
        return float(max(val, float(df)))

    def FirstStep(self) -> np.ndarray:
        """
        Draw a random initial point uniformly within the parameter bounds.

        Returns
        -------
        np.ndarray
            Initial parameter vector of shape ``(NumberOfParams,)``.
        """
        return self.mins + np.random.uniform(size=self.NumberOfParams) * (
            self.maxs - self.mins
        )

    def NextStep(self, Oldstep: np.ndarray) -> np.ndarray:
        """
        Propose the next step via a multivariate normal centred on the current step.

        The proposal is **not** clipped or resampled here. Enforcement of the
        uniform prior happens in :meth:`MainChain` by rejecting out-of-bounds
        proposals *before* evaluating chi-square. This preserves detailed
        balance — resampling inside the proposal would break it near boundaries.

        Parameters
        ----------
        Oldstep : np.ndarray
            Current parameter vector of shape ``(NumberOfParams,)``.

        Returns
        -------
        np.ndarray
            Proposed parameter vector (may be outside prior bounds).
        """
        return np.random.multivariate_normal(Oldstep, self.CovMat)

    def MetropolisHastings(self, Oldchi2: float, Newchi2: float) -> bool:
        """
        Apply the Metropolis-Hastings acceptance criterion.

        The acceptance probability is ``min(1, exp(-(Newchi2 - Oldchi2) / 2))``.

        Parameters
        ----------
        Oldchi2 : float
            Chi-square of the current step.
        Newchi2 : float
            Chi-square of the proposed step.

        Returns
        -------
        bool
            True if the proposed step is accepted, False otherwise.
        """
        delta = Newchi2 - Oldchi2
        # Better or equal step: always accept (avoids overflow in exp).
        if delta <= 0:
            return True
        return bool(np.exp(-delta / 2.0) >= np.random.uniform())

    @abstractmethod
    def chisquare(self, Params: np.ndarray) -> float:
        """
        Compute the chi-square (–2 × log-likelihood) for a parameter vector.

        Subclasses **must** override this method.

        The denominator in the chi-square must be the **known measurement
        uncertainty** (sigma), not the noise realization drawn for the data.
        For Gaussian noise with known sigma_i:

            chi2 = sum_i ((Y_i - model_i) / sigma_i)^2

        For uniform noise on [-R, R], the equivalent Gaussian sigma is
        R / sqrt(3) (the RMS of the uniform distribution).

        Parameters
        ----------
        Params : np.ndarray
            Parameter vector of shape ``(NumberOfParams,)``.

        Returns
        -------
        float
            Chi-square value (lower is better).
        """

    def MainChain(self, max_steps: int = 10_000_000) -> ChainResult:
        """
        Run the MCMC chain until ``TargetAcceptedPoints`` samples are collected
        or ``max_steps`` total proposals are made.

        **Uniform prior enforcement**: proposals outside ``[Mins, Maxs]`` are
        rejected *before* calling ``chisquare()``, preserving detailed balance.
        This differs from rejection-resampling inside the proposal, which would
        break detailed balance near boundaries.

        **Burn-in**: the returned ``ChainResult.samples`` includes all accepted
        samples, but the first ``ChainResult.n_adapt_samples`` were generated
        while the proposal covariance was still adapting and should be discarded:

            samples = diagnostics.burn_in(result.samples, result.n_adapt_samples)

        Parameters
        ----------
        max_steps : int
            Hard upper limit on total proposed steps. Prevents infinite loops
            when acceptance rate is very low. Default: 10,000,000.

        Returns
        -------
        ChainResult
            Named tuple with all chain outputs.
        """
        multiplicity = 0
        accepted = 0
        icov = 0
        one_time_update_cov = True
        est_cov_list: list[np.ndarray] = []
        all_samples: list[np.ndarray] = []
        adapt_end_count = 0          # accepted count when adaptation finishes
        adaptation_active = self.EstimateCovariance  # track if we started adapting

        OldStep = self.FirstStep()
        Oldchi2 = self.chisquare(OldStep)
        Bestchi2 = Oldchi2
        BestStep = OldStep.copy()

        step = 0

        with (open(self.outputfilename, "w") if self.write2file else _NullContext()) as outfile:
            write_fmt = "%1.6f \t" * self.NumberOfParams if self.write2file else ""

            while True:
                step += 1

                if accepted == self.TargetAcceptedPoints:
                    break

                # Guard against very low acceptance rates / degenerate chains
                if step >= max_steps:
                    logger.warning(
                        "max_steps=%d reached before collecting %d accepted samples "
                        "(got %d). Consider widening bounds, increasing SDs, or "
                        "reducing TargetAcceptedPoints.",
                        max_steps, self.TargetAcceptedPoints, accepted,
                    )
                    break

                if step % 1000 == 0:
                    logger.info(
                        "Step: %d | Accepted: %d / %d",
                        step, accepted, self.TargetAcceptedPoints,
                    )

                multiplicity += 1
                NewStep = self.NextStep(OldStep)

                # --- Enforce uniform prior (Fix 1: preserves detailed balance) ---
                # Out-of-bounds proposals are rejected here, not resampled inside
                # NextStep. Resampling would create an asymmetric proposal and
                # break detailed balance near the boundary.
                if np.any(NewStep < self.mins) or np.any(NewStep > self.maxs):
                    continue

                Newchi2 = self.chisquare(NewStep)

                if self.debug:
                    fmt = self.NumberOfParams * "{:10f} "
                    logger.debug(
                        "Step %d | Accepted %d | Old chi2=%.4f %s | New chi2=%.4f %s",
                        step, accepted,
                        Oldchi2, fmt.format(*OldStep),
                        Newchi2, fmt.format(*NewStep),
                    )

                # Scale down proposal once we first reach a good region
                if Newchi2 < self.goodchi2 and one_time_update_cov:
                    self.CovMat = self.alpha * np.diag(self.SD ** 2)
                    one_time_update_cov = False

                if self.MetropolisHastings(Oldchi2, Newchi2):
                    accepted += 1
                    multiplicity = 0
                    OldStep = NewStep
                    Oldchi2 = Newchi2
                    all_samples.append(NewStep.copy())

                    # Adaptive covariance estimation
                    if self.EstimateCovariance and icov < self.CovNum and Newchi2 < self.goodchi2:
                        icov += 1
                        est_cov_list.append(NewStep)
                        logger.info("Covariance estimation: %d / %d", icov, self.CovNum)

                    if self.EstimateCovariance and icov == self.CovNum and Newchi2 < self.goodchi2:
                        logger.info("Covariance estimated — updating proposal distribution.")
                        self.CovMat = np.cov(np.array(est_cov_list).T)
                        logger.debug("Estimated covariance matrix:\n%s", self.CovMat)
                        self.EstimateCovariance = False
                        adapt_end_count = accepted
                        logger.info(
                            "Adaptation complete after %d accepted samples. "
                            "Treat these as additional burn-in.",
                            adapt_end_count,
                        )

                    # Track global best
                    if Newchi2 < Bestchi2:
                        Bestchi2 = Newchi2
                        BestStep = NewStep.copy()
                        fmt = self.NumberOfParams * "{:10f} "
                        logger.info(
                            "New best chi2=%.5f at step=%d accepted=%d | params=%s",
                            Bestchi2, step, accepted, fmt.format(*BestStep),
                        )

                    if self.write2file:
                        print(
                            f"{step}\t{Newchi2:.6f}\t{multiplicity}\t"
                            + (write_fmt % tuple(NewStep)),
                            file=outfile,
                        )

        acceptance_ratio = accepted / step if step > 0 else 0.0

        # If adaptation was enabled but never completed (chain ended early),
        # all samples should be treated as adaptation-phase samples.
        if adaptation_active and self.EstimateCovariance:
            adapt_end_count = accepted
            logger.warning(
                "Adaptation did not complete (%d good samples collected, "
                "%d required). All %d samples should be treated as burn-in.",
                icov, self.CovNum, accepted,
            )

        logger.info(
            "Chain complete | Best chi2: %.5f | Acceptance ratio: %.5f | "
            "Adapt burn-in samples: %d",
            Bestchi2, acceptance_ratio, adapt_end_count,
        )

        return ChainResult(
            acceptance_ratio=acceptance_ratio,
            best_chi2=Bestchi2,
            best_params=BestStep,
            total_steps=step,
            accepted_steps=accepted,
            samples=np.array(all_samples) if all_samples else np.empty((0, self.NumberOfParams)),
            n_adapt_samples=adapt_end_count,
        )


class _NullContext:
    """No-op context manager used when write2file is False."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        pass


if __name__ == "__main__":
    print("PyChain — MCMC sampler using the Metropolis-Hastings algorithm.")
