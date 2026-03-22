"""
Markov Chain Monte Carlo (MCMC) sampling using the Metropolis-Hastings algorithm.

Copyright: MarkovianLabs
"""

import logging
from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np

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
        Use this for diagnostics (autocorrelation, ESS, corner plots, etc.).
    """

    acceptance_ratio: float
    best_chi2: float
    best_params: np.ndarray
    total_steps: int
    accepted_steps: int
    samples: np.ndarray


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
        Must have length equal to ``NumberOfParams`` and element-wise greater than ``Mins``.
    SDs : list[float] | None
        Initial proposal standard deviations for each parameter (all must be > 0).
        Must have length equal to ``NumberOfParams``.
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
        good samples (chi2 < ``goodchi2``). Default: True.
    CovNum : int
        Number of good samples used to estimate the proposal covariance. Must be > 1.
    goodchi2 : float
        Chi-square threshold below which a sample is considered a good fit. Must be > 0.
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
            raise ValueError("Each element of Maxs must be strictly greater than the corresponding element of Mins.")
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

        Proposals that fall outside the prior bounds are rejected and redrawn
        until a valid proposal is found.

        Parameters
        ----------
        Oldstep : np.ndarray
            Current parameter vector of shape ``(NumberOfParams,)``.

        Returns
        -------
        np.ndarray
            Proposed parameter vector of shape ``(NumberOfParams,)``.
        """
        proposal = np.random.multivariate_normal(Oldstep, self.CovMat)
        while np.any(proposal < self.mins) or np.any(proposal > self.maxs):
            proposal = np.random.multivariate_normal(Oldstep, self.CovMat)
        return proposal

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
        # If the new step is better (delta < 0), always accept without computing exp.
        if delta <= 0:
            return True
        likelihood_ratio = np.exp(-delta / 2.0)
        return bool(likelihood_ratio >= np.random.uniform())

    @abstractmethod
    def chisquare(self, Params: np.ndarray) -> float:
        """
        Compute the chi-square (negative log-likelihood proxy) for a parameter vector.

        Subclasses **must** override this method to define the likelihood of their model.

        Parameters
        ----------
        Params : np.ndarray
            Parameter vector of shape ``(NumberOfParams,)``.

        Returns
        -------
        float
            Chi-square value (lower is better).
        """

    def MainChain(self) -> ChainResult:
        """
        Run the MCMC chain until ``TargetAcceptedPoints`` samples are collected.

        Returns
        -------
        ChainResult
            A named tuple with acceptance ratio, best chi-square, best parameter
            vector, total steps, and accepted steps.
        """
        multiplicity = 0
        accepted = 0
        icov = 0
        one_time_update_cov = True
        est_cov_list: list[np.ndarray] = []
        all_samples: list[np.ndarray] = []

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

                if step % 1000 == 0:
                    logger.info(
                        "Step: %d | Accepted: %d / %d",
                        step, accepted, self.TargetAcceptedPoints,
                    )

                multiplicity += 1
                NewStep = self.NextStep(OldStep)
                Newchi2 = self.chisquare(NewStep)

                if self.debug:
                    fmt = self.NumberOfParams * "{:10f} "
                    logger.debug(
                        "Step %d | Accepted %d | Old chi2=%.4f %s | New chi2=%.4f %s",
                        step, accepted,
                        Oldchi2, fmt.format(*OldStep),
                        Newchi2, fmt.format(*NewStep),
                    )

                # Scale down proposal once we hit a good region
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

                    # Track global best
                    if Newchi2 < Bestchi2:
                        Bestchi2 = Newchi2
                        BestStep = NewStep.copy()
                        fmt = self.NumberOfParams * "{:10f} "
                        logger.info(
                            "New best chi2=%.5f at step=%d accepted=%d | params=%s",
                            Bestchi2, step, accepted, fmt.format(*BestStep),
                        )

                    # Write accepted sample to file
                    if self.write2file:
                        print(
                            f"{step}\t{Newchi2:.6f}\t{multiplicity}\t"
                            + (write_fmt % tuple(NewStep)),
                            file=outfile,
                        )

        acceptance_ratio = accepted / step
        logger.info(
            "Chain complete | Best chi2: %.5f | Acceptance ratio: %.5f",
            Bestchi2, acceptance_ratio,
        )

        return ChainResult(
            acceptance_ratio=acceptance_ratio,
            best_chi2=Bestchi2,
            best_params=BestStep,
            total_steps=step,
            accepted_steps=accepted,
            samples=np.array(all_samples),
        )


class _NullContext:
    """No-op context manager used when write2file is False."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        pass


if __name__ == "__main__":
    print("PyChain — MCMC sampler using the Metropolis-Hastings algorithm.")
