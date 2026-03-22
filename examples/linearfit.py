"""
Example: fit a linear model y = mx + c using PyChain MCMC.
"""

import numpy as np
import matplotlib.pyplot as plt
from pychain.mcmc import MCMC


class LinearFit(MCMC):
    """
    MCMC sampler for a linear model y = mx + c.

    Parameters
    ----------
    m : float
        Fiducial slope.
    c : float
        Fiducial intercept.
    RedStd : float
        Half-range of the uniform noise U(-RedStd, RedStd) added to the data.
        The RMS (sigma) of this distribution is RedStd / sqrt(3), which is used
        as the measurement uncertainty in the chi-square — not the noise
        realisation itself.
    """

    def __init__(self, m: float = 5.0, c: float = 25.0, RedStd: float = 15.0) -> None:
        super().__init__(
            TargetAcceptedPoints=10000,
            NumberOfParams=2,
            Mins=[0.0, 20.0],
            Maxs=[10.0, 30.0],
            SDs=[1.0, 2.0],
            alpha=0.2,
            write2file=True,
            outputfilename="chain.mcmc",
            randomseed=250192,
        )

        self.X = np.linspace(-10, 10, 25)
        # Noise realisations — used only to generate the data
        delta = np.random.uniform(low=-RedStd, high=RedStd, size=len(self.X))
        self.Y = (m * self.X + c) + delta

        # Known measurement uncertainty: RMS of Uniform(-RedStd, RedStd) = RedStd/sqrt(3)
        # This is the correct denominator for chi-square — NOT the noise draw.
        self.sigma = RedStd / np.sqrt(3.0)

    def FittingFunction(self, Params: np.ndarray) -> np.ndarray:
        """
        Evaluate the linear model.

        Parameters
        ----------
        Params : np.ndarray
            [m, c] — slope and intercept.

        Returns
        -------
        np.ndarray
            Model values y = m*X + c.
        """
        return Params[0] * self.X + Params[1]

    def chisquare(self, Params: np.ndarray) -> float:
        """
        Compute chi-square between data and the linear model.

        Uses the known noise RMS (sigma = RedStd / sqrt(3)) as the denominator,
        not the noise realisations. This is statistically correct: the chi-square
        denominator must be the measurement uncertainty, not the noise draw.

        Parameters
        ----------
        Params : np.ndarray
            [m, c].

        Returns
        -------
        float
            Sum of squared normalised residuals.
        """
        residuals = (self.Y - self.FittingFunction(Params)) / self.sigma
        return float(np.sum(residuals ** 2))


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    chain = LinearFit()
    result = chain.MainChain()
    print(f"Acceptance ratio : {result.acceptance_ratio:.4f}")
    print(f"Best chi2        : {result.best_chi2:.4f}")
    print(f"Best params [m,c]: {result.best_params}")
    print(f"Adapt burn-in    : {result.n_adapt_samples} samples")
