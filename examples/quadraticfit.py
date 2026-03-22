"""
Example: fit a quadratic model y = ax^2 + bx + c using PyChain MCMC.
"""

import numpy as np
import matplotlib.pyplot as plt
from pychain.mcmc import MCMC


class QuadraticFit(MCMC):
    """
    MCMC sampler for a quadratic model y = ax^2 + bx + c.

    Parameters
    ----------
    a : float
        Fiducial quadratic coefficient.
    b : float
        Fiducial linear coefficient.
    c : float
        Fiducial intercept.
    RedStd : float
        Half-range of the uniform noise added to the synthetic data.
    """

    def __init__(
        self,
        a: float = 1.0,
        b: float = 10.0,
        c: float = 25.0,
        RedStd: float = 15.0,
    ) -> None:
        super().__init__(
            TargetAcceptedPoints=10000,
            NumberOfParams=3,
            Mins=[-5.0, 5.0, 20.0],
            Maxs=[5.0, 15.0, 30.0],
            SDs=[0.7, 4.0, 7.0],
            alpha=0.01,
            write2file=True,
            outputfilename="chain.mcmc",
            randomseed=250192,
        )

        self.X = np.linspace(0, 10, 25)
        self.delta = np.random.uniform(low=-RedStd, high=RedStd, size=len(self.X))
        self.Y = (a * self.X ** 2 + b * self.X + c) + self.delta

    def FittingFunction(self, Params: np.ndarray) -> np.ndarray:
        """
        Evaluate the quadratic model.

        Parameters
        ----------
        Params : np.ndarray
            [a, b, c] — quadratic, linear, and constant coefficients.

        Returns
        -------
        np.ndarray
            Model values y = a*X^2 + b*X + c.
        """
        return Params[0] * self.X ** 2 + Params[1] * self.X + Params[2]

    def chisquare(self, Params: np.ndarray) -> float:
        """
        Compute chi-square between data and the quadratic model.

        Parameters
        ----------
        Params : np.ndarray
            [a, b, c].

        Returns
        -------
        float
            Sum of squared residuals normalised by noise.
        """
        residuals = (self.Y - self.FittingFunction(Params)) / self.delta
        return float(np.sum(residuals ** 2))


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    chain = QuadraticFit()
    result = chain.MainChain()
    print(f"Acceptance ratio    : {result.acceptance_ratio:.4f}")
    print(f"Best chi2           : {result.best_chi2:.4f}")
    print(f"Best params [a,b,c] : {result.best_params}")
