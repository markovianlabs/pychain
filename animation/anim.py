"""
Animated real-time visualisation of MCMC sampling for a linear model.

Usage
-----
    python anim.py [alpha] [delay]

    alpha  — proposal scale factor (default: 1.0)
    delay  — animation pause between frames in seconds (default: 0.001)
"""

import sys
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pychain.mcmc import MCMC

matplotlib.rc("xtick", labelsize=14)
matplotlib.rc("ytick", labelsize=14)

logger = logging.getLogger(__name__)


class Anim(MCMC):
    """
    MCMC sampler for a linear model with real-time animated visualisation.

    Parameters
    ----------
    m : float
        Fiducial slope.
    c : float
        Fiducial intercept.
    RedStd : float
        Half-range of the uniform noise added to the synthetic data.
    alpha : float
        Proposal scale factor.
    delay : float
        Pause duration (seconds) between animation frames.
    """

    def __init__(
        self,
        m: float = 5.0,
        c: float = 25.0,
        RedStd: float = 15.0,
        alpha: float = 1.5,
        delay: float = 0.001,
    ) -> None:
        super().__init__(
            TargetAcceptedPoints=5000,
            NumberOfParams=2,
            Mins=[0.0, 20.0],
            Maxs=[10.0, 30.0],
            SDs=[1.0, 2.0],
            alpha=alpha,
            write2file=True,
            outputfilename="chain.mcmc",
            randomseed=250192,
        )

        self.m = m
        self.c = c
        self.X = np.linspace(-10, 10, 25)
        self.delta = np.random.uniform(low=-RedStd, high=RedStd, size=len(self.X))
        self.Y = (m * self.X + c) + self.delta
        self.delay = delay

    def FittingFunction(self, Params: np.ndarray) -> np.ndarray:
        """
        Evaluate the linear model.

        Parameters
        ----------
        Params : np.ndarray
            [m, c].

        Returns
        -------
        np.ndarray
            Model values y = m*X + c.
        """
        return Params[0] * self.X + Params[1]

    def chisquare(self, Params: np.ndarray) -> float:
        """
        Compute chi-square between data and the linear model.

        Parameters
        ----------
        Params : np.ndarray
            [m, c].

        Returns
        -------
        float
            Sum of squared residuals normalised by noise.
        """
        residuals = (self.Y - self.FittingFunction(Params)) / self.delta
        return float(np.sum(residuals ** 2))

    def MainChain(self) -> float:
        """
        Run the MCMC chain with live plot updates.

        Returns
        -------
        float
            Acceptance ratio.
        """
        fig, axarr = plt.subplots(1, 2, figsize=(16, 7))

        axarr[1].set_xlim(4, 6)
        axarr[1].set_ylim(22, 28)
        axarr[1].set_xlabel(r"$\mathtt{m}$", fontsize=22)
        axarr[1].set_ylabel(r"$\mathtt{c}$", fontsize=22)

        OldStep = self.FirstStep()
        Oldchi2 = self.chisquare(OldStep)
        Bestchi2 = Oldchi2
        BestStep = OldStep.copy()

        multiplicity = 0
        acceptedpoints = 0
        xlist: list[float] = []
        ylist: list[float] = []

        i = 0
        while True:
            i += 1
            if acceptedpoints == self.TargetAcceptedPoints:
                break

            multiplicity += 1

            NewStep = self.NextStep(OldStep)
            Newchi2 = self.chisquare(NewStep)

            GoodPoint = self.MetropolisHastings(Oldchi2, Newchi2)

            if Newchi2 < 2 * len(self.Y):
                self.CovMat = self.alpha * np.diag(self.SD ** 2)

            if GoodPoint:
                if Newchi2 < Bestchi2:
                    BestStep = NewStep.copy()
                    Bestchi2 = Newchi2
                    logger.info("New best chi2=%.5f params=%s", Bestchi2, NewStep)

                acceptedpoints += 1
                multiplicity = 0

                axarr[0].clear()
                axarr[0].set_xlim(-10, 10)
                axarr[0].set_ylim(-50, 100)
                axarr[0].errorbar(
                    self.X, self.Y, np.abs(self.delta),
                    color="k", ms=8, ls="", marker="s",
                )
                axarr[0].plot(self.X, self.FittingFunction(OldStep), "k", ls="-", lw=2)
                axarr[0].set_title(
                    r"$\mathtt{Step:\ %i,\ AccPoints: %i}$" % (i + 1, acceptedpoints),
                    fontsize=22,
                )
                axarr[0].set_xlabel(r"$\mathtt{X}$", fontsize=22)
                axarr[0].set_ylabel(r"$\mathtt{Y}$", fontsize=22)

                xlist.append(OldStep[0])
                ylist.append(OldStep[1])
                axarr[1].plot(xlist[-3:-1], ylist[-3:-1], "k", lw=0.2)
                axarr[1].set_title(
                    r"$\mathtt{m=%1.3f,\ c=%1.3f,\ \chi^2=%1.3f}$"
                    % (OldStep[0], OldStep[1], Oldchi2),
                    fontsize=22,
                )

                plt.pause(self.delay)
                plt.draw()

                OldStep = NewStep
                Oldchi2 = Newchi2

        acceptance_ratio = acceptedpoints / i
        logger.info("Best chi2: %.5f | Acceptance ratio: %.5f", Bestchi2, acceptance_ratio)

        # Final frame showing best fit
        axarr[0].clear()
        axarr[0].set_xlim(-10, 10)
        axarr[0].set_ylim(-50, 100)
        axarr[0].errorbar(
            self.X, self.Y, np.abs(self.delta),
            color="k", ms=8, ls="", marker="s",
        )
        axarr[0].plot(self.X, self.FittingFunction(BestStep), "k", ls="-", lw=2)
        axarr[0].set_title(
            r"$\mathtt{Step:\ %i,\ AccPoints: %i}$" % (i + 1, acceptedpoints),
            fontsize=22,
        )
        axarr[0].set_xlabel(r"$\mathtt{X}$", fontsize=22)
        axarr[0].set_ylabel(r"$\mathtt{Y}$", fontsize=22)

        axarr[1].plot(BestStep[0], BestStep[1], "or", ms=12, label=r"$\mathtt{BestFit}$")
        axarr[1].plot(self.m, self.c, "og", ms=12, label=r"$\mathtt{Fiducial}$")
        axarr[1].legend(loc=1, fontsize=18, numpoints=1)
        axarr[1].set_title(
            r"$\mathtt{m=%1.3f,\ c=%1.3f,\ \chi^2=%1.3f}$"
            % (BestStep[0], BestStep[1], Bestchi2),
            fontsize=22,
        )

        plt.show()
        return acceptance_ratio


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) == 1:
        alpha = 1.0
        delay = 0.001
    elif len(sys.argv) == 3:
        alpha = float(sys.argv[1])
        delay = float(sys.argv[2])
    else:
        print("Usage: python anim.py [alpha] [delay]")
        sys.exit(1)

    co = Anim(alpha=alpha, delay=delay)
    co.MainChain()
