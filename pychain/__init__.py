"""
PyChain — Markov Chain Monte Carlo (MCMC) sampling using the Metropolis-Hastings algorithm.
"""

from pychain.mcmc import MCMC, ChainResult
from pychain import diagnostics

__all__ = ["MCMC", "ChainResult", "diagnostics"]
__version__ = "0.1.0"
