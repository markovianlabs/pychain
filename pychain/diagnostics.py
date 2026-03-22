"""
MCMC diagnostics: autocorrelation, effective sample size, Gelman-Rubin R-hat,
burn-in, thinning, and corner plots.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Burn-in and thinning
# ---------------------------------------------------------------------------

def burn_in(samples: np.ndarray, n_burn: int) -> np.ndarray:
    """
    Discard the first ``n_burn`` samples (the burn-in period).

    Parameters
    ----------
    samples : np.ndarray
        Shape ``(N, D)`` — N samples, D parameters.
    n_burn : int
        Number of initial samples to discard.

    Returns
    -------
    np.ndarray
        Remaining samples of shape ``(N - n_burn, D)``.
    """
    if n_burn < 0:
        raise ValueError("n_burn must be non-negative.")
    if n_burn >= len(samples):
        raise ValueError(
            f"n_burn ({n_burn}) must be less than the number of samples ({len(samples)})."
        )
    return samples[n_burn:]


def thin(samples: np.ndarray, step: int) -> np.ndarray:
    """
    Thin the chain by keeping every ``step``-th sample.

    Thinning reduces autocorrelation at the cost of fewer samples.

    Parameters
    ----------
    samples : np.ndarray
        Shape ``(N, D)``.
    step : int
        Thinning interval. Must be >= 1.

    Returns
    -------
    np.ndarray
        Thinned samples of shape ``(ceil(N / step), D)``.
    """
    if step < 1:
        raise ValueError("step must be >= 1.")
    return samples[::step]


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------

def autocorrelation(samples: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """
    Compute the normalised autocorrelation function (ACF) for each parameter.

    Parameters
    ----------
    samples : np.ndarray
        Shape ``(N, D)`` — N samples, D parameters.
    max_lag : int, optional
        Maximum lag to compute. Defaults to ``N // 2``.

    Returns
    -------
    np.ndarray
        Shape ``(max_lag, D)`` — ACF at each lag for each parameter.
        Lag 0 is always 1.0.
    """
    samples = samples.reshape(-1, 1) if samples.ndim == 1 else samples
    N, D = samples.shape

    if max_lag is None:
        max_lag = N // 2
    if max_lag >= N:
        raise ValueError(f"max_lag ({max_lag}) must be less than N ({N}).")
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1.")

    centered = samples - samples.mean(axis=0)
    variance = np.var(samples, axis=0)

    # Avoid division by zero for constant parameters
    safe_variance = np.where(variance == 0, 1.0, variance)

    acf = np.zeros((max_lag, D))
    for lag in range(max_lag):
        acf[lag] = np.mean(centered[: N - lag] * centered[lag:], axis=0) / safe_variance

    return acf


# ---------------------------------------------------------------------------
# Effective Sample Size
# ---------------------------------------------------------------------------

def effective_sample_size(samples: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """
    Estimate the Effective Sample Size (ESS) for each parameter.

    Uses the initial monotone positive sequence estimator (Geyer 1992):
    sums the ACF until the first non-positive value.

    ESS = N / (1 + 2 * Σ rho_k)

    Parameters
    ----------
    samples : np.ndarray
        Shape ``(N, D)``.
    max_lag : int, optional
        Maximum lag for ACF computation.

    Returns
    -------
    np.ndarray
        ESS for each parameter, shape ``(D,)``.
        Values are clipped to ``[1, N]``.
    """
    N = len(samples)
    acf = autocorrelation(samples, max_lag=max_lag)
    D = acf.shape[1]
    ess = np.zeros(D)

    for d in range(D):
        acf_sum = 0.0
        for k in range(1, len(acf)):
            if acf[k, d] <= 0:
                break
            acf_sum += acf[k, d]
        ess[d] = N / max(1.0 + 2.0 * acf_sum, 1e-10)

    return np.clip(ess, 1.0, N)


# ---------------------------------------------------------------------------
# Gelman-Rubin R-hat
# ---------------------------------------------------------------------------

def gelman_rubin(chains: list[np.ndarray]) -> np.ndarray:
    """
    Compute the Gelman-Rubin R-hat convergence diagnostic.

    R-hat ≈ 1.0 means the chains have converged.
    R-hat > 1.1 suggests further sampling is needed.

    Requires at least 2 independent chains of the same length.

    Parameters
    ----------
    chains : list[np.ndarray]
        List of M chains, each of shape ``(N, D)``.
        All chains must have identical shapes.

    Returns
    -------
    np.ndarray
        R-hat for each parameter, shape ``(D,)``.
    """
    if len(chains) < 2:
        raise ValueError("Gelman-Rubin requires at least 2 chains.")

    shapes = {c.shape for c in chains}
    if len(shapes) > 1:
        raise ValueError(
            f"All chains must have the same shape. Got shapes: {[c.shape for c in chains]}"
        )

    chains_arr = np.array(chains)  # (M, N, D)
    M, N, _ = chains_arr.shape

    chain_means = chains_arr.mean(axis=1)       # (M, D)
    grand_mean = chain_means.mean(axis=0)        # (D,)

    # Between-chain variance
    B = N / (M - 1) * np.sum((chain_means - grand_mean) ** 2, axis=0)

    # Within-chain variance
    W = chains_arr.var(axis=1, ddof=1).mean(axis=0)  # (D,)

    # Pooled variance estimate
    var_hat = (N - 1) / N * W + B / N

    # Guard against W == 0 (degenerate chain)
    safe_W = np.where(W == 0, 1e-10, W)
    r_hat = np.sqrt(var_hat / safe_W)

    return r_hat


# ---------------------------------------------------------------------------
# Corner plot
# ---------------------------------------------------------------------------

def corner_plot(
    samples: np.ndarray,
    param_names: Optional[list[str]] = None,
    truths: Optional[list[float]] = None,
    title: str = "Parameter Posteriors",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a corner plot of the MCMC parameter posteriors.

    Uses the ``corner`` library if installed, otherwise falls back to a
    matplotlib implementation.

    Parameters
    ----------
    samples : np.ndarray
        Shape ``(N, D)``.
    param_names : list[str], optional
        Axis labels for each parameter.
    truths : list[float], optional
        True (fiducial) values to mark on the plot.
    title : str
        Figure title.
    output_path : str, optional
        If provided, save the figure to this path (PNG/PDF/etc.).

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import corner  # type: ignore[import]

        fig = corner.corner(
            samples,
            labels=param_names,
            truths=truths,
            show_titles=True,
            title_kwargs={"fontsize": 12},
        )
        fig.suptitle(title, fontsize=14, y=1.02)
        logger.debug("Corner plot created using the `corner` library.")

    except ImportError:
        logger.warning(
            "`corner` library not installed — using matplotlib fallback. "
            "Install it with: pip install corner"
        )
        fig = _matplotlib_corner(samples, param_names, truths, title)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        logger.info("Corner plot saved to %s", output_path)

    return fig


def _matplotlib_corner(
    samples: np.ndarray,
    param_names: Optional[list[str]],
    truths: Optional[list[float]],
    title: str,
) -> plt.Figure:
    """Minimal corner plot using only matplotlib (no extra dependencies)."""
    D = samples.shape[1]
    names = param_names or [f"param_{i}" for i in range(D)]

    fig, axes = plt.subplots(D, D, figsize=(3 * D, 3 * D))
    fig.suptitle(title, fontsize=14)

    # Ensure axes is always a 2D array
    if D == 1:
        axes = np.array([[axes]])

    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
            elif i == j:
                ax.hist(samples[:, i], bins=40, color="steelblue", alpha=0.7)
                if truths is not None:
                    ax.axvline(truths[i], color="red", lw=1.5, ls="--", label="truth")
                ax.set_xlabel(names[i], fontsize=10)
            else:
                ax.scatter(samples[:, j], samples[:, i], s=1, alpha=0.3, color="steelblue")
                if truths is not None:
                    ax.axvline(truths[j], color="red", lw=1.5, ls="--")
                    ax.axhline(truths[i], color="red", lw=1.5, ls="--")
                ax.set_xlabel(names[j], fontsize=10)
                ax.set_ylabel(names[i], fontsize=10)

    plt.tight_layout()
    return fig
