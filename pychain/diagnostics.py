"""
MCMC diagnostics: autocorrelation, effective sample size, Gelman-Rubin R-hat,
burn-in, thinning, and corner plots.

References
----------
Geyer, C.J. (1992). Practical Markov Chain Monte Carlo. Statistical Science.
Vehtari, A. et al. (2021). Rank-normalization, folding, and localization:
    An improved R-hat for assessing convergence of MCMC.
    Bayesian Analysis, 16(2), 667–718.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from pychain._utils import norm_ppf as _norm_ppf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Burn-in and thinning
# ---------------------------------------------------------------------------

def burn_in(samples: np.ndarray, n_burn: int) -> np.ndarray:
    """
    Discard the first ``n_burn`` samples (the burn-in / warm-up period).

    Parameters
    ----------
    samples : np.ndarray
        Shape ``(N, D)`` — N samples, D parameters.
    n_burn : int
        Number of initial samples to discard. Use ``ChainResult.n_adapt_samples``
        as a minimum to discard the adaptive-covariance phase.

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

    **Note on thinning**: thinning reduces the *size* of the stored chain but
    does **not** improve statistical efficiency per unit of computation. Modern
    consensus (Geyer 1992, MacKay 2003) is that you lose information by
    discarding samples. The only legitimate reason to thin is to reduce memory
    or storage requirements. If you want lower autocorrelation, run a
    better-tuned chain instead.

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
# Internal helpers for rank-normalisation and FFT-based ACF
# ---------------------------------------------------------------------------

def _rank_normalize_joint(chains_1d: list[np.ndarray]) -> list[np.ndarray]:
    """
    Rank-normalize a list of 1D chains jointly (Vehtari et al. 2021).

    All chains are pooled, ranked together, then the Blom normal-score
    transform ``Phi^{-1}((r - 0.375) / (N_total + 0.25))`` is applied.
    The result is split back into separate arrays matching the input shapes.

    Parameters
    ----------
    chains_1d : list[np.ndarray]
        List of 1D arrays (one per chain / split-chain).

    Returns
    -------
    list[np.ndarray]
        Rank-normalised arrays in the same order and shapes as input.
    """
    all_vals = np.concatenate(chains_1d)
    N_total = len(all_vals)

    # Dense ranking (1-indexed), stable sort preserves order of ties
    order = np.argsort(all_vals, kind="stable")
    ranks = np.empty(N_total, dtype=float)
    ranks[order] = np.arange(1, N_total + 1, dtype=float)

    # Blom's normal-score transformation
    z_all = _norm_ppf((ranks - 0.375) / (N_total + 0.25))

    # Split back into per-chain arrays
    result: list[np.ndarray] = []
    idx = 0
    for c in chains_1d:
        n = len(c)
        result.append(z_all[idx: idx + n])
        idx += n
    return result


def _split_chains(chains: list[np.ndarray]) -> list[np.ndarray]:
    """
    Split each chain into two equal halves (for split-R-hat).

    Each chain of length N yields two chains of length N//2.
    The last sample is dropped if N is odd.
    """
    split: list[np.ndarray] = []
    for c in chains:
        half = len(c) // 2
        if half < 4:
            raise ValueError(
                f"Each chain must have at least 8 samples for split-R-hat, got {len(c)}."
            )
        split.append(c[:half])
        split.append(c[half: 2 * half])
    return split


def _rhat_1d(chains_1d: list[np.ndarray]) -> float:
    """
    Classic R-hat formula for a list of 1D chains (applied after rank-normalisation).

    Returns NaN if within-chain variance is zero (degenerate chain).
    """
    chains_arr = np.array(chains_1d)       # (M, N)
    M, N = chains_arr.shape

    chain_means = chains_arr.mean(axis=1)  # (M,)
    grand_mean = chain_means.mean()

    B = N / (M - 1) * np.sum((chain_means - grand_mean) ** 2)
    W = chains_arr.var(axis=1, ddof=1).mean()

    if W <= 0.0:
        return float("nan")

    var_hat = (N - 1) / N * W + B / N
    return float(np.sqrt(var_hat / W))


def _acf_fft(x: np.ndarray) -> np.ndarray:
    """
    Compute the full normalised ACF for a 1D array using FFT (O(N log N)).

    Zero-pads to 2N to avoid circular wrap-around. Returns an array of
    length N with ACF[0] = 1.
    """
    n = len(x)
    x_c = x - x.mean()
    f = np.fft.rfft(x_c, n=2 * n)
    acf_raw = np.fft.irfft(f * np.conj(f))[:n].real
    if acf_raw[0] == 0.0:
        return np.zeros(n)
    return acf_raw / acf_raw[0]


def _ess_1d(x: np.ndarray) -> float:
    """
    ESS for a 1D chain using FFT ACF and Geyer's initial positive sequence.

    Sums the ACF until the first non-positive value (initial positive sequence
    estimator), then applies ESS = N / (1 + 2 * sum_ACF).
    """
    n = len(x)
    acf = _acf_fft(x)
    acf_sum = 0.0
    for k in range(1, n):
        if acf[k] <= 0.0:
            break
        acf_sum += acf[k]
    return float(np.clip(n / (1.0 + 2.0 * acf_sum), 1.0, n))


# ---------------------------------------------------------------------------
# Public API: autocorrelation
# ---------------------------------------------------------------------------

def autocorrelation(samples: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """
    Compute the normalised autocorrelation function (ACF) for each parameter.

    Parameters
    ----------
    samples : np.ndarray
        Shape ``(N, D)`` — N samples, D parameters.
    max_lag : int, optional
        Maximum lag to return. Defaults to ``N // 2``.

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

    acf = np.zeros((max_lag, D))
    for d in range(D):
        full_acf = _acf_fft(samples[:, d])
        acf[:, d] = full_acf[:max_lag]

    return acf


# ---------------------------------------------------------------------------
# Public API: effective sample size (bulk + tail)
# ---------------------------------------------------------------------------

def effective_sample_size(
    samples: np.ndarray,
    kind: str = "bulk",
    max_lag: Optional[int] = None,   # deprecated; ignored (FFT computes full ACF)
) -> np.ndarray:
    """
    Estimate the Effective Sample Size (ESS) per parameter.

    Uses rank-normalisation (Vehtari et al. 2021) for robustness to non-normal
    posteriors, and FFT-based ACF computation (O(N log N)).

    Parameters
    ----------
    samples : np.ndarray
        Shape ``(N, D)``.
    kind : {'bulk', 'tail'}
        * ``'bulk'`` — ESS for the centre of the distribution (rank-normalised).
          This is the standard ESS reported by Stan and ArviZ.
        * ``'tail'`` — ESS for the tails (rank-normalised folded samples,
          sensitive to quantile estimation accuracy).
    max_lag : int, optional
        Deprecated and ignored. The FFT computes the full ACF automatically.

    Returns
    -------
    np.ndarray
        ESS for each parameter, shape ``(D,)``, clipped to ``[1, N]``.
    """
    if max_lag is not None:
        logger.warning(
            "max_lag is deprecated in effective_sample_size and is ignored. "
            "The FFT-based ACF uses the full chain automatically."
        )

    samples = samples.reshape(-1, 1) if samples.ndim == 1 else samples
    N, D = samples.shape

    if kind == "bulk":
        # Rank-normalize each parameter (jointly across its own samples)
        rn_list = _rank_normalize_joint([samples[:, d] for d in range(D)])
        return np.array([_ess_1d(rn_list[d]) for d in range(D)])

    elif kind == "tail":
        ess = np.zeros(D)
        for d in range(D):
            median_d = np.median(samples[:, d])
            folded = np.abs(samples[:, d] - median_d)
            rn = _rank_normalize_joint([folded])[0]
            ess[d] = _ess_1d(rn)
        return ess

    else:
        raise ValueError(f"kind must be 'bulk' or 'tail', got {kind!r}.")


def tail_ess(samples: np.ndarray) -> np.ndarray:
    """
    Convenience wrapper for tail ESS (see :func:`effective_sample_size`).

    Tail ESS measures how well the chain explores the tails of the posterior —
    a low tail ESS means quantile estimates (e.g. 95th percentile) are
    unreliable even if bulk ESS looks fine.

    Parameters
    ----------
    samples : np.ndarray
        Shape ``(N, D)``.

    Returns
    -------
    np.ndarray
        Tail ESS for each parameter, shape ``(D,)``.
    """
    return effective_sample_size(samples, kind="tail")


# ---------------------------------------------------------------------------
# Public API: Gelman-Rubin R-hat (Vehtari et al. 2021)
# ---------------------------------------------------------------------------

def gelman_rubin(chains: list[np.ndarray]) -> np.ndarray:
    """
    Rank-normalised split R-hat convergence diagnostic (Vehtari et al. 2021).

    Improvements over the classic Gelman & Rubin (1992) R-hat:

    * **Split chains**: each chain is split in half before analysis, so R-hat
      detects non-stationarity within a single chain (not just between chains).
    * **Rank-normalisation**: samples are transformed to approximate standard
      normals before computing R-hat, making it robust to non-Gaussian and
      heavy-tailed posteriors where the classic formula fails silently.
    * **Bulk + tail R-hat**: both the centre and tails of the distribution are
      checked; the maximum is returned.

    Convergence criterion: R-hat < 1.01 (stricter than the classic 1.1 rule).

    Parameters
    ----------
    chains : list[np.ndarray]
        List of M independent chains, each of shape ``(N, D)``.
        All chains must have the same shape. At least 2 chains required.
        Each chain must have at least 8 samples (for splitting).

    Returns
    -------
    np.ndarray
        R-hat for each parameter, shape ``(D,)``.
        Values close to 1.0 indicate convergence; > 1.01 warrants concern.
    """
    if len(chains) < 2:
        raise ValueError("Gelman-Rubin requires at least 2 chains.")

    shapes = {c.shape for c in chains}
    if len(shapes) > 1:
        raise ValueError(
            f"All chains must have the same shape. "
            f"Got shapes: {[c.shape for c in chains]}"
        )

    D = chains[0].shape[1]

    # Split each chain in half → 2M chains of length N//2
    split = _split_chains(chains)

    r_hat = np.zeros(D)
    for d in range(D):
        chains_1d = [s[:, d] for s in split]

        # Bulk R-hat: rank-normalize jointly, then classic R-hat formula
        rn_bulk = _rank_normalize_joint(chains_1d)
        bulk = _rhat_1d(rn_bulk)

        # Tail R-hat: fold around median, rank-normalize, then classic R-hat
        median_d = np.median(np.concatenate(chains_1d))
        folded = [np.abs(c - median_d) for c in chains_1d]
        rn_tail = _rank_normalize_joint(folded)
        tail = _rhat_1d(rn_tail)

        r_hat[d] = max(bulk, tail)

    return r_hat


# ---------------------------------------------------------------------------
# Public API: corner plot
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
        Shape ``(N, D)``. Apply burn-in and thinning before passing here.
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
