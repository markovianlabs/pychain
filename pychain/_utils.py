"""
Internal utilities shared across pychain modules.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Inverse normal CDF — no scipy dependency
# ---------------------------------------------------------------------------

def _norm_ppf_scalar(p: float) -> float:
    """
    Scalar inverse standard normal CDF (Phi^{-1}).

    Uses Acklam's rational approximation (2000), which achieves a maximum
    absolute error < 1.15e-9 across the full domain.

    Reference: Peter Acklam, https://web.archive.org/web/20151030215612/
               http://home.online.no/~pjacklam/notes/invnorm/
    """
    # Coefficients for the rational approximation
    a = (
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00,
    )
    b = (
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00,
    )
    d = (
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00,
    )

    p_low  = 0.02425
    p_high = 1.0 - p_low

    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}.")

    if p_low <= p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            q * (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    elif 0.0 < p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    else:  # p_high < p < 1
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )


_v_norm_ppf = np.vectorize(_norm_ppf_scalar)


def norm_ppf(p: np.ndarray) -> np.ndarray:
    """Vectorised inverse standard normal CDF (Phi^{-1}) — no scipy needed."""
    return _v_norm_ppf(np.asarray(p, dtype=float))
