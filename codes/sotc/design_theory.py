"""Closed-form statistics of FC2 estimators for the displacement designs.

Single-coordinate model. A normal coordinate Q with thermal variance
sigma^2 feels the force

    f = -w2 Q - c Q^2 - d Q^3 + eps,

where c and d are the cubic and quartic couplings of the coordinate to
itself and eps collects everything the FC2 model does not describe for
this coordinate in a given configuration (couplings to the other modes,
DFT noise), with zero mean and variance s2, independent of Q. The
least-squares slope through the origin over K force calls is

    phi_hat = -sum(f Q) / sum(Q^2)
            = w2 + c sum(Q^3)/sum(Q^2) + d sum(Q^4)/sum(Q^2) - sum(eps Q)/sum(Q^2),

and the target, the Gaussian-averaged curvature, is w2 + 3 d sigma^2.

Designs (K calls):

* ``canonical``: K independent Gaussian draws Q ~ N(0, sigma^2);
* ``paired``: K/2 Gaussian draws and their inverses;
* ``fixed``: K/2 pairs +-sigma (SOTC-D, one mode).

Results used in the manuscript (Sec. II E):

* the noise variance is s2 E[1/sum Q^2]; with sum Q^2 = sigma^2 chi^2 the
  inverse-moment factor is K/(K-2) (canonical), K/(K-4) (paired, because
  only K/2 amplitudes are independent) and 1 (fixed);
* pairs make sum(Q^3) = 0, so the cubic term vanishes; for unpaired draws
  it adds c^2 Var[sum Q^3 / sum Q^2] ~ 15 c^2 sigma^2 / K;
* fixed amplitudes have sum(Q^4)/sum(Q^2) = sigma^2: the bias is
  -2 d sigma^2 for one isolated mode at every budget; with M modes only the
  diagonal term of the renormalization is affected, which is Eq. (8).
  Gaussian amplitudes are biased too at finite budget, by -6 d sigma^2/(n+2)
  with n independent amplitudes, and add the variance of sum Q^4/sum Q^2;
* the mean-squared errors cross at a finite budget K*, beyond which the
  fixed-amplitude bias dominates (``crossover_calls``).
"""

from __future__ import annotations

import numpy as np


def independent_amplitudes(k_calls: int, design: str) -> int:
    """Number n of independent Gaussian amplitudes among K calls."""
    return k_calls if design == "canonical" else k_calls // 2


def inverse_moment_factor(k_calls: int, design: str) -> float:
    """sigma^2 K E[1/sum Q^2] for K calls: K/(K-2), K/(K-4), or 1."""
    if design == "fixed":
        return 1.0
    n = independent_amplitudes(k_calls, design)
    if n <= 2:
        return float("inf")
    # sum Q^2 = sigma^2 (K/n) chi^2_n and E[1/chi^2_n] = 1/(n - 2).
    return n / (n - 2)


def noise_variance(k_calls: int, design: str, sigma2: float, s2: float) -> float:
    """Variance of phi_hat from the unmodelled force eps (exact)."""
    return s2 / (sigma2 * k_calls) * inverse_moment_factor(k_calls, design)


def cubic_variance(k_calls: int, design: str, sigma2: float, c: float) -> float:
    """Variance of phi_hat from the cubic self-coupling c (exact).

    Pairs and fixed amplitudes cancel it; for n Gaussian amplitudes
    E[(sum Q^3 / sum Q^2)^2] = 15 n sigma^2 / ((n + 2)(n + 4)) by the
    Dirichlet(1/2) moments of Q_i^2 / sum Q^2."""
    if design != "canonical":
        return 0.0
    n = k_calls
    return 15.0 * c * c * sigma2 * n / ((n + 2) * (n + 4))


def quartic_bias(k_calls: int, design: str, sigma2: float, d: float) -> float:
    """Bias of phi_hat from the quartic self-coupling d (exact, one mode).

    Fixed amplitudes: -2 d sigma^2 at every budget. Gaussian amplitudes:
    E[sum Q^4 / sum Q^2] = 3 sigma^2 n / (n + 2), i.e. -6 d sigma^2 / (n + 2)."""
    if design == "fixed":
        return -2.0 * d * sigma2
    n = independent_amplitudes(k_calls, design)
    return -6.0 * d * sigma2 / (n + 2)


def quartic_variance(k_calls: int, design: str, sigma2: float, d: float) -> float:
    """Variance of phi_hat from the quartic self-coupling d (exact).

    Zero for fixed amplitudes; for n Gaussian amplitudes
    E[(sum Q^4/sum Q^2)^2] = sigma^4 n (9n + 96) / ((n + 4)(n + 6))."""
    if design == "fixed":
        return 0.0
    n = independent_amplitudes(k_calls, design)
    second = n * (9 * n + 96) / ((n + 4) * (n + 6))
    first = 3.0 * n / (n + 2)
    return d * d * sigma2 * sigma2 * (second - first * first)


def mse(k_calls: int, design: str, sigma2: float, c: float, d: float, s2: float) -> float:
    """Mean-squared error of phi_hat about the Gaussian-averaged curvature."""
    return (quartic_bias(k_calls, design, sigma2, d) ** 2
            + noise_variance(k_calls, design, sigma2, s2)
            + cubic_variance(k_calls, design, sigma2, c)
            + quartic_variance(k_calls, design, sigma2, d))


def crossover_calls(sigma2: float, c: float, d: float, s2: float, design: str = "paired",
                    k_max: int = 4096) -> int:
    """Smallest even budget at which fixed amplitudes stop having the smaller
    mean-squared error than the given Gaussian design (k_max if never)."""
    for k in range(6, k_max + 1, 2):
        if mse(k, "fixed", sigma2, c, d, s2) >= mse(k, design, sigma2, c, d, s2):
            return k
    return k_max


def simulate(k_calls, design, sigma2, w2, c, d, s2, n_rep, rng):
    """Monte Carlo of phi_hat for the single-coordinate model."""
    sigma = np.sqrt(sigma2)
    if design == "canonical":
        q = rng.normal(0.0, sigma, (n_rep, k_calls))
    elif design == "paired":
        h = rng.normal(0.0, sigma, (n_rep, k_calls // 2))
        q = np.concatenate([h, -h], axis=1)
    else:
        h = sigma * rng.choice([-1.0, 1.0], (n_rep, k_calls // 2))
        q = np.concatenate([h, -h], axis=1)
    eps = rng.normal(0.0, np.sqrt(s2), q.shape)
    f = -w2 * q - c * q ** 2 - d * q ** 3 + eps
    return -(f * q).sum(axis=1) / (q * q).sum(axis=1)


def pair_feature_rank(u: np.ndarray) -> int:
    """Number of distinct quadratic feature vectors u (x) u among the
    configurations u (K, n): an inverse pair repeats the same vector, so a
    paired design gives K/2 for the FC3 part, an unpaired one K."""
    feats = np.asarray([np.outer(x, x)[np.triu_indices(len(x))] for x in u])
    return int(np.linalg.matrix_rank(feats))
