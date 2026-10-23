"""D-optimal thermal-mode sign design (SOTC-D).

Each configuration excites every commensurate normal mode of a trial
harmonic model with exactly its quantum-thermal r.m.s. amplitude,

    u_i = sum_m s_m * sigma_m * e_im / sqrt(M_i),   s_m in {-1, +1},
    sigma_m^2 = hbar coth(hbar w_m / 2 k_B T) / (2 w_m),

so the single-configuration mode variances equal the canonical ones
deterministically.  The only freedom is the sign pattern.  Signs of the
K configurations are chosen by greedy single-flip ascent of

    log det( X^T X + eps I ),

where X is the force design matrix of a space-group/sum-rule adapted FC2
basis (D-optimal experimental design).  Each configuration is used with its
antithetic partner -u, which removes the force contribution that is even in
u (third order) from the linear FC2 fit exactly.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

HBAR = 1.054571817e-34
KB = 1.380649e-23
AMU = 1.66053906660e-27
EV = 1.602176634e-19
ANG = 1.0e-10


def thermal_modes(
    force_constants: np.ndarray,
    masses_amu: np.ndarray,
    temperature: float,
    omega_min_thz: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mode frequencies (THz), Cartesian mode patterns, thermal sigmas.

    ``patterns[m]`` is the (N,3) displacement pattern e_im/sqrt(M_i) in
    Angstrom per unit of mass-weighted coordinate; ``sigmas[m]`` is the r.m.s.
    mass-weighted amplitude, so ``sigmas[m]*patterns[m]`` is in Angstrom.
    Translations and unstable modes are excluded.
    """
    n = len(masses_amu)
    fc = np.asarray(force_constants, dtype=float).transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    inv_sqrt_m = np.repeat(1.0 / np.sqrt(np.asarray(masses_amu) * AMU), 3)
    dyn = fc * (EV / ANG ** 2) * np.outer(inv_sqrt_m, inv_sqrt_m)
    w2, vec = np.linalg.eigh(0.5 * (dyn + dyn.T))
    omega = np.sign(w2) * np.sqrt(np.abs(w2))
    freq_thz = omega / (2.0 * np.pi * 1.0e12)
    keep = freq_thz > omega_min_thz
    omega, vec, freq_thz = omega[keep], vec[:, keep], freq_thz[keep]
    sigma = np.sqrt(HBAR / (2.0 * omega) / np.tanh(HBAR * omega / (2.0 * KB * temperature)))
    patterns = (vec * inv_sqrt_m[:, None]).T.reshape(-1, n, 3) / ANG
    return freq_thz, patterns, sigma


def configurations(signs: np.ndarray, patterns: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Displacements (K, N, 3) in Angstrom for sign matrix (K, M)."""
    return np.einsum("km,m,mia->kia", signs, sigma, patterns)


def _design_blocks(u: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Force design rows for one configuration: (3N, n_basis)."""
    return -np.einsum("kijab,jb->iak", basis, u).reshape(-1, len(basis))


def d_optimal_signs(
    patterns: np.ndarray,
    sigma: np.ndarray,
    basis: np.ndarray,
    n_configurations: int,
    seed: int = 0,
    sweeps: int = 4,
    eps: float = 1.0e-8,
) -> Tuple[np.ndarray, Dict]:
    """Greedy single-flip D-optimal sign search.

    The objective is evaluated through the n_basis x n_basis information
    matrix, which is updated incrementally when one configuration changes.
    """
    rng = np.random.default_rng(seed)
    n_modes = len(sigma)
    signs = rng.choice([-1.0, 1.0], size=(n_configurations, n_modes))
    # Per-mode design contributions: X(u) is linear in u, so the rows of a
    # configuration are sum_m s_m sigma_m X(pattern_m).
    mode_rows = np.asarray([
        sigma[m] * _design_blocks(patterns[m], basis) for m in range(n_modes)
    ])  # (M, 3N, P)
    rows = np.einsum("km,mrp->krp", signs, mode_rows)
    info = np.einsum("krp,krq->pq", rows, rows)
    scale = np.trace(info) / info.shape[0]
    ridge = eps * scale * np.eye(info.shape[0])

    def logdet(matrix):
        sign, value = np.linalg.slogdet(matrix + ridge)
        return value if sign > 0 else -np.inf

    history = [logdet(info)]
    initial = history[0]
    for _ in range(sweeps):
        improved = 0
        for k in range(n_configurations):
            for m in rng.permutation(n_modes):
                delta = -2.0 * signs[k, m] * mode_rows[m]
                new_rows = rows[k] + delta
                trial = info - rows[k].T @ rows[k] + new_rows.T @ new_rows
                value = logdet(trial)
                if value > history[-1] + 1.0e-10:
                    signs[k, m] *= -1.0
                    rows[k] = new_rows
                    info = trial
                    history.append(value)
                    improved += 1
        if improved == 0:
            break
    normalized = info / np.sqrt(np.outer(np.diag(info), np.diag(info)))
    eig = np.linalg.eigvalsh(normalized)
    report = {
        "n_modes": int(n_modes),
        "n_basis": int(info.shape[0]),
        "logdet_initial": float(initial),
        "logdet_final": float(history[-1]),
        "accepted_flips": int(len(history) - 1),
        "normalized_information_condition": float(eig[-1] / max(eig[0], 1e-300)),
    }
    return signs, report


def sequential_d_optimal(
    patterns: np.ndarray,
    sigma: np.ndarray,
    basis: np.ndarray,
    n_configurations: int,
    seed: int = 0,
    sweeps: int = 6,
    eps: float = 1.0e-8,
    prior_information: np.ndarray | None = None,
) -> Tuple[np.ndarray, Dict]:
    """Nested D-optimal design: configuration k maximizes the information
    determinant given configurations 1..k-1, so every prefix is a design.

    ``prior_information`` (X^T X of force data already computed, in the same
    basis) conditions the design on existing configurations, e.g. those made
    with an earlier trial model."""
    rng = np.random.default_rng(seed)
    n_modes = len(sigma)
    mode_rows = np.asarray([
        sigma[m] * _design_blocks(patterns[m], basis) for m in range(n_modes)
    ])
    n_basis = mode_rows.shape[2]
    scale = np.mean([np.trace(r.T @ r) for r in mode_rows]) * n_modes / n_basis
    ridge = eps * scale * np.eye(n_basis)
    prior = (np.zeros((n_basis, n_basis)) if prior_information is None
             else np.array(prior_information, dtype=float))
    signs = np.zeros((n_configurations, n_modes))
    logdets = []

    def logdet(matrix):
        sign, value = np.linalg.slogdet(matrix + ridge)
        return value if sign > 0 else -np.inf

    for k in range(n_configurations):
        s = rng.choice([-1.0, 1.0], size=n_modes)
        rows = np.einsum("m,mrp->rp", s, mode_rows)
        best = logdet(prior + rows.T @ rows)
        for _ in range(sweeps):
            improved = False
            for m in rng.permutation(n_modes):
                new_rows = rows - 2.0 * s[m] * mode_rows[m]
                value = logdet(prior + new_rows.T @ new_rows)
                if value > best + 1.0e-10:
                    s[m] *= -1.0
                    rows, best, improved = new_rows, value, True
            if not improved:
                break
        signs[k] = s
        prior = prior + rows.T @ rows
        logdets.append(float(best))
    return signs, {"n_modes": int(n_modes), "n_basis": int(n_basis),
                   "logdet_by_prefix": logdets}


# Equal-weight values of Q_e^2 / sigma_e^2 with mean 1 and mean square 3,
# the second and fourth moments of a Gaussian mode amplitude.
MOMENT_MATCHED_LEVELS = np.array([0.0, 0.0, 2.0 - np.sqrt(2.0), 2.0 + np.sqrt(2.0)])


def moment_matched_factors(n_pairs: int, n_modes: int, rng) -> np.ndarray:
    """Amplitude factors (n_pairs, n_modes) of moment-matched SOTC-D.

    Within every block of four pairs each mode takes each value of
    sqrt(MOMENT_MATCHED_LEVELS) once, in an order drawn independently for
    every mode; ``sigma * factors[k]`` then replaces ``sigma`` for pair k.
    The design reproduces <Q_e^2> = sigma_e^2 and <Q_e^4> = 3 sigma_e^4, so
    the fixed-amplitude bias -V4_aeee sigma_e^2 / 3 vanishes."""
    n_levels = len(MOMENT_MATCHED_LEVELS)
    n_blocks = -(-n_pairs // n_levels)
    order = np.concatenate([
        np.stack([rng.permutation(n_levels) for _ in range(n_modes)], axis=1)
        for _ in range(n_blocks)])
    return np.sqrt(MOMENT_MATCHED_LEVELS[order[:n_pairs]])


def information_matrix(u: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """X^T X of the FC2 force design for displacements (K, N, 3)."""
    rows = [_design_blocks(ui, basis) for ui in np.asarray(u)]
    return sum(r.T @ r for r in rows)


def walsh_signs(n_modes: int, n_configurations: int) -> np.ndarray:
    """Deterministic, mutually orthogonal sign patterns (K, M).

    Pattern 0 alternates sign from one mode to the next, as in the
    Zacharias--Giustino special displacement; pattern k alternates in blocks
    of 2**k modes (Walsh functions), so different configurations excite
    the modes with orthogonal sign sequences."""
    index = np.arange(n_modes)
    return np.asarray([
        np.where((index >> k) & 1, -1.0, 1.0) for k in range(n_configurations)
    ])


def zg_modes(phonon, temperature: float, omega_min_thz: float = 0.05):
    """Real commensurate modes in Zacharias--Giustino ordering.

    For every commensurate q of one half of the Brillouin zone (one member
    of each +/-q pair, plus the time-reversal invariant points) and every
    branch, the real pattern Re[e(q,nu) exp(iq.R)]/sqrt(M) is returned with
    the thermal amplitude that makes the configuration's mean-square
    displacement equal to the canonical one.  Modes are ordered by q, then
    by frequency, which is the ordering on which the ZG sign alternation
    acts.  Eigenvector gauges are taken as returned by the diagonalizer.
    """
    from phonopy.harmonic.dynmat_to_fc import get_commensurate_points

    smat = np.rint(np.asarray(phonon.supercell.cell) @ np.linalg.inv(
        np.asarray(phonon.primitive.cell))).astype(int).T
    qpoints = get_commensurate_points(smat)
    p2s = np.asarray(phonon.primitive.p2s_map)
    s2p = np.asarray(phonon.primitive.s2p_map)
    pmap = {int(atom): i for i, atom in enumerate(p2s)}
    positions = np.asarray(phonon.supercell.scaled_positions) @ np.asarray(
        phonon.supercell.cell)
    pcell = np.asarray(phonon.primitive.cell)
    masses = np.asarray(phonon.supercell.masses)
    n = len(masses)
    seen, chosen = set(), []
    for q in qpoints:
        key = tuple(np.round(np.mod(q, 1.0), 8) % 1.0)
        minus = tuple(np.round(np.mod(-q, 1.0), 8) % 1.0)
        if key in seen:
            continue
        seen.update({key, minus})
        chosen.append((q, key == minus))
    patterns, sigmas, freqs = [], [], []
    for q, trim in chosen:
        f, vecs = phonon.get_frequencies_with_eigenvectors(q)
        order = np.argsort(f)
        # Lattice vector of each supercell atom relative to its primitive image.
        reference = positions - positions[s2p]
        phase = np.exp(2j * np.pi * (reference @ np.linalg.inv(pcell)) @ q)
        for nu in order:
            if f[nu] <= omega_min_thz:
                continue
            e = vecs[:, nu].reshape(-1, 3)
            pat = np.real(e[[pmap[int(a)] for a in s2p]] * phase[:, None])
            pat /= np.sqrt(masses)[:, None]
            norm = np.sqrt(np.sum(masses[:, None] * pat ** 2))
            if norm < 1.0e-12:
                pat = np.imag(e[[pmap[int(a)] for a in s2p]] * phase[:, None])
                pat /= np.sqrt(masses)[:, None]
                norm = np.sqrt(np.sum(masses[:, None] * pat ** 2))
            pat /= norm
            omega = 2.0 * np.pi * 1.0e12 * f[nu]
            var = HBAR / (2.0 * omega) / np.tanh(HBAR * omega / (2.0 * KB * temperature))
            # A non-TRIM real pattern carries the variance of both +/-q modes.
            sigmas.append(np.sqrt(var * (1.0 if trim else 2.0)))
            patterns.append(pat / np.sqrt(AMU) / ANG)
            freqs.append(f[nu])
    return np.asarray(freqs), np.asarray(patterns), np.asarray(sigmas)


def thermal_rattle(rng, phonon, temperature: float, n_configurations: int):
    """hiPhive-style rattle at the thermal amplitude: independent Gaussian
    Cartesian displacements with the per-atom r.m.s. of the trial model."""
    _, patterns, sigma = thermal_modes(
        np.asarray(phonon.force_constants), np.asarray(phonon.supercell.masses),
        temperature,
    )
    var = np.einsum("m,mia->ia", sigma ** 2, patterns ** 2)
    rms = np.sqrt(var.mean(axis=1))
    return rng.standard_normal((n_configurations, len(rms), 3)) * rms[None, :, None]
