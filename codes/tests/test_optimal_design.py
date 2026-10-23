"""Tests of the SOTC-D design primitives and of its estimator theory."""

import itertools

import numpy as np

from sotc.optimal_design import (
    AMU,
    ANG,
    EV,
    HBAR,
    KB,
    MOMENT_MATCHED_LEVELS,
    _design_blocks,
    configurations,
    information_matrix,
    moment_matched_factors,
    sequential_d_optimal,
    thermal_modes,
    walsh_signs,
)


def _toy_fc(n_atoms=4, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((3 * n_atoms, 3 * n_atoms))
    fc = a @ a.T + 3 * n_atoms * np.eye(3 * n_atoms)
    return fc.reshape(n_atoms, 3, n_atoms, 3).transpose(0, 2, 1, 3)


def test_thermal_modes_reproduce_quantum_covariance():
    fc = _toy_fc()
    masses = np.array([12.0, 30.0, 12.0, 50.0])
    temperature = 400.0
    freq, patterns, sigma = thermal_modes(fc, masses, temperature)
    n = len(masses)
    cov = np.einsum("m,mi,mj->ij", sigma ** 2, patterns.reshape(len(sigma), -1),
                    patterns.reshape(len(sigma), -1))
    # Direct canonical covariance <u u^T> in Angstrom^2.
    inv_sqrt_m = np.repeat(1.0 / np.sqrt(masses * AMU), 3)
    dyn = fc.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n) * (EV / ANG ** 2)
    dyn = dyn * np.outer(inv_sqrt_m, inv_sqrt_m)
    w2, vec = np.linalg.eigh(dyn)
    omega = np.sqrt(w2)
    var = HBAR / (2 * omega) / np.tanh(HBAR * omega / (2 * KB * temperature))
    direct = (vec * inv_sqrt_m[:, None]) @ np.diag(var) @ (vec * inv_sqrt_m[:, None]).T
    assert len(freq) == 3 * n
    np.testing.assert_allclose(cov, direct / ANG ** 2, rtol=1e-10, atol=1e-14)


def test_every_configuration_has_exact_mode_amplitudes():
    fc = _toy_fc(seed=1)
    masses = np.full(4, 20.0)
    _, patterns, sigma = thermal_modes(fc, masses, 300.0)
    signs = np.random.default_rng(3).choice([-1.0, 1.0], size=(5, len(sigma)))
    u = configurations(signs, patterns, sigma)
    # Project back onto mass-weighted normal coordinates.
    p = patterns.reshape(len(sigma), -1) * np.sqrt(np.repeat(masses, 3))
    q = u.reshape(len(u), -1) * np.sqrt(np.repeat(masses, 3)) @ p.T
    q /= np.sum(p ** 2, axis=1)
    np.testing.assert_allclose(np.abs(q), np.broadcast_to(sigma, q.shape), rtol=1e-10)


def test_walsh_signs_are_orthogonal_and_start_alternating():
    s = walsh_signs(64, 6)
    np.testing.assert_array_equal(s[0][:4], [1, -1, 1, -1])
    np.testing.assert_allclose(s @ s.T, 64 * np.eye(6))


def test_sequential_d_optimal_beats_random_signs():
    fc = _toy_fc(n_atoms=4, seed=2)
    masses = np.full(4, 20.0)
    _, patterns, sigma = thermal_modes(fc, masses, 300.0)
    rng = np.random.default_rng(0)
    basis = rng.standard_normal((10, 4, 4, 3, 3))
    signs, report = sequential_d_optimal(patterns, sigma, basis, 2, seed=1)
    best = np.linalg.slogdet(information_matrix(configurations(signs, patterns, sigma), basis))[1]
    random_values = [
        np.linalg.slogdet(information_matrix(configurations(
            rng.choice([-1.0, 1.0], size=(2, len(sigma))), patterns, sigma), basis))[1]
        for _ in range(50)
    ]
    assert best > np.max(random_values)
    # A prior information matrix is honoured: the conditioned design adds
    # information on top of it.
    prior = information_matrix(configurations(signs[:1], patterns, sigma), basis)
    cond, _ = sequential_d_optimal(patterns, sigma, basis, 1, seed=2,
                                   prior_information=prior)
    total = prior + information_matrix(configurations(cond, patterns, sigma), basis)
    assert np.linalg.slogdet(total)[1] > np.linalg.slogdet(prior + 1e-9 * np.eye(10))[1]


def _quartic_model(m, seed):
    rng = np.random.default_rng(seed)
    v4 = rng.standard_normal((m, m, m, m)) * 0.05
    v4 = sum(np.transpose(v4, p) for p in itertools.permutations(range(4))) / 24.0
    v3 = rng.standard_normal((m, m, m)) * 0.1
    v3 = sum(np.transpose(v3, p) for p in itertools.permutations(range(3))) / 6.0
    omega2 = rng.uniform(1.0, 3.0, m)
    sigma = rng.uniform(0.3, 1.0, m)
    return omega2, v3, v4, sigma


def _forces(q, omega2, v3, v4):
    return (-omega2 * q - 0.5 * np.einsum("abc,sb,sc->sa", v3, q, q)
            - np.einsum("abcd,sb,sc,sd->sa", v4, q, q, q) / 6.0)


def test_fixed_amplitude_bias_is_the_single_mode_self_contraction():
    """Regression FC2 of the fixed-amplitude design (exact average over all
    sign patterns) differs from the canonical (Gaussian) effective FC2 only
    by -V4_aeee sigma_e^2 / 3, i.e. one of the M terms of the quartic
    renormalization sum."""
    m = 6
    omega2, v3, v4, sigma = _quartic_model(m, seed=4)
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=m)))
    q = signs * sigma
    f = _forces(q, omega2, v3, v4)
    design = -(f.T @ q) / len(q) / sigma ** 2  # E[-f_a Q_e] / sigma_e^2
    gaussian = np.diag(omega2) + 0.5 * np.einsum("abbe,b->ae", v4, sigma ** 2)
    predicted = -np.einsum("aeee,e->ae", v4, sigma ** 2) / 3.0
    np.testing.assert_allclose(design - gaussian, predicted, atol=1e-12)
    # The bias is small compared with the renormalization it perturbs.
    renorm = 0.5 * np.einsum("abbe,b->ae", v4, sigma ** 2)
    assert np.linalg.norm(predicted) < np.linalg.norm(renorm)


def test_moment_matched_amplitudes_remove_the_fixed_amplitude_bias():
    """With squared amplitudes taking the moment-matched levels independently
    per mode (exact average over levels and signs), the regression FC2 equals
    the canonical effective FC2 to quartic order."""
    m = 4
    omega2, v3, v4, sigma = _quartic_model(m, seed=5)
    levels = np.sqrt(MOMENT_MATCHED_LEVELS)
    chi = np.array(list(itertools.product(levels, repeat=m)))
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=m)))
    q = (chi[:, None, :] * signs[None, :, :]).reshape(-1, m) * sigma
    f = _forces(q, omega2, v3, v4)
    design = -(f.T @ q) / len(q) / sigma ** 2
    gaussian = np.diag(omega2) + 0.5 * np.einsum("abbe,b->ae", v4, sigma ** 2)
    np.testing.assert_allclose(design, gaussian, atol=1e-12)


def test_moment_matched_factors_have_gaussian_moments_in_every_block():
    rng = np.random.default_rng(0)
    factors = moment_matched_factors(10, 7, rng)
    assert factors.shape == (10, 7)
    block = factors[:4] ** 2
    np.testing.assert_allclose(block.mean(axis=0), 1.0)
    np.testing.assert_allclose((block ** 2).mean(axis=0), 3.0)
    # the orders of different modes are independent
    assert any(not np.array_equal(factors[:4, 0], factors[:4, j]) for j in range(1, 7))


def test_antithetic_pairs_cancel_cubic_forces_exactly():
    m = 6
    omega2, v3, v4, sigma = _quartic_model(m, seed=5)
    rng = np.random.default_rng(0)
    q = rng.choice([-1.0, 1.0], size=(3, m)) * sigma
    q = np.concatenate([q, -q])
    with_cubic = np.linalg.lstsq(q, -_forces(q, omega2, v3, v4), rcond=None)[0]
    without = np.linalg.lstsq(q, -_forces(q, omega2, 0 * v3, v4), rcond=None)[0]
    np.testing.assert_allclose(with_cubic, without, atol=1e-12)
    unpaired = q[:3]
    a = np.linalg.lstsq(unpaired, -_forces(unpaired, omega2, v3, v4), rcond=None)[0]
    b = np.linalg.lstsq(unpaired, -_forces(unpaired, omega2, 0 * v3, v4), rcond=None)[0]
    assert np.abs(a - b).max() > 1e-3


def test_design_rows_are_linear_force_constant_maps():
    rng = np.random.default_rng(7)
    basis = rng.standard_normal((5, 3, 3, 3, 3))
    u = rng.standard_normal((3, 3))
    coef = rng.standard_normal(5)
    fc = np.einsum("k,kijab->ijab", coef, basis)
    np.testing.assert_allclose(_design_blocks(u, basis) @ coef,
                               -np.einsum("ijab,jb->ia", fc, u).ravel())


def _spring_phonon():
    """fcc nearest-neighbour central-spring model in a 2x2x2 phonopy cell."""
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    a = 4.05
    cell = 0.5 * a * np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    unit = PhonopyAtoms(symbols=["Al"], cell=cell, scaled_positions=[[0., 0., 0.]])
    phonon = Phonopy(unit, np.diag([2, 2, 2]))
    pos = np.asarray(phonon.supercell.positions)
    scell = np.asarray(phonon.supercell.cell)
    n = len(pos)
    fc = np.zeros((n, n, 3, 3))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for image in itertools.product((-1, 0, 1), repeat=3):
                d = pos[j] + np.asarray(image) @ scell - pos[i]
                if abs(np.linalg.norm(d) - a / np.sqrt(2)) < 1e-6:
                    e = d / np.linalg.norm(d)
                    fc[i, j] -= 2.0 * np.outer(e, e)
    for i in range(n):
        fc[i, i] = -fc[i].sum(axis=0)
    phonon.force_constants = fc
    return phonon


def test_zg_modes_and_rattle_match_canonical_mean_square():
    from sotc.optimal_design import thermal_rattle, zg_modes

    phonon = _spring_phonon()
    masses = np.asarray(phonon.supercell.masses)
    _, patterns, sigma = thermal_modes(np.asarray(phonon.force_constants), masses, 300.0)
    canonical = np.einsum("m,mia->ia", sigma ** 2, patterns ** 2)
    freq, zp, zs = zg_modes(phonon, 300.0)
    # Real ZG patterns are mass-orthogonal and every configuration carries
    # the canonical mean-square displacement per atom (on average over atoms).
    p = zp.reshape(len(zs), -1) * np.sqrt(np.repeat(masses, 3))
    gram = p @ p.T
    np.testing.assert_allclose(gram, np.diag(np.diag(gram)), atol=1e-9 * gram.max())
    u = configurations(walsh_signs(len(zs), 3), zp, zs)
    np.testing.assert_allclose(np.mean(u ** 2), np.mean(canonical), rtol=0.02)
    rattle = thermal_rattle(np.random.default_rng(0), phonon, 300.0, 4000)
    np.testing.assert_allclose(np.mean(rattle ** 2, axis=0).mean(axis=1),
                               canonical.mean(axis=1), rtol=0.03)
