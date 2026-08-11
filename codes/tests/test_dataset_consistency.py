"""Regression tests for cross-file scientific consistency."""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sotc.constants import AMU_TO_KG, HBAR, KB, coth
from sotc.ifc_extractor import IFCExtractor
from sotc.phonons import PhononCalculator
from sotc.runner import SOTCRunner


class _EmptyIFC:
    N = 1
    _Phi = {}


def _einstein_calculator(mass_amu=26.9815385, frequency_thz=5.0):
    calc = PhononCalculator(
        _EmptyIFC(), np.zeros((1, 3)), np.eye(3), np.array([mass_amu])
    )
    omega = 2.0 * np.pi * frequency_thz * 1.0e12
    frequencies = np.full((1, 3), omega)
    eigenvectors = np.eye(3, dtype=complex)[None, :, :]

    def table(_mesh, return_eigvecs=False):
        if return_eigvecs:
            return frequencies, eigenvectors, [np.zeros(3)]
        return frequencies

    calc._all_frequencies = table
    return calc, omega


def test_einstein_mode_sum_uses_total_msd_and_u_iso_factor_three():
    calc, omega = _einstein_calculator()
    T = 300.0
    result = calc.mean_square_displacements(T, q_mesh=(1, 1, 1))
    mass = calc.masses_amu[0] * AMU_TO_KG
    expected_total = (
        3.0 * HBAR / (2.0 * mass * omega)
        * coth(HBAR * omega / (2.0 * KB * T)) * 1.0e20
    )
    assert np.isclose(result["total_msd_ang2_by_atom"][0], expected_total)
    assert np.isclose(result["u_iso_ang2_by_atom"][0], expected_total / 3.0)
    assert np.isclose(
        result["debye_waller_b_ang2_by_atom"][0],
        8.0 * np.pi**2 * expected_total / 3.0,
    )


def test_projected_dos_and_mode_sum_msd_agree_for_einstein_solid():
    calc, _ = _einstein_calculator(frequency_thz=5.0)
    T = 300.0
    mode_sum = calc.mean_square_displacements(T, q_mesh=(1, 1, 1))
    nu = np.linspace(4.5, 5.5, 10001)
    sigma = 0.02
    gaussian = np.exp(-0.5 * ((nu - 5.0) / sigma) ** 2)
    gaussian /= np.trapezoid(gaussian, nu)
    pdos = (3.0 * gaussian)[None, :]
    dos_sum = calc.mean_square_displacements_from_projected_dos(T, nu, pdos)
    assert np.isclose(dos_sum["pdos_mode_count_by_atom"][0], 3.0, rtol=1e-8)
    assert np.isclose(
        dos_sum["u_iso_ang2_by_atom"][0],
        mode_sum["u_iso_ang2_by_atom"][0],
        rtol=5e-5,
    )


def test_ridge_does_not_inflate_reported_data_rank(tmp_path):
    n = 4
    cell = np.diag([4.0, 8.0, 8.0])
    positions = np.zeros((n, 3))
    positions[:, 0] = np.arange(n)
    extractor = IFCExtractor(
        positions, cell, np.ones(n), r_cutoff=1.1, ridge_alpha=1e-3
    )
    displacement = np.arange(3 * n, dtype=float).reshape(n, 3) * 1e-3
    force = np.zeros((n, 3))
    extractor.fit([displacement], [force])
    report = extractor.diagnostics()
    assert report["data_rank"] <= 3 * n
    assert report["data_rank"] <= report["raw_parameter_count"]
    assert report["ridge_creates_information"] is False
    extractor.save_diagnostics(str(tmp_path))
    assert (tmp_path / "design_matrix.npz").exists()
    assert (tmp_path / "fit_diagnostics.json").exists()


def test_compound_ifc_classes_do_not_pool_distinct_basis_pairs():
    cell = np.diag([4.0, 8.0, 8.0])
    positions = np.zeros((4, 3))
    positions[:, 0] = np.arange(4)
    mono = IFCExtractor(positions, cell, np.ones(4), r_cutoff=1.1)
    binary = IFCExtractor(
        positions,
        cell,
        np.ones(4),
        r_cutoff=1.1,
        basis_indices=np.array([0, 1, 0, 1]),
    )
    assert binary.n_unique_displacements() > mono.n_unique_displacements()
    assert all(pair in {(0, 1), (1, 0)} for pair in binary._unique_basis_pairs)


def test_runner_uses_crystallographic_orbits_for_fit_classes():
    a = 3.57
    hcp_cell = np.array([
        [a, 0.0, 0.0],
        [-a / 2.0, a * np.sqrt(3.0) / 2.0, 0.0],
        [0.0, 0.0, 1.633 * a],
    ])
    hcp_positions = np.array([
        [0.0, 0.0, 0.0],
        [a / 2.0, a / (2.0 * np.sqrt(3.0)), 1.633 * a / 2.0],
    ])
    hcp = SOTCRunner(
        element="He", mass_amu=4.002602,
        prim_cell=hcp_cell, prim_positions=hcp_positions,
        T=1.6, T_D=26.0, verbosity=0,
    )
    assert np.array_equal(hcp.basis_class_indices, [0, 0])

    rocksalt_cell = 0.5 * a * np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    rocksalt = SOTCRunner(
        element="Na", mass_amu=22.9898,
        elements=["Na", "Cl"], masses_amu=[22.9898, 35.453],
        prim_cell=rocksalt_cell,
        prim_positions=np.array([[0.0, 0.0, 0.0], [a / 2.0, 0.0, 0.0]]),
        T=300.0, T_D=300.0, verbosity=0,
    )
    assert np.array_equal(rocksalt.basis_class_indices, [0, 1])


def test_ifc_archive_round_trip(tmp_path):
    cell = np.diag([4.0, 8.0, 8.0])
    positions = np.zeros((4, 3))
    positions[:, 0] = np.arange(4)
    extractor = IFCExtractor(positions, cell, np.ones(4), r_cutoff=1.1)
    rng = np.random.default_rng(4)
    displacements = [rng.normal(0.0, 0.01, (4, 3)) for _ in range(3)]
    forces = [np.zeros((4, 3)) for _ in displacements]
    extractor.fit(displacements, forces)
    path = tmp_path / "ifc.npz"
    extractor.save(str(path))
    loaded = IFCExtractor.load(str(path))
    assert loaded._Phi.keys() == extractor._Phi.keys()
    for key in extractor._Phi:
        assert np.allclose(loaded._Phi[key], extractor._Phi[key])
