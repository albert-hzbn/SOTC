"""Synthetic validation for higher-order IFCs and DFPT LO--TO support."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sotc.anharmonic_ifc import (
    FiniteDifferenceAnharmonicIFC,
    ModeSpaceAnharmonicIFC,
)
from sotc.constants import AMU_TO_KG, EV_TO_J, RAD_S_TO_THZ
from sotc.phonons import PhononCalculator
from sotc.qe_dfpt import parse_ph_dielectric_born, parse_ph_gamma_frequencies


def test_cubic_quartic_coefficients_are_recovered_from_forces():
    rng = np.random.default_rng(14)
    truth = ModeSpaceAnharmonicIFC(n_atoms=2, max_order=4, rcond=1e-12)
    truth.coefficients = rng.normal(0.0, 0.3, len(truth.monomials))
    x = rng.uniform(-1.0, 1.0, (160, truth.n_modes))
    displacements = (x @ truth.mode_basis.T).reshape((-1, 2, 3))
    forces = truth.predict_forces(displacements)

    fitted = ModeSpaceAnharmonicIFC(n_atoms=2, max_order=4, rcond=1e-12)
    fitted.fit(
        displacements, forces, validation_fraction=0.25, random_seed=3
    )
    diagnostics = fitted.diagnostics()
    assert diagnostics["identifiable"] is True
    assert diagnostics["training_subset_identifiable"] is True
    assert diagnostics["validation_force_rmse"] < 1e-10
    assert np.allclose(fitted.coefficients, truth.coefficients, atol=1e-10)


def test_fitted_forces_are_negative_energy_gradient():
    rng = np.random.default_rng(9)
    model = ModeSpaceAnharmonicIFC(n_atoms=2, max_order=4)
    model.coefficients = rng.normal(0.0, 0.2, len(model.monomials))
    x = rng.normal(0.0, 0.2, model.n_modes)
    displacement = (x @ model.mode_basis.T).reshape((1, 2, 3))
    force = model.predict_forces(displacement).reshape(-1)
    direction = model.mode_basis[:, 0]
    step = 1e-6
    plus = displacement.reshape(-1) + step * direction
    minus = displacement.reshape(-1) - step * direction
    gradient = (
        model.energy(plus.reshape((1, 2, 3)))[0]
        - model.energy(minus.reshape((1, 2, 3)))[0]
    ) / (2.0 * step)
    assert np.isclose(-gradient, direction @ force, rtol=1e-7, atol=1e-9)


def test_high_order_fit_refuses_rank_deficiency_by_default():
    model = ModeSpaceAnharmonicIFC(n_atoms=2, max_order=4)
    displacements = np.zeros((2, 2, 3))
    forces = np.zeros_like(displacements)
    with pytest.raises(ValueError, match="not identifiable"):
        model.fit(displacements, forces)


def test_qe_dfpt_parser_and_charge_tensor_orientation():
    output = """
     Dielectric constant in cartesian axis
          (   5.100000  0.100000  0.000000 )
          (   0.100000  5.200000  0.000000 )
          (   0.000000  0.000000  5.300000 )

     Effective charges (d Force / dE) in cartesian axis
          atom    1
          Ex  (   2.100000  0.200000  0.300000 )
          Ey  (   0.400000  2.200000  0.500000 )
          Ez  (   0.600000  0.700000  2.300000 )
          atom    2
          Ex  (  -2.100000 -0.200000 -0.300000 )
          Ey  (  -0.400000 -2.200000 -0.500000 )
          Ez  (  -0.600000 -0.700000 -2.300000 )

     freq (    1) =      -0.012345 [THz] =      -0.411452 [cm-1]
     freq (    2) =       4.567890 [THz] =     152.368000 [cm-1]
    """
    dielectric, born = parse_ph_dielectric_born(output, 2)
    assert np.allclose(np.diag(dielectric), [5.1, 5.2, 5.3])
    assert np.allclose(born.sum(axis=0), 0.0)
    assert np.allclose(np.array([1.0, 0.0, 0.0]) @ born[0], [2.1, 0.2, 0.3])
    assert np.allclose(parse_ph_gamma_frequencies(output), [-0.012345, 4.567890])


class _TwoAtomEmptyIFC:
    N = 2
    _Phi = {}
    _pair_vecs = {}


def test_nonanalytic_term_creates_positive_longitudinal_shift():
    born = np.array([2.0 * np.eye(3), -2.0 * np.eye(3)])
    dielectric = 4.0 * np.eye(3)
    calc = PhononCalculator(
        _TwoAtomEmptyIFC(),
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        4.0 * np.eye(3),
        np.array([20.0, 30.0]),
        born_charges=born,
        dielectric_tensor=dielectric,
    )
    audit = calc.loto_frequencies_near_gamma(np.array([1.0, 0.0, 0.0]))
    assert np.allclose(audit["without_nac_thz"], 0.0)
    assert audit["with_nac_thz"][-1] > 0.0
    assert np.count_nonzero(np.asarray(audit["shift_thz"]) > 1e-10) == 1
    coulomb_ev_ang = 14.3996
    volume_ang3 = 4.0**3
    prefactor = EV_TO_J / (1.0e-20 * AMU_TO_KG)
    expected_omega2 = (
        4.0 * np.pi * coulomb_ev_ang / volume_ang3
        * prefactor / 4.0
        * (2.0**2 / 20.0 + (-2.0)**2 / 30.0)
    )
    expected_thz = np.sqrt(expected_omega2) * RAD_S_TO_THZ
    assert np.isclose(audit["with_nac_thz"][-1], expected_thz, rtol=1e-12)


def test_nac_inputs_are_validated():
    with pytest.raises(ValueError, match="supplied together"):
        PhononCalculator(
            _TwoAtomEmptyIFC(), np.zeros((2, 3)), np.eye(3),
            np.array([20.0, 30.0]), born_charges=np.zeros((2, 3, 3))
        )
    with pytest.raises(ValueError, match="positive definite"):
        PhononCalculator(
            _TwoAtomEmptyIFC(), np.zeros((2, 3)), np.eye(3),
            np.array([20.0, 30.0]),
            born_charges=np.zeros((2, 3, 3)),
            dielectric_tensor=np.diag([1.0, 1.0, -1.0]),
        )


def test_dense_fc2_fc3_fc4_reproduce_polynomial_forces_and_export(tmp_path):
    from sotc.anharmonic_io import (
        write_fourphonon_fc4,
        write_phono3py_fc3,
        write_shengbte_fc3,
    )

    rng = np.random.default_rng(31)
    model = ModeSpaceAnharmonicIFC(n_atoms=2, max_order=4)
    model.coefficients = rng.normal(0.0, 0.1, len(model.monomials))
    fc2 = model.dense_cartesian_tensor(2)
    fc3 = model.dense_cartesian_tensor(3)
    fc4 = model.dense_cartesian_tensor(4)
    u = rng.normal(0.0, 0.1, (2, 3))
    tensor_force = (
        -np.einsum("ijab,jb->ia", fc2, u)
        -0.5 * np.einsum("ijkabc,jb,kc->ia", fc3, u, u)
        -(1.0 / 6.0) * np.einsum("ijklabcd,jb,kc,ld->ia", fc4, u, u, u)
    )
    assert np.allclose(
        tensor_force, model.predict_forces(u[None])[0], atol=1e-12
    )

    hdf5_path = tmp_path / "fc3.hdf5"
    write_phono3py_fc3(hdf5_path, fc3)
    import h5py
    with h5py.File(hdf5_path, "r") as handle:
        assert handle["fc3"].shape == (2, 2, 2, 3, 3, 3)
        assert handle["fc3"].attrs["physical_unit"] == "eV/angstrom^3"
        assert np.allclose(handle["fc3"][:], fc3)

    mapping = np.array([0, 1])
    translations = np.zeros((2, 3))
    n3 = write_shengbte_fc3(
        tmp_path / "FORCE_CONSTANTS_3RD", fc3,
        np.array([0, 1]), mapping, translations,
    )
    n4 = write_fourphonon_fc4(
        tmp_path / "FORCE_CONSTANTS_4TH", fc4,
        np.array([0, 1]), mapping, translations,
    )
    assert int((tmp_path / "FORCE_CONSTANTS_3RD").read_text().splitlines()[0]) == n3
    assert int((tmp_path / "FORCE_CONSTANTS_4TH").read_text().splitlines()[0]) == n4
    assert n3 == 8
    assert n4 == 16


def test_complete_real_space_fc3_fc4_against_analytic_pair_potential():
    k2, k3, k4 = 1.7, -0.8, 2.3

    def analytic_forces(displacement):
        q = displacement[1, 0] - displacement[0, 0]
        derivative = k2 * q + 0.5 * k3 * q**2 + (k4 / 6.0) * q**3
        forces = np.zeros((2, 3))
        forces[0, 0] = derivative
        forces[1, 0] = -derivative
        return forces

    extractor = FiniteDifferenceAnharmonicIFC(
        n_atoms=2, force_function=analytic_forces,
        displacement_step_ang=0.05,
    )
    fc3, fc4 = extractor.extract_fc3_fc4()
    relative_coordinate = np.array([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    expected3_flat = k3 * np.einsum(
        "i,j,k->ijk", relative_coordinate, relative_coordinate,
        relative_coordinate,
    )
    expected4_flat = k4 * np.einsum(
        "i,j,k,l->ijkl", relative_coordinate, relative_coordinate,
        relative_coordinate, relative_coordinate,
    )

    def atom_major(flat, order):
        interleaved = flat.reshape((2, 3) * order)
        return interleaved.transpose(
            tuple(range(0, 2 * order, 2))
            + tuple(range(1, 2 * order, 2))
        )

    assert np.allclose(fc3, atom_major(expected3_flat, 3), atol=1e-11)
    assert np.allclose(fc4, atom_major(expected4_flat, 4), atol=1e-10)
    assert extractor.force_evaluations > 0
    assert extractor.diagnostics(3)["acoustic_sum_rule_max_abs"] < 1e-11
    assert extractor.diagnostics(4)["acoustic_sum_rule_max_abs"] < 1e-10


def test_fourphonon_export_block_matches_documented_sparse_layout(tmp_path):
    from sotc.anharmonic_io import write_fourphonon_fc4

    fc4 = np.zeros((1, 1, 1, 1, 3, 3, 3, 3))
    fc4[0, 0, 0, 0, 0, 1, 2, 0] = 4.25
    path = tmp_path / "FORCE_CONSTANTS_4TH"
    blocks = write_fourphonon_fc4(
        path, fc4, np.array([0]), np.array([0]), np.zeros((1, 3)),
        threshold_ev_ang4=1e-14,
    )
    lines = [line for line in path.read_text().splitlines() if line]
    assert blocks == 1
    assert lines[:6] == [
        "1", "1",
        "0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00",
        "0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00",
        "0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00",
        "1 1 1 1",
    ]
    tensor_lines = lines[6:]
    assert len(tensor_lines) == 81
    assert "1 2 3 1 4.2500000000000000e+00" in tensor_lines


def test_compact_fc234_force_evaluation_matches_full_translation_expansion():
    from itertools import product
    from sotc.symfc_fit import forces_from_compact_fcs, forces_from_fcs

    rng = np.random.default_rng(918)
    permutations = np.array([[0, 1], [1, 0]])
    p2s_map = np.array([0])
    compact = {
        order: rng.normal(size=(1,) + (2,) * (order - 1) + (3,) * order)
        for order in (2, 3, 4)
    }
    full = {}
    for order, values in compact.items():
        tensor = np.zeros((2,) * order + (3,) * order)
        for translation, permutation in enumerate(permutations):
            del translation
            for trailing in product(range(2), repeat=order - 1):
                source = (0,) + trailing
                target = (permutation[0],) + tuple(permutation[i] for i in trailing)
                tensor[target] = values[source]
        full[order] = tensor
    u = rng.normal(scale=0.1, size=(5, 2, 3))
    assert np.allclose(
        forces_from_compact_fcs(u, compact, permutations, p2s_map),
        forces_from_fcs(u, full),
        atol=1e-12,
    )


def test_compact_fc3_fc4_export_uses_reference_slots(tmp_path):
    from sotc.anharmonic_io import write_fourphonon_fc4, write_shengbte_fc3

    fc3 = np.zeros((1, 2, 2, 3, 3, 3))
    fc4 = np.zeros((1, 2, 2, 2, 3, 3, 3, 3))
    fc3[0, 1, 0, 0, 1, 2] = -2.5
    fc4[0, 1, 0, 1, 0, 1, 2, 0] = 3.75
    references = np.array([1])
    basis = np.array([0, 0])
    vectors = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    n3 = write_shengbte_fc3(
        tmp_path / "FC3", fc3, references, basis, vectors,
        threshold_ev_ang3=1e-14,
    )
    n4 = write_fourphonon_fc4(
        tmp_path / "FC4", fc4, references, basis, vectors,
        threshold_ev_ang4=1e-14,
    )
    assert n3 == 1
    assert n4 == 1
    assert (tmp_path / "FC3").read_text().splitlines()[0] == "1"
    assert (tmp_path / "FC4").read_text().splitlines()[0] == "1"


def test_dense_fc4_allocation_guard():
    model = ModeSpaceAnharmonicIFC(n_atoms=3, max_order=4)
    with pytest.raises(MemoryError, match="symmetry/locality-reduced"):
        model.dense_cartesian_tensor(4, max_elements=100)


def test_order_comparison_uses_same_holdout_and_detects_quartic(tmp_path):
    from sotc.validation import compare_anharmonic_orders

    rng = np.random.default_rng(77)
    truth = ModeSpaceAnharmonicIFC(n_atoms=2, max_order=4)
    truth.coefficients = rng.normal(0.0, 0.5, len(truth.monomials))
    x = rng.uniform(-0.7, 0.7, (120, truth.n_modes))
    u = (x @ truth.mode_basis.T).reshape((-1, 2, 3))
    f = truth.predict_forces(u)
    report = compare_anharmonic_orders(
        u, f, tmp_path, max_modes=3, test_fraction=0.2, random_seed=2
    )
    assert report["orders"]["2"]["identifiable"] is True
    assert report["orders"]["3"]["identifiable"] is True
    assert report["orders"]["4"]["identifiable"] is True
    assert report["orders"]["4"]["external_projected_force_rmse_ev_ang"] < 1e-10
    assert report["quartic_supported_by_external_cv"] is True
    assert (tmp_path / "mode_ifc_order4.npz").exists()
    assert (tmp_path / "order_comparison.json").exists()
