"""
Tests for sqtc.phonons — PhononCalculator

Verifies:
  1. Dynamical matrix is Hermitian at every q-point
  2. Acoustic modes at Γ have ω ≈ 0 (sum rule)
  3. Frequencies from a known spring constant match analytic dispersion
  4. C_V → 3R at high T (Dulong–Petit)
  5. DOS integrates to 1 when normalised
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from sqtc.ifc_extractor import IFCExtractor
from sqtc.phonons import PhononCalculator
from sqtc.constants import KB, HBAR, AMU_TO_KG, R_GAS


def make_sc_chain_with_ifcs(n: int = 8, a: float = 3.57, k_spring: float = 0.5):
    """
    Set up IFCExtractor for a 1D chain with known spring constant, then fit
    IFCs from synthetic forces, and return a PhononCalculator.

    Primitive cell: 1 atom, lattice constant a.
    Supercell: n atoms, length n·a.
    """
    from tests.test_ifc import make_linear_chain, chain_forces

    cell, pos, masses = make_linear_chain(n, a)
    extractor = IFCExtractor(pos, cell, masses, r_cutoff=a * 1.5, symmetrise=True)

    rng = np.random.default_rng(42)
    disps, forces = [], []
    for _ in range(4 * n):
        u = np.zeros((n, 3))
        u[:, 0] = rng.normal(0, 0.05, n)
        disps.append(u)
        forces.append(chain_forces(pos + u, cell, k_spring))

    extractor.fit(disps, forces)

    prim_cell = np.diag([a, 10.0, 10.0])
    prim_pos  = np.array([[0.0, 0.0, 0.0]])
    prim_mass = np.array([masses[0]])

    calc = PhononCalculator(
        ifc_extractor=extractor,
        prim_positions=prim_pos,
        prim_cell=prim_cell,
        masses_amu=prim_mass,
    )
    return calc, k_spring, a, masses[0]


class TestPhononCalculator:
    @pytest.fixture
    def phcalc(self):
        return make_sc_chain_with_ifcs()

    def test_dynamical_matrix_hermitian(self, phcalc):
        """D(q) must be Hermitian at every q."""
        calc, *_ = phcalc
        for q in [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.25, 0.1, 0.0]]:
            D = calc.dynamical_matrix(np.array(q))
            assert np.allclose(D, D.conj().T, atol=1e-10), (
                f"D(q={q}) is not Hermitian: max off-diag = "
                f"{np.abs(D - D.conj().T).max():.2e}"
            )

    def test_acoustic_modes_at_gamma(self, phcalc):
        """All modes at Γ = [0,0,0] must have |ω| < threshold."""
        calc, *_ = phcalc
        freqs = calc.frequencies_thz_at_q(np.array([0.0, 0.0, 0.0]))
        # The 1D chain has 1 atom/cell → 1 branch which is acoustic
        assert abs(freqs[0]) < 0.5, (  # 0.5 THz loose threshold
            f"Acoustic mode at Γ: {freqs[0]:.4f} THz (should be ≈ 0)"
        )

    def test_zone_boundary_frequency(self, phcalc):
        """
        For a 1D chain, ω(π/a) = 2√(k/M).  Compare SQTC phonons to this.
        """
        calc, k_spring, a, M_amu = phcalc
        k_si = k_spring * 1.60218e-19 / (1e-10)**2  # eV/Å² → J/m²
        M_kg = M_amu * AMU_TO_KG
        omega_ZB_analytic = 2.0 * np.sqrt(k_si / M_kg)  # rad/s
        nu_ZB_analytic = omega_ZB_analytic / (2 * np.pi * 1e12)  # THz

        # Zone boundary in fractional coords: q = 0.5 along x
        freqs_ZB = calc.frequencies_thz_at_q(np.array([0.5, 0.0, 0.0]))
        nu_ZB_sqtc = max(abs(f) for f in freqs_ZB)

        assert abs(nu_ZB_sqtc - nu_ZB_analytic) / nu_ZB_analytic < 0.2, (
            f"ω_ZB: SQTC={nu_ZB_sqtc:.4f} THz, analytic={nu_ZB_analytic:.4f} THz"
        )

    def test_dos_normalisation(self, phcalc):
        """DOS should integrate to ≈ 1 (3N_b modes per cell)."""
        calc, *_ = phcalc
        dos_data = calc.dos(q_mesh=(20, 1, 1), n_bins=100)
        nu_bins = dos_data["frequencies_thz"]
        dos_vals = dos_data["dos"]
        d_nu = nu_bins[1] - nu_bins[0]
        integral = np.sum(dos_vals) * d_nu
        # 1 atom/cell → integral ≈ 1
        assert abs(integral - 1.0) < 0.2, (
            f"DOS integral = {integral:.4f} (expected 1)"
        )

    def test_heat_capacity_dulong_petit(self, phcalc):
        """C_V → 3R per mole at very high T."""
        calc, _, _, M_amu = phcalc
        T_hot = 10000.0
        cv = calc.heat_capacity_v_jmolk(T_hot, q_mesh=(20, 1, 1))
        assert abs(cv - 3.0 * R_GAS) / (3.0 * R_GAS) < 0.05, (
            f"C_V at {T_hot} K = {cv:.2f} J/mol/K  (3R = {3*R_GAS:.2f})"
        )

    def test_zero_point_energy_positive(self, phcalc):
        """ZPE must be positive."""
        calc, *_ = phcalc
        zpe = calc.zero_point_energy(q_mesh=(20, 1, 1))
        assert zpe > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
