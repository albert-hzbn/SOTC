"""
Tests for sqtc.correlators — DebyeCorrelator

Verifies:
  1. Zero-point MSD formula: ⟨u²⟩₀ = 9ℏ²/(4Mk_BT_D)
  2. Pair correlator at R=0 equals MSD (== 3× the 1D variance)
  3. High-temperature classical limit: ⟨u²⟩_T → 3k_BT / (Mω_D²) as T → ∞
  4. C_V → 3R (Dulong–Petit) at high T
  5. Coherence length is positive
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from sotc.correlators import DebyeCorrelator
from sotc.constants import HBAR, KB, AMU_TO_KG, R_GAS

# He parameters
T_D = 26.0
M_amu = 4.002602

debye = DebyeCorrelator(T_D=T_D, M_amu=M_amu)

# ── Analytic formula for zero-point MSD ──────────────────────────────────────
M_kg = M_amu * 1.66053906660e-27
omega_D = KB * T_D / HBAR
msd_0_analytic = 9.0 * HBAR**2 / (4.0 * M_kg * KB * T_D)       # m²
msd_0_analytic_ang2 = msd_0_analytic / (1e-10)**2                # Å²


class TestDebyeCorrelatorZeroPoint:
    def test_zero_point_msd_formula(self):
        """⟨u²⟩_0K from correlator matches analytic expression."""
        msd_sqtc = debye.msd_ang2(T=0.001)  # near-zero temperature
        assert abs(msd_sqtc - msd_0_analytic_ang2) / msd_0_analytic_ang2 < 1e-3

    def test_pair_correlator_at_zero_separation(self):
        """Pair correlator at R=0 should equal MSD/3 (1D projection)."""
        msd = debye.msd_ang2(T=1.6)
        c2_r0 = debye.pair_correlator_ang2(R_ang=0.0, T=1.6)
        # C̄₂(0,T) = ⟨u²⟩/3  (isotropic: each Cartesian component = MSD/3)
        assert abs(c2_r0 - msd / 3.0) / (msd / 3.0) < 0.05


class TestDebyeCorrelatorHighT:
    def test_classical_msd_limit(self):
        """
        At T >> T_D, Debye MSD → 3k_BT/(Mω_D²) (equipartition).
        The Debye density of states g(ω) = 9ω²/ω_D³ gives:
          ⟨u²⟩ = (9ℏ/Mω_D³) ∫₀^ω_D ω·coth(ℏω/2kT) dω
        In the classical limit coth(x)→1/x, so:
          ⟨u²⟩ → (9kT/Mω_D³) ∫₀^ω_D dω = 9kT/(Mω_D²)
        Note: NOT 3kT/Mω_D² — the factor 9 (not 3) comes from the
        density-of-states prefactor already containing the 3 branches.
        """
        T_hot = 1000.0
        msd_hot = debye.msd_ang2(T_hot)
        omega_D_val = KB * T_D / HBAR
        msd_classical_ang2 = 9.0 * KB * T_hot / (M_kg * omega_D_val**2) / 1e-20
        assert abs(msd_hot - msd_classical_ang2) / msd_classical_ang2 < 0.05

    def test_dulong_petit_cv(self):
        """C_V → 3R at high temperature (Dulong–Petit)."""
        cv_hot = debye.heat_capacity_v_jmolk(T=10 * T_D)
        assert abs(cv_hot - 3.0 * R_GAS) / (3.0 * R_GAS) < 0.02

    def test_cv_low_T_cubic(self):
        """C_V ~ T³ at very low T (Debye law)."""
        cv_low1 = debye.heat_capacity_v_jmolk(T=0.1)
        cv_low2 = debye.heat_capacity_v_jmolk(T=0.2)
        # Should scale as (0.2/0.1)³ = 8
        ratio = cv_low2 / cv_low1
        assert abs(ratio - 8.0) < 0.5  # loose tolerance at numerically tricky T


class TestDebyeCorrelatorTable:
    def test_correlator_table_shape(self):
        R_vals = np.linspace(0.5, 8.0, 10)
        table = debye.correlator_table(R_vals, T=1.6)
        assert table.shape == (10,)

    def test_correlator_decays_with_R(self):
        """Correlator should decrease with increasing R."""
        R_vals = np.linspace(1.0, 8.0, 8)
        table = debye.correlator_table(R_vals, T=1.6)
        # Correlator at small R should be larger than at large R
        assert table[0] >= table[-1]

    def test_coherence_length_positive(self):
        xi = debye.coherence_length(T=1.6, gruneisen=2.0)
        assert xi > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
