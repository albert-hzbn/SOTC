"""
Tests for sqtc.cell_design — HNFEnumerator and DisplacementOptimizer

Verifies:
  1. HNF matrices have correct determinants for each n
  2. Generated supercells have correct number of atoms
  3. Displacement optimizer returns correct shape
  4. SQTC quality functional Q is non-negative
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from sotc.cell_design import HNFEnumerator, DisplacementOptimizer
from sotc.correlators import DebyeCorrelator

# Simple cubic primitive cell for testing
a = 3.5  # Å
prim_cell = np.eye(3) * a
n_b = 1   # 1 atom per primitive cell

enumerator = HNFEnumerator(prim_cell, n_b, n_min=2, n_max=8)


class TestHNFEnumerator:
    def test_hnf_determinants(self):
        """All returned HNF matrices should have determinant equal to n."""
        for n in [2, 3, 4, 6, 8]:
            H_mats = enumerator._hnf_matrices(n)
            for H in H_mats:
                det = round(abs(np.linalg.det(H)))
                assert det == n, f"HNF with det={det} ≠ {n}"

    def test_hnf_upper_triangular(self):
        """HNF matrices must be upper triangular."""
        for n in [2, 4, 6]:
            H_mats = enumerator._hnf_matrices(n)
            for H in H_mats:
                assert np.all(np.tril(H, -1) == 0), "HNF not upper triangular"

    def test_generate_returns_candidates(self):
        """generate() should return at least one (H, sc_cell) pair."""
        candidates = enumerator.generate(max_per_n=2)
        assert len(candidates) > 0

    def test_supercell_positions_count(self):
        """Tiled supercell positions should have n_b × det(H) atoms."""
        candidates = enumerator.generate(max_per_n=1)
        for H, sc_cell in candidates:
            n_sc = int(round(abs(np.linalg.det(H))))
            # Build positions by hand (same as runner._choose_supercell)
            pos = []
            for i in range(H[0, 0]):
                for j in range(H[1, 1]):
                    for k in range(H[2, 2]):
                        shift = np.array([i, j, k], dtype=float) @ prim_cell
                        pos.append(np.zeros(3) + shift)  # 1 atom/prim cell
            assert len(pos) == n_sc * n_b


class TestDisplacementOptimizer:
    """Smoke tests for displacement optimizer."""

    @pytest.fixture
    def sc_setup(self):
        """4×4×1 supercell of the simple cubic structure."""
        n = 8
        cell = prim_cell * n
        pos = []
        for i in range(n):
            for j in range(n):
                pos.append(np.array([i, j, 0], dtype=float) * a)
        pos = np.array(pos)
        masses = np.ones(len(pos)) * 4.0026  # He mass
        return cell, pos, masses

    def test_optimise_returns_correct_shape(self, sc_setup):
        cell, pos, masses = sc_setup
        N = len(pos)
        debye = DebyeCorrelator(T_D=26.0, M_amu=4.0026)
        optimizer = DisplacementOptimizer(
            target_correlator=debye, T=1.6, xi_T=5.0, r_max=7.0
        )
        u_opt, Q_final = optimizer.optimise(
            positions=pos, cell=cell, masses_amu=masses,
            n_sa_steps=100, T_sa_init=0.1, T_sa_final=1e-4,
        )
        assert u_opt.shape == (N, 3)
        assert isinstance(Q_final, float)

    def test_generate_ensemble_count(self, sc_setup):
        cell, pos, masses = sc_setup
        debye = DebyeCorrelator(T_D=26.0, M_amu=4.0026)
        optimizer = DisplacementOptimizer(
            target_correlator=debye, T=1.6, xi_T=5.0, r_max=7.0
        )
        ensemble = optimizer.generate_ensemble(
            positions=pos, cell=cell, masses_amu=masses, n_members=3
        )
        assert len(ensemble) == 3
        for u in ensemble:
            assert u.shape == pos.shape

    def test_quality_non_negative(self, sc_setup):
        cell, pos, masses = sc_setup
        debye = DebyeCorrelator(T_D=26.0, M_amu=4.0026)
        optimizer = DisplacementOptimizer(
            target_correlator=debye, T=1.6, xi_T=5.0, r_max=7.0
        )
        # Zero displacements → Q should be finite and ≥ 0
        u_zero = np.zeros_like(pos)
        shells, target_C2 = optimizer.find_shells(pos, cell)
        Q = optimizer.quality(u_zero, pos, cell, shells, target_C2)
        assert Q >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
