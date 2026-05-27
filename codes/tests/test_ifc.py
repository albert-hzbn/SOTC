"""
Tests for sqtc.ifc_extractor — IFCExtractor

Verifies:
  1. IFCs extracted from a known 1D spring chain match analytic values
  2. Acoustic sum rule is satisfied after fitting
  3. Force prediction RMSE is below threshold for harmonic data
  4. Design matrix has correct shape
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from sotc.ifc_extractor import IFCExtractor


def make_linear_chain(n: int, a: float):
    """
    Simple 1D chain of n atoms with lattice spacing a.
    Cell: (n*a, 1, 1)
    """
    cell = np.diag([n * a, 10.0, 10.0])
    pos = np.zeros((n, 3))
    pos[:, 0] = np.arange(n) * a
    masses = np.ones(n) * 4.0026
    return cell, pos, masses


def chain_forces(positions, cell, k_spring):
    """
    Forces on 1D nearest-neighbour harmonic chain.
    F_i = -k(u_i - u_{i-1}) - k(u_i - u_{i+1})  [only x-component]
    Uses minimum-image convention.
    """
    n = len(positions)
    forces = np.zeros((n, 3))
    a = cell[0, 0] / n  # equilibrium spacing

    for i in range(n):
        j_left  = (i - 1) % n
        j_right = (i + 1) % n

        dx_left  = positions[j_left,  0] - positions[i, 0]
        dx_right = positions[j_right, 0] - positions[i, 0]

        # Minimum image
        Lx = cell[0, 0]
        dx_left  -= Lx * round(dx_left  / Lx)
        dx_right -= Lx * round(dx_right / Lx)

        # Harmonic: F = k * Δx (spring from equilibrium = stretched by Δu)
        # Here positions are already displaced, so Δx = actual separation − a
        dx_left_eq  = dx_left  + a   # equilibrium is -a to the left
        dx_right_eq = dx_right - a   # equilibrium is +a to the right

        forces[i, 0] += k_spring * dx_left_eq + k_spring * dx_right_eq

    return forces


class TestIFCExtractor:
    """Tests on a simple 1D harmonic chain."""

    @pytest.fixture
    def chain_setup(self):
        n = 8
        a = 3.57
        k_spring = 0.5   # eV/Å²
        cell, pos, masses = make_linear_chain(n, a)
        return cell, pos, masses, k_spring, n, a

    def test_design_matrix_shape(self, chain_setup):
        cell, pos, masses, k, n, a = chain_setup
        extractor = IFCExtractor(pos, cell, masses, r_cutoff=a*1.5, symmetrise=False)
        u = np.zeros_like(pos)
        u[0, 0] = 0.1  # small displacement on atom 0
        M = extractor._build_design_matrix(u)
        # Shape: (3N, 9 * n_pairs) for a single snapshot
        assert M.shape[0] == 3 * n
        assert M.shape[1] > 0

    def test_ifc_recovery(self, chain_setup):
        """
        Fit IFCs from harmonic chain data; nearest-neighbour Φ_xx should ≈ k.
        """
        cell, pos, masses, k_spring, n, a = chain_setup
        extractor = IFCExtractor(pos, cell, masses, r_cutoff=a*1.5, symmetrise=True)

        # Generate multiple snapshots with random small displacements
        rng = np.random.default_rng(0)
        disps, forces = [], []
        for _ in range(4 * n):
            u = np.zeros((n, 3))
            u[:, 0] = rng.normal(0, 0.05, n)  # small x displacements only
            pos_disp = pos + u
            F = chain_forces(pos_disp, cell, k_spring)
            disps.append(u)
            forces.append(F)

        Phi = extractor.fit(disps, forces)

        # Check nearest-neighbour Φ_{01,xx} ≈ −k_spring
        # (off-diagonal IFC for a stretch-restoring spring is negative)
        nn_pair = None
        for (i, j), phi in Phi.items():
            if i == 0:
                dx = pos[j, 0] - pos[i, 0]
                dx -= cell[0, 0] * round(dx / cell[0, 0])
                if abs(abs(dx) - a) < 0.01:
                    nn_pair = (i, j, phi)
                    break

        assert nn_pair is not None, "No nearest-neighbour pair (0, ?) found"
        phi_xx = nn_pair[2][0, 0]
        assert abs(phi_xx - (-k_spring)) / k_spring < 0.20, (
            f"Φ_xx = {phi_xx:.4f}, expected ≈ {-k_spring:.4f}"
        )

    def test_acoustic_sum_rule(self, chain_setup):
        """After fitting, on-site IFC should equal −Σ_{j≠i} Φ_{ij} (ASR)."""
        cell, pos, masses, k_spring, n, a = chain_setup
        extractor = IFCExtractor(pos, cell, masses, r_cutoff=a*1.5, symmetrise=True)

        rng = np.random.default_rng(1)
        disps, forces = [], []
        for _ in range(3 * n):
            u = np.zeros((n, 3))
            u[:, 0] = rng.normal(0, 0.05, n)
            disps.append(u)
            forces.append(chain_forces(pos + u, cell, k_spring))

        Phi = extractor.fit(disps, forces)

        # Check ASR on atom 0: Φ_{00} + Σ_{j≠0} Φ_{0j} ≈ 0
        phi_00 = Phi.get((0, 0), np.zeros((3, 3)))
        phi_sum = phi_00.copy()
        for (i, j), phi in Phi.items():
            if i == 0 and j != 0:
                phi_sum += phi

        asr_residual = np.abs(phi_sum).max()
        assert asr_residual < 0.1, f"ASR residual = {asr_residual:.4f} eV/Å²"

    def test_fit_report(self, chain_setup):
        """fit_report() should return R² > 0.9 for harmonic data."""
        cell, pos, masses, k_spring, n, a = chain_setup
        extractor = IFCExtractor(pos, cell, masses, r_cutoff=a*1.5, symmetrise=True)

        rng = np.random.default_rng(2)
        disps, forces = [], []
        for _ in range(3 * n):
            u = np.zeros((n, 3))
            u[:, 0] = rng.normal(0, 0.05, n)
            disps.append(u)
            forces.append(chain_forces(pos + u, cell, k_spring))

        extractor.fit(disps, forces)
        report = extractor.fit_report(disps, forces)
        assert report["r2"] > 0.9, f"R² = {report['r2']:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
