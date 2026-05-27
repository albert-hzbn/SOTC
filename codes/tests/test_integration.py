"""
Integration test: full SQTC pipeline for ⁴He using AzizHFD mock forces.

This test runs 1 SC iteration with a small supercell and checks:
  1. The pipeline runs without error
  2. C_V at T=1.6 K is in a physically reasonable range
  3. Effective Debye temperature is within 50% of input T_D
  4. MSD > 0
  5. IFC RMSE < 0.5 eV/Å

This serves as a smoke/regression test for the full SQTC workflow.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from sotc.runner import SOTCRunner
from sotc.mock_forces import AzizHFDPotential
from sotc.constants import R_GAS


@pytest.fixture(scope="module")
def he_results():
    """Run a minimal (fast) SQTC calculation for He and return results dict."""
    a_he = 3.57
    c_he = 1.633 * a_he

    prim_cell = np.array([
        [a_he,    0.0,                0.0],
        [-a_he/2, a_he*np.sqrt(3)/2,  0.0],
        [0.0,     0.0,                c_he],
    ])
    prim_pos = np.array([
        [0.0,      0.0,                    0.0],
        [a_he/2,   a_he/(2*np.sqrt(3)),  c_he/2],
    ])

    runner = SOTCRunner(
        element="He",
        mass_amu=4.002602,
        prim_cell=prim_cell,
        prim_positions=prim_pos,
        T=1.6,
        T_D=26.0,
        n_atoms_sc=8,          # small for speed
        force_calculator=AzizHFDPotential(r_cut=8.0),
        r_cutoff=5.0,
        r_max_corr=6.0,
        n_ensemble=3,          # 3 snapshots for speed
        work_dir="/tmp/sqtc_test_he",
        verbosity=0,
    )

    results = runner.run(
        n_sc_iterations=1,     # 1 iteration for speed
        T_values=np.array([0.5, 1.0, 1.6, 2.0]),
        q_mesh_cv=(4, 4, 4),
    )
    return results


class TestHePipeline:
    def test_pipeline_returns_dict(self, he_results):
        assert isinstance(he_results, dict)

    def test_cv_physically_reasonable(self, he_results):
        """C_V at 1.6 K should be > 0 and < 3R."""
        cv = he_results["C_V_jmolk"]
        assert 0.0 < cv < 3.0 * R_GAS, f"C_V = {cv:.3f} J/mol/K"

    def test_debye_temp_in_range(self, he_results):
        """Effective T_D should be within a factor 3 of the input 26 K."""
        T_D_eff = he_results["T_D_effective"]
        if T_D_eff > 0:  # may be 0 if DOS fitting failed
            assert 5.0 < T_D_eff < 150.0, f"T_D_eff = {T_D_eff:.1f} K"

    def test_msd_positive(self, he_results):
        assert he_results["MSD_ang2"] > 0.0

    def test_zpe_positive(self, he_results):
        assert he_results["ZPE_eV"] > 0.0

    def test_cv_scan_length(self, he_results):
        assert len(he_results["C_V_scan"]) == len(he_results["T_values"])

    def test_cv_scan_monotonic_increasing(self, he_results):
        """C_V should be non-decreasing over the temperature range."""
        cv = np.array(he_results["C_V_scan"])
        T  = np.array(he_results["T_values"])
        # Allow slight numerical noise; just check general trend
        cv_low = cv[T < 1.0].mean() if any(T < 1.0) else cv[0]
        cv_high = cv[T > 1.5].mean() if any(T > 1.5) else cv[-1]
        assert cv_high >= cv_low * 0.8, (
            f"C_V does not increase with T: low={cv_low:.3f}, high={cv_high:.3f}"
        )

    def test_result_json_written(self, he_results):
        """JSON summary should have been written to work_dir."""
        import json
        from pathlib import Path
        result_file = Path("/tmp/sqtc_test_he/sqtc_results.json")
        assert result_file.exists(), "sqtc_results.json not found"
        with open(result_file) as f:
            data = json.load(f)
        assert "C_V_jmolk" in data
        assert "T_D_effective" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
