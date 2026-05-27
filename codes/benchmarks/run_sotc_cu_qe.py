#!/usr/bin/env python3
"""
SQTC + Quantum ESPRESSO benchmark: fcc Cu at 1200 K  (strongly anharmonic)
===========================================================================
Cu at T/T_melt = 0.88 is a stringent anharmonicity test; SQTC typically
requires 6-8 self-consistent iterations to converge the renormalised IFCs.

QE settings  (SSSP efficiency pseudopotential)
-----------------------------------------------
  Pseudopotential : Cu.paw.z_11.ld1.psl.v1.0.0-low.upf  (PAW, 11 val. e⁻)
  ecutwfc         : 45 Ry  (≈ 612 eV)  [SSSP recommended]
  ecutrho         : 360 Ry  (8 × ecutwfc for PAW)
  k-mesh (SC)     : 2×2×2  (scales from 8×8×8 on the primitive cell)
  Smearing        : Marzari-Vanderbilt cold smearing, σ = 0.02 Ry
  SCF convergence : ΔE_tot < 1×10⁻⁸ Ry

Usage
-----
    QE_PW_CMD="mpirun -np 64 pw.x" python3 codes/examples/run_sotc_cu_qe.py
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "../"))

from sotc import SOTCRunner
from sotc.qe_io import QEForceCalculator

# ── Physical parameters: fcc Cu ──────────────────────────────────────────────

a_cu     = 3.67           # Å  (experimental fcc Cu at ~1200 K)
M_Cu     = 63.546         # amu
T_D_Cu   = 310.0          # K  (seed Debye temperature)
T_design = 1200.0         # K  (T/T_melt = 0.88 — strongly anharmonic)

N_SC = 32                 # 32-atom 2×2×2 fcc supercell

prim_cell_cu = 0.5 * a_cu * np.array([
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
])

prim_pos_cu = np.array([[0.0, 0.0, 0.0]])

WORK_DIR = Path("examples/sotc_cu_qe_run")

# ── QE settings ───────────────────────────────────────────────────────────────

# Default: pseudopotentials/ bundled alongside this repository
# Override with the SSSP_DIR environment variable if needed.
SSP_DIR = Path(
    os.environ.get(
        "SSSP_DIR",
        str(Path(__file__).parent.parent.parent / "pseudopotentials"),
    )
)
CU_UPF = "Cu.paw.z_11.ld1.psl.v1.0.0-low.upf"   # PAW, 11 valence electrons

PW_CMD = os.environ.get(
    "QE_PW_CMD",
    "mpiexec -np 1 " + os.path.expanduser("~/Softwares/qe-7.5/bin/pw.x"),
)

# ── QE force calculator ───────────────────────────────────────────────────────

qe_calc = QEForceCalculator(
    species=["Cu"],
    pseudopotentials={"Cu": CU_UPF},
    pseudo_dir=SSP_DIR,
    ecutwfc=45.0,                               # Ry  (SSSP efficiency)
    ecutrho=360.0,                              # Ry  (8 × ecutwfc for PAW)
    kpts=(2, 2, 2),                             # 2×2×2 for 32-atom fcc supercell
    smearing="marzari-vanderbilt",
    degauss=0.02,                               # Ry
    conv_thr=1.0e-8,                            # Ry
    pw_cmd=PW_CMD,
    workdir=WORK_DIR / "qe_scratch",
    prefix="cu",
    n_parallel=1,
)

# ── SQTC runner ───────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 68)
    print(" SQTC + Quantum ESPRESSO: fcc Cu at 1200 K")
    print("=" * 68)
    print(f"  Supercell size   : {N_SC} atoms")
    print(f"  r_cutoff         : 4.5 Å  (1st + 2nd NN for fcc Cu)")
    print(f"  T_design         : {T_design} K  (T/T_melt = {T_design/1358:.2f})")
    print(f"  T_D (input)      : {T_D_Cu} K")
    print(f"  ecutwfc          : {qe_calc.ecutwfc:.0f} Ry  ({qe_calc.ecutwfc * 13.6057:.0f} eV)")
    print(f"  ecutrho          : {qe_calc.ecutrho:.0f} Ry")
    print(f"  k-mesh (SC)      : {qe_calc.kpts}")
    print(f"  pw_cmd           : {qe_calc.pw_cmd}")
    print()

    runner = SOTCRunner(
        element="Cu",
        mass_amu=M_Cu,
        prim_cell=prim_cell_cu,
        prim_positions=prim_pos_cu,
        T=T_design,
        T_D=T_D_Cu,
        n_atoms_sc=N_SC,
        force_calculator=qe_calc,
        r_cutoff=4.5,
        r_max_corr=9.0,
        n_ensemble=8,
        work_dir=WORK_DIR,
        verbosity=1,
        ridge_alpha=1e-3,
        symmetrize_bonds=True,
    )

    results = runner.run(
        n_sc_iterations=10,
        epsilon_conv=0.003,
        min_iterations=4,
        mixing=0.20,
        T_values=np.array([100., 200., 300., 500., 700., 900., 1000., 1200.]),
        q_mesh_cv=(8, 8, 8),
    )

    print("\n" + "─" * 60)
    print(" SQTC Cu benchmark summary")
    print("─" * 60)
    print(f"  Converged  : {results['converged']}")
    print(f"  Iterations : {results['n_iterations']}")
    print(f"  T_D        : {results['T_D_effective']:.1f} K  (expt: 315 K)")
    print()
    print(f"  {'T [K]':>8}  {'C_V [J/mol/K]':>14}  {'Expt':>10}")
    print("  " + "-" * 38)
    expt = {300.: "~24.5", 700.: "~25.2", 1200.: "~27.0"}
    for T_v, cv_v in zip(results["T_values"], results["C_V_scan"]):
        print(f"  {T_v:8.0f}  {cv_v:14.4f}  {expt.get(T_v, '—'):>10}")
    print()
    print(f"  Results saved to {WORK_DIR}/sotc_results.json")


if __name__ == "__main__":
    main()
