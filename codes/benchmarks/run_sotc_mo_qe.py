#!/usr/bin/env python3
"""
SQTC + Quantum ESPRESSO benchmark: bcc Mo at 1500 K  (moderately anharmonic)
=============================================================================
BCC Mo is harmonically stable (no imaginary modes), making it a clean test
of anharmonic IFC renormalisation in a refractory transition metal.

QE settings  (SSSP efficiency pseudopotential)
-----------------------------------------------
  Pseudopotential : Mo_ONCV_PBE-1.0.oncvpsp.upf  (NC ONCV, SSSP efficiency)
  ecutwfc         : 40 Ry  (≈ 544 eV)  [SSSP recommended]
  ecutrho         : 160 Ry  (4 × ecutwfc for NC ONCV)
  k-mesh (SC)     : 2×2×2  (appropriate for 64-atom bcc supercell)
  Smearing        : Marzari-Vanderbilt cold smearing, σ = 0.02 Ry
  SCF convergence : ΔE_tot < 1×10⁻⁸ Ry

Usage
-----
    QE_PW_CMD="mpirun -np 64 pw.x" python3 codes/examples/run_sotc_mo_qe.py
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "../"))

from sotc import SOTCRunner
from sotc.qe_io import QEForceCalculator

# ── Physical parameters: bcc Mo ──────────────────────────────────────────────

a_mo     = 3.16           # Å  (experimental bcc Mo at ~1500 K)
M_Mo     = 95.96          # amu
T_D_Mo   = 430.0          # K  (seed Debye temperature, calorimetric ≈ 450 K)
T_design = 1500.0         # K  (T/T_melt = 0.52)

N_SC = 64                 # 64-atom bcc supercell

# BCC primitive cell (1 atom per cell, body-centred)
prim_cell_mo = 0.5 * a_mo * np.array([
    [-1.0,  1.0,  1.0],
    [ 1.0, -1.0,  1.0],
    [ 1.0,  1.0, -1.0],
])

prim_pos_mo = np.array([[0.0, 0.0, 0.0]])

WORK_DIR = Path("examples/sotc_mo_qe_run")

# ── QE settings ───────────────────────────────────────────────────────────────

# Default: pseudopotentials/ bundled alongside this repository
# Override with the SSSP_DIR environment variable if needed.
SSP_DIR = Path(
    os.environ.get(
        "SSSP_DIR",
        str(Path(__file__).parent.parent.parent / "pseudopotentials"),
    )
)
MO_UPF = "Mo_ONCV_PBE-1.0.oncvpsp.upf"   # NC ONCV pseudopotential

PW_CMD = os.environ.get(
    "QE_PW_CMD",
    "mpiexec -np 1 " + os.path.expanduser("~/Softwares/qe-7.5/bin/pw.x"),
)

# ── QE force calculator ───────────────────────────────────────────────────────

qe_calc = QEForceCalculator(
    species=["Mo"],
    pseudopotentials={"Mo": MO_UPF},
    pseudo_dir=SSP_DIR,
    ecutwfc=40.0,                               # Ry  (SSSP efficiency for Mo ONCV)
    ecutrho=160.0,                              # Ry  (4 × ecutwfc for NC ONCV)
    kpts=(2, 2, 2),                             # 2×2×2 for 64-atom bcc supercell
    smearing="marzari-vanderbilt",
    degauss=0.02,                               # Ry  (Mo refractory metal)
    conv_thr=1.0e-8,                            # Ry
    pw_cmd=PW_CMD,
    workdir=WORK_DIR / "qe_scratch",
    prefix="mo",
    n_parallel=1,
)

# ── SQTC runner ───────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 68)
    print(" SQTC + Quantum ESPRESSO: bcc Mo at 1500 K")
    print("=" * 68)
    print(f"  Supercell size   : {N_SC} atoms (bcc)")
    print(f"  r_cutoff         : 5.0 Å  (1st + 2nd NN for bcc Mo)")
    print(f"  T_design         : {T_design} K  (T/T_melt = {T_design/2896:.2f})")
    print(f"  T_D (input)      : {T_D_Mo} K")
    print(f"  ecutwfc          : {qe_calc.ecutwfc:.0f} Ry  ({qe_calc.ecutwfc * 13.6057:.0f} eV)")
    print(f"  ecutrho          : {qe_calc.ecutrho:.0f} Ry")
    print(f"  k-mesh (SC)      : {qe_calc.kpts}")
    print(f"  pw_cmd           : {qe_calc.pw_cmd}")
    print()

    runner = SOTCRunner(
        element="Mo",
        mass_amu=M_Mo,
        prim_cell=prim_cell_mo,
        prim_positions=prim_pos_mo,
        T=T_design,
        T_D=T_D_Mo,
        n_atoms_sc=N_SC,
        force_calculator=qe_calc,
        r_cutoff=5.0,
        r_max_corr=10.0,
        n_ensemble=1,
        work_dir=WORK_DIR,
        verbosity=1,
        ridge_alpha=1e-3,
        symmetrize_bonds=True,
    )

    results = runner.run(
        n_sc_iterations=10,
        epsilon_conv=0.003,
        mixing=0.30,
        T_values=np.array([300., 600., 900., 1000., 1200., 1500.]),
        q_mesh_cv=(8, 8, 8),
    )

    print("\n" + "─" * 60)
    print(" SQTC Mo benchmark summary")
    print("─" * 60)
    print(f"  Converged  : {results['converged']}")
    print(f"  Iterations : {results['n_iterations']}")
    print(f"  T_D        : {results['T_D_effective']:.1f} K  (expt calorimetric: ~450 K)")
    print()
    print(f"  {'T [K]':>8}  {'C_V [J/mol/K]':>14}  {'Expt':>10}")
    print("  " + "-" * 38)
    expt = {600.: "~25.3", 1000.: "~26.3", 1200.: "~26.7", 1500.: "~27.5"}
    for T_v, cv_v in zip(results["T_values"], results["C_V_scan"]):
        print(f"  {T_v:8.0f}  {cv_v:14.4f}  {expt.get(T_v, '—'):>10}")
    print()
    print(f"  Results saved to {WORK_DIR}/sotc_results.json")


if __name__ == "__main__":
    main()
