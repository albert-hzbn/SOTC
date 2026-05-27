#!/usr/bin/env python3
"""
SQTC + Quantum ESPRESSO benchmark: MgO rocksalt at 1500 K  (low anharmonicity)
================================================================================
MgO is the stiff-ionic benchmark: high T_D (~740 K), small Grüneisen parameter
(γ ≈ 1.5).  SQTC converges close to harmonic DFT at this temperature.

QE settings  (SSSP efficiency pseudopotentials)
-----------------------------------------------
  Mg : Mg.pbe-n-kjpaw_psl.0.3.0.UPF  (PAW)
  O  : O.pbe-n-kjpaw_psl.0.1.UPF     (PAW)
  ecutwfc         : 80 Ry  (O drives the cutoff; Mg=35, O=80)
  ecutrho         : 640 Ry  (8 × ecutwfc for PAW)
  k-mesh (SC)     : 1×1×1  (64-atom rocksalt supercell, Γ-only)
  Occupations     : fixed  (insulator, band gap ~7.8 eV)
  SCF convergence : ΔE_tot < 1×10⁻⁸ Ry

Usage
-----
    QE_PW_CMD="mpirun -np 128 pw.x" python3 codes/examples/run_sotc_mgo_qe.py
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "../"))

from sotc import SOTCRunner
from sotc.qe_io import QEForceCalculator

# ── Physical parameters: MgO rocksalt ────────────────────────────────────────

a_mgo    = 4.26           # Å  (experimental MgO at ~1500 K)
M_Mg     = 24.305         # amu
M_O      = 15.999         # amu
T_D_MgO  = 740.0          # K  (seed; PBE harmonic gives 720–760 K)
T_design = 1500.0         # K  (T/T_D = 2.0; classical regime)

N_SC = 64                 # 64-atom rocksalt supercell (32 f.u.)

# Rocksalt = FCC Bravais + 2-atom basis
prim_cell_mgo = 0.5 * a_mgo * np.array([
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
])
prim_pos_mgo = np.array([
    [0.0,      0.0, 0.0],   # Mg at origin
    [a_mgo/2,  0.0, 0.0],   # O  at (a/2, 0, 0)
])

WORK_DIR = Path("examples/sotc_mgo_qe_run")

# ── QE settings ───────────────────────────────────────────────────────────────

# Default: pseudopotentials/ bundled alongside this repository
# Override with the SSSP_DIR environment variable if needed.
SSP_DIR = Path(
    os.environ.get(
        "SSSP_DIR",
        str(Path(__file__).parent.parent.parent / "pseudopotentials"),
    )
)
SSP_DIR = SSP_DIR  # alias kept for clarity

MG_UPF = "Mg.pbe-n-kjpaw_psl.0.3.0.UPF"
O_UPF  = "O.pbe-n-kjpaw_psl.0.1.UPF"

PW_CMD = os.environ.get(
    "QE_PW_CMD",
    "mpiexec -np 1 " + os.path.expanduser("~/Softwares/qe-7.5/bin/pw.x"),
)

# ── QE force calculator ───────────────────────────────────────────────────────
# MgO is an insulator (band gap ~7.8 eV): use smearing=None → occupations='fixed'

qe_calc = QEForceCalculator(
    species=["Mg", "O"],
    pseudopotentials={"Mg": MG_UPF, "O": O_UPF},
    pseudo_dir=SSP_DIR,
    ecutwfc=80.0,                               # Ry  (O PAW drives cutoff)
    ecutrho=640.0,                              # Ry  (8 × ecutwfc for PAW)
    kpts=(1, 1, 1),                             # Γ-only for 64-atom supercell
    smearing=None,                              # insulator: fixed occupations
    conv_thr=1.0e-8,                            # Ry
    pw_cmd=PW_CMD,
    workdir=WORK_DIR / "qe_scratch",
    prefix="mgo",
    n_parallel=1,
)

# ── SQTC runner ───────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 68)
    print(" SQTC + Quantum ESPRESSO: MgO rocksalt at 1500 K")
    print("=" * 68)
    print(f"  Supercell size   : {N_SC} atoms (32 f.u.)")
    print(f"  r_cutoff         : 3.2 Å  (captures Mg-O and Mg-Mg NN shells)")
    print(f"  T_design         : {T_design} K  (T/T_D = {T_design/T_D_MgO:.1f})")
    print(f"  T_D (input)      : {T_D_MgO} K")
    print(f"  ecutwfc          : {qe_calc.ecutwfc:.0f} Ry  ({qe_calc.ecutwfc * 13.6057:.0f} eV)")
    print(f"  ecutrho          : {qe_calc.ecutrho:.0f} Ry")
    print(f"  k-mesh (SC)      : {qe_calc.kpts}  (Γ-only, insulator)")
    print(f"  pw_cmd           : {qe_calc.pw_cmd}")
    print()

    runner = SOTCRunner(
        element="Mg",
        mass_amu=M_Mg,
        elements=["Mg", "O"],
        masses_amu=[M_Mg, M_O],
        prim_cell=prim_cell_mgo,
        prim_positions=prim_pos_mgo,
        T=T_design,
        T_D=T_D_MgO,
        n_atoms_sc=N_SC,
        force_calculator=qe_calc,
        r_cutoff=3.2,
        r_max_corr=9.0,
        n_ensemble=3,
        work_dir=WORK_DIR,
        verbosity=1,
        ridge_alpha=1e-3,
        symmetrize_bonds=True,
    )

    results = runner.run(
        n_sc_iterations=10,
        epsilon_conv=0.003,
        min_iterations=4,
        mixing=0.30,
        T_values=np.array([300., 600., 900., 1000., 1200., 1500.]),
        q_mesh_cv=(8, 8, 8),
    )

    print("\n" + "─" * 60)
    print(" SQTC MgO benchmark summary")
    print("─" * 60)
    print(f"  Converged  : {results['converged']}")
    print(f"  Iterations : {results['n_iterations']}")
    print(f"  T_D        : {results['T_D_effective']:.1f} K  (expt: ~760 K)")
    print()
    print(f"  {'T [K]':>8}  {'C_V [J/mol/K]':>14}  {'Expt':>10}")
    print("  " + "-" * 38)
    expt = {300.: "~37.5", 600.: "~47.6", 900.: "~49.5", 1200.: "~50.1", 1500.: "~50.8"}
    for T_v, cv_v in zip(results["T_values"], results["C_V_scan"]):
        print(f"  {T_v:8.0f}  {cv_v:14.4f}  {expt.get(T_v, '—'):>10}")
    print()
    print(f"  Results saved to {WORK_DIR}/sotc_results.json")


if __name__ == "__main__":
    main()
