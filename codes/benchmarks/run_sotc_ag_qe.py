#!/usr/bin/env python3
"""
SQTC + Quantum ESPRESSO benchmark: fcc Ag at 1000 K  (moderately anharmonic)
=============================================================================
Ag at T/T_melt = 0.81 is a good complement to Au and Cu: lighter noble metal,
higher T_D (~225 K), and well-measured phonon dispersion and heat capacity.

QE settings  (SSSP efficiency v1.3 pseudopotential)
-----------------------------------------------
  Pseudopotential : Ag_ONCV_PBE-1.0.oncvpsp.upf  (NC ONCV SG15)
  ecutwfc         : 50 Ry  (≈ 680 eV)  [SSSP recommended]
  ecutrho         : 200 Ry  (4 × ecutwfc for NC)
  k-mesh (SC)     : 2×2×2  (scales from 8×8×8 on the primitive cell)
  Smearing        : Marzari-Vanderbilt cold smearing, σ = 0.02 Ry
  SCF convergence : ΔE_tot < 1×10⁻⁸ Ry

Experimental references:
  T_D (calorimetric, low-T)  ≈ 225 K   (Kittel)
  T_D (effective, room-T)    ≈ 215 K   (Grimvall 1981)
  α(300 K) = 18.9×10⁻⁶ K⁻¹            (CRC Handbook)
  Cv(300 K) ≈ 25.3 J/mol/K             (NIST Webbook)
  T_melt = 1235 K

Usage
-----
    QE_PW_CMD="mpirun -np 64 pw.x" python3 codes/benchmarks/run_sotc_ag_qe.py
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "../"))

from sotc import SOTCRunner
from sotc.qe_io import QEForceCalculator

# ── Physical parameters: fcc Ag ──────────────────────────────────────────────

a_ag     = 4.12           # Å  (experimental fcc Ag at ~1000 K; DFT-PBE ≈ 4.15 Å)
M_Ag     = 107.8682       # amu
T_D_Ag   = 220.0          # K  (seed Debye temperature)
T_design = 1000.0         # K  (T/T_melt = 0.81 — strongly anharmonic)

N_SC = 32                 # 32-atom 2×2×2 fcc supercell

prim_cell_ag = 0.5 * a_ag * np.array([
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
])

prim_pos_ag = np.array([[0.0, 0.0, 0.0]])

WORK_DIR = Path("examples/sotc_ag_qe_run")

# ── QE settings ───────────────────────────────────────────────────────────────

SSP_DIR = Path(
    os.environ.get(
        "SSSP_DIR",
        str(Path(__file__).parent.parent.parent / "pseudopotentials"),
    )
)
AG_UPF = "Ag_ONCV_PBE-1.0.oncvpsp.upf"   # NC ONCV SG15

PW_CMD = os.environ.get(
    "QE_PW_CMD",
    "mpiexec -np 1 " + os.path.expanduser("~/Softwares/qe-7.5/bin/pw.x"),
)

# ── QE force calculator ───────────────────────────────────────────────────────

qe_calc = QEForceCalculator(
    species=["Ag"],
    pseudopotentials={"Ag": AG_UPF},
    pseudo_dir=SSP_DIR,
    ecutwfc=50.0,                               # Ry  (SSSP efficiency)
    ecutrho=200.0,                              # Ry  (4 × ecutwfc for NC ONCV)
    kpts=(2, 2, 2),                             # 2×2×2 for 32-atom fcc supercell
    smearing="marzari-vanderbilt",
    degauss=0.02,                               # Ry
    conv_thr=1.0e-8,                            # Ry
    pw_cmd=PW_CMD,
    workdir=WORK_DIR / "qe_scratch",
    prefix="ag",
    n_parallel=1,
)

# ── SQTC runner ───────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 68)
    print(" SQTC + Quantum ESPRESSO: fcc Ag at 1000 K")
    print("=" * 68)
    print(f"  Supercell size   : {N_SC} atoms")
    print(f"  r_cutoff         : 4.5 Å  (1st + 2nd NN for fcc Ag)")
    print(f"  T_design         : {T_design} K  (T/T_melt = {T_design/1235:.2f})")
    print(f"  T_D (input)      : {T_D_Ag} K")
    print(f"  ecutwfc          : {qe_calc.ecutwfc:.0f} Ry")
    print(f"  ecutrho          : {qe_calc.ecutrho:.0f} Ry")
    print(f"  k-mesh (SC)      : {qe_calc.kpts}")
    print(f"  pw_cmd           : {qe_calc.pw_cmd}")
    print()

    runner = SOTCRunner(
        element="Ag",
        mass_amu=M_Ag,
        prim_cell=prim_cell_ag,
        prim_positions=prim_pos_ag,
        T=T_design,
        T_D=T_D_Ag,
        n_atoms_sc=N_SC,
        force_calculator=qe_calc,
        r_cutoff=4.5,
        r_max_corr=9.0,
        n_ensemble=5,       # T/T_melt = 0.81; 5 snaps balance cost vs noise
        work_dir=WORK_DIR,
        verbosity=1,
        ridge_alpha=1e-3,
        symmetrize_bonds=True,
    )

    T_values = np.array([100., 200., 300., 400., 500., 700., 900., 1000., 1100.])

    res = runner.run(
        n_sc_iterations=12,
        epsilon_conv=0.003,
        min_iterations=4,
        mixing=0.25,
        T_values=T_values,
        q_mesh_cv=(20, 20, 20),
    )

    print("\n── Final properties ──")
    cv300 = None
    for T, cv in zip(res["T_values"], res["C_V_scan"]):
        print(f"  Cv({T:.0f} K) = {cv:.4f} J/(mol·K)")
        if abs(T - 300.0) < 1:
            cv300 = cv

    print(f"\n  T_D (spectral-moment) = {res['T_D_effective']:.2f} K  (expt low-T: ~225 K)")
    if cv300 is not None:
        print(f"  Cv(300 K)             = {cv300:.4f} J/(mol·K)  (expt: 25.3)")
    print(f"  ZPE                   = {res.get('ZPE_meV', float('nan')):.2f} meV/atom")
    print(f"  Converged             : {res['converged']}")
    print(f"  Iterations            : {res['n_iterations']}")


if __name__ == "__main__":
    main()
