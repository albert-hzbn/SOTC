#!/usr/bin/env python3
"""
SQTC + Quantum ESPRESSO QHA: NaCl rocksalt thermal expansion benchmark
=======================================================================
Runs SQTC at three volumes (V₋, V₀, V₊, δ=3%) then builds a QHA to compute
the thermal expansion coefficient α(T), equilibrium volume V(T), C_P(T), and
bulk modulus B(T) for NaCl.

NaCl has a large Grüneisen parameter (γ ≈ 1.6) and very small bulk modulus
(B₀ ≈ 24 GPa), resulting in the largest thermal expansion coefficient among
the benchmark materials: α(300 K) ≈ 40×10⁻⁶ K⁻¹.

Note: LO-TO splitting is absent in short-range IFC framework; SQTC recovers
TO-branch phonons only. This slightly underestimates F_vib but preserves the
dominant Grüneisen contribution.

QE settings  (SSSP efficiency v1.3 pseudopotentials)
-----------------------------------------------
  Na : na_pbe_v1.5.uspp.F.UPF  (USPP GBRV-1.5)
  Cl : cl_pbe_v1.4.uspp.F.UPF  (USPP GBRV-1.4)
  SQTC ecutwfc    : 60 Ry  (Cl drives the cutoff; 480 Ry ecutrho)
  Static ecutwfc  : 65 Ry  (slightly tighter for EOS accuracy)
  k-mesh SQTC SC  : 1×1×1  (64-atom insulating supercell, Γ-only)
  k-mesh static   : 6×6×6  (2-atom primitive cell)
  Occupations     : fixed  (insulator, band gap ~8.9 eV)

Experimental references:
  α(300 K) = 40.4×10⁻⁶ K⁻¹  (Touloukian et al. 1977)
  α(600 K) = 48.0×10⁻⁶ K⁻¹  (Touloukian et al. 1977)
  B₀ = 24.5 GPa              (Simmons & Wang 1971)
  V₀ ≈ 22.4 Å³/atom          (at 300 K, expt; PBE ≈ 22.6 Å³/atom)

Usage
-----
    QE_PW_CMD="mpirun -np 64 pw.x" python3 codes/benchmarks/run_sotc_nacl_qha_qe.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "../"))

from sotc import SOTCRunner, SOTCQuasiHarmonic
from sotc.ifc_extractor import IFCExtractor
from sotc.phonons import PhononCalculator
from sotc.qe_io import QEForceCalculator

# ── Physical parameters: NaCl rocksalt ───────────────────────────────────────

a_nacl   = 5.66           # Å  (DFT-PBE equilibrium; expt 5.64 Å at RT)
M_Na     = 22.9898        # amu
M_Cl     = 35.453         # amu
T_D_NaCl = 310.0          # K  (seed; calorimetric expt ≈ 321 K)
T_design = 300.0          # K  (SQTC displacement design temperature for QHA)
delta    = 0.03           # ±3% volume step

N_SC = 64                 # 64-atom rocksalt supercell (32 f.u.)

# Rocksalt FCC primitive cell (2-atom basis)
prim_cell_nacl = 0.5 * a_nacl * np.array([
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
])
prim_pos_nacl = np.array([
    [0.0,        0.0, 0.0],   # Na at origin
    [a_nacl/2,   0.0, 0.0],   # Cl at (a/2, 0, 0)
])
# V0 = volume of 2-atom primitive cell (Å³/f.u.)
V0_ang3 = abs(float(np.linalg.det(prim_cell_nacl)))   # Å³/f.u.

a_scales   = [(1 - delta) ** (1.0 / 3.0), 1.0, (1 + delta) ** (1.0 / 3.0)]
volumes    = [V0_ang3 * (1 - delta), V0_ang3, V0_ang3 * (1 + delta)]
vol_labels = ["Vm", "V0", "Vp"]

T_values = np.arange(50, 801, 50, dtype=float)

WORK_BASE = Path("examples/sotc_nacl_qha_qe_run")
WORK_BASE.mkdir(exist_ok=True)

# ── QE settings ───────────────────────────────────────────────────────────────

SSP_DIR = Path(
    os.environ.get(
        "SSSP_DIR",
        str(Path(__file__).parent.parent.parent / "pseudopotentials"),
    )
)
NA_UPF = "na_pbe_v1.5.uspp.F.UPF"
CL_UPF = "cl_pbe_v1.4.uspp.F.UPF"

PW_CMD = os.environ.get(
    "QE_PW_CMD",
    "mpiexec -np 1 " + os.path.expanduser("~/Softwares/qe-7.5/bin/pw.x"),
)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — SQTC at 3 volumes
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 68)
print(" SQTC QHA: NaCl rocksalt — Step 1: SQTC phonons at 3 volumes")
print("=" * 68)

phonon_calcs = []
sotc_results_per_vol = []

for iv, (ascale, vol) in enumerate(zip(a_scales, volumes)):
    label = vol_labels[iv]
    work_dir = WORK_BASE / f"sotc_{label}"
    cell_v = ascale * prim_cell_nacl

    a_v = a_nacl * ascale
    prim_pos_v = np.array([
        [0.0,   0.0, 0.0],
        [a_v/2, 0.0, 0.0],
    ])

    print(f"\n{'─'*60}")
    print(f" Volume {label}:  a = {a_nacl * ascale:.4f} Å   V = {vol:.4f} Å³/f.u.")
    print(f"{'─'*60}")

    results_json = work_dir / "sotc_results.json"
    if results_json.exists():
        print(f"  Loading cached SQTC results from {results_json}")
        with open(results_json) as f:
            cached = json.load(f)
        ifc_path = work_dir / f"iter_{cached['selected_iteration']:02d}" / "ifc_extractor.npz"
        if ifc_path.exists():
            ifc = IFCExtractor.load(str(ifc_path))
            calc = PhononCalculator(
                ifc_extractor=ifc,
                prim_positions=prim_pos_v,
                prim_cell=cell_v,
                masses_amu=np.array([M_Na, M_Cl]),
            )
            phonon_calcs.append(calc)
            sotc_results_per_vol.append(cached)
            continue
        else:
            print(f"  IFC file not found at {ifc_path}, re-running SQTC.")

    qe_calc = QEForceCalculator(
        species=["Na", "Cl"],
        pseudopotentials={"Na": NA_UPF, "Cl": CL_UPF},
        pseudo_dir=SSP_DIR,
        ecutwfc=60.0,
        ecutrho=480.0,
        kpts=(1, 1, 1),
        smearing=None,          # insulator: fixed occupations
        conv_thr=1.0e-8,
        pw_cmd=PW_CMD,
        workdir=work_dir / "qe_scratch",
        prefix=f"nacl_{label.lower()}",
        n_parallel=1,
    )

    runner = SOTCRunner(
        element="Na",
        mass_amu=M_Na,
        elements=["Na", "Cl"],
        masses_amu=[M_Na, M_Cl],
        prim_cell=cell_v,
        prim_positions=prim_pos_v,
        T=T_design,
        T_D=T_D_NaCl,
        n_atoms_sc=N_SC,
        force_calculator=qe_calc,
        r_cutoff=4.3 * ascale,
        r_max_corr=9.0 * ascale,
        n_ensemble=3,
        work_dir=str(work_dir),
        verbosity=1,
        ridge_alpha=1e-3,
        symmetrize_bonds=True,
    )

    res = runner.run(
        n_sc_iterations=10,
        epsilon_conv=0.003,
        min_iterations=4,
        mixing=0.30,
        T_values=np.array([200.0, 300.0, 400.0]),
        q_mesh_cv=(6, 6, 6),
    )
    phonon_calcs.append(res["phonon_calculator"])
    sotc_results_per_vol.append(res)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Static DFT for E₀(V)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 68)
print(" SQTC QHA: NaCl rocksalt — Step 2: Static DFT E₀(V)")
print("=" * 68)

static_energies = []

for iv, (ascale, vol) in enumerate(zip(a_scales, volumes)):
    label = vol_labels[iv]
    static_dir = WORK_BASE / f"static_{label}"
    energy_file = static_dir / "energy_eV.txt"

    if energy_file.exists():
        E0 = float(energy_file.read_text().strip())
        static_energies.append(E0)
        print(f"  {label}: E₀ = {E0:.6f} eV  (cached)")
        continue

    static_dir.mkdir(parents=True, exist_ok=True)
    a_v = a_nacl * ascale
    cell_v = ascale * prim_cell_nacl
    prim_pos_v = np.array([[0.0, 0.0, 0.0], [a_v/2, 0.0, 0.0]])

    qe_static = QEForceCalculator(
        species=["Na", "Cl"],
        pseudopotentials={"Na": NA_UPF, "Cl": CL_UPF},
        pseudo_dir=SSP_DIR,
        ecutwfc=65.0,
        ecutrho=520.0,
        kpts=(6, 6, 6),         # dense for 2-atom primitive cell
        smearing=None,
        conv_thr=1.0e-9,
        pw_cmd=PW_CMD,
        workdir=static_dir / "qe_scratch",
        prefix=f"nacl_static_{label.lower()}",
        n_parallel=1,
    )

    print(f"  {label}: Running static DFT  (a = {a_v:.4f} Å)...")
    E0 = qe_static.energy(positions=prim_pos_v, cell=cell_v)
    energy_file.write_text(str(E0))
    static_energies.append(E0)
    print(f"  {label}: E₀ = {E0:.6f} eV")

print(f"\n  EOS energies : {[f'{e:.6f}' for e in static_energies]} eV/f.u.")
print(f"  Volumes      : {[f'{v:.4f}' for v in volumes]} Å³/f.u.")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — QHA thermal expansion
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 68)
print(" SQTC QHA: NaCl rocksalt — Step 3: QHA thermal expansion")
print("=" * 68)

qha = SOTCQuasiHarmonic(
    volumes_ang3=volumes,
    static_energies_ev=static_energies,
    phonon_calcs=phonon_calcs,
    q_mesh=(20, 20, 20),
    method="both",
)

print(f"  B₀ (static parabolic EOS) = {qha.B0_GPa:.2f} GPa  (expt: 24.5 GPa)")
print(f"  V₀ = {qha.V0_ang3:.4f} Å³/f.u.  (expt @ 300 K: 44.8 Å³/f.u.)")

results = qha.run(T_values)

np.savez(
    WORK_BASE / "qha_results.npz",
    **{k: np.asarray(v) for k, v in results.items()},
)
print(f"\n  Saved QHA results → {WORK_BASE}/qha_results.npz")

print(f"\n  {'T [K]':>7}  {'V [Å³]':>9}  {'α×10⁶ [K⁻¹]':>13}  "
      f"{'α_gruen×10⁶':>13}  {'C_P [J/mol/K]':>14}  {'B [GPa]':>9}")
print("  " + "─" * 75)
for i, T in enumerate(results["T_K"]):
    if T in (100, 200, 300, 400, 600, 800):
        gruen = results.get("gruen_alpha_K", [float("nan")] * len(results["T_K"]))
        print(f"  {T:7.0f}  {results['V_ang3'][i]:9.4f}  "
              f"{results['alpha_K'][i]*1e6:13.2f}  "
              f"{gruen[i]*1e6:13.2f}  "
              f"{results['Cp_jmolk'][i]:14.4f}  "
              f"{results['B_GPa'][i]:9.2f}")

print()
print("  Experimental references (NaCl rocksalt):")
print("    α(300 K) = 40.4×10⁻⁶ K⁻¹  (Touloukian et al. 1977)")
print("    α(600 K) = 48.0×10⁻⁶ K⁻¹  (Touloukian et al. 1977)")
print("    B₀ = 24.5 GPa              (Simmons & Wang 1971)")
