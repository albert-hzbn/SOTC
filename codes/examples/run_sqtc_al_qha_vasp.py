#!/usr/bin/env python3
"""
SQTC + QHA for fcc Al: thermal expansion benchmark
====================================================

Runs SQTC at three volumes (V₀, V₀×(1−δ), V₀×(1+δ), δ=0.02) plus three
static primitive-cell DFT calculations for the EOS.  Then:

  1. Builds SQTCQuasiHarmonic with method="both"
  2. Computes α(T), V(T), C_P(T), B(T) via full QHA minimisation
  3. Also computes α(T) via the Grüneisen shortcut
  4. Compares both against:
       (a) the previous single-volume SQTC C_V (sqtc_al_fast_vasp_run)
       (b) phonopy-QHA PBE literature values for fcc Al
       (c) experimental data

Literature / experimental references (Al fcc, P=0)
----------------------------------------------------
  α(T=300 K) expt:  23.1 × 10⁻⁶ K⁻¹  (Nix & MacNair, 1941)
  α(T=600 K) expt:  26.2 × 10⁻⁶ K⁻¹  (Touloukian et al., 1977)
  V₀ expt:          16.48 Å³/atom at 300 K
  B₀ expt:          76.5 GPa (Simmons & Wang, 1971)

  phonopy-QHA PBE (Grabowski et al., PRB 2007):
    α(300 K) ≈ 22.0 × 10⁻⁶ K⁻¹
    α(900 K) ≈ 28.0 × 10⁻⁶ K⁻¹

  TDEP-QHA PBE (Hellman et al., PRB 2013):
    α(300 K) ≈ 23.5 × 10⁻⁶ K⁻¹

  SSCHA-QHA (Bianco et al., PRB 2017):
    α(300 K) ≈ 22.8 × 10⁻⁶ K⁻¹

Usage
-----
  Submit via slurm_al_qha.sh, or run locally:
    python3 codes/examples/run_sqtc_al_qha_vasp.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))

from sqtc import SQTCRunner, SQTCQuasiHarmonic
from sqtc.phonons import PhononCalculator
from sqtc.vasp_io import VASPWriter, VASPRunner

# ── Machine / VASP settings ──────────────────────────────────────────────────

VASP_STD      = os.environ.get("VASP_STD", "/u/alli/Softwares/DFT_v/s")
PP_BASE_DIR   = Path(os.environ.get("VASP_PP_BASE",
                     "/u/alli/Softwares/VASP/Pseudopotentials"))
NCORES_PER_JOB = 64
NCORE_INCAR    = 8
VASP_CMD       = f"srun --nodes=1 --ntasks={NCORES_PER_JOB} --exclusive {VASP_STD}"

# ── fcc Al geometry ───────────────────────────────────────────────────────────

a_al   = 4.05          # Å  (DFT-PBE equilibrium lattice constant)
delta  = 0.02          # ±2% volume step  →  ±0.67% linear step per axis

prim_cell_al = 0.5 * a_al * np.array([
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
])
prim_pos_al  = np.array([[0.0, 0.0, 0.0]])
V0_ang3      = abs(float(np.linalg.det(prim_cell_al)))   # Å³/atom  (~16.57)

# Three volumes: V₋  V₀  V₊
# Scale the lattice uniformly: a_scaled = a × (1±δ)^(1/3)
a_scales = [(1 - delta) ** (1.0 / 3.0), 1.0, (1 + delta) ** (1.0 / 3.0)]
volumes  = [V0_ang3 * (1 - delta), V0_ang3, V0_ang3 * (1 + delta)]

M_Al_amu = 26.9815385
T_D_Al   = 390.0

VASP_SETTINGS = {
    "encut":    450.0,
    "functional": "PBE",
    "ncore":    NCORE_INCAR,
    "kgrid":    (4, 4, 4),
    "pp_base_dir": PP_BASE_DIR,
    "pp_set":   "PAW_PBE",
    "extra_incar": {
        "KSPACING": "0.25",
        "EDIFF":    "1E-7",
        "PREC":     "Accurate",
        "ISMEAR":   "1",
        "SIGMA":    "0.10",
    },
    "timeout": 3600,
}

WORK_BASE = Path("sqtc_al_qha_run")
WORK_BASE.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — SQTC at 3 volumes
# ─────────────────────────────────────────────────────────────────────────────

T_design   = 300.0          # SQTC displacement design temperature [K]
T_values   = np.arange(50, 1001, 50, dtype=float)   # evaluation grid

phonon_calcs = []
sqtc_results_per_vol = []

vol_labels = ["Vm", "V0", "Vp"]

for iv, (ascale, vol) in enumerate(zip(a_scales, volumes)):
    label = vol_labels[iv]
    work_dir = WORK_BASE / f"sqtc_{label}"
    cell_v = ascale * prim_cell_al

    print(f"\n{'='*65}")
    print(f" SQTC run  {label}:  a = {a_al*ascale:.4f} Å   V = {vol:.4f} Å³")
    print(f"{'='*65}")

    # Cache: re-use existing run if already converged
    results_json = work_dir / "sqtc_results.json"
    if results_json.exists():
        print(f"  Loading cached results from {results_json}")
        with open(results_json) as f:
            cached = json.load(f)
        # Rebuild PhononCalculator from saved IFC
        ifc_path = work_dir / f"iter_{cached['selected_iteration']:02d}" / "ifc_extractor.npz"
        if ifc_path.exists():
            from sqtc.ifc_extractor import IFCExtractor
            ifc = IFCExtractor.load(ifc_path)
            calc = PhononCalculator(
                ifc_extractor=ifc,
                prim_positions=np.zeros((1, 3)),
                prim_cell=cell_v,
                masses_amu=np.array([M_Al_amu]),
            )
            phonon_calcs.append(calc)
            sqtc_results_per_vol.append(cached)
            continue

    runner = SQTCRunner(
        element="Al",
        mass_amu=M_Al_amu,
        prim_cell=cell_v,
        prim_positions=prim_pos_al,
        T=T_design,
        T_D=T_D_Al,
        n_atoms_sc=32,
        force_calculator=None,
        vasp_cmd=VASP_CMD,
        vasp_settings=VASP_SETTINGS,
        r_cutoff=4.5 * ascale,
        r_max_corr=8.0 * ascale,
        n_ensemble=6,
        work_dir=str(work_dir),
        verbosity=1,
        ridge_alpha=1e-3,
        symmetrize_bonds=True,
    )

    res = runner.run(
        n_sc_iterations=12,
        epsilon_conv=0.003,
        mixing=0.30,
        T_values=np.array([200.0, 300.0, 400.0]),
        q_mesh_cv=(6, 6, 6),
    )
    phonon_calcs.append(res["phonon_calculator"])
    sqtc_results_per_vol.append(res)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Static primitive-cell DFT for E_0(V)  (NSW=0, 4-atom prim cell)
# ─────────────────────────────────────────────────────────────────────────────
# Uses a denser k-mesh (8×8×8) to get a precise total energy.

print("\n" + "="*65)
print(" Static DFT  —  E₀(V) for EOS")
print("="*65)

static_energies = []

for iv, (ascale, vol) in enumerate(zip(a_scales, volumes)):
    label = vol_labels[iv]
    static_dir = WORK_BASE / f"static_{label}"
    static_dir.mkdir(exist_ok=True)
    energy_file = static_dir / "energy.txt"

    if energy_file.exists():
        static_energies.append(float(energy_file.read_text().strip()))
        print(f"  {label}: E = {static_energies[-1]:.6f} eV  (cached)")
        continue

    cell_v    = ascale * prim_cell_al
    positions = np.array([[0.0, 0.0, 0.0]])
    species   = ["Al"]

    writer = VASPWriter(directory=static_dir)
    writer.write_poscar(cell=cell_v, species=species, positions=positions)
    writer.write_incar(
        encut=520.0,
        functional="PBE",
        ncore=NCORE_INCAR,
        params={
            "EDIFF":  "1E-8",
            "ISMEAR": "1",
            "SIGMA":  "0.10",
            "PREC":   "Accurate",
            "NSW":    "0",
            "ISIF":   "2",
        },
    )
    writer.write_kpoints(kgrid=(8, 8, 8))
    writer.write_potcar(species=species, pp_base_dir=PP_BASE_DIR, pp_set="PAW_PBE")

    vasp_runner = VASPRunner(vasp_cmd=VASP_CMD, base_dir=static_dir)
    import subprocess
    cmd = VASP_CMD.split()
    proc = subprocess.run(cmd, cwd=static_dir, capture_output=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"Static VASP failed for {label}:\n{proc.stderr.decode()[-2000:]}")

    # Parse energy
    outcar = static_dir / "OUTCAR"
    import re
    text = outcar.read_text()
    matches = re.findall(r"energy  without entropy=\s*([-\d.E+]+)", text)
    if not matches:
        matches = re.findall(r"TOTEN\s*=\s*([-\d.E+]+)\s*eV", text)
    E0 = float(matches[-1])
    energy_file.write_text(str(E0))
    static_energies.append(E0)
    print(f"  {label}: E = {E0:.6f} eV")

print(f"\n  EOS: E = {static_energies}")
print(f"  V   = {[f'{v:.4f}' for v in volumes]} Å³")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — QHA + Grüneisen thermal expansion
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*65)
print(" QHA thermal expansion  (method='both')")
print("="*65)

qha = SQTCQuasiHarmonic(
    volumes_ang3       = volumes,
    static_energies_ev = static_energies,
    phonon_calcs       = phonon_calcs,
    q_mesh             = (20, 20, 20),
    method             = "both",
)

print(f"  B₀ (static parabolic EOS) = {qha.B0_GPa:.2f} GPa  (expt: 76.5 GPa)")
print(f"  V₀ = {qha.V0_ang3:.4f} Å³  (expt @ 300 K: 16.48 Å³)")

results = qha.run(T_values)

np.savez(
    WORK_BASE / "qha_results.npz",
    **{k: np.asarray(v) for k, v in results.items()},
)
print(f"\n  Saved QHA results → {WORK_BASE / 'qha_results.npz'}")

# Print table at key temperatures
print(f"\n  {'T [K]':>8}  {'V [Å³]':>10}  {'α×10⁶ [K⁻¹]':>14}  "
      f"{'α_gruen×10⁶':>14}  {'C_P [J/mol/K]':>14}  {'B [GPa]':>9}")
print("  " + "-"*80)
for i, T in enumerate(results["T_K"]):
    if T in (100, 200, 300, 400, 600, 800, 1000):
        print(f"  {T:8.0f}  {results['V_ang3'][i]:10.4f}  "
              f"{results['alpha_K'][i]*1e6:14.2f}  "
              f"{results['gruen_alpha_K'][i]*1e6:14.2f}  "
              f"{results['Cp_jmolk'][i]:14.4f}  "
              f"{results['B_GPa'][i]:9.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Comparison: previous SQTC (single volume) + literature
# ─────────────────────────────────────────────────────────────────────────────

prev_json = Path("sqtc_al_fast_vasp_run/sqtc_results.json")
prev_cv   = None
if prev_json.exists():
    with open(prev_json) as f:
        prev = json.load(f)
    prev_cv = np.array(prev["C_V_scan"])
    prev_T  = np.array(prev["T_values"])
    print(f"\n  Previous SQTC (single V, 300 K):")
    print(f"    T_D_eff = {prev['T_D_effective']:.1f} K")
    for T_p, cv_p in zip(prev_T, prev_cv):
        print(f"    C_V({T_p:.0f} K) = {cv_p:.4f} J/mol/K")

# Literature α(T) for Al (Touloukian et al. 1977, polynomial fit)
def alpha_expt_al(T_K: np.ndarray) -> np.ndarray:
    """Polynomial fit to Touloukian et al. (1977) Table 4 for Al.
    Valid 100–900 K.  Returns α [K⁻¹]."""
    # Coefficients for α/10⁻⁶ K⁻¹ = a0 + a1*T + a2*T²
    a0, a1, a2 = 18.84, 0.0142, -4.05e-6
    return (a0 + a1 * T_K + a2 * T_K**2) * 1e-6

# Literature C_P for Al (Dinsdale, CALPHAD 1991)
def cp_calphad_al(T_K: np.ndarray) -> np.ndarray:
    """CALPHAD C_P [J/mol/K] for fcc Al, 298–933 K."""
    return 28.08 + 0.00510 * T_K + 8.62e5 / T_K**2   # Dinsdale 1991

T_lit = np.arange(100, 1001, 10, dtype=float)

# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("fcc Al: SQTC-QHA benchmark", fontsize=13)

T     = results["T_K"]
alpha = results["alpha_K"]
agrun = results["gruen_alpha_K"]
Cp    = results["Cp_jmolk"]
V     = results["V_ang3"]
B     = results["B_GPa"]

# ── α(T) ──────────────────────────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(T, alpha * 1e6, "b-",  lw=2,   label="SQTC-QHA (Helmholtz)")
ax.plot(T, agrun * 1e6, "b--", lw=1.5, label="SQTC Grüneisen")
ax.plot(T_lit, alpha_expt_al(T_lit) * 1e6, "k-", lw=1.2, label="Expt (Touloukian 1977)")
# Scatter: phonopy-QHA PBE Grabowski 2007
T_grab = np.array([200, 300, 400, 600, 900])
a_grab = np.array([19.5, 22.0, 24.0, 25.5, 28.0])
ax.scatter(T_grab, a_grab, marker="s", s=40, c="gray",  label="phonopy-QHA (Grabowski 2007)")
# TDEP Hellman 2013
T_tdep = np.array([300, 600, 900])
a_tdep = np.array([23.5, 25.8, 28.5])
ax.scatter(T_tdep, a_tdep, marker="^", s=40, c="green", label="TDEP-QHA (Hellman 2013)")
ax.set_xlabel("T [K]")
ax.set_ylabel("α [10⁻⁶ K⁻¹]")
ax.set_title("Thermal expansion coefficient")
ax.legend(fontsize=7)
ax.set_xlim(0, 1000)
ax.set_ylim(0, 35)

# ── V(T) ──────────────────────────────────────────────────────────────────────
ax = axes[0, 1]
ax.plot(T, V, "b-", lw=2, label="SQTC-QHA")
ax.axhline(16.48, color="k", lw=1.2, ls="--", label="Expt V₀ @ 300 K")
ax.set_xlabel("T [K]")
ax.set_ylabel("V [Å³/atom]")
ax.set_title("Equilibrium volume")
ax.legend(fontsize=8)
ax.set_xlim(0, 1000)

# ── C_P(T) ────────────────────────────────────────────────────────────────────
ax = axes[1, 0]
ax.plot(T, Cp, "b-", lw=2, label="SQTC-QHA C_P")
ax.plot(T, results["Cv_jmolk"], "b:", lw=1.5, label="SQTC C_V (middle V)")
ax.plot(T_lit[T_lit <= 900], cp_calphad_al(T_lit[T_lit <= 900]),
        "k-", lw=1.2, label="CALPHAD (Dinsdale 1991)")
# Previous single-volume C_V
if prev_cv is not None:
    ax.scatter(prev_T, prev_cv, marker="o", c="red", zorder=5,
               label="SQTC prev. (sqtc_al_fast_vasp_run)")
ax.set_xlabel("T [K]")
ax.set_ylabel("C_P, C_V  [J/(mol·K)]")
ax.set_title("Heat capacity")
ax.legend(fontsize=7)
ax.set_xlim(0, 1000)
ax.set_ylim(20, 35)

# ── B(T) ──────────────────────────────────────────────────────────────────────
ax = axes[1, 1]
ax.plot(T, B, "b-", lw=2, label="SQTC-QHA B(T)")
ax.axhline(76.5, color="k", lw=1.2, ls="--", label="Expt B₀ (Simmons & Wang 1971)")
ax.axhline(qha.B0_GPa, color="gray", lw=1, ls=":", label=f"SQTC-QHA B₀ = {qha.B0_GPa:.1f} GPa")
ax.set_xlabel("T [K]")
ax.set_ylabel("B [GPa]")
ax.set_title("Bulk modulus")
ax.legend(fontsize=8)
ax.set_xlim(0, 1000)

plt.tight_layout()
out_png = WORK_BASE / "al_qha_comparison.png"
out_pdf = WORK_BASE / "al_qha_comparison.pdf"
plt.savefig(out_png, dpi=150)
plt.savefig(out_pdf)
print(f"\n  Saved plots → {out_png}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Summary table
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print(" SUMMARY: fcc Al  (PBE-GGA, 0 GPa)")
print("="*70)
print(f"  B₀ [GPa]:    SQTC-QHA = {qha.B0_GPa:6.2f}   Expt = 76.50")
print(f"  V₀ [Å³]:     SQTC-QHA = {qha.V0_ang3:6.4f}   Expt = 16.48")
print()

# Find closest temperature index
def _at_T(arr, T):
    i = int(np.argmin(np.abs(results["T_K"] - T)))
    return arr[i]

print(f"  {'T [K]':>6}  {'α_QHA×10⁶':>12}  {'α_Gruen×10⁶':>12}  "
      f"{'α_expt×10⁶':>12}  {'C_P [J/mol/K]':>14}")
print("  " + "-"*68)
for T_c in [100, 200, 300, 400, 600, 800, 1000]:
    a_q = _at_T(results["alpha_K"],       T_c) * 1e6
    a_g = _at_T(results["gruen_alpha_K"], T_c) * 1e6
    a_e = alpha_expt_al(np.array([float(T_c)]))[0] * 1e6 if T_c >= 100 else float("nan")
    cp  = _at_T(results["Cp_jmolk"], T_c)
    print(f"  {T_c:6.0f}  {a_q:12.2f}  {a_g:12.2f}  {a_e:12.2f}  {cp:14.4f}")

print("\nDone.")
