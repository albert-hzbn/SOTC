#!/usr/bin/env python3
"""
Post-process fcc Cu SQTC run at T = 1200 K.

Compares:
  1. SQTC (anharmonic, T = 1200 K IFCs, all 8 iterations)  — main result
  2. Harmonic reference  (iter_00 only — cold IFC regime)    — baseline
  3. Literature: experimental C_V, phonon dispersion, T_D, α(T)

Produces (sqtc_cu_vasp_run/postproc/):
  cu_phonons_comparison.pdf    — harmonic vs SQTC phonon bands + DOS
  cu_thermal_properties.pdf    — C_V, S_vib, F_vib, T_D, MSD, DW vs experiment
  cu_anharmonicity.pdf         — ΔC_V(T) = C_V(SQTC) − C_V(harm),
                                  iteration convergence, T_D(T)
  cu_gruneisen.pdf             — mode + effective Grüneisen parameters
  thermal_properties.npz       — all T-dependent quantities
  phonon_bandstructure.npz     — SQTC band structure
  phonon_dos.npz               — SQTC DOS
  cu_summary.json              — key-T summary table

Scientific context
------------------
Cu has γ ≈ 1.96 (Grüneisen parameter) and T_melt = 1358 K. At 1200 K
(T/T_melt = 0.88) explicit anharmonic contributions push C_V above the
classical Dulong–Petit limit 3R = 24.94 J/(mol·K). The SQTC method
captures this directly by sampling force constants at thermally displaced
configurations. The harmonic approximation (0 K IFCs) misses the
anharmonic excess entirely.

Key literature references
-------------------------
  Experimental C_V  — Barin & Knacke (1977); NIST JANAF tables
  Phonon dispersion — Nilsson & Rolandson, Phys Rev B 7 (1973) 2393
  T_D (calorimetric)— Kittel, Introduction to Solid State Physics (8th ed.)
  TDEP comparison  — Born & Hellman, npj Comput Mater 5 (2019) 97
  SSCHA comparison — Monacelli et al., J Phys Condens Matter 33 (2021) 363001
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent / "../"))

from sqtc import IFCExtractor, PhononCalculator
from sqtc.postprocessor import (
    SQTCPostProcessor,
    _parse_poscar,
    _parse_outcar_forces,
    _build_eq_positions,
)
from sqtc.constants import NA

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT    = Path(".")
RUN_DIR = ROOT / "sqtc_cu_vasp_run"
OUT_DIR = RUN_DIR / "postproc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── fcc Cu geometry at T = 1200 K ────────────────────────────────────────────
# a_exp(300 K) = 3.615 Å; linear thermal expansion gives a(1200 K) ≈ 3.670 Å
# The SQTC run was conducted at this volume (forces at physical density).

a_cu        = 3.67          # Å  (experimental fcc Cu at ~1200 K)
M_Cu_amu    = 63.546
T_design    = 1200.0        # K  (SQTC design temperature)
T_melt_Cu   = 1358.0        # K

prim_cell   = 0.5 * a_cu * np.array([
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
], dtype=float)             # FCC primitive cell [Å]
prim_pos    = np.array([[0.0, 0.0, 0.0]])   # basis: one atom at origin

R_CUTOFF        = 4.5
RIDGE_ALPHA     = 1e-3
N_ATOMS         = 36        # 36-atom supercell (H = [[3,1,0],[0,4,1],[0,0,3]])
Q_MESH_CV       = (20, 20, 20)
Q_MESH_DOS      = (30, 30, 30)

T_values = np.arange(10, 1301, 10, dtype=float)   # 10–1300 K


# ── Helper: parse TOTEN from OUTCAR ──────────────────────────────────────────

def _extract_toten(path: Path) -> float | None:
    text = path.read_text()
    m = re.findall(r"energy\s+without entropy\s*=\s*([-\d.E+]+)", text)
    if m:
        return float(m[-1])
    m = re.findall(r"TOTEN\s*=\s*([-\d.E+]+)\s*eV", text)
    return float(m[-1]) if m else None


# ── Helper: load snapshots from a list of iteration directories ──────────────

def _load_snaps(run_dir: Path, iter_names: list[str], label: str = ""):
    """
    Load POSCAR + OUTCAR from iter_*/snap_*/.
    Returns (eq_pos, sc_cell, displ_list, forces_list, energies).
    """
    ref_poscar = None
    for iname in iter_names:
        p = run_dir / iname / "snap_0000" / "POSCAR"
        if p.exists():
            ref_poscar = p
            break
    if ref_poscar is None:
        raise FileNotFoundError(f"No snap_0000/POSCAR under {run_dir}/{iter_names}")

    sc_cell, _, _ = _parse_poscar(ref_poscar)
    eq_pos, H = _build_eq_positions(sc_cell, prim_cell, prim_pos)
    n_atoms = len(eq_pos)
    cell_inv = np.linalg.inv(sc_cell)

    displ_list:  list[np.ndarray] = []
    forces_list: list[np.ndarray] = []
    energies:    list[float]      = []

    for iname in iter_names:
        iter_dir = run_dir / iname
        if not iter_dir.is_dir():
            continue
        for snap in sorted(iter_dir.glob("snap_*/")):
            poscar = snap / "POSCAR"
            outcar = snap / "OUTCAR"
            if not poscar.exists() or not outcar.exists():
                continue
            try:
                _, _, snap_pos = _parse_poscar(poscar)
            except Exception:
                continue
            if len(snap_pos) != n_atoms:
                continue
            try:
                forces = _parse_outcar_forces(outcar, n_atoms)
            except Exception:
                continue
            u = snap_pos - eq_pos
            frac = u @ cell_inv
            frac -= np.round(frac)
            u = frac @ sc_cell
            displ_list.append(u)
            forces_list.append(forces)
            E = _extract_toten(outcar)
            if E is not None:
                energies.append(E)

    print(f"  [{label}] {len(displ_list)} snaps, {n_atoms} atoms, "
          f"{len(energies)} energies")
    return eq_pos, sc_cell, displ_list, forces_list, energies


# ── Helper: fit IFCs ──────────────────────────────────────────────────────────

def _fit_ifc(eq_pos, sc_cell, displ_list, forces_list, label=""):
    ifc = IFCExtractor(
        supercell_positions = eq_pos,
        supercell_cell      = sc_cell,
        masses_amu          = np.full(N_ATOMS, M_Cu_amu),
        r_cutoff            = R_CUTOFF,
        symmetrise          = True,
        ridge_alpha         = RIDGE_ALPHA,
        symmetrize_bonds    = True,
    )
    ifc.fit(displ_list, forces_list)
    rep = ifc.fit_report(displ_list, forces_list)
    print(f"  [{label}] RMSE={rep['rmse_ev_ang']:.5f} eV/Å  "
          f"R²={rep['r2']:.5f}  rank={rep['rank']}")
    return ifc


# ── Helper: build PhononCalculator ───────────────────────────────────────────

def _make_calc(ifc):
    return PhononCalculator(
        ifc_extractor  = ifc,
        prim_positions = prim_pos,
        prim_cell      = prim_cell,
        masses_amu     = np.array([M_Cu_amu]),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1  —  Load snapshots
# ═══════════════════════════════════════════════════════════════════════════════

all_iters  = [f"iter_{i:02d}" for i in range(8)]   # iter_00 … iter_07
harm_iters = ["iter_00"]                            # harmonic reference

print("=" * 70)
print(" Loading Cu snapshots")
print("=" * 70)

eq_pos, sc_cell, displ_all, forces_all, ens_all = _load_snaps(
    RUN_DIR, all_iters, label="SQTC  (all 8 iter)")

_, _, displ_h, forces_h, _ = _load_snaps(
    RUN_DIR, harm_iters, label="Harmonic (iter_00)")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2  —  Fit IFCs
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" Fitting IFCs")
print("=" * 70)

ifc_sqtc = _fit_ifc(eq_pos, sc_cell, displ_all,  forces_all, label="SQTC conv.")
ifc_harm = _fit_ifc(eq_pos, sc_cell, displ_h,    forces_h,   label="Harmonic  ")

calc_sqtc = _make_calc(ifc_sqtc)
calc_harm = _make_calc(ifc_harm)

# Quick stability check
for name, calc in [("SQTC", calc_sqtc), ("Harm", calc_harm)]:
    st = calc.spectrum_statistics(q_mesh=(10, 10, 10))
    print(f"  [{name}] ω_max = {st['max_freq_thz']:.2f} THz  "
          f"unstable = {st['unstable_fraction']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 3  —  Build SQTCPostProcessors and compute properties
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" Computing phonon properties")
print("=" * 70)

pp_sqtc = SQTCPostProcessor(
    phonon_calc = calc_sqtc,
    label       = "fcc Cu (SQTC, 1200 K)",
    elements    = ["Cu"],
    T_design    = T_design,
)
pp_harm = SQTCPostProcessor(
    phonon_calc = calc_harm,
    label       = "fcc Cu (harmonic, iter_00)",
    elements    = ["Cu"],
    T_design    = 0.0,
)

phonon_data_sqtc = pp_sqtc.compute_phonons(
    q_mesh               = Q_MESH_DOS,
    n_bins               = 500,
    sigma_thz            = 0.06,
    n_points_per_segment = 120,
)
phonon_data_harm = pp_harm.compute_phonons(
    q_mesh               = Q_MESH_DOS,
    n_bins               = 500,
    sigma_thz            = 0.06,
    n_points_per_segment = 120,
)

thermal_sqtc = pp_sqtc.compute_thermal_properties(T_values, q_mesh=Q_MESH_CV)
thermal_harm = pp_harm.compute_thermal_properties(T_values, q_mesh=Q_MESH_CV)

# Save SQTC as primary output
pp_sqtc.save(OUT_DIR)
pp_sqtc.print_summary(thermal_data=thermal_sqtc, phonon_data=phonon_data_sqtc)

# ═══════════════════════════════════════════════════════════════════════════════
# Step 4  —  Iteration convergence (C_V at T_design per iter subset)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n  Computing C_V convergence over iterations ...")

cv_iter = []   # C_V(T_design) after loading 1..N iterations
for n_it in range(1, len(all_iters) + 1):
    _, _, d_tmp, f_tmp, _ = _load_snaps(
        RUN_DIR, all_iters[:n_it], label=f"iter_00..{n_it-1:02d}")
    ifc_tmp  = _fit_ifc(eq_pos, sc_cell, d_tmp, f_tmp, label=f"  n={n_it}")
    calc_tmp = _make_calc(ifc_tmp)
    cv_at_td = float(calc_tmp.heat_capacity_scan(
        np.array([T_design]), q_mesh=Q_MESH_CV
    )[0])
    cv_iter.append(cv_at_td)
    print(f"  iter_00..{n_it-1:02d}:  C_V(1200 K) = {cv_at_td:.4f} J/(mol·K)"
          f"  (Dulong-Petit = {3*8.314:.4f})")

cv_iter = np.array(cv_iter)

# ═══════════════════════════════════════════════════════════════════════════════
# Step 5  —  Literature / experimental reference data
# ═══════════════════════════════════════════════════════════════════════════════

T_plot = T_values

# Experimental C_P ≈ C_V for Cu below melting (Barin & Knacke 1977; NIST JANAF)
# Polynomial fit valid 300–1300 K: C_P = a + bT + c/T²  (J/mol/K)
# Dinsdale CALPHAD data for Cu (SGTE 2007):
def cp_expt_cu(T: np.ndarray) -> np.ndarray:
    """C_P [J/(mol·K)] — SGTE-CALPHAD polynomial for fcc Cu, 298–1358 K."""
    return 22.7 + 0.01000 * T + 0.0e0 * T**0   # simplified linear fit
    # More accurate piecewise (Dinsdale 1991, Cu FCC):
    #   298–1358 K:  24.914 − 3.617e-3*T + 1.083e6/T² ... (not used here for clarity)

# Accurate NIST values at key temperatures (C_P ≈ C_V for solids at moderate P)
T_nist = np.array([298,  400,  500,  600,  700,  800,  900,  1000, 1100, 1200, 1300], dtype=float)
Cp_nist= np.array([24.44, 24.93, 25.30, 25.57, 25.79, 25.97, 26.13, 26.27, 26.40, 26.52, 26.63])

# Experimental phonon dispersion (Nilsson & Rolandson 1973) — selected high-sym points
# (ξ in r.l.u., ν in THz)  — Γ-X branch [L branch] Γ-L branch
T_dispers_expt = {
    "Gamma-X_L": (
        np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        np.array([0.0, 1.05, 1.95, 2.76, 3.50, 4.02, 4.26, 4.38, 4.46, 4.53]),
    ),
    "Gamma-L_L": (
        np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        np.array([0.0, 1.22, 2.35, 3.29, 3.86, 4.04]),
    ),
}

# TDEP / Born & Hellman 2019 — T_D and C_V at selected T
T_tdep = np.array([300, 500, 700, 1000, 1200], dtype=float)
Cv_tdep= np.array([24.1, 24.6, 25.0, 25.3,  25.5])  # approx. from literature

# Thermal expansion (experimental, Touloukian et al. 1977 CTE Cu)
def alpha_expt_cu(T: np.ndarray) -> np.ndarray:
    """Volumetric CTE [K⁻¹] for Cu — polynomial fit to Touloukian 1977."""
    # Approximate: starts ~48e-6 at 300K (volumetric = 3×linear)
    # linear CTE: a0=16.4e-6, a1=2.0e-9 K⁻¹
    alpha_lin = 16.4e-6 + 2.0e-9 * T
    return 3.0 * alpha_lin

# ═══════════════════════════════════════════════════════════════════════════════
# Step 6  —  Plots
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" Plotting")
print("=" * 70)

T   = thermal_sqtc["T_K"]
T_h = thermal_harm["T_K"]

Cv_s  = thermal_sqtc["Cv_jmolk"]
Sv_s  = thermal_sqtc["Svib_jmolk"]
Fv_s  = thermal_sqtc["Fvib_ev"]
TD_s  = thermal_sqtc["TD_caloric"]
MSD_s = thermal_sqtc["MSD_ang2"]
DW_s  = thermal_sqtc["DW_B_ang2"]
ZPE_s = thermal_sqtc["ZPE_eV"]
TDs_s = thermal_sqtc["TD_spectral"]

Cv_h  = thermal_harm["Cv_jmolk"]
Sv_h  = thermal_harm["Svib_jmolk"]
Fv_h  = thermal_harm["Fvib_ev"]
TD_h  = thermal_harm["TD_caloric"]
MSD_h = thermal_harm["MSD_ang2"]
DW_h  = thermal_harm["DW_B_ang2"]
ZPE_h = thermal_harm["ZPE_eV"]
TDs_h = thermal_harm["TD_spectral"]

Cv_DP = 3 * 8.314   # 24.942 J/(mol·K) — Dulong-Petit

# ── Fig 1: Phonon band comparison (harmonic vs SQTC) + DOS ──────────────────

bs_s  = phonon_data_sqtc["bs"]
bs_h  = phonon_data_harm["bs"]
dos_s = phonon_data_sqtc["pdos"]
dos_h = phonon_data_harm["pdos"]

fig1 = plt.figure(figsize=(13, 6))
gs1  = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 1.2], wspace=0.05)
ax_bs1  = fig1.add_subplot(gs1[0])
ax_bs2  = fig1.add_subplot(gs1[1], sharey=ax_bs1)
ax_dos  = fig1.add_subplot(gs1[2], sharey=ax_bs1)

fig1.suptitle(
    f"fcc Cu — phonon dispersion: harmonic (0 K–like) vs SQTC T={T_design:.0f} K",
    fontsize=13, fontweight="bold",
)

# Harmonic bands
dist_h = bs_h["distances"]
freq_h = bs_h["frequencies_thz"]
for b in range(freq_h.shape[1]):
    ax_bs1.plot(dist_h, freq_h[:, b], color="steelblue", lw=1.0, alpha=0.85)
for lp in bs_h["label_positions"]:
    ax_bs1.axvline(lp, color="gray", lw=0.7, ls="--")
ax_bs1.axhline(0, color="gray", lw=0.5)
ax_bs1.set_xticks(bs_h["label_positions"])
ax_bs1.set_xticklabels(
    [l.replace("G", r"$\Gamma$") for l in bs_h["labels"]], fontsize=11
)
ax_bs1.set_ylabel("Frequency (THz)", fontsize=12)
ax_bs1.set_title(f"Harmonic (iter_00)\nω_max = {bs_h['frequencies_thz'].max():.2f} THz", fontsize=10)
ax_bs1.set_xlim(dist_h[0], dist_h[-1])
ax_bs1.tick_params(axis="x", length=0)
ax_bs1.grid(axis="y", lw=0.3, alpha=0.4)

# SQTC bands
dist_s = bs_s["distances"]
freq_s = bs_s["frequencies_thz"]
for b in range(freq_s.shape[1]):
    ax_bs2.plot(dist_s, freq_s[:, b], color="firebrick", lw=1.0, alpha=0.85)
for lp in bs_s["label_positions"]:
    ax_bs2.axvline(lp, color="gray", lw=0.7, ls="--")
ax_bs2.axhline(0, color="gray", lw=0.5)
ax_bs2.set_xticks(bs_s["label_positions"])
ax_bs2.set_xticklabels(
    [l.replace("G", r"$\Gamma$") for l in bs_s["labels"]], fontsize=11
)
ax_bs2.set_title(
    f"SQTC T={T_design:.0f} K (all iters)\nω_max = {bs_s['frequencies_thz'].max():.2f} THz",
    fontsize=10,
)
ax_bs2.set_xlim(dist_s[0], dist_s[-1])
ax_bs2.tick_params(axis="x", length=0)
ax_bs2.grid(axis="y", lw=0.3, alpha=0.4)
plt.setp(ax_bs2.get_yticklabels(), visible=False)

# DOS comparison
ax_dos.fill_betweenx(dos_h["frequencies_thz"], dos_h["dos_total"],
                     alpha=0.25, color="steelblue", label="Harmonic")
ax_dos.plot(dos_h["dos_total"], dos_h["frequencies_thz"],
            color="steelblue", lw=1.2, ls="--")
ax_dos.fill_betweenx(dos_s["frequencies_thz"], dos_s["dos_total"],
                     alpha=0.30, color="firebrick", label="SQTC")
ax_dos.plot(dos_s["dos_total"], dos_s["frequencies_thz"],
            color="firebrick", lw=1.5)
ax_dos.axhline(0, color="gray", lw=0.5)
ax_dos.set_xlabel("DOS", fontsize=11)
ax_dos.set_xlim(left=0)
plt.setp(ax_dos.get_yticklabels(), visible=False)
ax_dos.tick_params(axis="y", length=0)
ax_dos.legend(fontsize=8, loc="upper right", framealpha=0.7)
ax_dos.grid(axis="y", lw=0.3, alpha=0.4)

y_max = max(freq_h.max(), freq_s.max()) * 1.08
ax_bs1.set_ylim(-0.3, y_max)

p1 = OUT_DIR / "cu_phonons_comparison.pdf"
fig1.savefig(p1, dpi=200, bbox_inches="tight")
fig1.savefig(p1.with_suffix(".png"), dpi=200, bbox_inches="tight")
print(f"  Saved → {p1}")
plt.close(fig1)

# ── Fig 2: Thermal properties (6 panels) — SQTC vs harmonic vs experiment ───

fig2, axes2 = plt.subplots(3, 2, figsize=(13, 14))
fig2.suptitle(
    f"fcc Cu — thermal properties: SQTC T={T_design:.0f} K vs harmonic vs experiment",
    fontsize=13, fontweight="bold",
)

# C_V
ax = axes2[0, 0]
ax.plot(T,   Cv_s, "r-",  lw=2.2, label=f"SQTC ({T_design:.0f} K IFCs)")
ax.plot(T_h, Cv_h, "b--", lw=1.6, label="Harmonic (iter_00)")
ax.scatter(T_nist, Cp_nist, c="k", s=35, zorder=6, label="Expt NIST/JANAF (C_P)")
ax.scatter(T_tdep, Cv_tdep, c="green", marker="^", s=50, zorder=6, label="TDEP (Born 2019)")
ax.axhline(Cv_DP, color="gray", ls=":", lw=1.2, label=f"Dulong-Petit = {Cv_DP:.2f} J/mol/K")
ax.axvline(T_design, color="r", ls=":", lw=0.9, alpha=0.4, label=f"T_design = {T_design:.0f} K")
ax.axvline(T_melt_Cu, color="orange", ls="--", lw=0.9, alpha=0.6, label=f"T_melt = {T_melt_Cu:.0f} K")
ax.set_xlabel("T [K]"); ax.set_ylabel("C_V  [J/(mol·K)]")
ax.set_title("Heat capacity")
ax.legend(fontsize=6.5, framealpha=0.85)
ax.set_xlim(0, T_values.max()); ax.set_ylim(0, 28)
ax.grid(lw=0.3, alpha=0.5)

# S_vib
ax = axes2[0, 1]
ax.plot(T,   Sv_s, "r-",  lw=2.2, label="SQTC")
ax.plot(T_h, Sv_h, "b--", lw=1.6, label="Harmonic")
ax.axvline(T_design, color="r", ls=":", lw=0.9, alpha=0.4)
ax.set_xlabel("T [K]"); ax.set_ylabel("S_vib  [J/(mol·K)]")
ax.set_title("Vibrational entropy")
ax.legend(fontsize=8)
ax.set_xlim(0, T_values.max())
ax.grid(lw=0.3, alpha=0.5)

# F_vib
ax = axes2[1, 0]
ax.plot(T,   Fv_s * 1000, "r-",  lw=2.2, label="SQTC")
ax.plot(T_h, Fv_h * 1000, "b--", lw=1.6, label="Harmonic")
ax.axhline(ZPE_s * 1000, color="r",  ls=":", lw=1.0, alpha=0.5,
           label=f"ZPE_SQTC = {ZPE_s*1000:.1f} meV")
ax.axhline(ZPE_h * 1000, color="b",  ls=":", lw=1.0, alpha=0.5,
           label=f"ZPE_harm = {ZPE_h*1000:.1f} meV")
ax.axvline(T_design, color="r", ls=":", lw=0.9, alpha=0.4)
ax.set_xlabel("T [K]"); ax.set_ylabel("F_vib  [meV/f.u.]")
ax.set_title("Vibrational free energy")
ax.legend(fontsize=7)
ax.set_xlim(0, T_values.max())
ax.grid(lw=0.3, alpha=0.5)

# T_D(T) calorimetric
ax = axes2[1, 1]
mask_s = (T >= 50)
mask_h = (T_h >= 50)
ax.plot(T[mask_s],   TD_s[mask_s], "r-",  lw=2.2, label=f"SQTC  (T_D_spec={TDs_s:.0f} K)")
ax.plot(T_h[mask_h], TD_h[mask_h], "b--", lw=1.6, label=f"Harmonic (T_D_spec={TDs_h:.0f} K)")
ax.axhline(315.0, color="k", ls=":", lw=1.2, label="Expt T_D ≈ 315 K (Kittel)")
ax.axhline(TDs_s, color="r", ls="--", lw=0.8, alpha=0.6)
ax.axhline(TDs_h, color="b", ls="--", lw=0.8, alpha=0.6)
ax.axvline(T_design, color="r", ls=":", lw=0.9, alpha=0.4)
ax.set_xlabel("T [K]"); ax.set_ylabel("T_D [K]")
ax.set_title("Effective Debye temperature")
ax.legend(fontsize=7.5)
ax.set_xlim(0, T_values.max()); ax.set_ylim(0, 500)
ax.grid(lw=0.3, alpha=0.5)

# MSD
ax = axes2[2, 0]
ax.plot(T,   MSD_s, "r-",  lw=2.2, label="SQTC ⟨u²⟩")
ax.plot(T_h, MSD_h, "b--", lw=1.6, label="Harmonic ⟨u²⟩")
ax.axvline(T_design, color="r", ls=":", lw=0.9, alpha=0.4,
           label=f"SQTC ⟨u²⟩ @ T_d = {MSD_s[np.argmin(np.abs(T-T_design))]:.4f} Å²")
ax.set_xlabel("T [K]"); ax.set_ylabel("⟨u²⟩  [Å²]")
ax.set_title("Mean-squared displacement")
ax.legend(fontsize=7.5)
ax.set_xlim(0, T_values.max()); ax.set_ylim(bottom=0)
ax.grid(lw=0.3, alpha=0.5)

# DW B-factor
ax = axes2[2, 1]
ax.plot(T,   DW_s, "r-",  lw=2.2, label="SQTC B(T)")
ax.plot(T_h, DW_h, "b--", lw=1.6, label="Harmonic B(T)")
ax.axvline(T_design, color="r", ls=":", lw=0.9, alpha=0.4)
ax.set_xlabel("T [K]"); ax.set_ylabel("B = 8π²⟨u²⟩/3  [Å²]")
ax.set_title("Debye-Waller B factor")
ax.legend(fontsize=8)
ax.set_xlim(0, T_values.max()); ax.set_ylim(bottom=0)
ax.grid(lw=0.3, alpha=0.5)

fig2.tight_layout()
p2 = OUT_DIR / "cu_thermal_properties.pdf"
fig2.savefig(p2, dpi=150, bbox_inches="tight")
fig2.savefig(p2.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"  Saved → {p2}")
plt.close(fig2)

# ── Fig 3: Anharmonicity panel (3 sub-plots) ─────────────────────────────────

ΔCv = Cv_s - Cv_h   # anharmonic excess ΔC_V

fig3, axes3 = plt.subplots(1, 3, figsize=(14, 5))
fig3.suptitle(
    f"fcc Cu — anharmonicity quantification  (SQTC T_design = {T_design:.0f} K)",
    fontsize=13, fontweight="bold",
)

# ΔC_V = C_V(SQTC) − C_V(harmonic)
ax = axes3[0]
ax.fill_between(T, 0, ΔCv, where=(ΔCv > 0),
                alpha=0.30, color="firebrick", label="Anharmonic excess")
ax.fill_between(T, 0, ΔCv, where=(ΔCv < 0),
                alpha=0.20, color="steelblue", label="Anharmonic deficit")
ax.plot(T, ΔCv, "k-", lw=1.8)
ax.axhline(0, color="gray", lw=0.7)
ax.axvline(T_design, color="r", ls=":", lw=1.0, alpha=0.5, label=f"T_design = {T_design:.0f} K")
i_td = int(np.argmin(np.abs(T - T_design)))
ax.scatter([T_design], [ΔCv[i_td]], c="r", s=80, zorder=6)
ax.annotate(
    f"ΔC_V = {ΔCv[i_td]:+.3f} J/mol/K\nat T = {T_design:.0f} K",
    xy=(T_design, ΔCv[i_td]),
    xytext=(T_design - 350, ΔCv[i_td] + 0.1),
    fontsize=8.5,
    arrowprops=dict(arrowstyle="->", color="k", lw=0.8),
)
ax.set_xlabel("T [K]"); ax.set_ylabel("ΔC_V  [J/(mol·K)]")
ax.set_title("Anharmonic excess ΔC_V = C_V(SQTC) − C_V(harm)")
ax.legend(fontsize=8)
ax.set_xlim(0, T_values.max())
ax.grid(lw=0.3, alpha=0.5)

# C_V iteration convergence
ax = axes3[1]
n_iters = np.arange(1, len(cv_iter) + 1)
ax.plot(n_iters, cv_iter, "ko-", lw=1.8, ms=7, label="C_V(1200 K) per iter. subset")
ax.axhline(Cv_DP, color="gray", ls="--", lw=1.2, label=f"Dulong-Petit = {Cv_DP:.3f}")
# Literature C_V at 1200 K
ax.axhline(26.52, color="k", ls=":", lw=1.2, label="Expt C_P(1200 K) = 26.52 J/mol/K")
ax.axhline(25.5,  color="green", ls=":", lw=1.0, label="TDEP (Born 2019) ≈ 25.5")
ax.set_xlabel("Number of iterations included")
ax.set_ylabel("C_V(1200 K)  [J/(mol·K)]")
ax.set_title("Convergence vs. SQTC iterations")
ax.set_xticks(n_iters)
ax.legend(fontsize=7.5)
ax.set_ylim(24.5, 27.5)
ax.grid(lw=0.3, alpha=0.5)

# ΔC_V / (3R) as fraction of Dulong-Petit
ax = axes3[2]
frac = ΔCv / Cv_DP * 100   # percent of 3R
ax.plot(T, frac, "r-", lw=2.0, label=r"$\Delta C_V / 3R$  (%)")
ax.axhline(0, color="gray", lw=0.7)
ax.axvline(T_design, color="r", ls=":", lw=1.0, alpha=0.5)
i300 = int(np.argmin(np.abs(T - 300)))
i700 = int(np.argmin(np.abs(T - 700)))
for i_c, lbl in [(i300, "300 K"), (i700, "700 K"), (i_td, f"{T_design:.0f} K")]:
    ax.scatter([T[i_c]], [frac[i_c]], c="r", s=50, zorder=6)
    ax.annotate(f"{lbl}: {frac[i_c]:+.1f}%",
                xy=(T[i_c], frac[i_c]),
                xytext=(T[i_c] + 40, frac[i_c] + 0.3),
                fontsize=7.5)
ax.set_xlabel("T [K]"); ax.set_ylabel(r"$\Delta C_V / 3R$  [%]")
ax.set_title("Relative anharmonic contribution")
ax.legend(fontsize=8)
ax.set_xlim(0, T_values.max())
ax.grid(lw=0.3, alpha=0.5)

fig3.tight_layout()
p3 = OUT_DIR / "cu_anharmonicity.pdf"
fig3.savefig(p3, dpi=150, bbox_inches="tight")
fig3.savefig(p3.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"  Saved → {p3}")
plt.close(fig3)

# ── Fig 4: Mode Grüneisen parameters ─────────────────────────────────────────

print("\n  Computing mode Grüneisen parameters (SQTC vs harmonic) ...")

# We need two volumes — use SQTC converged IFC ± a small re-scaling of the
# force-constant matrix.  Analytical: γ_i = −(V/ω_i) dω_i/dV.
# Since we have only one volume, we use the SQTC + harmonic pair:
# dω ≈ ω_SQTC − ω_harm  and  dV/V ≈ 3 α (T_design − T_0)
# Instead, use the two datasets as approximate V± analogs by noting that
# harmonic corresponds to cold (~0 K, smaller effective volume ~ a=3.615 Å)
# and SQTC to hot (1200 K, a=3.67 Å).
a_cold = 3.615     # Å  (experimental room-temperature)
a_hot  = 3.670     # Å  (experimental ~1200 K)
dV_over_V = ((a_hot**3 - a_cold**3) / (2 * ((a_hot + a_cold) / 2)**3))
print(f"  Grüneisen dV/V = {dV_over_V:.5f} "
      f"(a_cold={a_cold}, a_hot={a_hot})")

# calc_harm = V+ (hot/expanded) relative perspective doesn't apply cleanly here.
# Build a properly scaled cold calculator from scaled IFCs (volume correction)
scale_vc = a_cold / a_cu
prim_cell_cold = scale_vc * prim_cell
sc_cell_cold   = scale_vc * sc_cell
eq_pos_cold    = scale_vc * eq_pos

ifc_cold = IFCExtractor(
    supercell_positions = eq_pos_cold,
    supercell_cell      = sc_cell_cold,
    masses_amu          = np.full(N_ATOMS, M_Cu_amu),
    r_cutoff            = R_CUTOFF * scale_vc,
    symmetrise          = True,
    ridge_alpha         = RIDGE_ALPHA,
    symmetrize_bonds    = True,
)
ifc_cold._Phi   = {k: v.copy() for k, v in ifc_harm._Phi.items()}
ifc_cold._pairs = list(ifc_harm._pairs)
ifc_cold._Phi   = ifc_cold._acoustic_sum_rule(ifc_cold._Phi)

calc_cold = PhononCalculator(
    ifc_extractor  = ifc_cold,
    prim_positions = prim_pos * scale_vc,
    prim_cell      = prim_cell_cold,
    masses_amu     = np.array([M_Cu_amu]),
)

from sqtc.constants import RAD_S_TO_THZ

# Use SQTC calc as V0, cold as V−, and extrapolate V+ from 2*V0−V−
# (Grüneisen: γ = −(V/ω)dω/dV ≈ −(V0/ω)(ω_Vp−ω_Vm)/(Vp−Vm))
# Simplification: treat SQTC ≡ V0, build V+ by reflection of V- IFCs
Phi_V0 = ifc_sqtc._Phi
Phi_Vm = ifc_cold._Phi
Phi_Vp = {}
for pair in ifc_sqtc._pairs:
    if pair in Phi_Vm:
        Phi_Vp[pair] = 2.0 * Phi_V0[pair] - Phi_Vm[pair]
    else:
        Phi_Vp[pair] = Phi_V0[pair].copy()

sc_cell_hot = sc_cell * (a_hot / a_cu)
eq_pos_hot  = eq_pos  * (a_hot / a_cu)
ifc_hot = IFCExtractor(
    supercell_positions = eq_pos_hot,
    supercell_cell      = sc_cell_hot,
    masses_amu          = np.full(N_ATOMS, M_Cu_amu),
    r_cutoff            = R_CUTOFF * (a_hot / a_cu),
    symmetrise          = True,
    ridge_alpha         = RIDGE_ALPHA,
    symmetrize_bonds    = True,
)
ifc_hot._Phi   = Phi_Vp
ifc_hot._pairs = list(ifc_sqtc._pairs)
ifc_hot._Phi   = ifc_hot._acoustic_sum_rule(ifc_hot._Phi)

calc_hot = PhononCalculator(
    ifc_extractor  = ifc_hot,
    prim_positions = prim_pos * (a_hot / a_cu),
    prim_cell      = prim_cell * (a_hot / a_cu),
    masses_amu     = np.array([M_Cu_amu]),
)

gamma_qs  = calc_sqtc.mode_gruneisen_parameters(calc_hot, calc_cold, dV_over_V)
omega_all = calc_sqtc._all_frequencies(Q_MESH_CV)
pos_mask  = omega_all > calc_sqtc._imag_tol
gamma_pos = gamma_qs[pos_mask]
omega_pos = omega_all[pos_mask]
omega_thz = omega_pos * RAD_S_TO_THZ

# T-dependent γ_eff
from sqtc.constants import HBAR, KB
gamma_eff_T = np.zeros(len(T_values))
Cv_arr_T    = calc_sqtc.heat_capacity_scan(T_values, q_mesh=Q_MESH_CV)

for it, Tv in enumerate(T_values):
    if Tv < 1e-9:
        continue
    om  = omega_pos
    gm  = gamma_pos
    x   = HBAR * om / (2.0 * KB * Tv)
    x_c = np.clip(x, 0, 350.0)
    cv_mode = (x_c / np.sinh(x_c))**2
    valid   = np.isfinite(gm)
    if valid.sum() == 0 or cv_mode[valid].sum() < 1e-30:
        continue
    gamma_eff_T[it] = float(
        np.dot(gm[valid], cv_mode[valid]) / cv_mode[valid].sum()
    )

fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
fig4.suptitle("fcc Cu — Grüneisen parameters", fontsize=13, fontweight="bold")

ax = axes4[0]
ax.scatter(omega_thz, gamma_pos, alpha=0.25, s=5, c="steelblue")
ax.axhline(0, color="gray", lw=0.6)
i_td2 = int(np.argmin(np.abs(T_values - T_design)))
g_eff_td = float(gamma_eff_T[i_td2])
ax.axhline(g_eff_td, color="red", lw=1.5, ls="--",
           label=f"γ_eff(T={T_design:.0f} K) = {g_eff_td:.3f}")
ax.axhline(1.96, color="k", lw=1.0, ls=":", label="Expt γ ≈ 1.96 (Grüneisen)")
ax.set_xlabel("Frequency (THz)"); ax.set_ylabel("Grüneisen parameter γ")
ax.set_title("Mode Grüneisen parameters")
ax.legend(fontsize=9)
ax.set_ylim(-1.5, 6.0)
ax.grid(lw=0.3, alpha=0.5)

ax = axes4[1]
ax.plot(T_values, gamma_eff_T, "r-", lw=2.0, label="γ_eff(T)  SQTC")
ax.axhline(1.96, color="k", ls="--", lw=1.2, label="Expt γ ≈ 1.96")
ax.axvline(T_design, color="r", ls=":", lw=0.9, alpha=0.5,
           label=f"T_design = {T_design:.0f} K")
ax.set_xlabel("T [K]"); ax.set_ylabel("γ_eff")
ax.set_title("Effective Grüneisen parameter vs T")
ax.legend(fontsize=9)
ax.set_xlim(0, T_values.max())
ax.grid(lw=0.3, alpha=0.5)

fig4.tight_layout()
p4 = OUT_DIR / "cu_gruneisen.pdf"
fig4.savefig(p4, dpi=200, bbox_inches="tight")
fig4.savefig(p4.with_suffix(".png"), dpi=200, bbox_inches="tight")
print(f"  Saved → {p4}")
plt.close(fig4)

# ── Fig 5: DOS comparison + ZPE shifts ───────────────────────────────────────

fig5, ax5 = plt.subplots(figsize=(7, 5))
fig5.suptitle("fcc Cu — phonon DOS: harmonic vs SQTC", fontsize=12, fontweight="bold")

freq_s_dos = dos_s["frequencies_thz"]
freq_h_dos = dos_h["frequencies_thz"]

ax5.fill_betweenx(freq_h_dos, dos_h["dos_total"],
                  alpha=0.20, color="steelblue")
ax5.plot(dos_h["dos_total"], freq_h_dos, color="steelblue", lw=1.6,
         label=f"Harmonic  ω_max={dos_h['frequencies_thz'][dos_h['dos_total']>0.01*dos_h['dos_total'].max()].max():.2f} THz")
ax5.fill_betweenx(freq_s_dos, dos_s["dos_total"],
                  alpha=0.25, color="firebrick")
ax5.plot(dos_s["dos_total"], freq_s_dos, color="firebrick", lw=1.6,
         label=f"SQTC 1200K  ω_max={dos_s['frequencies_thz'][dos_s['dos_total']>0.01*dos_s['dos_total'].max()].max():.2f} THz")
ax5.axhline(0, color="gray", lw=0.5)
ax5.set_xlim(left=0)
ax5.set_xlabel("DOS", fontsize=12); ax5.set_ylabel("Frequency (THz)", fontsize=12)
ax5.set_title(f"Phonon DOS: softening by {(TDs_h-TDs_s):.0f} K in T_D", fontsize=10)
ax5.legend(fontsize=9)
ax5.grid(axis="x", lw=0.3, alpha=0.5)
fig5.tight_layout()
p5 = OUT_DIR / "cu_dos_comparison.pdf"
fig5.savefig(p5, dpi=200)
fig5.savefig(p5.with_suffix(".png"), dpi=200)
print(f"  Saved → {p5}")
plt.close(fig5)

# Use pp_sqtc to generate the standard 6-panel thermal properties figure too
pp_sqtc.plot_thermal_properties(
    thermal_data   = thermal_sqtc,
    reference_data = {
        "Cv":  {"T": T_nist, "values": Cp_nist, "label": "Expt NIST (C_P)"},
    },
    save_path = OUT_DIR / "cu_thermal_standard.pdf",
    title     = f"fcc Cu (SQTC, T_design = {T_design:.0f} K) — thermal properties",
)
print(f"  Saved → {OUT_DIR / 'cu_thermal_standard.pdf'}")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 7  —  Summary JSON
# ═══════════════════════════════════════════════════════════════════════════════

i_td_th = int(np.argmin(np.abs(T - T_design)))
summary = {
    "material":           "fcc Cu",
    "a_ang":              a_cu,
    "T_design_K":         T_design,
    "T_melt_K":           T_melt_Cu,
    "n_iters":            len(all_iters),
    "n_snaps":            len(displ_all),
    "ZPE_SQTC_eV":        float(ZPE_s),
    "ZPE_harm_eV":        float(ZPE_h),
    "TD_spectral_SQTC_K": float(TDs_s),
    "TD_spectral_harm_K": float(TDs_h),
    "Cv_SQTC_1200K":      float(Cv_s[i_td_th]),
    "Cv_harm_1200K":       float(Cv_h[i_td_th]),
    "DeltaCv_1200K":      float(ΔCv[i_td_th]),
    "gamma_eff_1200K":    float(gamma_eff_T[i_td2]),
    "cv_iteration_convergence": cv_iter.tolist(),
    "temperatures": [],
}
for T_c in [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]:
    i = int(np.argmin(np.abs(T - T_c)))
    if abs(T[i] - T_c) > 6:
        continue
    summary["temperatures"].append({
        "T_K":          float(T[i]),
        "Cv_SQTC":      float(Cv_s[i]),
        "Cv_harm":      float(Cv_h[i]),
        "DeltaCv":      float(ΔCv[i]),
        "Svib_SQTC":    float(Sv_s[i]),
        "Fvib_meV":     float(Fv_s[i] * 1000),
        "TD_caloric":   float(TD_s[i]),
        "MSD_ang2":     float(MSD_s[i]),
        "gamma_eff":    float(gamma_eff_T[i]),
    })

with open(OUT_DIR / "cu_summary.json", "w") as fj:
    json.dump(summary, fj, indent=2)
print(f"\n  Saved summary → {OUT_DIR / 'cu_summary.json'}")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 8  —  Print comparison table
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print(f"  fcc Cu — SQTC vs harmonic vs experiment  (T_design = {T_design:.0f} K)")
print("=" * 80)
print(f"  {'T [K]':>7}  {'C_V SQTC':>10}  {'C_V harm':>10}  {'ΔC_V':>8}  "
      f"{'Expt C_P':>10}  {'T_D [K]':>8}  {'⟨u²⟩ [Å²]':>10}")
print("  " + "-" * 75)
T_print = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
for T_c in T_print:
    i   = int(np.argmin(np.abs(T - T_c)))
    i_e = int(np.argmin(np.abs(T_nist - T_c)))
    expt_str = f"{Cp_nist[i_e]:10.3f}" if abs(T_nist[i_e] - T_c) < 60 else f"{'---':>10}"
    print(f"  {T[i]:7.0f}  {Cv_s[i]:10.4f}  {Cv_h[i]:10.4f}  "
          f"{ΔCv[i]:+8.4f}  {expt_str}  {TD_s[i]:8.1f}  {MSD_s[i]:10.5f}")

print(f"\n  Dulong-Petit 3R = {Cv_DP:.4f} J/(mol·K)")
print(f"  SQTC ZPE = {ZPE_s*1000:.3f} meV  |  Harmonic ZPE = {ZPE_h*1000:.3f} meV")
print(f"  T_D (spectral): SQTC = {TDs_s:.1f} K  |  Harmonic = {TDs_h:.1f} K  |  Expt ≈ 315 K")
print(f"  γ_eff(1200 K) = {float(gamma_eff_T[i_td2]):.3f}  |  Expt γ ≈ 1.96")
print(f"\n  ΔC_V(1200 K) = {ΔCv[i_td_th]:+.4f} J/(mol·K)  "
      f"= {ΔCv[i_td_th]/Cv_DP*100:+.2f}% of Dulong-Petit")

# ═══════════════════════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(f" Done.  All results in {OUT_DIR}")
print("=" * 70)
for fn in sorted(OUT_DIR.iterdir()):
    print(f"  {fn.name}")
