"""
SQTC Post-processor  —  phonons + temperature-dependent properties.

Provides two public classes:

  SQTCPostProcessor
      Generic post-processor for a fitted :class:`PhononCalculator`.
      Computes, plots and saves:
        • Phonon band structure + partial/total DOS
        • C_V(T), S_vib(T), F_vib(T), ZPE, T_D(T)
        • Mean-squared displacement <u²>(T), Debye-Waller B(T)

  SQTCRunLoader
      Reads force/displacement data from a completed SQTC work directory
      (POSCAR + OUTCAR in each snapshot sub-folder), re-fits the IFCs, and
      returns a ready-to-use :class:`SQTCPostProcessor`.

Typical usage::

    # Post-process a finished run without re-running SQTC
    loader = SQTCRunLoader(
        work_dir    = "sqtc_al_fast_vasp_run",
        prim_cell   = prim_cell_al,
        prim_pos    = prim_pos_al,
        masses_amu  = [26.98],
        elements    = ["Al"],
        r_cutoff    = 4.5,
        ridge_alpha = 1e-3,
        symmetrize_bonds = True,
    )
    pp = loader.build()              # → SQTCPostProcessor
    T  = np.arange(10, 1001, 10)

    phonon_data  = pp.compute_phonons()
    thermal_data = pp.compute_thermal_properties(T)

    pp.plot_phonons(save_path="phonons.pdf")
    pp.plot_thermal_properties(T, save_path="thermal.pdf")
    pp.save(out_dir="sqtc_al_fast_vasp_run/postproc")

Alternatively, if you already have a PhononCalculator::

    from sqtc.postprocessor import SQTCPostProcessor
    pp = SQTCPostProcessor(phonon_calc, label="fcc Al", elements=["Al"])
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from .constants import HBAR, KB, NA, AMU_TO_KG
    from .ifc_extractor import IFCExtractor
    from .phonons import PhononCalculator
except ImportError:
    # Running as __main__ directly: add parent of sqtc/ to sys.path
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from sqtc.constants import HBAR, KB, NA, AMU_TO_KG
    from sqtc.ifc_extractor import IFCExtractor
    from sqtc.phonons import PhononCalculator

# ── Default element colour palette ────────────────────────────────────────────
_DEFAULT_COLORS: Dict[str, str] = {
    "H":  "#1f77b4", "He": "#aec7e8",
    "Li": "#e7969c", "Be": "#c5b0d5", "B":  "#c49c94", "C":  "#555555",
    "N":  "#7f7f7f", "O":  "#e377c2", "F":  "#98df8a", "Na": "#17becf",
    "Mg": "#bcbd22", "Al": "#1f77b4", "Si": "#ffbb78", "P":  "#f7b6d2",
    "S":  "#dbdb8d", "Cl": "#ff7f0e", "K":  "#9edae5", "Ca": "#6b6ecf",
    "Ti": "#9467bd", "V":  "#8c564b", "Cr": "#e7ba52", "Mn": "#843c39",
    "Fe": "#ad494a", "Co": "#d6616b", "Ni": "#2ca02c", "Cu": "#d62728",
    "Zn": "#cedb9c", "Ga": "#b5cf6b", "Ge": "#637939", "As": "#8c6d31",
    "Se": "#bd9e39", "Br": "#e7cb94", "Sr": "#393b79", "Y":  "#5254a3",
    "Zr": "#6b6ecf", "Nb": "#9c9ede", "Mo": "#8c564b", "Ru": "#c7c7c7",
    "Rh": "#7b4173", "Pd": "#ce6dbd", "Ag": "#aec7e8", "In": "#d9d9d9",
    "Sn": "#bcbcbc", "Sb": "#a0a0a0", "Te": "#848484", "Cs": "#252525",
    "Ba": "#3f007d", "La": "#6a51a3", "Hf": "#807dba", "Ta": "#9e9ac8",
    "W":  "#7f7f7f", "Re": "#636363", "Os": "#525252", "Ir": "#3f3f3f",
    "Pt": "#c0c0c0", "Au": "#ffbb78", "Tl": "#e6550d", "Pb": "#fd8d3c",
    "Bi": "#fdae6b",
}


# ── Standard atomic masses (IUPAC 2021) ───────────────────────────────────────
_STANDARD_MASSES: Dict[str, float] = {
    "H": 1.008,   "He": 4.003,  "Li": 6.941,  "Be": 9.012,
    "B": 10.81,   "C": 12.011,  "N": 14.007,  "O": 15.999,
    "F": 18.998,  "Ne": 20.180, "Na": 22.990,  "Mg": 24.305,
    "Al": 26.982, "Si": 28.085, "P": 30.974,  "S": 32.06,
    "Cl": 35.45,  "Ar": 39.948, "K": 39.098,  "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942,  "Cr": 51.996,
    "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693,
    "Cu": 63.546, "Zn": 65.38,  "Ga": 69.723, "Ge": 72.630,
    "As": 74.922, "Se": 78.971, "Br": 79.904, "Kr": 83.798,
    "Rb": 85.468, "Sr": 87.62,  "Y": 88.906,  "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.96,  "Ru": 101.07, "Rh": 102.906,
    "Pd": 106.42, "Ag": 107.868,"Cd": 112.411,"In": 114.818,
    "Sn": 118.710,"Sb": 121.760,"Te": 127.60, "I": 126.904,
    "Cs": 132.905,"Ba": 137.327,"La": 138.905,"Ce": 140.116,
    "Hf": 178.49, "Ta": 180.948,"W": 183.84,  "Re": 186.207,
    "Os": 190.23, "Ir": 192.217,"Pt": 195.084,"Au": 196.967,
    "Hg": 200.59, "Tl": 204.383,"Pb": 207.2,  "Bi": 208.980,
}


class SQTCPostProcessor:
    """
    Generic post-processor for a fitted :class:`PhononCalculator`.

    Parameters
    ----------
    phonon_calc : PhononCalculator
        A fully fitted PhononCalculator (IFCs already extracted).
    label : str
        Human-readable compound label, e.g. "fcc Al" or "NaCl".
    elements : list of str, optional
        Element symbols for each basis atom in the primitive cell.
        Used for partial-DOS colouring and plot legends.
    element_colors : dict, optional
        Override the default element → colour mapping.
    T_design : float, optional
        SQTC design temperature [K]; shown on plot titles.
    """

    def __init__(
        self,
        phonon_calc: PhononCalculator,
        label: str = "",
        elements: Optional[List[str]] = None,
        element_colors: Optional[Dict[str, str]] = None,
        T_design: float = 300.0,
    ):
        self.calc      = phonon_calc
        self.label     = label
        self.elements  = elements or []
        self.colors    = dict(_DEFAULT_COLORS)
        if element_colors:
            self.colors.update(element_colors)
        self.T_design  = T_design

        # Cached results (populated on demand)
        self._phonon_data:  Optional[Dict] = None
        self._thermal_data: Optional[Dict] = None

    # ── Phonon band structure + DOS ───────────────────────────────────────────

    def compute_phonons(
        self,
        q_mesh: Tuple[int, int, int] = (30, 30, 30),
        n_bins: int = 500,
        sigma_thz: float = 0.10,
        n_points_per_segment: int = 100,
    ) -> Dict:
        """
        Compute phonon band structure + partial/total DOS.

        Returns
        -------
        dict with keys:
            ``bs``   — result of :meth:`~PhononCalculator.compute_band_structure`
            ``pdos`` — result of :meth:`~PhononCalculator.compute_partial_dos`
        """
        bs   = self.calc.compute_band_structure(
            n_points_per_segment=n_points_per_segment
        )
        pdos = self.calc.compute_partial_dos(
            q_mesh=q_mesh, n_bins=n_bins, sigma_thz=sigma_thz
        )
        stats = self.calc.spectrum_statistics(q_mesh=(10, 10, 10))
        self._phonon_data = {"bs": bs, "pdos": pdos, "stats": stats}
        return self._phonon_data

    # ── Temperature-dependent properties ─────────────────────────────────────

    def compute_thermal_properties(
        self,
        T_values: np.ndarray,
        q_mesh: Tuple[int, int, int] = (20, 20, 20),
    ) -> Dict:
        """
        Compute all temperature-dependent properties.

        Returns
        -------
        dict with keys:
            ``T_K``         — temperatures [K]
            ``Cv_jmolk``    — C_V [J/(mol·K)]
            ``Svib_jmolk``  — S_vib [J/(mol·K)]
            ``Fvib_ev``     — F_vib [eV/f.u.]
            ``ZPE_eV``      — zero-point energy [eV/f.u.]  (scalar)
            ``TD_spectral`` — spectral-moment Debye temperature [K] (scalar)
            ``TD_caloric``  — calorimetric T_D at each T [K]
            ``MSD_ang2``    — mean-squared displacement <u²> [Å²]
            ``DW_B_ang2``   — Debye-Waller B = 8π²<u²>/3 [Å²]
        """
        T_arr = np.asarray(T_values, dtype=float)
        n_T   = len(T_arr)

        # ZPE and spectral T_D (T-independent)
        ZPE_eV      = self.calc.zero_point_energy(q_mesh=q_mesh)
        TD_spectral = self.calc.debye_temperature_from_dos(q_mesh=q_mesh)

        # C_V scan (vectorised)
        Cv_arr = self.calc.heat_capacity_scan(T_arr, q_mesh=q_mesh)

        # Precompute frequency mesh once
        all_omegas          = self.calc._all_frequencies(q_mesh)
        n_q_pts             = all_omegas.shape[0]
        omegas_flat         = all_omegas.ravel()
        valid_mask, pos_mask, omega_use = self.calc._classify_omegas(omegas_flat)
        omega_safe = np.where(pos_mask, omega_use, np.inf)

        Svib_arr  = np.zeros(n_T)
        Fvib_arr  = np.zeros(n_T)
        TD_arr    = np.zeros(n_T)
        MSD_arr   = np.zeros(n_T)
        DW_arr    = np.zeros(n_T)

        for it, T in enumerate(T_arr):
            Fvib_arr[it] = self.calc.vibrational_free_energy(T, q_mesh=q_mesh)
            Svib_arr[it] = self.calc.vibrational_entropy(T, q_mesh=q_mesh) * NA

            # Calorimetric T_D — reuse Cv already computed by heat_capacity_scan
            TD_arr[it] = self.calc._calorimetric_debye_temperature(Cv_arr[it], T)

            # Mean-squared displacement
            if T < 1e-9:
                coth_vals = np.ones_like(omega_use)
            else:
                x = np.clip(HBAR * omega_use / (2.0 * KB * T), 0, 350.0)
                coth_vals = np.where(x > 1e-10, 1.0 / np.tanh(x), 1.0 / x)

            # Average over all basis-atom masses (for multi-species: use mean mass)
            M_mean_SI = float(np.mean(self.calc.masses_amu)) * AMU_TO_KG
            contrib = np.where(pos_mask, coth_vals / omega_safe, 0.0)
            msd_SI = (HBAR / (2.0 * M_mean_SI)) * np.sum(contrib) / n_q_pts
            MSD_arr[it] = msd_SI * 1e20   # m² → Å²
            DW_arr[it]  = 8.0 * np.pi**2 * MSD_arr[it] / 3.0

        self._thermal_data = {
            "T_K":         T_arr,
            "Cv_jmolk":    Cv_arr,
            "Svib_jmolk":  Svib_arr,
            "Fvib_ev":     Fvib_arr,
            "ZPE_eV":      ZPE_eV,
            "TD_spectral": TD_spectral,
            "TD_caloric":  TD_arr,
            "MSD_ang2":    MSD_arr,
            "DW_B_ang2":   DW_arr,
        }
        return self._thermal_data

    # ── Plotting: phonon band + DOS ───────────────────────────────────────────

    def plot_phonons(
        self,
        phonon_data: Optional[Dict] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
        title: Optional[str] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ):
        """
        Plot phonon band structure + partial/total DOS side-by-side.

        Parameters
        ----------
        phonon_data  : output of :meth:`compute_phonons`; computed if None.
        save_path    : save figure to this path (pdf/png auto-detected).
        show         : call plt.show() after drawing.
        title        : override auto title.
        ylim         : (ymin, ymax) in THz; auto if None.
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        if phonon_data is None:
            phonon_data = self._phonon_data or self.compute_phonons()

        bs   = phonon_data["bs"]
        pdos = phonon_data["pdos"]

        dists  = bs["distances"]
        freqs  = bs["frequencies_thz"]    # (n_q, n_branches)

        freq_bins = pdos["frequencies_thz"]
        dos_total = pdos["dos_total"]
        dos_parts = pdos["dos_partial"]   # (n_b, n_bins)

        unique_elems = list(dict.fromkeys(self.elements)) if self.elements else []
        dos_by_elem  = {}
        if unique_elems:
            for elem in unique_elems:
                mask = np.array([e == elem for e in self.elements])
                dos_by_elem[elem] = dos_parts[mask].sum(axis=0)

        fig = plt.figure(figsize=(11, 5))
        gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharey=ax1)

        # Band structure
        for b in range(freqs.shape[1]):
            ax1.plot(dists, freqs[:, b], color="steelblue", lw=1.2, alpha=0.9)
        for lpos in bs["label_positions"]:
            ax1.axvline(x=lpos, color="gray", lw=0.7, ls="--")
        ax1.axhline(y=0, color="gray", lw=0.5)
        ax1.set_xticks(bs["label_positions"])
        ax1.set_xticklabels(
            [lbl.replace("G", r"$\Gamma$") for lbl in bs["labels"]],
            fontsize=12,
        )
        ax1.set_xlim(dists[0], dists[-1])
        ax1.set_ylabel("Frequency (THz)", fontsize=12)
        ax1.tick_params(axis="x", length=0)
        ax1.grid(axis="y", lw=0.3, alpha=0.5)
        auto_title = (
            title if title is not None
            else f"{self.label} — SQTC phonons at {self.T_design:.0f} K"
        )
        ax1.set_title(auto_title, fontsize=12)

        # DOS
        n_unique = len(unique_elems)
        if n_unique > 1:
            for elem in unique_elems:
                d   = dos_by_elem[elem]
                col = self.colors.get(elem, "#aaaaaa")
                ax2.fill_betweenx(freq_bins, d, alpha=0.30, color=col, label=elem)
                ax2.plot(d, freq_bins, color=col, lw=1.2, ls="--", alpha=0.8)
            ax2.plot(dos_total, freq_bins, color="k", lw=1.5, label="Total", zorder=5)
            ax2.legend(fontsize=8, loc="upper right", framealpha=0.7)
        else:
            col = self.colors.get(
                (unique_elems[0] if unique_elems else ""), "steelblue"
            )
            ax2.fill_betweenx(freq_bins, dos_total, alpha=0.35, color=col)
            ax2.plot(dos_total, freq_bins, color=col, lw=1.5)

        ax2.axhline(y=0, color="gray", lw=0.5)
        ax2.set_xlabel("DOS", fontsize=11)
        ax2.set_xlim(left=0)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.tick_params(axis="y", length=0)
        ax2.grid(axis="y", lw=0.3, alpha=0.5)

        # y-limits
        if ylim is not None:
            ax1.set_ylim(*ylim)
        else:
            y_min = min(-0.3, float(freqs.min()) * 1.1)
            pos_dos = freq_bins[dos_total > 0.01 * dos_total.max()] if dos_total.max() > 0 else freq_bins
            y_max = max(float(freqs.max()), float(pos_dos.max()) if len(pos_dos) else 0) * 1.07
            ax1.set_ylim(bottom=y_min, top=y_max)

        fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.12, wspace=0.05)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200)
            # Also save companion format
            alt = save_path.with_suffix(".png" if save_path.suffix == ".pdf" else ".pdf")
            fig.savefig(alt, dpi=200)

        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    # ── Plotting: temperature-dependent properties ────────────────────────────

    def plot_thermal_properties(
        self,
        thermal_data: Optional[Dict] = None,
        T_values: Optional[np.ndarray] = None,
        reference_data: Optional[Dict] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
        title: Optional[str] = None,
    ):
        """
        Plot C_V, S_vib, F_vib, T_D, <u²>, and B(T).

        Parameters
        ----------
        thermal_data    : output of :meth:`compute_thermal_properties`.
                          Computed if None (requires T_values).
        T_values        : temperatures [K]; required if thermal_data is None.
        reference_data  : dict of reference curves/points for comparison.

                          Format::

                              {
                                "Cv":  {"T": ..., "values": ..., "label": "Expt"},
                                "Svib":{"T": ..., "values": ..., "label": "..."},
                                "Cp":  {"T": ..., "values": ..., "label": "..."},
                                "TD":  {"T": ..., "values": ..., "label": "..."},
                                "MSD": {"T": ..., "values": ..., "label": "..."},
                                "DW":  {"T": ..., "values": ..., "label": "..."},
                              }

        save_path       : save figure here.
        show            : call plt.show().
        title           : override auto suptitle.
        """
        import matplotlib.pyplot as plt

        if thermal_data is None:
            if self._thermal_data is not None:
                thermal_data = self._thermal_data
            elif T_values is not None:
                thermal_data = self.compute_thermal_properties(T_values)
            else:
                raise ValueError("Provide thermal_data or T_values.")

        T   = thermal_data["T_K"]
        Cv  = thermal_data["Cv_jmolk"]
        S   = thermal_data["Svib_jmolk"]
        F   = thermal_data["Fvib_ev"]
        ZPE = thermal_data["ZPE_eV"]
        TD  = thermal_data["TD_caloric"]
        TDs = thermal_data["TD_spectral"]
        MSD = thermal_data["MSD_ang2"]
        DW  = thermal_data["DW_B_ang2"]
        ref = reference_data or {}

        fig, axes = plt.subplots(3, 2, figsize=(12, 13))
        suptitle = title if title is not None else (
            f"{self.label} — Temperature-dependent properties (SQTC)"
        )
        fig.suptitle(suptitle, fontsize=13)

        def _add_ref(ax, key, marker="o", ms=40, color="red"):
            if key in ref:
                rd = ref[key]
                T_r, v_r, lbl = rd["T"], rd["values"], rd.get("label", "Ref")
                ax.scatter(T_r, v_r, marker=marker, s=ms, c=color,
                           zorder=5, label=lbl)

        # C_V
        ax = axes[0, 0]
        ax.plot(T, Cv, "b-", lw=2, label="SQTC C_V")
        ax.axhline(3 * 8.314 * self.calc.n_b, color="gray", ls="--", lw=1,
                   label=f"Dulong-Petit ({3*self.calc.n_b:.0f}R)")
        _add_ref(ax, "Cv");  _add_ref(ax, "Cp", marker="s", color="darkred")
        ax.set_xlabel("T [K]"); ax.set_ylabel("C_V  [J/(mol·K)]")
        ax.set_title("Heat capacity"); ax.legend(fontsize=7.5)
        ax.set_xlim(0, T.max()); ax.set_ylim(bottom=0)

        # S_vib
        ax = axes[0, 1]
        ax.plot(T, S, "b-", lw=2, label="SQTC S_vib")
        _add_ref(ax, "Svib")
        ax.set_xlabel("T [K]"); ax.set_ylabel("S_vib  [J/(mol·K)]")
        ax.set_title("Vibrational entropy"); ax.legend(fontsize=8)
        ax.set_xlim(0, T.max())

        # F_vib
        ax = axes[1, 0]
        ax.plot(T, F * 1000, "b-", lw=2, label="SQTC F_vib")
        ax.axhline(ZPE * 1000, color="gray", ls="--", lw=1,
                   label=f"ZPE = {ZPE*1000:.2f} meV")
        _add_ref(ax, "Fvib")
        ax.set_xlabel("T [K]"); ax.set_ylabel("F_vib  [meV/f.u.]")
        ax.set_title("Vibrational free energy"); ax.legend(fontsize=8)
        ax.set_xlim(0, T.max())

        # T_D(T)
        ax = axes[1, 1]
        ax.plot(T, TD, "b-", lw=2, label="T_D calorimetric (SQTC)")
        ax.axhline(TDs, color="orange", ls="--", lw=1.5,
                   label=f"T_D spectral = {TDs:.0f} K")
        _add_ref(ax, "TD", marker="^", color="green")
        ax.set_xlabel("T [K]"); ax.set_ylabel("T_D  [K]")
        ax.set_title("Effective Debye temperature"); ax.legend(fontsize=7.5)
        ax.set_xlim(0, T.max())

        # MSD
        ax = axes[2, 0]
        ax.plot(T, MSD, "b-", lw=2, label="SQTC <u²>")
        _add_ref(ax, "MSD", marker="D", color="firebrick")
        ax.set_xlabel("T [K]"); ax.set_ylabel("<u²>  [Å²]")
        ax.set_title("Mean-squared displacement"); ax.legend(fontsize=8)
        ax.set_xlim(0, T.max()); ax.set_ylim(bottom=0)

        # DW B-factor
        ax = axes[2, 1]
        ax.plot(T, DW, "b-", lw=2, label="SQTC B(T)")
        _add_ref(ax, "DW", marker="D", color="firebrick")
        ax.set_xlabel("T [K]"); ax.set_ylabel("B = 8π²<u²>/3  [Å²]")
        ax.set_title("Debye-Waller B factor"); ax.legend(fontsize=8)
        ax.set_xlim(0, T.max()); ax.set_ylim(bottom=0)

        plt.tight_layout()
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
            alt = save_path.with_suffix(".png" if save_path.suffix == ".pdf" else ".pdf")
            fig.savefig(alt, dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    # ── Save to disk ──────────────────────────────────────────────────────────

    def save(
        self,
        out_dir: Union[str, Path],
        phonon_data: Optional[Dict]  = None,
        thermal_data: Optional[Dict] = None,
    ) -> None:
        """
        Save phonon and thermal results to ``out_dir`` as .npz + .json files.

        Uses cached data from previous ``compute_*`` calls if arguments are None.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        pd_ = phonon_data  or self._phonon_data
        td_ = thermal_data or self._thermal_data

        if pd_ is not None:
            bs   = pd_["bs"]
            pdos = pd_["pdos"]
            np.savez(
                out_dir / "phonon_bandstructure.npz",
                distances=bs["distances"],
                frequencies_thz=bs["frequencies_thz"],
                q_points=bs["q_points"],
                labels=np.array(bs["labels"], dtype=str),
                label_positions=np.array(bs["label_positions"]),
            )
            np.savez(
                out_dir / "phonon_dos.npz",
                frequencies_thz=pdos["frequencies_thz"],
                dos_total=pdos["dos_total"],
                dos_partial=pdos["dos_partial"],
                elements=np.array(self.elements, dtype=str) if self.elements else np.array([]),
            )

        if td_ is not None:
            np.savez(
                out_dir / "thermal_properties.npz",
                **{k: np.asarray(v) for k, v in td_.items()},
            )
            # JSON summary at key temperatures
            T_arr    = td_["T_K"]
            key_Ts   = [10, 50, 100, 200, 298, 300, 400, 500, 600, 700, 800, 900, 1000]
            summary  = {
                "label":           self.label,
                "elements":        self.elements,
                "T_design_K":      self.T_design,
                "ZPE_eV":          float(td_["ZPE_eV"]),
                "TD_spectral_K":   float(td_["TD_spectral"]),
                "temperatures":    [],
            }
            for T_c in key_Ts:
                i = int(np.argmin(np.abs(T_arr - T_c)))
                if abs(T_arr[i] - T_c) > 5:
                    continue
                row = {"T_K": float(T_arr[i])}
                for key in ("Cv_jmolk", "Svib_jmolk", "TD_caloric", "MSD_ang2", "DW_B_ang2"):
                    row[key] = float(td_[key][i])
                row["Fvib_meV"] = float(td_["Fvib_ev"][i]) * 1000
                summary["temperatures"].append(row)
            with open(out_dir / "thermal_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

    def print_summary(
        self,
        thermal_data: Optional[Dict] = None,
        phonon_data: Optional[Dict]  = None,
    ) -> None:
        """Print a formatted summary table to stdout."""
        pd_ = phonon_data  or self._phonon_data
        td_ = thermal_data or self._thermal_data

        print(f"\n{'='*78}")
        print(f"  {self.label}  —  SQTC phonon + thermal summary")
        print(f"{'='*78}")

        if pd_ is not None:
            stats = pd_["stats"]
            print(f"  ω_max = {stats['max_freq_thz']:.2f} THz  |  "
                  f"unstable = {stats['unstable_fraction']:.4f}")
            print(f"  Band path: {' - '.join(pd_['bs']['labels'])}")

        if td_ is not None:
            T_arr = td_["T_K"]
            print(f"\n  ZPE = {td_['ZPE_eV']*1000:.3f} meV  |  "
                  f"T_D (spectral) = {td_['TD_spectral']:.1f} K")
            print(f"\n  {'T [K]':>7}  {'C_V':>12}  {'S_vib':>12}  "
                  f"{'F_vib [meV]':>12}  {'T_D [K]':>8}  {'<u²> [Å²]':>10}  {'B [Å²]':>8}")
            print("  " + "-" * 78)
            for T_c in [10, 100, 200, 300, 400, 600, 800, 1000]:
                i = int(np.argmin(np.abs(T_arr - T_c)))
                if abs(T_arr[i] - T_c) > 5:
                    continue
                print(f"  {T_arr[i]:7.0f}  "
                      f"{td_['Cv_jmolk'][i]:12.4f}  "
                      f"{td_['Svib_jmolk'][i]:12.4f}  "
                      f"{td_['Fvib_ev'][i]*1000:12.3f}  "
                      f"{td_['TD_caloric'][i]:8.1f}  "
                      f"{td_['MSD_ang2'][i]:10.5f}  "
                      f"{td_['DW_B_ang2'][i]:8.5f}")
        print()


# ── POSCAR / OUTCAR parsing helpers ───────────────────────────────────────────

def _parse_poscar(path: Path):
    """
    Parse a VASP POSCAR/CONTCAR.

    Returns
    -------
    cell : (3,3) float [Å]  — lattice vectors (rows)
    species : list of str   — per-atom species labels
    positions : (n,3) float [Å]  — Cartesian positions
    """
    lines = Path(path).read_text().splitlines()
    scale = float(lines[1])
    cell  = scale * np.array([
        [float(x) for x in lines[2].split()],
        [float(x) for x in lines[3].split()],
        [float(x) for x in lines[4].split()],
    ])
    # VASP5 has species-name line (line 5); VASP4 goes straight to counts.
    try:
        counts = [int(x) for x in lines[5].split()]
        species_names_line = None
        counts_line_idx    = 5
    except ValueError:
        species_names_line = lines[5].split()
        counts             = [int(x) for x in lines[6].split()]
        counts_line_idx    = 6

    n_atoms = sum(counts)
    if species_names_line is not None:
        species = []
        for s, n in zip(species_names_line, counts):
            species += [s] * n
    else:
        species = ["X"] * n_atoms

    # Skip optional "Selective dynamics" line
    coord_idx = counts_line_idx + 1
    if lines[coord_idx].strip().lower()[0] == "s":
        coord_idx += 1
    coord_type = lines[coord_idx].strip().lower()[0]
    pos_start  = coord_idx + 1
    pos_lines  = [l for l in lines[pos_start: pos_start + n_atoms] if l.strip()]
    positions  = np.array([[float(x) for x in l.split()[:3]] for l in pos_lines])
    if coord_type == "d":
        positions = positions @ cell
    return cell, species, positions


def _parse_outcar_forces(path: Path, n_atoms: int) -> np.ndarray:
    """Extract the last set of forces from OUTCAR → (n_atoms, 3) eV/Å."""
    text = Path(path).read_text()
    pattern = (
        r"TOTAL-FORCE \(eV/Angst\)\s*[-\s]+\n"
        r"((?:\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\n)+)"
    )
    blocks = re.findall(pattern, text)
    if not blocks:
        raise ValueError(f"No TOTAL-FORCE block found in {path}")
    rows   = [r for r in blocks[-1].strip().split("\n") if r.strip()]
    forces = np.array([[float(x) for x in row.split()[3:6]] for row in rows])
    return forces[:n_atoms]


def _parse_qe_pwo(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a Quantum ESPRESSO pw.x output file via ASE.

    Returns
    -------
    positions : (n_atoms, 3) float [Å]
    forces    : (n_atoms, 3) float [eV/Å]
    cell      : (3, 3) float [Å]  (row vectors)
    """
    try:
        from ase.io import read as _ase_read
    except ImportError as exc:
        raise ImportError(
            "ASE is required to read QE output files. "
            "Install it with: pip install ase"
        ) from exc
    atoms = _ase_read(str(path), format="espresso-out", index=-1)
    return (
        atoms.get_positions().copy(),
        atoms.get_forces().copy(),
        np.array(atoms.cell),
    )


def _build_eq_positions(
    sc_cell: np.ndarray,
    prim_cell: np.ndarray,
    prim_pos: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct equilibrium supercell positions from the HNF matrix
    H = round(sc_cell @ prim_cell⁻¹).

    Replicates the runner's enumeration loop exactly so that atom ordering
    is identical to the POSCAR written during the run.

    Returns
    -------
    eq_positions : (n_total, 3) float [Å]
    H            : (3,3) int — HNF transformation matrix
    """
    H     = np.round(sc_cell @ np.linalg.inv(prim_cell)).astype(int)
    H_inv = np.linalg.inv(H.astype(float))
    max_idx = max(
        H[0, 0],
        abs(H[0, 1]) + H[1, 1],
        abs(H[0, 2]) + abs(H[1, 2]) + H[2, 2],
    ) + 1
    sc_pos = []
    for n1 in range(max_idx):
        for n2 in range(max_idx):
            for n3 in range(max_idx):
                m    = np.array([n1, n2, n3], dtype=float)
                frac = m @ H_inv
                if np.all(frac >= -1e-9) and np.all(frac < 1.0 - 1e-9):
                    shift = m @ prim_cell
                    for p in prim_pos:
                        sc_pos.append(p + shift)
    return np.array(sc_pos), H


def _unsort_to_prim_order(
    arrays: List[np.ndarray],
    elements_per_atom: List[str],
    n_total: int,
) -> List[np.ndarray]:
    """
    Undo VASP's alphabetical-species sorting (applied by VASPRunner).

    VASP sorts atoms alphabetically by species when writing POSCAR.
    If the primitive cell uses a non-alphabetical order (e.g. Na before Cl),
    the POSCAR atom order differs from the runner's internal repeating order.
    This function computes the inverse permutation and applies it.
    """
    n_b  = len(elements_per_atom)
    n_fu = n_total // n_b
    species  = elements_per_atom * n_fu       # original repeating order
    sorted_i = sorted(range(n_total), key=lambda i: species[i])
    inv      = np.empty(n_total, dtype=int)
    for new_i, old_i in enumerate(sorted_i):
        inv[old_i] = new_i
    return [arr[inv] for arr in arrays]


def _needs_unsort(elements_per_atom: List[str]) -> bool:
    """True if VASP alphabetical sort would change the atom order."""
    if len(elements_per_atom) <= 1:
        return False
    if len(set(elements_per_atom)) == 1:
        return False
    # Compare original order with sorted order
    n_fu = 4   # arbitrary; permutation depends only on element labels
    orig = elements_per_atom * n_fu
    return orig != sorted(orig)


# ── SQTCRunLoader ─────────────────────────────────────────────────────────────

class SQTCRunLoader:
    """
    Load force/displacement data from a completed SQTC work directory.

    Supports both VASP (``POSCAR`` + ``OUTCAR``) and Quantum ESPRESSO
    (``espresso.pwo``) snapshot formats.

    Parameters
    ----------
    work_dir         : path to the SQTC run directory (e.g. ``sqtc_al_fast_vasp_run``).
    prim_cell        : (3,3) primitive cell lattice vectors [Å].
    prim_pos         : (n_b, 3) primitive basis positions [Å].
    masses_amu       : list of masses per basis atom [amu].
    elements         : element symbols per basis atom; used for partial-DOS labels.
    r_cutoff         : IFC regression cutoff radius [Å].
    ridge_alpha      : Tikhonov regularisation strength.
    symmetrize_bonds : central-force symmetry projection.
    label            : human-readable label (defaults to work_dir stem).
    T_design         : SQTC design temperature [K] (read from results.json if None).
    use_all_iters    : if True, pool snapshots from all iterations (more data,
                       potentially better IFC fit).  If False, only use the
                       iteration indicated by ``selected_iteration`` in
                       ``sqtc_results.json``.
    calculator       : ``'auto'`` (default), ``'vasp'``, or ``'qe'``.
                       ``'auto'`` detects the format by checking snap directory
                       contents.
    qe_scratch_dir   : path to the ``QEForceCalculator`` workdir containing
                       ``snap_NNNN/espresso.pwo`` files.  Defaults to
                       ``work_dir/qe_scratch`` when ``calculator='qe'``.
    """

    def __init__(
        self,
        work_dir: Union[str, Path],
        prim_cell: np.ndarray,
        prim_pos: np.ndarray,
        masses_amu: List[float],
        elements: Optional[List[str]] = None,
        r_cutoff: float = 6.0,
        ridge_alpha: float = 0.0,
        symmetrize_bonds: bool = False,
        label: str = "",
        T_design: Optional[float] = None,
        use_all_iters: bool = True,
        calculator: str = "auto",
        qe_scratch_dir: Optional[Union[str, Path]] = None,
    ):
        self.work_dir         = Path(work_dir)
        self.prim_cell        = np.asarray(prim_cell, dtype=float)
        self.prim_pos         = np.asarray(prim_pos,  dtype=float)
        self.masses_amu       = list(masses_amu)
        self.elements         = elements or []
        self.r_cutoff         = r_cutoff
        self.ridge_alpha      = ridge_alpha
        self.symmetrize_bonds = symmetrize_bonds
        self.label            = label or self.work_dir.stem
        self.T_design         = T_design
        self.use_all_iters    = use_all_iters
        self.calculator       = calculator.lower()
        self.qe_scratch_dir   = Path(qe_scratch_dir) if qe_scratch_dir else None

    def _load_results_json(self) -> Optional[Dict]:
        p = self.work_dir / "sqtc_results.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return None

    def _iter_dirs_to_load(self, saved: Optional[Dict]) -> List[Path]:
        """Return the list of iter_* directories whose snaps we should read."""
        if self.use_all_iters:
            dirs = sorted(self.work_dir.glob("iter_*/"))
        elif saved is not None and "selected_iteration" in saved:
            sel  = saved["selected_iteration"]
            dirs = [self.work_dir / f"iter_{sel - 1:02d}"]
        else:
            # Fall back to last iteration
            dirs = sorted(self.work_dir.glob("iter_*/"))[-1:]
        return [d for d in dirs if d.is_dir()]

    def _qe_scratch_path(self) -> Path:
        """Resolve the QE scratch directory (defaults to work_dir/qe_scratch)."""
        return self.qe_scratch_dir if self.qe_scratch_dir else self.work_dir / "qe_scratch"

    def _auto_detect_calculator(self) -> str:
        """
        Inspect the run directory to determine calculator type.

        Checks for VASP POSCAR files first, then QE espresso.pwo files.
        Returns ``'vasp'`` or ``'qe'``.
        """
        vasp_snaps = list(self.work_dir.glob("iter_*/snap_*/POSCAR"))
        if vasp_snaps:
            return "vasp"
        qe_snaps = list(self._qe_scratch_path().glob("snap_*/espresso.pwo"))
        if qe_snaps:
            return "qe"
        raise RuntimeError(
            f"Cannot detect calculator type in {self.work_dir}: "
            "no iter_*/snap_*/POSCAR (VASP) or qe_scratch/snap_*/espresso.pwo (QE) found."
        )

    def build(self) -> "SQTCPostProcessor":
        """
        Auto-dispatch to the appropriate loader based on ``self.calculator``.

        Detects VASP vs QE automatically when ``calculator='auto'`` (default).
        """
        calc = self.calculator
        if calc == "auto":
            calc = self._auto_detect_calculator()
        if calc == "qe":
            return self._build_from_qe()
        return self._build_from_vasp()

    def _build_from_vasp(self) -> "SQTCPostProcessor":
        """
        Load POSCAR/OUTCAR snapshots, fit IFCs, build PhononCalculator, and
        return a ready-to-use :class:`SQTCPostProcessor`.
        """
        saved = self._load_results_json()
        if self.T_design is None and saved is not None:
            self.T_design = saved.get("T", 300.0)

        n_b = len(self.masses_amu)
        do_unsort = _needs_unsort(self.elements) if self.elements else False

        # Determine consistent supercell atom count from last snap's POSCAR.
        last_poscar = sorted(self.work_dir.glob("iter_*/snap_0000/POSCAR"))[-1]
        _, _, ref_pos = _parse_poscar(last_poscar)
        n_atoms = len(ref_pos)

        # Reconstruct equilibrium positions
        sc_cell_ref, _, _ = _parse_poscar(last_poscar)
        # Re-read to get cell without positions confusion
        sc_cell_ref = _parse_poscar(last_poscar)[0]

        # ── Derive actual sublattice basis positions from POSCAR ──────────────
        # For binary structures (rocksalt, zincblende), VASP re-sorts atoms
        # alphabetically by species, which may differ from the runner's original
        # element ordering.  The canonical prim_pos in _build_primitive_cell
        # assigns species to basis positions assuming a specific ordering; if the
        # runner used a different ordering the displacement u = snap - eq would
        # be wrong (by ~a/2) giving R²≈-1.
        #
        # Fix: for each species, average the wrapped fractional coordinates of
        # ALL atoms of that species (noise decreases as 1/√N_atoms, so much more
        # robust than using a single displaced atom).  Map the resulting mean
        # fractional to the canonical Cartesian basis position:
        #   (0,0,0)       → Cartesian (0,0,0)          [origin sublattice]
        #   (1/2,1/2,1/2) → Cartesian (a/2,0,0)        [rocksalt 2nd basis]
        #   (1/4,1/4,1/4) → Cartesian (a/4,a/4,a/4)    [zincblende 2nd basis]
        # The (a/2,0,0) choice is the runner's universal convention and must be
        # used instead of the wrapped (a/2,a/2,a/2) to keep atom ordering consistent
        # with _build_eq_positions.
        if n_b > 1 and self.elements and len(set(self.elements)) > 1:
            _, _vasp_sp, _vasp_pos = _parse_poscar(last_poscar)
            _species_positions: Dict[str, List[np.ndarray]] = {
                el: [] for el in set(self.elements)
            }
            for _sp, _pos in zip(_vasp_sp, _vasp_pos):
                if _sp in _species_positions:
                    _species_positions[_sp].append(_pos)
            _prim_inv = np.linalg.inv(self.prim_cell)
            _a_lat = float((4.0 * abs(np.linalg.det(self.prim_cell))) ** (1.0 / 3.0))
            _new_pp: List[np.ndarray] = []
            for _el in self.elements:
                _poss = _species_positions.get(_el, [])
                if not _poss:
                    break
                # Circular mean of wrapped fractionals (handles the 0/1 boundary):
                _fracs_w = np.array([(_pos @ _prim_inv) % 1.0 for _pos in _poss])
                # atoms near (0,0,0) have fracs in [0,ε] ∪ [1-ε,1] after wrapping
                # (positive and negative displacements), so arithmetic mean ≈ 0.5.
                # Circular mean: project onto unit circle, then take arc-tangent.
                _theta = 2.0 * np.pi * _fracs_w          # (N, 3) angles
                _cx = np.mean(np.cos(_theta), axis=0)    # (3,) cosine sums
                _cy = np.mean(np.sin(_theta), axis=0)    # (3,) sine sums
                _mf = np.round(
                    (np.arctan2(_cy, _cx) / (2.0 * np.pi)) % 1.0 * 4
                ) / 4
                _mf = _mf % 1.0   # 1.0 ≡ 0.0 (same crystal site)
                # Map to canonical Cartesian basis position
                if np.allclose(_mf, [0.0, 0.0, 0.0], atol=0.1):
                    _new_pp.append(np.array([0.0, 0.0, 0.0]))
                elif np.allclose(_mf, [0.5, 0.5, 0.5], atol=0.1):
                    # Rocksalt convention: runner always uses (a/2, 0, 0)
                    _new_pp.append(np.array([_a_lat / 2.0, 0.0, 0.0]))
                elif np.allclose(_mf, [0.25, 0.25, 0.25], atol=0.1):
                    # Zincblende convention: (a/4, a/4, a/4)
                    _new_pp.append(np.array([_a_lat / 4.0] * 3))
                else:
                    # Unknown pattern: fall back to rounded fractional converted back
                    _new_pp.append(_mf @ self.prim_cell)
            if len(_new_pp) == n_b:
                self.prim_pos = np.array(_new_pp, dtype=float)

        eq_positions, H = _build_eq_positions(sc_cell_ref, self.prim_cell, self.prim_pos)
        masses_sc = np.tile(self.masses_amu, n_atoms // n_b)

        if len(eq_positions) != n_atoms:
            raise RuntimeError(
                f"Reconstructed {len(eq_positions)} atoms, expected {n_atoms}.  "
                f"H={H.tolist()}"
            )

        # Collect snapshots
        displacements_list: List[np.ndarray] = []
        forces_list:        List[np.ndarray] = []
        n_loaded = 0

        for iter_dir in self._iter_dirs_to_load(saved):
            for snap in sorted(iter_dir.glob("snap_*/")):
                poscar = snap / "POSCAR"
                outcar = snap / "OUTCAR"
                if not poscar.exists() or not outcar.exists():
                    continue
                try:
                    snap_cell, snap_sp, snap_pos = _parse_poscar(poscar)
                except Exception as e:
                    print(f"  [SQTCRunLoader] Skip {snap.name} (POSCAR): {e}")
                    continue
                if len(snap_sp) != n_atoms:
                    continue   # different supercell size — skip
                try:
                    forces = _parse_outcar_forces(outcar, n_atoms)
                except Exception as e:
                    print(f"  [SQTCRunLoader] Skip {snap.name} (OUTCAR): {e}")
                    continue

                if do_unsort:
                    snap_pos, forces = _unsort_to_prim_order(
                        [snap_pos, forces], self.elements, n_atoms
                    )

                # Displacement with minimum-image convention
                u = snap_pos - eq_positions
                cell_inv = np.linalg.inv(sc_cell_ref)
                frac = u @ cell_inv
                frac -= np.round(frac)
                u = frac @ sc_cell_ref

                displacements_list.append(u)
                forces_list.append(forces)
                n_loaded += 1

        if n_loaded == 0:
            raise RuntimeError(
                f"No valid snapshots found in {self.work_dir}"
            )

        # Fit IFCs
        ifc = IFCExtractor(
            supercell_positions=eq_positions,
            supercell_cell=sc_cell_ref,
            masses_amu=masses_sc,
            r_cutoff=self.r_cutoff,
            symmetrise=True,
            ridge_alpha=self.ridge_alpha,
            symmetrize_bonds=self.symmetrize_bonds,
        )
        ifc.fit(displacements_list, forces_list)
        rep = ifc.fit_report(displacements_list, forces_list)

        print(f"  [{self.label}] Loaded {n_loaded} snaps  "
              f"RMSE={rep['rmse_ev_ang']:.5f} eV/Å  R²={rep['r2']:.5f}  "
              f"rank={rep['rank']}")

        phonon_calc = PhononCalculator(
            ifc_extractor=ifc,
            prim_positions=self.prim_pos,
            prim_cell=self.prim_cell,
            masses_amu=np.array(self.masses_amu),
        )

        return SQTCPostProcessor(
            phonon_calc=phonon_calc,
            label=self.label,
            elements=self.elements,
            T_design=self.T_design,
        )

    def _build_from_qe(self) -> "SQTCPostProcessor":
        """
        Load QE ``espresso.pwo`` snapshots, fit IFCs, and return a
        ready-to-use :class:`SQTCPostProcessor`.

        Snapshot layout (written by :class:`QEForceCalculator`)::

            <qe_scratch_dir>/
                snap_0000/espresso.pwo   ← equilibrium (F_eq ≈ 0)
                snap_0001/espresso.pwo   ← displaced config 1
                snap_0002/espresso.pwo   ← displaced config 2
                …

        The first snap whose forces satisfy ``|F|_max < 0.1 eV/Å`` is treated
        as the equilibrium; its forces are subtracted from all others so that
        the IFC regression receives ΔF = F(u) − F(eq).

        Because QEForceCalculator tiles species in the runner's natural order
        (not alphabetically sorted like VASP), no species-reordering is needed.
        """
        saved = self._load_results_json()
        if self.T_design is None and saved is not None:
            self.T_design = saved.get("T", 300.0)

        qe_scratch = self._qe_scratch_path()
        all_snap_dirs = sorted(qe_scratch.glob("snap_*/"))
        pwo_dirs = [d for d in all_snap_dirs if (d / "espresso.pwo").exists()]

        if not pwo_dirs:
            raise RuntimeError(
                f"No espresso.pwo files found under {qe_scratch}. "
                "Check that QEForceCalculator.workdir matches --qe-scratch-dir."
            )

        # Read all snaps and identify equilibrium (smallest |F|_max)
        snap_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # (pos, forces, cell)
        for snap_dir in pwo_dirs:
            try:
                pos, forces, cell = _parse_qe_pwo(snap_dir / "espresso.pwo")
                snap_data.append((pos, forces, cell))
            except Exception as e:
                print(f"  [SQTCRunLoader] Skip {snap_dir.name} (pwo): {e}")

        if not snap_data:
            raise RuntimeError(f"No readable espresso.pwo snapshots under {qe_scratch}")

        # The equilibrium snap is snap_0000 by construction (runner calls
        # _equilibrium_forces before _evaluate_forces).  If the first snap has
        # tiny forces, use it; otherwise fall back to searching.
        def _f_max(forces):
            return float(np.abs(forces).max())

        eq_idx = 0
        if _f_max(snap_data[0][1]) > 0.5:  # unexpectedly large → search
            eq_idx = int(np.argmin([_f_max(s[1]) for s in snap_data]))
            print(f"  [SQTCRunLoader] Equilibrium snap detected as snap_{eq_idx:04d} "
                  f"(|F|_max={_f_max(snap_data[eq_idx][1]):.4f} eV/Å)")

        eq_pos, F_eq, sc_cell_ref = snap_data[eq_idx]
        n_atoms = len(eq_pos)
        n_b = len(self.masses_amu)

        # Rebuild ideal equilibrium positions from primitive cell
        eq_positions, H = _build_eq_positions(sc_cell_ref, self.prim_cell, self.prim_pos)
        masses_sc = np.tile(self.masses_amu, n_atoms // n_b)

        if len(eq_positions) != n_atoms:
            raise RuntimeError(
                f"Reconstructed {len(eq_positions)} atoms, expected {n_atoms}.  "
                f"H={H.tolist()}"
            )

        # Collect displaced snaps
        displacements_list: List[np.ndarray] = []
        forces_list:        List[np.ndarray] = []
        n_loaded = 0

        for i, (snap_pos, snap_forces, snap_cell) in enumerate(snap_data):
            if i == eq_idx:
                continue  # skip equilibrium snap itself
            if len(snap_pos) != n_atoms:
                continue  # mismatched supercell size

            # Displacement with minimum-image convention
            u = snap_pos - eq_positions
            cell_inv = np.linalg.inv(sc_cell_ref)
            frac = u @ cell_inv
            frac -= np.round(frac)
            u = frac @ sc_cell_ref

            # Force difference ΔF = F(u) − F(eq)
            delta_F = snap_forces - F_eq

            displacements_list.append(u)
            forces_list.append(delta_F)
            n_loaded += 1

        if n_loaded == 0:
            raise RuntimeError(f"No valid displaced QE snapshots found in {qe_scratch}")

        # Fit IFCs
        ifc = IFCExtractor(
            supercell_positions=eq_positions,
            supercell_cell=sc_cell_ref,
            masses_amu=masses_sc,
            r_cutoff=self.r_cutoff,
            symmetrise=True,
            ridge_alpha=self.ridge_alpha,
            symmetrize_bonds=self.symmetrize_bonds,
        )
        ifc.fit(displacements_list, forces_list)
        rep = ifc.fit_report(displacements_list, forces_list)

        print(f"  [{self.label}] Loaded {n_loaded} QE snaps  "
              f"RMSE={rep['rmse_ev_ang']:.5f} eV/Å  R²={rep['r2']:.5f}  "
              f"rank={rep['rank']}")

        phonon_calc = PhononCalculator(
            ifc_extractor=ifc,
            prim_positions=self.prim_pos,
            prim_cell=self.prim_cell,
            masses_amu=np.array(self.masses_amu),
        )

        return SQTCPostProcessor(
            phonon_calc=phonon_calc,
            label=self.label,
            elements=self.elements,
            T_design=self.T_design,
        )


# =============================================================================
# CLI  —  python codes/sqtc/postprocessor.py --run-dir <dir> --structure fcc …
# =============================================================================

def _analytical_r_cutoff(structure: str, a: float) -> float:
    """
    Compute a sensible IFC cutoff radius from the structure type and lattice
    constant, using the midpoint between the 3rd and 4th NN shell distances.

    This corresponds to including 3 NN shells for most structures, which
    matches the typical choice in SQTC benchmark runs.

    FCC shells (units of a): 1/√2, 1, √(3/2), √2, √(5/2), √3, ...
    BCC shells (units of a): √3/2, 1, √2,      √3, √(11)/2,  2, ...
    """
    s = structure.lower()
    if s == "fcc":
        # 3rd: a*√(3/2)=1.225a,  4th: a*√2=1.414a  → midpoint: 1.320a
        r_cut = a * 0.5 * (np.sqrt(1.5) + np.sqrt(2.0))
    elif s == "bcc":
        # 3rd: a*√2=1.414a,  4th: a*√3=1.732a  → midpoint: 1.573a
        r_cut = a * 0.5 * (np.sqrt(2.0) + np.sqrt(3.0))
    elif s == "sc":
        # 3rd: a*√3,  4th: 2a  → midpoint: 1.866a
        r_cut = a * 0.5 * (np.sqrt(3.0) + 2.0)
    elif s in ("rocksalt", "nacl", "zincblende"):
        # FCC-based, but typically use tighter cutoff capturing ~2 unlike-atom shells
        # 2nd shell distance ≈ a/√2,  3rd ≈ a → midpoint: 0.5*(1/√2+1)*a ≈ 0.854a
        # For rocksalt a_conv ~ 4-6 Å, this gives ~3-5 Å — reasonable
        r_cut = a * 0.5 * (1.0 / np.sqrt(2.0) + 1.0)
    elif s == "hcp":
        # 1st: a,  2nd: c/2 or a depending on c/a,  3rd: a*√2 approx
        # Use conservative 1.2a
        r_cut = a * 1.2
    else:
        r_cut = a * 1.2   # generic fallback
    return round(r_cut, 3)


def _detect_structure_spglib(sc_cell: np.ndarray,
                             species: List[str]) -> Optional[str]:
    """
    Auto-detect crystal structure type from supercell geometry using spglib.

    Strategy
    --------
    Displaced SQTC snapshots always have SG #1, so we do NOT use atomic
    positions.  Instead we:

    1. Derive `a` from V_prim = V_sc / n_formula_units for each candidate
       Bravais lattice (BCC, FCC, HCP, SC).
    2. Build the ideal primitive cell and check that H = sc_cell @ inv(prim)
       is an integer matrix (integrality test).
    3. Run spglib on the *ideal* reconstructed primitive cell to confirm the
       space group.  Detected groups:

       SG 229  Im-3m   → bcc
       SG 225  Fm-3m   → fcc  (1 atom/prim)  or  rocksalt  (2 atoms/prim, 2 sp)
       SG 216  F-43m   → zincblende  (2 atoms/prim, 2 species)
       SG 227  Fd-3m   → diamond / zincblende  (2 atoms/prim, 1 species)
       SG 221  Pm-3m   → sc
       SG 194  P6_3/mmc → hcp

    Returns the structure-type string or None if auto-detection fails.
    """
    try:
        import spglib as _spg
    except ImportError:
        return None

    n_atoms = len(species)
    V_sc    = float(abs(np.linalg.det(sc_cell)))
    n_uniq  = len(set(species))

    def _h_ok(prim: np.ndarray, tol: float = 0.025) -> bool:
        """True if H = sc_cell @ inv(prim) is integer to within *tol*."""
        try:
            H = sc_cell @ np.linalg.inv(prim)
            return float(np.max(np.abs(H - np.round(H)))) < tol
        except np.linalg.LinAlgError:
            return False

    def _spg_no(lattice: np.ndarray, frac, nums,
                symprec: float = 1e-3) -> Optional[int]:
        """Return spglib space-group number, or None on failure."""
        try:
            ds = _spg.get_symmetry_dataset(
                (lattice,
                 np.array(frac, dtype=float),
                 np.array(nums, dtype=int)),
                symprec=symprec,
            )
            return int(ds.number) if ds is not None else None
        except Exception:
            return None

    # ── BCC (1 atom/prim, Im-3m, SG 229) ────────────────────────────────────
    if n_uniq == 1:
        a    = float((2.0 * V_sc / n_atoms) ** (1.0 / 3.0))
        prim = 0.5 * a * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=float)
        if _h_ok(prim):
            if _spg_no(prim, [[0, 0, 0]], [1]) == 229:
                return "bcc"

    # ── FCC pure metal (1 atom/prim, Fm-3m, SG 225) ─────────────────────────
    if n_uniq == 1:
        a    = float((4.0 * V_sc / n_atoms) ** (1.0 / 3.0))
        prim = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        if _h_ok(prim):
            if _spg_no(prim, [[0, 0, 0]], [1]) == 225:
                return "fcc"

    # ── FCC-based binary (2 atoms/prim): rocksalt or zincblende ─────────────
    if n_uniq == 2 and n_atoms % 2 == 0:
        n_fu = n_atoms // 2
        a    = float((4.0 * V_sc / n_fu) ** (1.0 / 3.0))
        prim = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        if _h_ok(prim):
            # rocksalt: 2nd basis at frac (1/2,1/2,1/2) → SG 225
            if _spg_no(prim, [[0, 0, 0], [0.5, 0.5, 0.5]], [1, 2]) == 225:
                return "rocksalt"
            # zincblende: 2nd basis at frac (1/4,1/4,1/4) → SG 216
            if _spg_no(prim, [[0, 0, 0], [0.25, 0.25, 0.25]], [1, 2]) == 216:
                return "zincblende"

    # ── Diamond cubic (1 species, 2 atoms/prim, Fd-3m, SG 227) ──────────────
    if n_uniq == 1 and n_atoms % 2 == 0:
        n_fu = n_atoms // 2
        a    = float((4.0 * V_sc / n_fu) ** (1.0 / 3.0))
        prim = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        if _h_ok(prim):
            if _spg_no(prim, [[0, 0, 0], [0.25, 0.25, 0.25]], [1, 1]) == 227:
                return "zincblende"   # diamond is a special case of zincblende

    # ── Simple cubic (1 atom/prim, Pm-3m, SG 221) ───────────────────────────
    if n_uniq == 1:
        a    = float((V_sc / n_atoms) ** (1.0 / 3.0))
        prim = a * np.eye(3)
        if _h_ok(prim):
            if _spg_no(prim, [[0, 0, 0]], [1]) == 221:
                return "sc"

    # ── HCP (2 atoms/prim, P6_3/mmc, SG 194) ────────────────────────────────
    if n_uniq == 1 and n_atoms % 2 == 0:
        # Try ideal c/a = sqrt(8/3) ≈ 1.633; V_prim = sqrt(3)/2 * a^2 * c
        ideal_ca = float(np.sqrt(8.0 / 3.0))
        V_p = V_sc / (n_atoms // 2)
        a   = float((2.0 * V_p / (np.sqrt(3.0) * ideal_ca)) ** (1.0 / 3.0))
        c   = ideal_ca * a
        prim = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
        if _h_ok(prim, tol=0.1):   # looser tol: real c/a may differ from ideal
            # HCP basis in fractional: (0,0,0) and (1/3, 2/3, 1/2)
            if _spg_no(prim, [[0, 0, 0], [1.0/3, 2.0/3, 0.5]], [1, 1]) == 194:
                return "hcp"

    return None


def _auto_detect_run_params(run_dir: Path, structure: str,
                            qe_scratch_dir: Optional[Path] = None):
    """
    Derive equilibrium lattice parameter, element symbols, standard atomic
    masses, and T_design from snapshots + sqtc_results.json in *run_dir*.

    Supports both VASP (``POSCAR``) and QE (``espresso.pwo``) snapshots.

    Parameters
    ----------
    run_dir        : SQTC run directory.
    structure      : crystal structure string.
    qe_scratch_dir : path to QE scratch dir; used when no POSCARs are found.

    Returns
    -------
    lattice  : list of float [Ang] -- [a] or [a, c] for hcp
    elements : list of str         -- unique species in snapshot order
    masses   : list of float [amu]
    T_design : float | None
    """
    # ── Try VASP POSCAR first ─────────────────────────────────────────────────
    poscars = sorted(run_dir.glob("iter_*/snap_*/POSCAR"))
    if poscars:
        cell, species, _ = _parse_poscar(poscars[0])
        # Unique elements in POSCAR order
        seen: List[str] = []
        for s in species:
            if s not in seen:
                seen.append(s)
        elements = seen
        n_total = len(species)
    else:
        # ── Fall back to QE espresso.pwo ──────────────────────────────────────
        scratch = qe_scratch_dir or (run_dir / "qe_scratch")
        pwos = sorted(scratch.glob("snap_*/espresso.pwo"))
        if not pwos:
            raise FileNotFoundError(
                f"No POSCAR files found under {run_dir}/iter_*/snap_*/ and "
                f"no espresso.pwo files found under {scratch}/snap_*/"
            )
        pos, _, cell = _parse_qe_pwo(pwos[0])
        # Element symbols from espresso.pwo via ASE
        try:
            from ase.io import read as _ase_read
            _atoms = _ase_read(str(pwos[0]), format="espresso-out", index=-1)
            _syms = _atoms.get_chemical_symbols()
        except Exception:
            _syms = ["X"] * len(pos)
        seen_qe: List[str] = []
        for s in _syms:
            if s not in seen_qe:
                seen_qe.append(s)
        elements = seen_qe
        n_total = len(_syms)

    # Primitive cell volume
    V_sc = abs(float(np.linalg.det(cell)))
    st = structure.lower()
    n_basis = 2 if st in ("hcp", "rocksalt", "nacl", "zincblende") else 1
    V_prim = V_sc / (n_total / n_basis)   # volume of one primitive cell

    # Lattice constant from structure
    if st == "fcc":
        a = (4.0 * V_prim) ** (1.0 / 3.0)
        lattice = [round(a, 5)]
    elif st == "bcc":
        a = (2.0 * V_prim) ** (1.0 / 3.0)
        lattice = [round(a, 5)]
    elif st == "sc":
        a = V_prim ** (1.0 / 3.0)
        lattice = [round(a, 5)]
    elif st in ("rocksalt", "nacl", "zincblende"):
        a = (4.0 * V_prim) ** (1.0 / 3.0)   # FCC primitive: V = a^3/4
        lattice = [round(a, 5)]
    elif st == "hcp":
        # Assume ideal c/a = sqrt(8/3); refine from V_prim = (sqrt(3)/2) a^2 c
        c_over_a = float(np.sqrt(8.0 / 3.0))
        a = (V_prim / (np.sqrt(3.0) / 2.0 * c_over_a)) ** (1.0 / 3.0)
        lattice = [round(a, 5), round(a * c_over_a, 5)]
    else:
        raise ValueError(f"Cannot auto-detect lattice for structure '{structure}'")

    # Masses from standard table
    missing = [el for el in elements if el not in _STANDARD_MASSES]
    if missing:
        raise ValueError(
            f"Element(s) {missing} not in mass table -- use --mass to supply them."
        )
    masses = [_STANDARD_MASSES[el] for el in elements]

    # T_design, r_cutoff, ridge_alpha, symmetrize_bonds from sqtc_results.json
    T_design:        Optional[float] = None
    r_cutoff:        Optional[float] = None
    ridge_alpha:     Optional[float] = None
    symmetrize_bonds: Optional[bool] = None
    rjson = run_dir / "sqtc_results.json"
    if rjson.exists():
        with open(rjson) as fh:
            data = json.load(fh)
        T_design         = data.get("T")
        r_cutoff         = data.get("r_cutoff")
        ridge_alpha      = data.get("ridge_alpha")
        symmetrize_bonds = data.get("symmetrize_bonds")   # None for old runs

    # If r_cutoff not stored (old run), infer from NN shell distances in POSCAR
    if r_cutoff is None:
        r_cutoff = _analytical_r_cutoff(structure, lattice[0])

    # If symmetrize_bonds not stored (old run), derive a structure-based default.
    # FCC metals and light rocksalts (NaCl, MgO, LiF) used True in all original
    # runner scripts; BCC, zincblende, and heavy rocksalts (PbTe) used False.
    if symmetrize_bonds is None:
        st2 = structure.lower()
        if st2 == "fcc":
            symmetrize_bonds = True
        elif st2 in ("rocksalt", "nacl"):
            _heavy = {"Pb", "Te", "Bi", "Tl", "In", "Sn", "Ge"}
            symmetrize_bonds = not any(el in _heavy for el in elements)
        else:
            symmetrize_bonds = False

    return lattice, elements, masses, T_design, r_cutoff, ridge_alpha, symmetrize_bonds


def _build_primitive_cell(structure: str, lattice) -> tuple:
    """Build (prim_cell [Å], prim_pos [Å]) from structure name + lattice params."""
    a = float(lattice[0])
    s = structure.lower()
    if s == "fcc":
        prim_cell = 0.5*a*np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
        prim_pos  = np.array([[0.0, 0.0, 0.0]])
    elif s == "bcc":
        prim_cell = 0.5*a*np.array([[-1,1,1],[1,-1,1],[1,1,-1]], dtype=float)
        prim_pos  = np.array([[0.0, 0.0, 0.0]])
    elif s == "sc":
        prim_cell = a*np.eye(3)
        prim_pos  = np.array([[0.0, 0.0, 0.0]])
    elif s == "hcp":
        c = float(lattice[1]) if len(lattice) >= 2 else a*np.sqrt(8.0/3.0)
        prim_cell = np.array([[a,0,0],[-0.5*a,0.5*np.sqrt(3)*a,0],[0,0,c]])
        prim_pos  = np.array([[0.0,0.0,0.0],[0.0, a/np.sqrt(3), 0.5*c]])
    elif s in ("rocksalt", "nacl"):
        prim_cell = 0.5*a*np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
        prim_pos  = np.array([[0.0,0.0,0.0], prim_cell.T @ np.array([0.5,0.5,0.5])])
    elif s == "zincblende":
        prim_cell = 0.5*a*np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
        prim_pos  = np.array([[0.0,0.0,0.0], prim_cell.T @ np.array([0.25,0.25,0.25])])
    else:
        raise ValueError(
            f"Unknown structure '{structure}'. "
            "Supported: fcc, bcc, sc, hcp, rocksalt, zincblende"
        )
    return prim_cell, prim_pos


def _cli_main():

        import argparse
        import matplotlib
        matplotlib.use("Agg")

        parser = argparse.ArgumentParser(
            description="Post-process a completed SQTC run for any material.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
    Examples
    --------
      python codes/sqtc/postprocessor.py --run-dir sqtc_au_vasp_run \\
          --structure fcc --lattice 4.08 --elements Au --mass 196.967 --T-design 300

      python codes/sqtc/postprocessor.py --run-dir sqtc_cu_vasp_run \\
          --structure fcc --lattice 3.67 --elements Cu --mass 63.546 --T-design 1200

      python codes/sqtc/postprocessor.py --run-dir sqtc_nacl_vasp_run \\
          --structure rocksalt --lattice 5.64 --elements Na Cl --mass 22.99 35.45

      python codes/sqtc/postprocessor.py --run-dir sqtc_bccmo_vasp_run \\
          --structure bcc --lattice 3.147 --elements Mo --mass 95.96 --T-design 300

      python codes/sqtc/postprocessor.py --run-dir sqtc_mgo_vasp_run \\
          --structure rocksalt --lattice 4.21 --elements Mg O --mass 24.305 15.999 \\
          --r-cutoff 5.5

      python codes/sqtc/postprocessor.py --run-dir sqtc_si_vasp_run \\
          --structure zincblende --lattice 5.43 --elements Si Si --mass 28.085 28.085

      # Quantum ESPRESSO run (auto-detects QE from qe_scratch/ sub-directory):
      python codes/sqtc/postprocessor.py --run-dir sqtc_al_qe_run \\
          --calculator qe --structure fcc

      python codes/sqtc/postprocessor.py --run-dir sqtc_al_qe_run \\
          --calculator qe --qe-scratch-dir sqtc_al_qe_run/qe_scratch \\
          --structure fcc --lattice 4.05 --elements Al --mass 26.982 --T-design 300
    """,
        )
        # Required
        parser.add_argument("--run-dir",     required=True,
                            help="Path to the SQTC run directory")
        parser.add_argument("--structure",   default=None,
                            choices=["fcc","bcc","sc","hcp","rocksalt","zincblende"],
                            help="Crystal structure type [auto-detected via spglib if omitted]")
        # Auto-detected from POSCAR/pwo when omitted
        parser.add_argument("--lattice",     nargs="+", type=float, default=None,
                            help="Lattice constant(s) in Angstrom [auto-detected from snapshot]")
        parser.add_argument("--elements",    nargs="+", default=None,
                            help="Element symbol(s) per basis atom [auto-detected from snapshot]")
        parser.add_argument("--mass",        nargs="+", type=float, default=None,
                            help="Atomic mass(es) in amu [auto-detected via standard masses]")
        # Optional
        parser.add_argument("--T-design",    type=float, default=None,
                            help="SQTC design temperature [K] (auto-read from sqtc_results.json if omitted)")
        parser.add_argument("--label",       default=None,
                            help="Human-readable label (default: run directory name)")
        parser.add_argument("--r-cutoff",    type=float, default=6.0,
                            help="IFC cutoff radius [Angstrom] (default: 6.0)")
        parser.add_argument("--ridge-alpha", type=float, default=1e-3,
                            help="Tikhonov regularisation (default: 1e-3)")
        parser.add_argument("--symmetrize-bonds", action="store_true", default=None,
                            help="Apply central-force bond symmetrization "
                                 "[auto-read from sqtc_results.json if omitted]")
        parser.add_argument("--T-min",       type=float, default=10.0,
                            help="Start of temperature scan [K] (default: 10)")
        parser.add_argument("--T-max",       type=float, default=None,
                            help="End of temperature scan [K] (default: max(T_design*1.05, 1000))")
        parser.add_argument("--T-step",      type=float, default=10.0,
                            help="Temperature step [K] (default: 10)")
        parser.add_argument("--q-mesh-cv",   type=int, nargs=3, default=[20,20,20],
                            metavar=("N1","N2","N3"),
                            help="q-mesh for C_V / thermal properties (default: 20 20 20)")
        parser.add_argument("--q-mesh-dos",  type=int, nargs=3, default=[30,30,30],
                            metavar=("N1","N2","N3"),
                            help="q-mesh for DOS + band structure (default: 30 30 30)")
        parser.add_argument("--out-dir",     default=None,
                            help="Output directory (default: <run-dir>/postproc)")
        parser.add_argument("--calculator",  default="auto",
                            choices=["auto", "vasp", "qe"],
                            help="Force calculator type: 'auto' (default), 'vasp', or 'qe'")
        parser.add_argument("--qe-scratch-dir", default=None,
                            help="Path to QEForceCalculator workdir containing snap_NNNN/ "
                                 "sub-directories (default: <run-dir>/qe_scratch)")
        args = parser.parse_args()

        run_dir = Path(args.run_dir)
        if not run_dir.is_dir():
            parser.error(f"Run directory not found: {run_dir}")

        qe_scratch_dir = Path(args.qe_scratch_dir) if args.qe_scratch_dir else None

        # Resolve calculator type early so auto-detect can choose the right files
        calculator = args.calculator
        if calculator == "auto":
            _has_vasp = bool(list(run_dir.glob("iter_*/snap_*/POSCAR"))[:1])
            _qe_root = qe_scratch_dir or (run_dir / "qe_scratch")
            _has_qe  = bool(list(_qe_root.glob("snap_*/espresso.pwo"))[:1])
            if _has_vasp:
                calculator = "vasp"
            elif _has_qe:
                calculator = "qe"
            # else: leave as 'auto' — SQTCRunLoader will raise a clear error

        # ── Auto-detect structure via spglib if --structure not supplied ──────
        structure = args.structure
        if structure is None:
            _sc_cell = _sc_species = None
            if calculator in ("vasp", "auto"):
                _poscar_paths = sorted(run_dir.glob("iter_*/snap_*/POSCAR"))
                if _poscar_paths:
                    _sc_cell, _sc_species, _ = _parse_poscar(_poscar_paths[0])
            if _sc_cell is None and calculator in ("qe", "auto"):
                _qe_root = qe_scratch_dir or (run_dir / "qe_scratch")
                _pwos = sorted(_qe_root.glob("snap_*/espresso.pwo"))
                if _pwos:
                    try:
                        from ase.io import read as _ase_read
                        _atoms = _ase_read(str(_pwos[0]), format="espresso-out", index=-1)
                        _sc_cell = np.array(_atoms.cell)
                        _sc_species = _atoms.get_chemical_symbols()
                    except Exception:
                        pass
            if _sc_cell is None:
                parser.error(
                    "No snapshots found; cannot auto-detect structure. "
                    "Please supply --structure {fcc,bcc,sc,hcp,rocksalt,zincblende}."
                )
            structure = _detect_structure_spglib(_sc_cell, _sc_species)
            if structure is None:
                parser.error(
                    "spglib could not identify the crystal structure automatically. "
                    "Please supply --structure {fcc,bcc,sc,hcp,rocksalt,zincblende}."
                )
            print(f"  [spglib] Detected structure: {structure} "
                  f"(SG lookup on ideal primitive cell)")

        # ── Auto-detect missing lattice / elements / mass / T_design ─────────
        try:
            auto_lattice, auto_elements, auto_masses, auto_T, auto_rcutoff, auto_ridge, auto_sym_bonds = \
                _auto_detect_run_params(run_dir, structure, qe_scratch_dir=qe_scratch_dir)
        except Exception as exc:
            parser.error(f"Auto-detection failed: {exc}")

        lattice     = args.lattice     if args.lattice     is not None else auto_lattice
        elements    = args.elements    if args.elements    is not None else auto_elements
        masses      = args.mass        if args.mass        is not None else auto_masses
        T_design    = args.T_design    if args.T_design    is not None else auto_T
        r_cutoff    = args.r_cutoff    if args.r_cutoff    != 6.0      else (auto_rcutoff or args.r_cutoff)
        ridge_alpha = args.ridge_alpha if args.ridge_alpha != 1e-3     else (auto_ridge   or args.ridge_alpha)
        # symmetrize_bonds: explicit CLI flag > stored/derived value from _auto_detect_run_params
        symmetrize_bonds = bool(auto_sym_bonds) if args.symmetrize_bonds is None else args.symmetrize_bonds

        if len(elements) != len(masses):
            parser.error(
                f"--elements and --mass must have same length "
                f"({len(elements)} vs {len(masses)})"
            )

        out_dir = Path(args.out_dir) if args.out_dir else run_dir / "postproc"
        out_dir.mkdir(parents=True, exist_ok=True)
        label = args.label or run_dir.stem.replace("_", " ")

        # Build primitive cell
        prim_cell, prim_pos = _build_primitive_cell(structure, lattice)

        print(f"\n{'='*68}")
        print(f"  SQTC PostProcessor -- {label}")
        print(f"{'='*68}")
        print(f"  run_dir    : {run_dir}")
        print(f"  calculator : {calculator}")
        if calculator == "qe":
            print(f"  qe_scratch : {qe_scratch_dir or run_dir / 'qe_scratch'}")
        print(f"  structure  : {structure}  a={lattice}")
        print(f"  elements   : {elements}   masses={masses} amu")
        print(f"  T_design   : {T_design} K")
        print(f"  r_cutoff   : {r_cutoff} A   ridge_alpha={ridge_alpha}   symmetrize_bonds={symmetrize_bonds}")
        print(f"  out_dir    : {out_dir}")

        # Load + fit
        loader = SQTCRunLoader(
            work_dir=run_dir, prim_cell=prim_cell, prim_pos=prim_pos,
            masses_amu=masses, elements=elements,
            r_cutoff=r_cutoff, ridge_alpha=ridge_alpha,
            symmetrize_bonds=symmetrize_bonds, label=label,
            T_design=T_design, use_all_iters=True,
            calculator=calculator, qe_scratch_dir=qe_scratch_dir,
        )
        pp = loader.build()

        T_design_eff = pp.T_design or 300.0
        T_max = args.T_max or max(T_design_eff * 1.05, 1000.0)
        T_values = np.arange(args.T_min, T_max + args.T_step, args.T_step)
        q_cv  = tuple(args.q_mesh_cv)
        q_dos = tuple(args.q_mesh_dos)

        # Compute
        print(f"\n  Computing phonon band structure + DOS  (q={q_dos}) ...")
        phonon_data = pp.compute_phonons(q_mesh=q_dos, n_bins=500, sigma_thz=0.06)

        print(f"  Computing thermal properties  "
              f"(q={q_cv}, T={T_values[0]:.0f}-{T_values[-1]:.0f} K) ...")
        thermal_data = pp.compute_thermal_properties(T_values, q_mesh=q_cv)

        pp.print_summary(thermal_data=thermal_data, phonon_data=phonon_data)
        pp.save(out_dir=out_dir)
        print(f"\n  Saved data -> {out_dir}")

        # Plots
        pp.plot_phonons(
            phonon_data=phonon_data,
            save_path=out_dir / f"{run_dir.stem}_phonons.pdf",
            title=f"{label} -- phonons  (T_design={T_design_eff:.0f} K)",
        )
        pp.plot_thermal_properties(
            thermal_data=thermal_data,
            save_path=out_dir / f"{run_dir.stem}_thermal.pdf",
            title=f"{label} -- thermal properties  (T_design={T_design_eff:.0f} K)",
        )
        print(f"  Saved -> {out_dir / (run_dir.stem + '_phonons.pdf')}")
        print(f"  Saved -> {out_dir / (run_dir.stem + '_thermal.pdf')}")

        # Results table
        T_arr = thermal_data["T_K"]
        Cv  = thermal_data["Cv_jmolk"]
        Sv  = thermal_data["Svib_jmolk"]
        TD  = thermal_data["TD_caloric"]
        MSD = thermal_data["MSD_ang2"]
        ZPE = float(thermal_data["ZPE_eV"])
        TDs = float(thermal_data["TD_spectral"])

        print(f"\n  w_max={phonon_data['stats']['max_freq_thz']:.3f} THz | "
              f"ZPE={ZPE*1000:.2f} meV | T_D(spectral)={TDs:.1f} K")
        print(f"\n  {'T [K]':>7}  {'Cv [J/mol/K]':>13}  {'Svib [J/mol/K]':>15}  "
              f"{'TD [K]':>7}  {'<u2> [A2]':>10}")
        print("  " + "-"*60)
        key_Ts = sorted({10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                         int(round(T_design_eff / 10) * 10)})
        for T_c in key_Ts:
            i = int(np.argmin(np.abs(T_arr - T_c)))
            if abs(T_arr[i] - T_c) > 6 or T_arr[i] > T_max + 5:
                continue
            print(f"  {T_arr[i]:7.0f}  {Cv[i]:13.4f}  {Sv[i]:15.4f}  "
                  f"{TD[i]:7.1f}  {MSD[i]:10.5f}")

        print(f"\n{'='*68}")
        print(f"  Done. Output: {out_dir}")
        print(f"{'='*68}")
        for fp in sorted(out_dir.iterdir()):
            print(f"  {fp.name}")


if __name__ == "__main__":
    import sys
    sys.exit(_cli_main())
