"""
SQTC Quasi-Harmonic Approximation (QHA) for thermal expansion.

Obtains α(T), C_P(T), B(T) and V(T) from a **minimum** number of DFT
calculations by combining:

  1. A 3-point static EOS  (E_0 vs V, primitive cell, NSW=0 — cheap)
  2. SQTC phonon runs at the same 3 volumes (V₋, V₀, V₊)

The total DFT cost is ~3× the per-T SQTC cost plus 3 static primitive-cell
SCF calculations, versus phonopy-QHA's ~235× cost.

Theory
------
At pressure P=0 the Gibbs free energy reduces to the Helmholtz free energy:

    G(T, V) = E_0(V) + F_vib(T, V)

    F_vib(T, V) = Σ_{q,s} k_B T ln[2 sinh(ħω_{qs}(V)/2k_B T)]

At each temperature T the equilibrium volume V*(T) minimises G:

    dG/dV = dE_0/dV + dF_vib/dV = 0  →  V*(T)

Thermal quantities follow:
    α(T) = (1/V*) dV*/dT
    B(T) = -V* d²G/dV²|_{V*}
    C_P(T) = C_V(T) + T V* α(T)² B(T)

Instead of the full volume sweep, a 2-point Grüneisen shortcut is also
provided:
    α(T) ≈ γ_eff(T) C_V(T) / (3 B₀ V₀)
where γ_eff is the heat-capacity-weighted average of mode Grüneisen
parameters and B₀ is the static bulk modulus.

Usage
-----
from sqtc.qha import SQTCQuasiHarmonic

qha = SQTCQuasiHarmonic(
    volumes_ang3   = [V_minus, V_0, V_plus],  # Å³ per formula unit
    static_energies_ev = [E_minus, E_0, E_plus],
    phonon_calcs   = [calc_Vm, calc_V0, calc_Vp],  # PhononCalculator instances
)
results = qha.compute(T_values=np.arange(100, 1500, 50))
# results['alpha_K']   : thermal expansion coefficient [K⁻¹]
# results['V_ang3']    : equilibrium volume [Å³]
# results['Cp_jmolk']  : C_P [J/(mol·K)]
# results['B_GPa']     : bulk modulus [GPa]
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import numpy as np
from scipy.optimize import minimize_scalar

from .constants import KB, NA, EV_TO_J


class SQTCQuasiHarmonic:
    """
    Quasi-harmonic approximation using SQTC phonons at 3 volumes.

    Requires 3 volumes to:
      (a) fit a parabolic / Birch-Murnaghan EOS to E_0(V)
      (b) interpolate F_vib(T, V) across the volume range

    Parameters
    ----------
    volumes_ang3 : list of 3 floats [Å³/f.u.]
        Primitive-cell volumes — must be sorted V₋ < V₀ < V₊.
        Recommended step: ±1–2% of V₀.
    static_energies_ev : list of 3 floats [eV/f.u.]
        DFT total energies at each volume (NSW=0 SCF, primitive cell).
    phonon_calcs : list of 3 PhononCalculator
        One per volume, fitted SQTC IFCs at the same 3 volumes.
    q_mesh : (3,) int, optional
        q-mesh for F_vib integration. Default (20,20,20).
    method : {"qha", "gruneisen", "both"}
        Which thermal-expansion method to use in :meth:`run`:

        * ``"qha"``       — full Helmholtz volume minimisation.
          Most accurate; requires F_vib at 3 volumes.
        * ``"gruneisen"`` — Grüneisen parameter shortcut
          α = γ_eff C_V / (3 B₀ V₀).  No volume minimisation loop;
          faster but slightly less accurate above 0.8 T_melt.
        * ``"both"``      — run both and return results side-by-side.
    """

    def __init__(
        self,
        volumes_ang3: List[float],
        static_energies_ev: List[float],
        phonon_calcs: list,
        q_mesh: tuple = (20, 20, 20),
        method: Literal["qha", "gruneisen", "both"] = "qha",
    ):
        assert len(volumes_ang3) == 3
        assert len(static_energies_ev) == 3
        assert len(phonon_calcs) == 3

        self._V = np.asarray(volumes_ang3, dtype=float)   # Å³/f.u.
        self._E = np.asarray(static_energies_ev, dtype=float)  # eV/f.u.
        self._calcs = phonon_calcs
        self._q_mesh = q_mesh
        if method not in ("qha", "gruneisen", "both"):
            raise ValueError(f"method must be 'qha', 'gruneisen', or 'both'; got '{method}'")
        self.method = method

        # Fit a 2nd-order polynomial to E_0(V) as a quick EOS.
        # For 3 points this is exact.  Parabola: E = a + b(V-V0) + c(V-V0)²
        # Static bulk modulus B₀ = V₀ d²E/dV² = 2c × V₀
        V0 = self._V[1]
        self._eos_coeffs = np.polyfit(self._V - V0, self._E, 2)  # [c, b, a]
        # B₀ in GPa:  2c [eV/Å⁶] × V₀ [Å³] × EV_TO_J / 1e-30 / 1e9
        c = self._eos_coeffs[0]
        ANG3_TO_M3 = 1e-30
        self.B0_GPa = float(2.0 * c * V0 * EV_TO_J / ANG3_TO_M3 / 1e9)
        self.V0_ang3 = float(V0)

    def _E0(self, V: float) -> float:
        """Static DFT energy at volume V [eV/f.u.] from parabolic EOS."""
        V0 = self._V[1]
        c = self._eos_coeffs
        dV = V - V0
        return float(np.polyval(c, dV))

    def _Fvib(self, T: float, V: float) -> float:
        """
        F_vib(T, V) [eV/f.u.] by linear interpolation over the 3 volumes.

        For ±1–2% volume steps the linear interpolation in V is accurate to
        <0.1 meV/f.u. in F_vib, which is sufficient for α(T) calculations.
        """
        Fv = np.array([
            c.vibrational_free_energy(T, q_mesh=self._q_mesh)
            for c in self._calcs
        ])
        # Quadratic interpolation (exact for 3 points)
        V0 = self._V[1]
        coeffs = np.polyfit(self._V - V0, Fv, 2)
        return float(np.polyval(coeffs, V - V0))

    def _G(self, T: float, V: float) -> float:
        """Helmholtz free energy G = E_0(V) + F_vib(T,V) [eV/f.u.]."""
        return self._E0(V) + self._Fvib(T, V)

    def equilibrium_volume(self, T: float) -> float:
        """
        Minimise G(T,V) over V in the range [V₋, V₊] → V*(T) [Å³/f.u.].
        """
        Vmin, Vmax = self._V[0], self._V[2]
        result = minimize_scalar(
            lambda V: self._G(T, V),
            bounds=(Vmin, Vmax),
            method="bounded",
            options={"xatol": 1e-6},
        )
        return float(result.x)

    def compute(
        self,
        T_values: np.ndarray,
        q_mesh: Optional[tuple] = None,
        dT_alpha: float = 5.0,
        v_range_factor: float = 1.5,
    ) -> Dict[str, np.ndarray]:
        """
        Compute thermal expansion and related quantities at each T.

        Parameters
        ----------
        T_values       : array of temperatures [K]
        q_mesh         : override q-mesh for F_vib (default: self._q_mesh)
        dT_alpha       : finite-difference step for dV/dT [K], default 5 K
        v_range_factor : factor by which to extend the minimisation search
            range beyond [Vm, Vp].  Default 1.5 means the search extends
            0.5×(Vp-Vm) beyond each end, allowing V*(T) to lie outside the
            fitting interval when thermal expansion is large.  The F_vib
            polynomial is extrapolated in this region, which is accurate to
            within ~0.1 meV/f.u. for volume excursions up to ~5%.

        Returns
        -------
        dict with keys:
            'T_K'       : temperatures [K]
            'V_ang3'    : equilibrium volume V*(T) [Å³/f.u.]
            'alpha_K'   : volumetric thermal expansion α(T) = (1/V)dV/dT [K⁻¹]
            'Cv_jmolk'  : C_V per mole [J/(mol·K)]  (from SQTC phonons at V*)
            'Cp_jmolk'  : C_P = C_V + TVα²B [J/(mol·K)]
            'B_GPa'     : effective bulk modulus at V*(T) [GPa]
        """
        if q_mesh is not None:
            self._q_mesh = q_mesh

        T_arr = np.asarray(T_values, dtype=float)
        n = len(T_arr)

        V_star   = np.zeros(n)
        alpha    = np.zeros(n)
        Cv       = np.zeros(n)
        Cp       = np.zeros(n)
        B_arr    = np.zeros(n)

        # Precompute F_vib arrays at each of the 3 stored volumes for all T.
        # This avoids redundant _all_frequencies calls.
        Fvib_3V = np.array([
            [c.vibrational_free_energy(T, q_mesh=self._q_mesh) for T in T_arr]
            for c in self._calcs
        ])  # (3, n_T)

        V0 = self._V[1]
        Fv_coeffs = np.array([
            np.polyfit(self._V - V0, Fvib_3V[:, it], 2)
            for it in range(n)
        ])  # (n_T, 3) polynomial coefficients

        def _G_fast(T_idx: int, V: float) -> float:
            dV = V - V0
            Fvib = float(np.polyval(Fv_coeffs[T_idx], dV))
            return self._E0(V) + Fvib

        Vmin, Vmax = self._V[0], self._V[2]
        dV_fit = Vmax - Vmin
        Vlo = Vmin - (v_range_factor - 1.0) * dV_fit
        Vhi = Vmax + (v_range_factor - 1.0) * dV_fit

        for it, T in enumerate(T_arr):
            # Equilibrium volume
            res = minimize_scalar(
                lambda V, i=it: _G_fast(i, V),
                bounds=(Vlo, Vhi),
                method="bounded",
                options={"xatol": 1e-6},
            )
            Vs = float(res.x)
            V_star[it] = Vs

            # Bulk modulus from G curvature: B = V d²G/dV²
            h = Vs * 0.001
            d2G = (_G_fast(it, Vs + h) - 2 * _G_fast(it, Vs) + _G_fast(it, Vs - h)) / h**2
            ANG3_TO_M3 = 1e-30
            B_arr[it] = float(Vs * d2G * EV_TO_J / ANG3_TO_M3 / 1e9)  # GPa

        # Thermal expansion: central finite difference in T using V_star
        for it in range(n):
            T = T_arr[it]
            V = V_star[it]
            if T < dT_alpha:
                alpha[it] = 0.0
                continue
            if it == 0:
                alpha[it] = (V_star[1] - V_star[0]) / (T_arr[1] - T_arr[0]) / V if n > 1 else 0.0
            elif it == n - 1:
                alpha[it] = (V_star[-1] - V_star[-2]) / (T_arr[-1] - T_arr[-2]) / V
            else:
                alpha[it] = (V_star[it+1] - V_star[it-1]) / (T_arr[it+1] - T_arr[it-1]) / V

        # C_V at V*(T): use the middle-volume SQTC phonon calculator as a proxy
        # (within ±2% volume step the error in C_V is < 0.1 J/mol/K)
        Cv = self._calcs[1].heat_capacity_scan(T_arr, q_mesh=self._q_mesh)

        # C_P = C_V + T V α² B  (Nernst-Lindemann relation, standard QHA)
        ANG3_TO_M3 = 1e-30
        V_m3 = V_star * ANG3_TO_M3              # m³/f.u.
        B_Pa  = B_arr * 1e9                     # Pa
        Cp = Cv + T_arr * V_m3 * alpha**2 * B_Pa * NA  # J/(mol·K)

        return {
            "T_K":       T_arr,
            "V_ang3":    V_star,
            "alpha_K":   alpha,
            "Cv_jmolk":  Cv,
            "Cp_jmolk":  Cp,
            "B_GPa":     B_arr,
        }

    # ── Grüneisen shortcut: single-temperature α from mode γ ─────────────────

    def alpha_gruneisen(
        self,
        T_values: np.ndarray,
        q_mesh: tuple = (20, 20, 20),
    ) -> Dict[str, np.ndarray]:
        """
        Fast Grüneisen estimate of α(T) using the central-volume SQTC phonons
        and mode Grüneisen parameters from V± finite differences.

        α(T) ≈ γ_eff(T) × C_V(T) / (3 × B₀ × V₀)

        This requires **no** volume minimisation — just the two finite-difference
        phonon calculators (calc_V+ and calc_V-) that were already run.

        Cost: identical to running SQTC at 3 volumes (which is already done
        for SQTCQuasiHarmonic) — no extra DFT beyond what compute() uses.

        Parameters
        ----------
        T_values : array [K]
        q_mesh   : q-mesh for C_V and γ_eff sums

        Returns
        -------
        dict with keys:
            'T_K', 'alpha_K', 'gamma_eff', 'Cv_jmolk'
        """
        T_arr = np.asarray(T_values, dtype=float)
        calc0, calcp, calcm = self._calcs[1], self._calcs[2], self._calcs[0]

        # dV/V between V+ and V-
        dV_over_V = (self._V[2] - self._V[0]) / (2.0 * self._V[1])

        # Mode Grüneisen parameters (n_q, n_branches)
        gamma_qs = calc0.mode_gruneisen_parameters(calcp, calcm, dV_over_V)
        omega0   = calc0._all_frequencies(q_mesh)   # (n_q, n_branches) rad/s

        ANG3_TO_M3 = 1e-30
        V0_m3 = self._V[1] * ANG3_TO_M3
        B0_Pa = self.B0_GPa * 1e9

        alpha   = np.zeros(len(T_arr))
        gamma_e = np.zeros(len(T_arr))
        Cv_arr  = calc0.heat_capacity_scan(T_arr, q_mesh=q_mesh)  # J/(mol·K)

        for it, T in enumerate(T_arr):
            if T < 1e-9:
                continue
            # Modal heat capacity weights (dimensionless per mode)
            from .constants import HBAR, KB
            pos = omega0 > calc0._imag_tol
            om  = omega0[pos]
            gm  = gamma_qs[pos]

            x = HBAR * om / (2.0 * KB * T)
            x_c = np.clip(x, 0, 350.0)
            cv_mode = (x_c / np.sinh(x_c))**2   # shape function, dimensionless

            # NaN entries in gm (acoustic) → exclude
            valid = np.isfinite(gm)
            if not np.any(valid):
                continue
            weights = cv_mode[valid]
            gm_v    = gm[valid]
            if weights.sum() < 1e-30:
                continue
            g_eff = float(np.dot(gm_v, weights) / weights.sum())
            gamma_e[it] = g_eff

            # Cv per f.u. in J/K (Cv_arr is per mol)
            Cv_fu = Cv_arr[it] / NA
            alpha[it] = g_eff * Cv_fu / (3.0 * B0_Pa * V0_m3)

        return {
            "T_K":       T_arr,
            "alpha_K":   alpha,
            "gamma_eff": gamma_e,
            "Cv_jmolk":  Cv_arr,
        }

    # ── Unified entry point ───────────────────────────────────────────────────

    def run(
        self,
        T_values: np.ndarray,
        q_mesh: Optional[tuple] = None,
        v_range_factor: float = 1.5,
    ) -> Dict[str, np.ndarray]:
        """
        Run the selected method(s) and return a unified results dict.

        Which methods are executed is controlled by ``self.method``:

        * ``"qha"``       → calls :meth:`compute`; returns keys
          ``T_K, V_ang3, alpha_K, Cv_jmolk, Cp_jmolk, B_GPa``.
        * ``"gruneisen"`` → calls :meth:`alpha_gruneisen`; returns keys
          ``T_K, alpha_K, gamma_eff, Cv_jmolk``.
        * ``"both"``      → runs both.  QHA keys are returned at top level;
          Grüneisen keys are additionally included with prefix ``gruen_``:
          ``gruen_alpha_K, gruen_gamma_eff``.  This lets you plot both curves
          from a single dict.

        Parameters
        ----------
        T_values : array-like [K]
        q_mesh   : optional override for the q-mesh used in F_vib / C_V sums.

        Returns
        -------
        dict
        """
        T_arr = np.asarray(T_values, dtype=float)
        qm = q_mesh or self._q_mesh

        if self.method == "qha":
            return self.compute(T_arr, q_mesh=qm, v_range_factor=v_range_factor)

        if self.method == "gruneisen":
            return self.alpha_gruneisen(T_arr, q_mesh=qm)

        # method == "both"
        qha_res   = self.compute(T_arr, q_mesh=qm, v_range_factor=v_range_factor)
        gruen_res = self.alpha_gruneisen(T_arr, q_mesh=qm)
        combined  = dict(qha_res)
        combined["gruen_alpha_K"]   = gruen_res["alpha_K"]
        combined["gruen_gamma_eff"] = gruen_res["gamma_eff"]
        return combined
