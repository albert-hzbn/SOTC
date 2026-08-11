"""Converged quasi-harmonic thermodynamics from direct volume calculations.

The implementation deliberately requires at least five directly calculated
volumes.  It never extrapolates phonon free energies beyond the volume grid.
At external pressure ``p`` the minimized potential is

    G(T, p, V) = E_static(V) + F_vib(T, V) + p V.

At p=0 the minimized quantity is the Helmholtz free energy.  Volumetric and
cubic linear expansion are reported separately:

    alpha_V = (1/V) dV/dT,       alpha_L = alpha_V/3,
    C_p-C_v = T V B_T alpha_V^2 = 9 T V B_T alpha_L^2.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit, minimize_scalar

from .constants import NA

GPA_PER_EV_ANG3 = 160.21766208
ANG3_TO_M3 = 1.0e-30


def _birch_murnaghan_energy(V, E0, V0, B0_eva3, B0_prime):
    """Third-order Birch--Murnaghan energy, with B0 in eV/Angstrom^3."""
    eta = (V0 / np.asarray(V, dtype=float)) ** (2.0 / 3.0)
    x = eta - 1.0
    return E0 + 9.0 * V0 * B0_eva3 / 16.0 * (
        B0_prime * x**3 + (6.0 - 4.0 * eta) * x**2
    )


class SOTCQuasiHarmonic:
    """Quasi-harmonic calculation on a direct, converged volume grid.

    Parameters
    ----------
    volumes_ang3, static_energies_ev, phonon_calcs
        Equal-length sequences with at least five directly calculated volumes,
        sorted internally.  Energies and volumes are per formula unit, and each
        phonon calculator must use the same formula-unit convention.
    eos_model
        ``"birch_murnaghan"`` (default) or a local ``"quadratic"`` model.
        The latter is named honestly and is intended only for a narrow grid.
    pressure_gpa
        External pressure.  The pV term is included explicitly.
    """

    def __init__(
        self,
        volumes_ang3: List[float],
        static_energies_ev: List[float],
        phonon_calcs: list,
        q_mesh: tuple = (20, 20, 20),
        method: Literal["qha", "gruneisen", "both"] = "qha",
        eos_model: Literal["birch_murnaghan", "quadratic"] = "birch_murnaghan",
        pressure_gpa: float = 0.0,
    ):
        if not (len(volumes_ang3) == len(static_energies_ev) == len(phonon_calcs)):
            raise ValueError("volumes, static energies, and phonon calculators must have equal length")
        if len(volumes_ang3) < 5:
            raise ValueError(
                "A converged QHA workflow requires at least five direct volumes; "
                f"received {len(volumes_ang3)}"
            )
        order = np.argsort(np.asarray(volumes_ang3, dtype=float))
        self._V = np.asarray(volumes_ang3, dtype=float)[order]
        self._E = np.asarray(static_energies_ev, dtype=float)[order]
        self._calcs = [phonon_calcs[int(i)] for i in order]
        if np.any(np.diff(self._V) <= 0.0):
            raise ValueError("Volume grid must contain distinct values")
        if method not in ("qha", "gruneisen", "both"):
            raise ValueError(f"Unsupported method: {method}")
        if eos_model not in ("birch_murnaghan", "quadratic"):
            raise ValueError(f"Unsupported EOS model: {eos_model}")
        self._q_mesh = tuple(q_mesh)
        self.method = method
        self.eos_model = eos_model
        self.pressure_gpa = float(pressure_gpa)
        self._fit_static_eos()

    def _fit_static_eos(self) -> None:
        imin = int(np.argmin(self._E))
        e_guess = float(self._E[imin])
        v_guess = float(self._V[imin])
        if self.eos_model == "birch_murnaghan":
            span = float(self._V[-1] - self._V[0])
            bounds = (
                [e_guess - 10.0, self._V[0] - 0.25 * span, 1.0e-5, 0.0],
                [e_guess + 10.0, self._V[-1] + 0.25 * span, 10.0, 12.0],
            )
            params, covariance = curve_fit(
                _birch_murnaghan_energy,
                self._V,
                self._E,
                p0=[e_guess, v_guess, 0.5, 4.0],
                bounds=bounds,
                maxfev=50000,
            )
            self._eos_params = np.asarray(params)
            self._eos_covariance = np.asarray(covariance)
            self.V0_ang3 = float(params[1])
            self.B0_GPa = float(params[2] * GPA_PER_EV_ANG3)
        else:
            coeff = np.polyfit(self._V, self._E, 2)
            if coeff[0] <= 0.0:
                raise ValueError("Quadratic static EOS has non-positive curvature")
            self._eos_params = coeff
            self._eos_covariance = np.full((3, 3), np.nan)
            self.V0_ang3 = float(-coeff[1] / (2.0 * coeff[0]))
            self.B0_GPa = float(
                self.V0_ang3 * 2.0 * coeff[0] * GPA_PER_EV_ANG3
            )
        predicted = np.asarray(self._E0(self._V))
        residual = self._E - predicted
        self.eos_diagnostics = {
            "model": self.eos_model,
            "n_volumes": int(len(self._V)),
            "volume_min_ang3": float(self._V[0]),
            "volume_max_ang3": float(self._V[-1]),
            "V0_ang3": self.V0_ang3,
            "B0_GPa": self.B0_GPa,
            "rmse_meV_per_fu": float(np.sqrt(np.mean(residual**2)) * 1000.0),
            "max_abs_residual_meV_per_fu": float(np.max(np.abs(residual)) * 1000.0),
            "residuals_meV_per_fu": (residual * 1000.0).tolist(),
        }
        self.eos_diagnostics.update(self._leave_one_out_eos())

    def _leave_one_out_eos(self) -> Dict:
        v0_values = []
        b_values = []
        if len(self._V) <= 5 and self.eos_model == "birch_murnaghan":
            # Four points determine BM3 exactly and do not provide a residual
            # test; still report parameter sensitivity, but label it clearly.
            label = "four-point BM3 parameter sensitivity"
        else:
            label = "leave-one-volume-out sensitivity"
        for omit in range(len(self._V)):
            keep = np.arange(len(self._V)) != omit
            V, E = self._V[keep], self._E[keep]
            try:
                if self.eos_model == "birch_murnaghan":
                    span = float(V[-1] - V[0])
                    params, _ = curve_fit(
                        _birch_murnaghan_energy,
                        V,
                        E,
                        p0=[float(E.min()), float(V[np.argmin(E)]), 0.5, 4.0],
                        bounds=(
                            [float(E.min()) - 10.0, V[0] - 0.25 * span, 1e-5, 0.0],
                            [float(E.min()) + 10.0, V[-1] + 0.25 * span, 10.0, 12.0],
                        ),
                        maxfev=50000,
                    )
                    v0_values.append(float(params[1]))
                    b_values.append(float(params[2] * GPA_PER_EV_ANG3))
                else:
                    c = np.polyfit(V, E, 2)
                    v0 = float(-c[1] / (2.0 * c[0]))
                    v0_values.append(v0)
                    b_values.append(float(v0 * 2.0 * c[0] * GPA_PER_EV_ANG3))
            except Exception:
                continue
        return {
            "sensitivity_test": label,
            "V0_leave_one_out_std_ang3": (
                float(np.std(v0_values, ddof=1)) if len(v0_values) > 1 else None
            ),
            "B0_leave_one_out_std_GPa": (
                float(np.std(b_values, ddof=1)) if len(b_values) > 1 else None
            ),
        }

    def _E0(self, V):
        if self.eos_model == "birch_murnaghan":
            return _birch_murnaghan_energy(V, *self._eos_params)
        return np.polyval(self._eos_params, V)

    def _pv_ev(self, V):
        return self.pressure_gpa * np.asarray(V) / GPA_PER_EV_ANG3

    def _fvib_values(self, T: float) -> np.ndarray:
        return np.array([
            calc.vibrational_free_energy(T, q_mesh=self._q_mesh)
            for calc in self._calcs
        ])

    def _free_energy_spline(self, T: float, include_pv: bool = True) -> CubicSpline:
        values = self._E0(self._V) + self._fvib_values(T)
        if include_pv:
            values = values + self._pv_ev(self._V)
        return CubicSpline(self._V, values, bc_type="natural", extrapolate=False)

    def _Fvib(self, T: float, V: float) -> float:
        if not self._V[0] <= V <= self._V[-1]:
            raise ValueError("F_vib extrapolation outside direct volume grid is forbidden")
        return float(CubicSpline(
            self._V, self._fvib_values(T), bc_type="natural", extrapolate=False
        )(V))

    def _G(self, T: float, V: float) -> float:
        if not self._V[0] <= V <= self._V[-1]:
            raise ValueError("Free-energy extrapolation outside direct volume grid is forbidden")
        return float(self._E0(V) + self._Fvib(T, V) + self._pv_ev(V))

    def equilibrium_volume(self, T: float) -> float:
        spline = self._free_energy_spline(T, include_pv=True)
        result = minimize_scalar(
            lambda V: float(spline(V)),
            bounds=(float(self._V[0]), float(self._V[-1])),
            method="bounded",
            options={"xatol": 1.0e-8},
        )
        if not result.success:
            raise RuntimeError(f"Volume minimization failed at T={T}: {result.message}")
        return float(result.x)

    def compute(
        self,
        T_values: np.ndarray,
        q_mesh: Optional[tuple] = None,
        dT_alpha: Optional[float] = None,
        v_range_factor: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """Compute QHA properties without volume extrapolation."""
        if v_range_factor != 1.0:
            raise ValueError("QHA volume extrapolation is disabled; use v_range_factor=1")
        if q_mesh is not None:
            self._q_mesh = tuple(q_mesh)
        T_arr = np.asarray(T_values, dtype=float)
        if len(T_arr) < 4 or np.any(np.diff(T_arr) <= 0.0):
            raise ValueError("T_values must contain at least four strictly increasing points")

        n_t = len(T_arr)
        V_star = np.zeros(n_t)
        B_gpa = np.zeros(n_t)
        helmholtz = np.zeros(n_t)
        gibbs = np.zeros(n_t)
        boundary = np.zeros(n_t, dtype=bool)
        cv_grid = np.zeros((len(self._V), n_t))

        for iv, calc in enumerate(self._calcs):
            cv_grid[iv] = calc.heat_capacity_scan(T_arr, q_mesh=self._q_mesh)

        Cv = np.zeros(n_t)
        for it, T in enumerate(T_arr):
            total = self._free_energy_spline(T, include_pv=True)
            result = minimize_scalar(
                lambda V: float(total(V)),
                bounds=(float(self._V[0]), float(self._V[-1])),
                method="bounded",
                options={"xatol": 1.0e-8},
            )
            V = float(result.x)
            V_star[it] = V
            margin = min(V - self._V[0], self._V[-1] - V)
            boundary[it] = margin < 0.01 * (self._V[-1] - self._V[0])
            curvature = float(total(V, 2))
            B_gpa[it] = V * curvature * GPA_PER_EV_ANG3
            if not np.isfinite(B_gpa[it]) or B_gpa[it] <= 0.0:
                raise RuntimeError(
                    f"Non-positive free-energy curvature at T={T} K: B={B_gpa[it]} GPa"
                )
            f_spline = self._free_energy_spline(T, include_pv=False)
            helmholtz[it] = float(f_spline(V))
            gibbs[it] = float(total(V))
            Cv[it] = float(CubicSpline(
                self._V, cv_grid[:, it], bc_type="natural", extrapolate=False
            )(V))

        # A single differentiable interpolant avoids inconsistent ad-hoc
        # finite-difference steps.  Sensitivity is evaluated by the validation
        # driver using coarser temperature grids and alternate splines.
        volume_of_t = CubicSpline(T_arr, V_star, bc_type="natural")
        dVdT = volume_of_t(T_arr, 1)
        alpha_v = dVdT / V_star
        alpha_l = alpha_v / 3.0

        V_m3 = V_star * ANG3_TO_M3
        B_pa = B_gpa * 1.0e9
        Cp = Cv + T_arr * V_m3 * B_pa * alpha_v**2 * NA
        return {
            "T_K": T_arr,
            "V_ang3": V_star,
            "alpha_V_K": alpha_v,
            "alpha_L_K": alpha_l,
            "alpha_K": alpha_v,  # compatibility alias, explicitly volumetric
            "Cv_jmolk": Cv,
            "Cp_jmolk": Cp,
            "B_GPa": B_gpa,
            "helmholtz_ev_fu": helmholtz,
            "gibbs_ev_fu": gibbs,
            "minimum_near_volume_boundary": boundary,
            "volume_grid_ang3": self._V.copy(),
        }

    def alpha_gruneisen(
        self,
        T_values: np.ndarray,
        q_mesh: tuple = (20, 20, 20),
    ) -> Dict[str, np.ndarray]:
        """Independent Gruneisen estimate of volumetric and linear expansion."""
        T_arr = np.asarray(T_values, dtype=float)
        i0 = int(np.argmin(np.abs(self._V - self.V0_ang3)))
        if i0 == 0 or i0 == len(self._V) - 1:
            raise ValueError("EOS equilibrium must be bracketed by phonon volumes")
        im, ip = i0 - 1, i0 + 1
        calc0, calcm, calcp = self._calcs[i0], self._calcs[im], self._calcs[ip]
        dlnv_half = 0.5 * (np.log(self._V[ip]) - np.log(self._V[im]))
        gamma_qs = calc0.mode_gruneisen_parameters(
            calcp, calcm, dlnv_half, q_mesh=q_mesh
        )
        omega0 = calc0._all_frequencies(q_mesh)
        V0_m3 = self._V[i0] * ANG3_TO_M3
        B0_pa = self.B0_GPa * 1.0e9
        Cv_arr = calc0.heat_capacity_scan(T_arr, q_mesh=q_mesh)
        alpha_v = np.zeros(len(T_arr))
        gamma_eff = np.zeros(len(T_arr))
        from .constants import HBAR, KB
        for it, T in enumerate(T_arr):
            if T <= 0.0:
                continue
            valid = (omega0 > calc0._imag_tol) & np.isfinite(gamma_qs)
            om = omega0[valid]
            gm = gamma_qs[valid]
            x = np.clip(HBAR * om / (2.0 * KB * T), 0.0, 350.0)
            weights = (x / np.sinh(x))**2
            if weights.size == 0 or weights.sum() <= 0.0:
                continue
            gamma_eff[it] = float(np.dot(gm, weights) / weights.sum())
            alpha_v[it] = gamma_eff[it] * (Cv_arr[it] / NA) / (B0_pa * V0_m3)
        return {
            "T_K": T_arr,
            "alpha_V_K": alpha_v,
            "alpha_L_K": alpha_v / 3.0,
            "alpha_K": alpha_v,
            "gamma_eff": gamma_eff,
            "Cv_jmolk": Cv_arr,
        }

    def run(
        self,
        T_values: np.ndarray,
        q_mesh: Optional[tuple] = None,
        v_range_factor: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        T_arr = np.asarray(T_values, dtype=float)
        qm = tuple(q_mesh or self._q_mesh)
        if self.method == "qha":
            return self.compute(T_arr, q_mesh=qm, v_range_factor=v_range_factor)
        if self.method == "gruneisen":
            return self.alpha_gruneisen(T_arr, q_mesh=qm)
        qha = self.compute(T_arr, q_mesh=qm, v_range_factor=v_range_factor)
        gr = self.alpha_gruneisen(T_arr, q_mesh=qm)
        combined = dict(qha)
        combined["gruen_alpha_V_K"] = gr["alpha_V_K"]
        combined["gruen_alpha_L_K"] = gr["alpha_L_K"]
        combined["gruen_alpha_K"] = gr["alpha_V_K"]
        combined["gruen_gamma_eff"] = gr["gamma_eff"]
        return combined
