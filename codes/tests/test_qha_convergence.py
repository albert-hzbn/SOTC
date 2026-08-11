"""Tests for QHA thermodynamic definitions and safeguards."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sotc.constants import NA
from sotc.qha import (
    ANG3_TO_M3,
    GPA_PER_EV_ANG3,
    SOTCQuasiHarmonic,
    _birch_murnaghan_energy,
)


class _AnalyticPhonons:
    def __init__(self, volume, v0=16.0, thermal_pressure=2.0e-5):
        self.volume = volume
        self.v0 = v0
        self.thermal_pressure = thermal_pressure

    def vibrational_free_energy(self, T, q_mesh=(1, 1, 1)):
        return -self.thermal_pressure * T * (self.volume - self.v0)

    def heat_capacity_scan(self, T_values, q_mesh=(1, 1, 1)):
        return np.full(len(T_values), 20.0)


def _qha():
    volumes = np.linspace(15.0, 17.0, 7)
    energies = _birch_murnaghan_energy(volumes, -3.0, 16.0, 0.5, 4.0)
    calcs = [_AnalyticPhonons(v) for v in volumes]
    return SOTCQuasiHarmonic(
        volumes.tolist(), energies.tolist(), calcs,
        q_mesh=(1, 1, 1), eos_model="birch_murnaghan",
    )


def test_qha_requires_at_least_five_direct_volumes():
    volumes = [15.5, 16.0, 16.5]
    energies = [0.1, 0.0, 0.1]
    calcs = [_AnalyticPhonons(v) for v in volumes]
    with pytest.raises(ValueError, match="at least five direct volumes"):
        SOTCQuasiHarmonic(volumes, energies, calcs)


def test_qha_eos_recovers_positive_bulk_modulus():
    qha = _qha()
    assert np.isclose(qha.V0_ang3, 16.0, rtol=2e-4)
    assert np.isclose(qha.B0_GPa, 0.5 * GPA_PER_EV_ANG3, rtol=2e-3)
    assert qha.eos_diagnostics["rmse_meV_per_fu"] < 1e-6


def test_qha_reports_distinct_volumetric_and_linear_expansion_and_cp_identity():
    qha = _qha()
    T = np.arange(50.0, 451.0, 50.0)
    result = qha.compute(T, q_mesh=(1, 1, 1))
    assert np.all(np.diff(result["V_ang3"]) > 0.0)
    assert np.allclose(result["alpha_L_K"], result["alpha_V_K"] / 3.0)
    expected_cp = (
        result["Cv_jmolk"]
        + T * result["V_ang3"] * ANG3_TO_M3
        * result["B_GPa"] * 1e9 * result["alpha_V_K"]**2 * NA
    )
    assert np.allclose(result["Cp_jmolk"], expected_cp)
    assert not np.any(result["minimum_near_volume_boundary"])


def test_qha_forbids_volume_extrapolation():
    qha = _qha()
    with pytest.raises(ValueError, match="extrapolation"):
        qha._Fvib(300.0, qha._V[-1] + 0.1)
    with pytest.raises(ValueError, match="extrapolation"):
        qha.compute(np.arange(50.0, 251.0, 50.0), v_range_factor=1.5)
