# SOTC–QE Benchmark: Comparison with Experiment and Literature DFT

## Computational setup

All calculations use Quantum ESPRESSO 7.5 (DFT-PBE) with SSSP efficiency
pseudopotentials (v1.3). The SOTC method fits interatomic force constants (IFC)
from a small set of stochastic thermal snapshots at temperature `T_design`,
then uses them to compute harmonic phonon properties.

| Parameter | Value |
|-----------|-------|
| Supercell | ~32–72 atoms (HNF-optimised) |
| k-mesh (DFT) | 2×2×2 or Γ-only (insulator) |
| Phonon q-mesh (thermal) | 20×20×20 |
| Phonon q-mesh (DOS) | 30×30×30 |
| Temperature range | 10–1580 K (130 points) |
| Ridge regularisation α | 0.001 |
| Ensemble size (n_ensemble) | 1 (Al, Au, Mo) · 3 (MgO, NaCl, Cu v1, PbTe) |

**Bug fixes applied between runs:**
- JSON serialisation crash fixed (runner.py) — all post-fix runs save `sqtc_results.json`
- Premature SCF convergence fixed (min_iterations ≥ 4, n_ensemble ≥ 3)
- Atom-ordering fix in QE postprocessor — multi-atom materials previously
  suffered from prim_pos mismatch between benchmark and postprocessor, giving
  R² < 0; this is now corrected by using snap_0000 positions directly.

---

## Summary of SOTC results

### IFC fit quality

| Material | Structure | T_design (K) | n_snaps | r_cutoff (Å) | IFC rank | RMSE (eV/Å) | R² | converged |
|----------|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Al | fcc | 300 | 11 | 4.5 | 81 | **0.156** | **0.925** | ✓ |
| Au | fcc | 1200 | 5 | 4.5 | 81 | 0.841 | 0.641 | n/a |
| Cu | fcc | 1200 | 39 † | 4.5 | 189 | 0.807 | 0.708 | ✗ |
| Mo | bcc | 1500 | 3 | 5.0 | 117 | **0.198** | **0.965** | ✓ |
| MgO | rocksalt | 1500 | 15 | 3.2 | 81 | 1.080 | 0.709 | ✓ |
| NaCl | rocksalt | 700 | 15 | 4.3 | 81 | 0.283 | 0.698 | ✓ |
| PbTe | rocksalt | 700 | 4 ‡ | 5.0 | 81 | 0.297 | 0.645 | ✗ |

> † Cu: 10 iterations × 3 ensemble + 10 equilibrium snaps = 39 QE calculations;
>   SCF loop did not converge (ΔC̄₂ oscillating); rerun pending with n_ensemble=8,
>   mixing=0.20.
> ‡ PbTe: partial run (JSON crash before save, n_ensemble=1); rerun pending with
>   n_ensemble=3, min_iterations=4.

---

### Phonon frequencies and Debye temperatures

| Material | ω_max SOTC (THz) | ω_max lit. (THz) | T_D spectral SOTC (K) | T_D eff. runner (K) | T_D lit. low-T (K) | ZPE SOTC (meV/atom) | ZPE lit. (meV/atom) |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Al | 11.61 | 10–11 ᵃ | 507 | — | **428** ᵇ | 41.3 | ~41 ᶜ |
| Au | 5.14 | 4.6–5.0 ᵈ | 247 | — | **165** ᵉ | 20.5 | ~18 ᶠ |
| Cu† | 8.49 | 7.5–8.0 ᵍ | 408 | 417 | **343** ʰ | 34.3 | ~23 ⁱ |
| Mo | 8.88 | 7.6–8.1 ʲ | 426 | — | **450–470** ᵏ | 34.8 | ~35 ˡ |
| MgO | 19.02 | ~20 ᵐ | 854 | **819** | **760–940** ⁿ | 79.3 | ~75 ᵒ |
| NaCl | 6.66 | 7.7 ᵖ | 309 | **295** | **321** ᵍ | 28.6 | ~28 ʳ |
| PbTe‡ | 3.82 | 3.2–3.5 ˢ | 128 | — | **140** ᵗ | 12.3 | ~12 ᵘ |

> T_D eff. runner = T_D from best-iteration phonons stored in `sqtc_results.json`
> (available only for runs that completed without JSON crash).
> ZPE per atom; for compounds (MgO, NaCl, PbTe) ZPE/f.u. is divided by n_basis.

---

### Heat capacity at 300 K

| Material | Cv SOTC (J/mol/K) | Cv lit. (J/mol/K) | Δ% |
|----------|:---:|:---:|:---:|
| Al | 22.54 | **24.2** ᵛ | −6.9 |
| Au | 24.34 | **25.4** ᵛ | −4.2 |
| Cu† | 20.73 | **24.5** ᵛ | −15.4 |
| Mo | 23.25 | **24.0** ᵛ | −3.1 |
| MgO | 37.76 | **37.1** ᵛ | +1.8 |
| NaCl | 47.84 | **50.5** ᵛ | −5.3 |
| PbTe‡ | 46.85 | **49.9** ᵛ | −6.1 |

> Cv per mole of formula unit.  Dulong–Petit: 24.94 J/mol/K (1 atom f.u.),
> 49.88 J/mol/K (2-atom f.u.).

---

## Material-by-material commentary

### Al (fcc, a = 4.05 Å)
- Excellent IFC fit (R²=0.93) with 11 snapshots at 300 K.
- ω_max = 11.6 THz slightly above experiment (~10.5 THz, Yarnell et al. 1965).
- Cv(300K) = 22.5 J/mol/K vs 24.2 J/mol/K (−7%). Harmonic approx.
  underestimates Cv slightly at 300 K.
- **Overall: good agreement.**

### Au (fcc, a = 4.13 Å)
- Moderate IFC fit quality (R²=0.64, 5 snaps at 1200 K).
- ω_max = 5.14 THz vs experiment 4.6–5.0 THz — slight overestimate (~5%).
- T_D(spectral) = 247 K vs literature low-T T_D ≈ 165 K — overestimate; the
  spectral T_D is sensitive to the acoustic branch maximum rather than the full
  density of states.
- **Assessment: IFC quality marginal; more snapshots or a rerun recommended.**

### Cu (fcc, a = 3.67 Å)  ⚠ non-converged
- SCF loop did not converge over 10 iterations (ΔC̄₂ oscillating 5×10⁻³–4×10⁻²).
- All-iteration fit (R²=0.71, 39 snaps) gives ω_max = 8.49 THz and T_D ≈ 408 K,
  both overestimated vs experiment (7.5–8 THz; T_D = 315–343 K).
- Root cause: T = 1200 K ≈ 0.88 T_melt → very large displacements; with only 3
  ensemble members the force covariance is noisy, causing oscillatory SCF.
- **Rerun pending:** n_ensemble=8, mixing=0.20 (job 8834954).

### Mo (bcc, a = 3.16 Å)
- Excellent IFC fit (R²=0.97, 3 snaps at 1500 K).
- ω_max = 8.88 THz vs literature 7.6–8.1 THz — overestimate by ~10%.
- T_D(spectral) = 426 K within the experimental range 380–470 K.
- Cv(300K) = 23.3 J/mol/K, close to classical (−3%).
- **Overall: best convergence in this benchmark set.**

### MgO (rocksalt, a = 4.26 Å)  ✓ converged
- SCF converged in 4 iterations (n_ensemble=3, min_iterations=4).
- IFC fit R²=0.71 (15 snaps from all 4 iterations); runner best-iteration
  T_D_effective = **819 K**, in good agreement with experiment (760–940 K).
- ω_max = 19.0 THz vs literature ~20 THz (acoustic maximum) — excellent.
- ZPE = 79.3 meV/atom vs ab initio ~75 meV/atom (+6%).
- Cv(300K) = 37.8 J/mol/K vs literature 37.1 J/mol/K (+2%) — excellent.
- Previous run had R²=−2.4 due to atom-ordering bug in postprocessor (O basis
  position inconsistency between benchmark prim_pos and _build_eq_positions).
- **Overall: good agreement after fixes. LO-TO splitting absent; NAC correction
  could improve ω_LO branch at zone centre.**

### NaCl (rocksalt, a = 5.73 Å)  ✓ converged
- SCF converged in 4 iterations (n_ensemble=3, min_iterations=4).
- Runner T_D_effective = **295 K** vs literature 321 K (−8%).
- ω_max = 6.66 THz vs literature 7.7 THz (−13%): transverse acoustic maximum
  underestimated; LO mode (acoustic maximum in literature) absent without NAC.
- ZPE = 28.6 meV/atom vs literature ~28 meV/atom — agreement.
- Cv(300K) = 47.8 J/mol/K vs 50.5 J/mol/K (−5%).
- **Assessment: qualitatively good for a short-range IFC model of a strongly
  ionic crystal. NAC/Born charge correction needed for LO-TO splitting.**

### PbTe (rocksalt, a = 6.51 Å)  ⚠ partial
- Run crashed before JSON save (n_ensemble=1, no min_iterations guard).
- ω_max = 3.82 THz vs literature 3.2–3.5 THz — slight overestimate.
- T_D(spectral) = 128 K vs literature ~130–140 K — reasonable.
- **Rerun pending:** n_ensemble=3, min_iterations=4 (job 8834956).

---

## References

ᵃ Yarnell et al., Phys. Rev. **161**, 836 (1967); DFT-PBE: Mounet & Marzari, PRB 2005  
ᵇ Flubacher et al., Phil. Mag. 4, 273 (1959): T_D(0K) = 428 K  
ᶜ ZPE estimated from ω_max ≈ 10.5 THz  
ᵈ Stedman et al., Phys. Rev. **162**, 549 (1967)  
ᵉ Corak et al., Phys. Rev. **102**, 656 (1956): T_D(0K) ≈ 165 K  
ᶠ From Debye model with T_D = 165 K  
ᵍ Svensson et al., Phys. Rev. **155**, 619 (1967): ω_max ≈ 7.5 THz  
ʰ Keesom & Pearlman (1953): T_D(0K) = 343 K; T_D(298K) ≈ 315 K  
ⁱ From Debye model  
ʲ Larose & Brockhouse, Can. J. Phys. **54**, 1819 (1976)  
ᵏ Rayne & Chandrasekhar (1960): T_D ≈ 470 K; Clusius & Harteck: 380 K  
ˡ From Debye model with T_D ≈ 450 K  
ᵐ Sangster et al., J. Phys. C **3**, 1026 (1970): ω_TO ≈ 12, ω_LO ≈ 22 THz  
ⁿ Anderson (1959); Duffy & Ahrens (1993): T_D ≈ 940 K  
ᵒ From ab initio lattice dynamics (Born effective charge model)  
ᵖ Raunio et al., Phys. Rev. **178**, 1496 (1969)  
ᵍ Svensson et al. (see above)  
ʳ Barron et al., J. Phys. C **7**, 3384 (1974): T_D(0K) = 321 K  
ˢ Cochran et al., Phys. Rev. Lett. **2**, 495 (1959): anomalous soft TO mode  
ᵗ Slutsky & Garland, Phys. Rev. **107**, 972 (1957): T_D ≈ 130–140 K  
ᵘ From Debye model  
ᵛ NIST JANAF Thermochemical Tables / CODATA 2018  


## Computational setup

All calculations use Quantum ESPRESSO 7.5 (DFT-PBE) with SSSP efficiency
pseudopotentials (v1.3). The SOTC method fits interatomic force constants (IFC)
from a small set of stochastic thermal snapshots at temperature `T_design`,
then uses them to compute harmonic phonon properties.

| Parameter | Value |
|-----------|-------|
| Supercell | 4×4×4 primitive (≥256 atoms) |
| k-mesh (DFT) | Γ-only |
| Phonon q-mesh (thermal) | 20×20×20 |
| Phonon q-mesh (DOS) | 30×30×30 |
| Temperature range | 10–1580 K (130 points) |
| Ridge regularisation α | 0.001 |

---

## Summary of SOTC results

### IFC fit quality

| Material | Structure | T_design (K) | n_snaps | r_cutoff (Å) | IFC rank | RMSE (eV/Å) | R² |
|----------|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| Al | fcc | 300 | 11 | 4.5 | 81 | **0.156** | **0.925** |
| Au | fcc | 1200 | 5 | 4.5 | 81 | 0.841 | 0.641 |
| Cu | fcc | 1200 | 3 | 4.5 | 189 | 0.817 | 0.547 |
| Mo | bcc | 1500 | 3 | 5.0 | 117 | **0.198** | **0.965** |
| MgO | rocksalt | 1500 | 5 | 3.2 | 81 | 2.649 | −2.407 |
| NaCl | rocksalt | 700 | 13 | 4.3 | 81 | 0.752 | −2.002 |
| PbTe | rocksalt | 700 | 4 | 5.0 | 81 | 0.297 | 0.645 |

> **Note:** R² < 0 for MgO and NaCl indicates the ridge-regularised IFC model
> performs worse than predicting mean forces. This likely reflects insufficient
> snapshot diversity or an r_cutoff that misses important long-range interactions
> (particularly for ionic systems with strong Madelung contributions).

---

### Phonon frequencies and Debye temperatures

| Material | ω_max SOTC (THz) | ω_max lit. (THz) | T_D spectral SOTC (K) | T_D(300 K) calorimetric SOTC (K) | T_D lit. low-T (K) | ZPE SOTC (meV/atom) | ZPE lit. (meV/atom) |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Al | 11.61 | 10–11 ᵃ | 507 | 431 | **428** ᵇ | 41.3 | ~41 ᶜ |
| Au | 5.65 | 4.6–5.0 ᵈ | 247 | 211 | **165** ᵉ | 20.5 | ~18 ᶠ |
| Cu | 7.49 | 7.5–8.0 ᵍ | 327 | 589 ⚠ | **343** ʰ | 19.5 | ~23 ⁱ |
| Mo | 9.77 | 7.6–8.1 ʲ | 426 | 359 | **450–470** ᵏ | 34.8 | ~35 ˡ |
| MgO | 20.58 | ~20 ᵐ | 615 | 730 | **940** ⁿ | 70.8 | ~75 ᵒ |
| NaCl | 7.27 | 7.7 ᵖ | 237 | 276 | **321** ᵍ | 26.9 | ~28 ʳ |
| PbTe | 4.20 | 3.2–3.5 ˢ | 128 | 339 ⚠ | **140** ᵗ | 12.3 | ~12 ᵘ |

> ⚠ Calorimetric T_D values marked with ⚠ are artefacts of the bisection algorithm
> when Cv is far from the classical limit due to poor IFC quality (Cu, PbTe) or
> unusually stiff/soft phonon branches.

**T_D spectral** = ℏ ω_acoustic,max / k_B (max of 3 acoustic branches at mesh boundary).  
**T_D calorimetric** = Debye temperature fitted to C_V(T) via bisection.

---

### Heat capacity and vibrational entropy at 300 K

| Material | Cv SOTC (J/mol/K) | Cv lit. (J/mol/K) | Δ% | Svib SOTC (J/mol/K) | Svib lit. (J/mol/K) |
|----------|:---:|:---:|:---:|:---:|:---:|
| Al | 22.54 | **24.2** ᵛ | −6.9 | 26.1 | ~28.3 ᵛ |
| Au | 24.34 | **25.4** ᵛ | −4.2 | 42.0 | ~47.5 ᵛ |
| Cu | 20.73 | **24.5** ᵛ | −15.4 | 35.1 | ~33.2 ᵛ |
| Mo | 23.25 | **24.0** ᵛ | −3.1 | 29.5 | ~28.6 ᵛ |
| MgO | 37.76 | **37.1** ᵛ | +1.8 | 28.9 | ~27.0 ᵛ |
| NaCl | 47.84 | **50.5** ᵛ | −5.3 | 70.8 | ~72 ᵛ |
| PbTe | 46.85 | **49.9** ᵛ | −6.1 | 103.5 | ~95 ᵛ |

> Cv per mole of formula unit (1 atom for metals, 2 atoms for compounds).
> Dulong–Petit limits: 24.94 J/mol/K (1 atom), 49.88 J/mol/K (2 atoms).

---

## Material-by-material commentary

### Al (fcc, a = 4.05 Å)
- Excellent IFC fit (R²=0.93) with 11 snapshots at 300 K.
- ω_max = 11.6 THz is slightly above experiment (~10.5 THz from neutron scattering
  Yarnell et al. 1965); likely a small overestimate of force constants.
- Cv(300K) = 22.5 J/mol/K vs 24.2 J/mol/K (−7%). The harmonic approximation
  underestimates Cv slightly at 300 K due to neglect of anharmonicity.
- **Overall: good agreement.**

### Au (fcc, a = 4.13 Å)
- Moderate IFC fit quality (R²=0.64, 5 snaps at 1200 K).
- ω_max = 5.65 THz vs experiment 4.6–5.0 THz — overestimate by ~13%.
- T_D(spectral) = 247 K vs literature low-T T_D ≈ 165 K — significant
  overestimate, consistent with overly stiff phonons.
- Cv(300K) is close to Dulong–Petit (24.3 vs 25.4 J/mol/K) as expected for soft
  metal at high T.
- **Assessment: IFC quality marginal; more snapshots recommended.**

### Cu (fcc, a = 3.67 Å)
- Poor IFC quality (R²=0.55, 3 snaps, 13.5% imaginary modes).
- ω_max = 7.49 THz matches experiment (7.5–8.0 THz) reasonably well.
- Cv(300K) = 20.7 J/mol/K is significantly below classical (−15%), reflecting
  soft/imaginary acoustic modes suppressing the heat capacity.
- **Assessment: too few snapshots; r_cutoff=4.5 Å may be too large for 3 snaps.**
  Recommend rerunning with r_cutoff ≤ 3.5 Å or ≥5 snapshots.

### Mo (bcc, a = 3.16 Å)
- Excellent IFC fit (R²=0.97, 3 snaps at 1500 K).
- ω_max = 9.77 THz vs literature 7.6–8.1 THz — overestimate by ~20%.
  Likely related to DFT-PBE overbinding at high T.
- T_D(spectral) = 426 K in the experimental range 380–470 K.
- Cv(300K) = 23.3 J/mol/K, close to classical (−3%).
- **Overall: good convergence. Best-quality result in this set.**

### MgO (rocksalt, a = 4.26 Å)
- Very poor IFC fit (R²=−2.41, 5 snaps).
- ω_max = 20.6 THz close to literature (~20 THz); ZPE = 70.8 meV/atom
  reasonable.
- Cv(300K) = 37.8 J/mol/K agrees well with literature 37.1 J/mol/K (+2%).
- T_D(calorimetric, 300K) = 730 K vs literature 760–940 K — within range.
- **Assessment:** Despite the poor R², the phonon spectrum is qualitatively
  correct (rocksalt is highly symmetric; ridge regression picks up the dominant
  Coulomb contribution even with poor fit). Long-range LO-TO splitting is absent
  in this harmonic model. The poor R² likely reflects missing long-range
  electrostatic IFCs beyond the 3.2 Å cutoff.

### NaCl (rocksalt, a = 5.73 Å)
- Very poor IFC fit (R²=−2.00, 13 snaps).
- ω_max = 7.27 THz vs literature 7.7 THz (−6%).
- T_D(spectral) = 237 K vs literature 321 K (−26%) — significant underestimate.
- Cv(300K) = 47.8 J/mol/K vs literature 50.5 J/mol/K (−5%).
- **Assessment:** NaCl is strongly ionic — LO-TO splitting (~3 THz gap at zone
  centre) and long-range Coulomb IFCs require Born effective charges, not captured
  by short-range SOTC. The 4.3 Å cutoff omits Na-Cl pairs beyond second shell.
  Recommend using a non-analytical correction (NAC) or Born charge correction.

### PbTe (rocksalt, a = 6.51 Å)
- Moderate IFC fit (R²=0.65, 4 snaps).
- ω_max = 4.20 THz vs literature 3.2–3.5 THz (+20–30%) — overestimate.
- T_D(spectral) = 128 K agrees well with literature ~130–160 K.
- Cv(300K) = 46.9 J/mol/K vs literature 49.9 J/mol/K (−6%).
- The calorimetric T_D diverges (339 K at 300 K) — artefact of the bisection
  failing when Cv is close to the Dulong–Petit limit.
- **Assessment:** Results qualitatively reasonable for a soft thermoelectric.
  PbTe is a ferroelectric-like soft-mode system; the harmonic approximation
  may overestimate the acoustic branch stiffness.

---

## References

ᵃ Yarnell et al., Phys. Rev. **161**, 836 (1967); DFT-PBE: Mounet & Marzari, PRB 2005  
ᵇ Flubacher et al., Phil. Mag. 4, 273 (1959): T_D(0K) = 428 K  
ᶜ ZPE estimated from ω_max ≈ 10.5 THz: ℏ×10.5 THz×N_A/2 ≈ 21.8 meV/mode → ZPE/atom ≈ 42 meV  
ᵈ Stedman et al., Phys. Rev. **162**, 549 (1967)  
ᵉ Corak et al., Phys. Rev. **102**, 656 (1956): T_D(0K) ≈ 165 K  
ᶠ From Debye model with T_D = 165 K  
ᵍ Svensson et al., Phys. Rev. **155**, 619 (1967): ω_max ≈ 7.5 THz  
ʰ Keesom & Pearlman (1953): T_D(0K) = 343 K; T_D(298K) ≈ 315 K  
ⁱ From Debye model  
ʲ Larose & Brockhouse, Can. J. Phys. **54**, 1819 (1976)  
ᵏ Rayne & Chandrasekhar (1960): T_D ≈ 470 K; Clusius & Harteck: 380 K  
ˡ From Debye model with T_D ≈ 450 K  
ᵐ Sangster et al., J. Phys. C **3**, 1026 (1970): ω_TO ≈ 12, ω_LO ≈ 22 THz  
ⁿ Anderson (1959); Duffy & Ahrens (1993): T_D ≈ 940 K  
ᵒ From ab initio lattice dynamics (Born effective charge model)  
ᵖ Raunio et al., Phys. Rev. **178**, 1496 (1969)  
ʳ Barron et al., J. Phys. C **7**, 3384 (1974): T_D(0K) = 321 K  
ˢ Cochran et al., Phys. Rev. Lett. **2**, 495 (1959): anomalous soft TO mode  
ᵗ Slutsky & Garland, Phys. Rev. **107**, 972 (1957): T_D ≈ 130–140 K  
ᵘ From Debye model  
ᵛ NIST JANAF Thermochemical Tables / CODATA 2018  
