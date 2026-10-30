# PRB reviewer comments → JAP manuscript

Each comment from the three PRB reports (BS15679) mapped to where the JAP
version (`sotc_jap.tex`, `sotc_jap_supplementary.tex`) addresses it, with the
evidence. Section/table/figure numbers refer to the compiled JAP PDFs.

## Reviewer 1

| # | Comment | Status | Where / evidence |
|---|---|---|---|
| 1 | TDEP, sTDEP, SCHA, SSCHA described incorrectly; "compact TDEP" | Resolved | Introduction: TDEP is a fitting method, SSCHA a variational free-energy minimization with stochastic sampling. Secs. II A and II E prove that the canonical effective FC2 is the Gaussian-averaged Hessian and that its self-consistent fixed point is the SCHA/sTDEP fixed point. "Compact TDEP" is removed. sTDEP is cited as Shulumba et al. (PRB 95, 014302), and Monacelli et al. (2021) is cited. |
| 2 | No parameters for the Phonopy/TDEP comparisons | Resolved | Every comparison is now computed by the author in the same cell with identical QE settings (Sec. IV A–C). Canonical (TDEP-type) sampling, sTDEP, SSCHA (python-sscha 1.6), special displacements, rattle and hiPhive fits are all specified (Sec. IV C), and results are in Secs. V C–I and Table VII. |
| 3 | MSD off by factor ≥ 2, undiscussed | Resolved | One convention, U_iso = ⟨u²⟩/3 (Sec. II B, after Eq. 5). Temperature-consistent harmonic models give U_iso(300 K) for Al, Cu, Au within 6 % of Peng et al., SOTC-D models within 2–11 % (Sec. V J). The manuscript states the convention once and compares only temperature-consistent models. |
| 4 | QHA vs Grüneisen disagreement, 300 K artifact | Resolved | Self-consistent QHA on a volume grid (Sec. V K, Table XIII, SM Table S11): α from (∂S/∂V)_T/3B_T; Grüneisen at V(T), B_T reproduces the minimization within 0.1e-6/K, the static-B0 shortcut is 13–44 % low. Al within 3 % of experiment at 300 K; PbTe (unstable harmonic trial) stable, about 20 % below experiment at 300 K. |
| 5 | Too few volumes; expanded-volume IFCs; parabolic vs Birch–Murnaghan; E0 breaks the variational form | Resolved | Seven volumes; FC2 computed directly at every volume (FD trial + SOTC-D-SC), not extrapolated; converged 125-atom QHA on the same volumes for comparison; static energy from a third-order Birch–Murnaghan fit; F(V) cubic and quartic fits compared; the E − E0 term is the SCHA anharmonic correction ⟨E − E0 − ½uΦu⟩, so F is the SCHA free energy (Eq. 11, Sec. V K). PbTe grid with the same construction. PBE-lattice sensitivity in Table S6. |
| 6 | LaTeX errors; sign of e^{iq·r} | Resolved | Eq. (4) and Supplementary Sec. S10 use the same e^{-iq·R} convention, and Eq. (4) states the eigenvector gauge (R vs R+τ). The documents compile without errors or undefined references. |

## Reviewer 2

| # | Comment | Status | Where / evidence |
|---|---|---|---|
| 1 | MSD discrepancies for Al, Cu | Resolved | As R1.3 (Sec. V J); Eq. (5) and U_iso = ⟨|u|²⟩/3 use one convention (the factor of 3). |
| 2 | Effective rank of the IFC matrix (Cu 162) vs N_ens = 1 | Resolved | Sec. V A reconciles it explicitly: the Cu runs used N_ens = 8 (not 1; Table I lists N_ens for every crystal), and the 4.5 Å cutoff includes the third fcc shell of Cu (4.49 Å) but not of Al/Ag/Au (4.96–5.06 Å), so Cu has 90 parameters against 30. In the symmetry-adapted basis each single iteration has full rank (Table II: 90/90 for Cu, condition 15.3). |
| 3 | Al thermal-expansion anomaly (α(300 K) = 36.9e-6 K⁻¹, non-monotonic) | Resolved | Recomputed on a volume grid with two independent 32-call grids: 13.1, 22.8, 23.3e-6/K at 100/300/400 K, monotonic, 1–3 % below experiment at 300 K; converged 125-atom QHA added; high-T non-resolution quantified (Sec. V K, Table XIII). |
| 4 | Neutron data not shown | Resolved | NaCl: the 39 tabulated 80 K neutron points and IR TO/LO limits are plotted (Supplementary Fig. S3, cited in Sec. V L). Al: the converged FD reference (X 5.72/9.35 THz, L 4.25/9.66 THz) is compared with the Born–von Kármán model fitted to inelastic neutron data at 10 K by Kresch et al. (PRB 77, 024301; thesis Table 9.2): X 5.75/9.45 THz, L 4.21/9.72 THz, all within 1.1%. The 300 K neutron values (5.39/9.37, 4.16/9.43 THz) are also quoted (Sec. V B; script codes/validation/al_neutron_bvk.py). |
| 5 | LO–TO splitting missing for polar materials | Resolved | DFPT Born charges and ε∞ at the production volume, mixed-space NAC (Sec. V L, Fig. 4). |
| 6 | Experimental lattice, no relaxation, residual forces | Resolved | Fits use ΔF = F(u) − F(0) with the undisplaced forces computed in the same cell (Sec. III A). At the ideal positions F(0) vanishes by site symmetry; the computed undisplaced-cell forces are 2e-6–1.4e-3 eV/Å rms (Mo largest), at most 1e-3 of the thermal force response (0.48–2.3 eV/Å), stated in Sec. III A (from primary_audit.json, reference_rms_force_by_iteration). PBE-lattice sensitivity is in Table S6 and Sec. V J. |
| i | Were the TDEP curves from the literature? | Resolved | No literature curves remain; all baselines are computed with matched settings. |
| ii | Phonopy not introduced | Resolved | Introduced at first use (Sec. IV C). |
| iii | Code details in the theory section | Resolved | The theory (Sec. II) is implementation-free; file names and keywords are in the README and data repository. |

## Reviewer 3

| # | Comment | Status | Where / evidence |
|---|---|---|---|
| 1 | Provenance of derivations | Resolved | Each equation states whether it is a standard result (with reference) or derived here, with a pointer to the Supplementary section: Eq. (4) → S10, Eq. (6) gradient → S11, Eq. (8) and the averaged-Hessian identity Eq. (2) → S4. |
| 2 | Shell reduction of C̄₂ undefined | Resolved | Eq. (5) gives species-resolved shells, binned at 0.05 Å, with an explicit trace and multiplicity. |
| 3 | "Functional" → function | Resolved | Q is called the objective throughout. |
| 4 | Exponential decay of the correlator unjustified (main concern) | Resolved | The decay argument and the "representability theorem" are withdrawn. ξ_w is stated to be a numerical weight, not a physical length. The performance claims now rest on estimator theory (Secs. II A and II E: bias O(1/M), exact cubic cancellation, D-optimality) and on matched numerical controls. The bias is removable exactly by moment-matched amplitudes (Secs. II A and II E, IV H; unit-tested by exact enumeration); it is measured at ~0.01 THz on the surrogates. |
| 5 | Model (Debye) dispersions as input; need a baseline against random displacements | Resolved | Every design now uses the same material-specific trial: same-cell DFT harmonic FD, then the iteration-1 effective FC2. Designs are compared with random canonical subsets and pairs at equal force calls (Tables IV–VI, Figs. 1–2), plus special displacements and rattle. The self-consistent bootstrapped variant starts from the DFT harmonic trial with no extra calls. |
| 6 | How can 2–5 calculations determine about 100 parameters? | Resolved | One call gives 3N equations, and the symmetry-adapted basis has 3–90 parameters (Tables II, VIII). The audit lists the priors and their bias (axial projection, ridge). When the data cannot determine the model (Cu with 2 calls), the paper says so explicitly. |
| 7 | Comparison with TDEP/SSCHA inappropriate (harmonic vs anharmonic) | Resolved | SOTC-D estimates the same object as TDEP/SSCHA: the temperature-dependent effective FC2, which includes quartic renormalization (Secs. II A and II E). The comparison is made at equal force calls against the same target: DFT canonical sampling and two-iteration DFT sTDEP in all 11 crystals (Tables IV–VI, Fig. 2; bootstrapped SOTC-D best in 11/11 at 8 calls, 9/11 at 16), and sTDEP and SSCHA on DFT-fitted surrogates of Al, W, NaCl, Si and Mg with a known self-consistent target (Tables IX, XI), and SSCHA, iterated sTDEP, ZG-SC and AIMD-TDEP with DFT forces for Al and Si against a converged self-consistent reference (Table VII): SOTC-D-SC needs the fewest calls or ties in 13 of 14 material–accuracy cases and 2–16× fewer calls than SSCHA in 13 of 14. Anharmonic free energies and lifetimes are stated to be out of scope. |
| 8 | Aggregate DOS quantities hide dispersion errors; Au instability | Resolved | All primary metrics are branch-resolved mesh RMSE (12³ mesh) and held-out forces. Dispersions are compared against converged large-cell references (Fig. 3, Table III), with convergence against the number of force calls (Figs. 1, 2). The Au instability is traced to one-configuration fits with the axial prior and removed by pooling or the symmetry basis (Sec. V A). |

## Second-round report (report 2.1)

| # | Comment | Status | Where / evidence |
|---|---|---|---|
| 1 | Introduction still superficial: TDEP/SCHA generic; "levels" not explained; mode-coupling theory misdescribed; sTDEP never defined | Resolved | Introduction rewritten: levels distinguished by ensemble and retained anharmonicity (QHA → self-consistent harmonic → TDEP/AIMD); SCHA defined by the Gibbs–Bogoliubov minimization and its stationarity condition; SSCHA by stochastic evaluation with importance reweighting; TDEP as a least-squares fitting procedure applied to canonical samples (usually AIMD), with mode-coupling theory as its formal basis; sTDEP defined (Gaussian quantum sampling + refit to self-consistency); sTDEP/SCHA fixed points shown equal (Sec. II A, Eqs. 1–3). |
| 2 | Why remove TDEP comparisons instead of giving parameters? | Resolved | Canonical sampling with TDEP-type least-squares fitting, unpaired and paired, one-shot and iterated (sTDEP, DFT forces, all 11 crystals), plus SSCHA on model potentials, with every parameter stated (Sec. IV C; Tables IV–VII, IX, XI). For Al and Si, sTDEP, SSCHA and AIMD-TDEP were run with DFT forces in the same cell against a converged self-consistent reference, with AIMD convergence measured against MD steps (Sec. V D, Table VII). Sec. VI discusses SOTC-D as the sampling step of sTDEP and its relation to SSCHA and AIMD-TDEP. |
| 3 | MSD section reads like a reply; Au instability implies inapplicability, not mentioned elsewhere | Resolved | Reply-style sentences removed; one U_iso convention. Au: instability shown to come from one-configuration fits with the axial prior; every SOTC-D model of Au with ≥ 4 calls is stable, and the converged FD reference is stable (Sec. V A). |
| 4 | Scope narrowed without QHA/higher-order extension | Resolved | Higher-order FC3/FC4 fits validated (Sec. V G) and used as anharmonic model potentials (Sec. V H); thermal expansion of Al and PbTe computed on volume grids (Sec. V K, Table XIII, SM Table S11). |
| — | Manuscript should read as a new paper | Done | Written as a fresh manuscript: SOTC-C and SOTC-D introduced as two designs of this work; no revision or reply language. Structure follows SSCHA/TDEP method papers: Theory (Sec. II), Implementation (Sec. III), Computational details (Sec. IV), Results (Sec. V). |

## Limitations stated in the manuscript (scientific scope, not open review items)

* At 16 force calls the advantage of SOTC-D over canonical sampling with the
  same trial disappears in about half the crystals (Fig. 1, Sec. V C).
* With two calls, no design determines the 90-parameter Cu model in the
  supercell basis.
* In the crystal-symmetry basis (3–12 parameters) the margins shrink:
  SOTC-D wins in 8–9 of 11 crystals (Table S2).
* Against DFT sTDEP at 16 calls, pooled sTDEP is more accurate for Cu and Mo
  (Table VI); in the crystal-symmetry basis bootstrapped SOTC-D is best in 8
  of 11 crystals at 8 calls and in 6 of 11 at 16 (Table S3).
* On the Mg surrogate, pooled sTDEP is slightly better than SOTC-D-SC at 32
  and 64 calls (0.015/0.011 vs 0.016/0.014 THz), and one-shot canonical
  sampling reaches 0.02 THz with 16 calls against 32 (Table X).
* Fixed amplitudes leave a bias floor of about 0.01 THz; the moment-matched
  variant removes it but raises the variance at small budgets (Sec. V H).
* SSCHA, iterated sTDEP, ZG-SC and SOTC-D-SC with DFT forces were run for Al and Si (600 K) and scored against a converged self-consistent reference (sTDEP, 4 × 128 configurations, two seeds agreeing within 0.03/0.01 THz; Table VII). For Al, SOTC-D-SC is best at 4–32 calls; for Si at 32 calls, ZG-SC (0.014 THz) and pooled sTDEP (0.03 THz) are better than SOTC-D-SC (0.05 THz). python-sscha stops after one population in every DFT run, so its result is the iteration-0 effective FC2. AIMD-TDEP: Al and Si (four trajectories each); fitted jointly with FC3 it lies 0.081 (Al) and 0.055 THz (Si) from the converged SC FC2.

## Verification

Each row above was checked against the compiled text by searching for the cited sentence or equation (script in the session log): all 31 items are present; no reply-style phrases ("earlier analysis", "archived", "published SOTC", "factor of three"), no "compact TDEP", no exponential-decay or representability claim remain.

## Independent coverage audit (after the AIMD-TDEP and neutron additions)

A fresh audit of the current manuscript against all four reports flagged the
items below. Each has been addressed as follows.

| Item | Action |
|---|---|
| A1.1f / A2.1: "all methods use stochastic samples"; auxiliary vs physical phonons | Intro rewritten. Φ_SC is the SCHA auxiliary FC2, not the free-energy-Hessian spectrum (bubble term, Bianco 2017). SCHA can be solved without sampling (self-consistent phonon theory). |
| A1.3: MSD not computed from SOTC models | Al U_iso(300 K) from SOTC-cell models (Sec. V thermo; `al_uiso_models.py`): effective FC2 0.0104, SOTC-D 8 calls 0.0102/0.0104/0.0106 Å², 3–7 % above the tabulated 0.0099. The truncated 32-atom harmonic FD gives 0.0150 (imaginary regions). |
| B4: neutron data not shown | Al neutron BvK models at 10 and 300 K are plotted with the converged and effective dispersions (SM Fig. S4). The 300 K discrepancy and its causes (truncation, DOS-fit non-uniqueness) are discussed. |
| R3.8: branch-resolved convergence vs number of calls; Au | SM Fig. S4: Al and Au at 2/4/8/16 calls. Au SOTC-D from the updated trial has no unstable modes at any budget. |
| Unsupported 4/16-call numbers | SM Tables S7 and S8 (matched designs; crystal-basis self-consistent comparison at 4 and 16 calls). |
| Abstract/conclusion "all crystals" claims | Qualified as supercell basis; crystal-basis counts are in Sec. V F and Table S2/S8. |
| Al 4-vs-256 SSCHA and 4-vs-500 MD claims | Qualified by the 0.1 THz resolution of the reference; called order-of-magnitude. |
| Small inconsistencies | NaCl sTDEP 64; Cu SOTC-C = 8 configurations of a 36-call workflow; NaCl 1.3-fold; surrogate 0.001–0.006 THz; Table III/IV reference difference stated in the caption; broken sentence fixed; SM title matches; MD/quantum rms order stated; trainstation attributed to hiPhive; AIMD thermostat 77 fs; EOS 7 lattice parameters ±3 %, third-order BM. |
| A2.4 / A1.4–5 / B3: QHA removed | Addressed with new calculations (Sec. V K, Table XII). Al: 7 volumes; self-consistent SOTC-D with 16 calls at 100–900 K on two independent grids (independent and common seeds); harmonic QHA in the 32-atom and converged 125-atom cells; α = (∂S/∂V)_T/3B_T. Converged QHA: +4 % at 300/400 K. SC: 0–7 % / 5–17 % below experiment at 300/400 K, reproducible to 500 K. The 32-atom QHA diverges above 500 K. Above 500 K the SC free-energy curvature is not resolved at 16 calls; this is stated as a limitation of the method. |

## Second independent audit (after the thermal-expansion addition)

| Item | Action |
|---|---|
| Methods paragraph said thermal expansion is not computed | Rewritten; points to Sec. V K. |
| QHA baseline unfair (small cell) | Converged 125-atom harmonic QHA added on the same volumes (`dft_qha_large.py`). |
| Non-monotonic α / B_T | 16-call reruns and a second, common-seed grid (`--common-seed`). The high-T non-reproducibility is now quantified and stated; fit forms, energy-term sign, and its bias are given. |
| AIMD-TDEP gap unexplained | Classical-amplitude Gaussian ensemble (40 DFT calls, `dft_classical_gaussian.py`, `classical_gaussian_compare.py`): quantum vs classical Gaussian 0.08 THz. Superseded (Oct 2026): the 0.23 THz classical-Gaussian–AIMD gap was an artefact of fitting FC2 alone to unpaired MD samples; the joint FC2+FC3 fit lies 0.056 THz from the classical Gaussian. |
| SSCHA-256 reference claim inaccurate | Rewritten with actual ranges; 0.23 THz projection effect reported; 4-vs-256 claim qualified in the conclusions. |
| Held-out force counts / 4-call numbers untabulated | Count row added to Table V; SM Table S9 (supercell basis, 4 and 16 calls). |
| MgO 0.40/0.41, 2–16 vs 4–16, "factors of two" | Fixed: 0.395 vs 0.405; budgets harmonized; "up to 2.6× vs random subsets". |
| Acronyms, SCPH citation, test-suite jargon, SM bibliography, Kresch \cite, SM paragraph | Fixed. |
| Raw Stedman–Nilsson points; neutron data for further crystals; LO–TO for SOTC-D in Fig. 3 | Not done; Al neutron BvK and NaCl data are the experimental comparisons. |

## Follow-up to the second audit

| Item | Action |
|---|---|
| Experimental validation beyond Al/NaCl (B4) | Near-neighbour force constants vs neutron BvK fits (Kresch thesis Tables 14.2/14.3), temperature-matched for Al (300 K) and Cu (1200 K, interpolated): SM Table S10, main-text paragraph. Al converged reference within 4 %. Cu effective/SOTC-D 16–18 % stiffer, which is the expected sign since the auxiliary FC2 lacks bubble softening. Ag, Au, Mo, W are not compared because the neutron data are at room temperature and the models at 1000–1500 K. |
| Debye–Waller of Cu/Au from SOTC models (A1.3) | SOTC-D-SC at room-temperature lattice, 300 K, 8 calls: Cu 0.0062, Au 0.0074 Å² (−11 %, −5 %), stable within 3 % from 2 to 8 calls. |
| AIMD/sTDEP fits not with the TDEP code | Stated explicitly (Sec. III A): same estimator (OLS in a symmetry-constrained basis); differences only in basis and optional higher-order terms. |
| 300 K TA(X) stiffness in main text | Numbers moved to the main text with the truncation explanation. |
| Raw Stedman–Nilsson points | Not available to the author; BvK model curves at 10/300 K are plotted instead (SM Fig. S4). |

## Final audit follow-up

| Item | Action |
|---|---|
| Non-monotonic α (B3, A1.4) | Both grids extended to 32 calls per volume and temperature. Up to 400 K: grids agree within 0.6, α monotonic (13.1, 22.8, 23.3), 1–3 %/7–8 % below experiment. Above 400 K: underestimated rise and unresolved curvature, stated in the text. |
| Fit description (A1.5) | Cubic F(V), quadratic S(V), per-atom quantities, E and E0 defined, energy term averaged over the second half and fitted linearly in V. Quartic sensitivity ≤0.3 up to 300 K. |
| Small corrections | Al U_iso 0.0101 (2–7 %); Cu two-call statement; Si "within 0.002 THz"; PBE/SCF/TA/LA/RMSE/rms defined; "neighbor" spelling; SM paragraph lists S14. |

## DFT self-consistent comparison and one-shot estimators (Oct 2026)

| Item | Action |
|---|---|
| SSCHA/sTDEP comparison limited to 0.1 THz reference | Converged SC reference for Al and Si (`dft_sc_protocols.py stdep --budget 512 --per-iter 128`); thermal-metric projection of the full-range SSCHA FC2 (Euclidean projection adds 0.06–0.10 THz); projection-free refit of the SSCHA population. Table VII. |
| One-shot estimators | ZG special displacements and thermal rattle at DFT in all 11 crystals (Tables IV, S-tables) and on the Al/W/NaCl/Si surrogates (SM Table S1); ZG-SC (one ZG pair per iteration) for Al and Si. SOTC-D best in 8/8/9 of 11 crystals at 2/4/8 calls; ZG better for Si and Mg (2 calls), PbTe and Si (4), Si (8). Rattle worst in 9 of 11. |
| ZG implementation | Bloch phase used lattice vector only; fixed to full position (regression test on rock-salt bond fluctuations). Mg, MgO, NaCl, PbTe, Si ZG DFT stages rerun. |
| PbTe thermal expansion | 72-atom cell, 7 volumes, 300/500/700 K, 32 calls: α(300 K) = 16.4e-6/K (14.2/15.7 at 8/16 calls) vs 20.4 measured; unresolved at 500–700 K. |
| AIMD-TDEP fit | Unpaired MD samples have nonzero odd moments, so an FC2-only fit absorbs cubic forces (0.22–0.23 THz). `aimd_tdep_compare.py` and `classical_gaussian_compare.py` now also fit FC2+FC3 (cutoff 4.0 Å). Joint fit: 0.081 (Al) / 0.055 THz (Si) from the converged SC FC2; Al gap is mainly zero-point (0.08), non-Gaussian part ≈0.06 THz. Text in Secs. IV, V D, VI corrected (previously "mostly non-Gaussian"). |

## Independent audit against the PRB reports (Oct 27 2026)

Requests re-extracted from bs15679_report_1_1/1_2/2_1 and reviwer3.md and checked against sotc_jap.tex, the supplement and tables:
TDEP/SSCHA/sTDEP/mode-coupling descriptions (Intro); parameters of all baselines (Sec. IV); MSD/U_iso vs experiment for Al, Cu, Au (Sec. V); trace-averaged correlator and U_iso (Eqs. 4–5, SM S10); rank/conditioning (Table II); neutron BvK and NaCl data (Table S10, SM Fig.); LO–TO with DFPT Born charges (Sec. V); residual forces and PBE-lattice effect (Sec. III A, tab_volume); thermal expansion with BM static energy on seven volumes, Al and PbTe (Sec. V K); random-displacement (rattle) baseline; AIMD/TDEP/SSCHA comparison (Table VII, Sec. V D); converged-reference dispersions and Au stability (Sec. V, Table III). No exponential-decay or "functional" claims remain.
Fixes from this audit: SSCHA DFT scope now "Al and Si" (Intro); eigenvector gauge of Eq. (4) stated (R vs R+τ, the source of the ZG bug); "matches 256-configuration SSCHA with four calls" corrected to 16 calls (Al) and 32 (Si), consistent with Table VII.

Full read-through (Oct 30 2026): sotc_jap.tex (all sections) and sotc_jap_supplementary.tex were read against each request of the four reports. All items are addressed in the text; the section/table/equation references in this map were checked against the compiled .aux. Further fixes from the read-through: Sec. III B said thermal expansion is computed "for Al only" (now Al and PbTe); Sec. V K retitled "Thermal expansion"; R1.5 upgraded from "Removed from scope" to Resolved with evidence.
Data checks (Oct 30 2026): Au SOTC-D unstable fraction 0.19 % at 2 calls and 0 at 4–16 calls for both trials and both bases (au design_comparison.json), matching Sec. V C; Cu SOTC-C archive has 4 iterations × 8 displaced + 4 references = 36 calls, matching Table I; the referee's "rank 162" is the rank of the old per-pair tensor parametrization (Al 81), explained by Cu's extra third shell and N_ens = 8 (Sec. V A).
Numbers behind referee-driven claims, checked against result files (Oct 30 2026):
- U_iso (R1.3, R2.1): robustness_summary.json (shifted 24³ mesh) Al 0.01036, Cu 0.00657, Au 0.00805, Ag 0.00957, Mo 0.00288, W 0.00217 Å²; al_uiso_models.json Al SOTC-D 8 calls 0.0101/0.0104/0.0106, cell FD 0.0150; sotc_uiso_room.json Cu 0.0062, Au 0.0074 (cell FD 0.0064/0.0088, 2→8 calls < 2 %). All match Sec. V J.
- Al α (R1.4–5, R2.3): qha_thermal_expansion{,_common}.json SC 32 calls 22.5/23.0 (300 K), 23.1/23.4 (400 K); converged QHA 24.1/26.2/40.2. Match Table XIII and text means.
- PbTe α: qha_thermal_expansion_common.json 14.2/15.7/16.4 (8/16/32 calls), quartic 15.8, no-ΔU 16.4.
- LO–TO (R2.5): DFPT Z*/ε∞ NaCl 1.111/2.471, MgO 1.973/3.167, PbTe 5.767/28.83; SOTC-C splittings 2.29/8.05/1.32 THz (production_sotc_nac_summary.json); NaCl FD TO/LO 4.37/7.26 THz, −11 %/−10 %, LO/TO ratio +1.7 % (nacl_loto_plot_summary.json). Match Sec. V L.
- Lines 1222–1295 of sotc_jap.tex (Sec. V K body) read; the whole main text and supplement have now been read.
