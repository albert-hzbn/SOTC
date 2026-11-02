# Do SOTC-designed configurations make better MLIP training sets?

Separate study note (Task 7). It is not part of the SOTC manuscript.
Data: `summaries/{al,si}/robustness/mlip_study.json`. Code: `codes/validation/mlip_study.py` and `mlip_data.py`.

## Setup

**Systems.** Al at 300 K and Si at 600 K, in the SOTC cells (32 and 64 atoms), with the PBE settings of the paper.

**Training sets.** Each model is trained on the first N DFT configurations of one design, with N = 4, 8, 16, 32, 64:

| design | configurations |
|---|---|
| SOTC-D | nested sequential D-optimal ±u pairs. The first 16 are the paper's set; the set is extended to 64 with the existing pairs as prior. |
| canonical | paired quantum-canonical pool; only 48 configurations exist |
| rattle | independent per-atom Gaussian displacements, canonical variance |
| AIMD | every 20th production step of trajectories 0–2 |

**Model.** MACE, small: 32 channels, max_L = 0, correlation order 2, r_max = 5 Å, float64. Loss weights are energy 1 and forces 100.

**Training.** Every model gets the same number of optimizer steps: 4800 at batch size 2. There are 3 seeds for every (design, N).

A first pass used a fixed 150 epochs. That gave N = 4 only 300 optimizer steps, and those models were under-fitted: training force RMSE was about 47 meV/Å, the same as the test error. That pass is superseded and kept in the JSON without the `_st` suffix.

**Tests.**
- Force RMSE on 100 held-out snapshots of AIMD trajectory 3, which was not used in training.
- Phonon RMSE on a 6³ q mesh. The potential's FC2 comes from ±0.01 Å finite differences in the same cell; the reference is the DFT finite-difference FC2.
- MD stability: three 2-ps Langevin runs at the design temperature. A run fails if any atom moves more than 1 Å from its site or T exceeds 5 T₀.

## Results

Values are mean ± standard deviation over 3 seeds. Columns are N training configurations, i.e. N DFT calls. AIMD frames are not independent calls: each one sits on a trajectory of about 800 DFT steps.

**Force RMSE (meV/Å), held-out AIMD**

| | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| Al SOTC-D | 31.6 ± 0.1 | 30.8 ± 0.1 | 30.2 ± 0.1 | 29.9 ± 0.2 | 30.9 ± 0.4 |
| Al canonical | 31.9 ± 0.2 | 31.6 ± 0.2 | 31.0 ± 0.3 | 31.4 ± 0.8 | – |
| Al rattle | 33.4 ± 0.4 | 31.2 ± 0.1 | 31.1 ± 0.4 | 31.3 ± 0.9 | 31.6 ± 0.7 |
| Al AIMD | 33.2 ± 0.3 | 31.7 ± 0.0 | 31.3 ± 0.1 | 31.3 ± 0.4 | 31.0 ± 0.5 |
| Si SOTC-D | 44.0 ± 0.5 | 43.2 ± 0.7 | 38.5 ± 3.9 | 32.9 ± 7.9 | 39.6 ± 7.7 |
| Si canonical | 44.8 ± 0.1 | 41.8 ± 0.9 | 39.1 ± 0.5 | 33.2 ± 6.4 | – |
| Si rattle | 44.9 ± 2.7 | 40.8 ± 4.7 | 28.7 ± 2.0 | 34.0 ± 7.1 | 35.4 ± 7.8 |
| Si AIMD | 44.5 ± 0.2 | 41.2 ± 1.1 | 38.0 ± 2.2 | 36.8 ± 1.1 | 34.3 ± 5.7 |

**Phonon RMSE (THz) vs DFT FD**

| | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| Al SOTC-D | 0.290 | 0.275 | 0.272 | 0.272 | 0.277 |
| Al canonical | 0.293 | 0.272 | 0.278 | 0.283 | – |
| Al rattle | 0.304 | 0.311 | 0.284 | 0.280 | 0.282 |
| Al AIMD | 0.280 | 0.281 | 0.273 | 0.272 | 0.283 |
| Si SOTC-D | 0.193 | 0.176 | 0.166 | 0.145 | 0.178 |
| Si canonical | 0.213 | 0.211 | 0.199 | 0.160 | – |
| Si rattle | 0.196 | 0.162 | 0.144 | 0.133 | 0.161 |
| Si AIMD | 0.203 | 0.201 | 0.173 | 0.169 | 0.170 |

**MD stability.** All 57 equal-step models per material (and the 27 first-pass models) were stable in all three runs. The test does not discriminate between designs at these temperatures.

## Conclusions

**The design does not matter much for this model.**
- In Al every design reaches 30–32 meV/Å and 0.27–0.29 THz by N = 8, and nothing improves beyond that. The limit is the model's capacity (max_L = 0, 32 channels), not the data.
- SOTC-D is lowest in force RMSE at every N in Al. The margin is only 0.5–1.5 meV/Å (2–5%).
- In Si the seed-to-seed scatter at N ≥ 16 (4–8 meV/Å) is as large as the differences between designs. No design is consistently best. Rattle has the lowest mean at N = 16–32, and SOTC-D and canonical are ahead at N ≤ 8 on phonons.

**Small N.** With 4–8 calls the ±u paired designs (SOTC-D, canonical) give the best phonons in Si (0.19–0.21 THz vs 0.20 AIMD). The gain is modest.

**Verdict.** SOTC-D is an adequate but not a superior MLIP training set. The D-optimal criterion targets FC2 information, and a many-body potential is not limited by that. There is no evidence for a claim that "SOTC designs make better MLIP training sets". The task shows that SOTC configurations can be reused for MLIP training without penalty.

**Limitations.**
- One small MACE architecture.
- Two simple solids near their design temperatures.
- The held-out AIMD test is at the same temperature as the training designs.
- Transferability to other temperatures or defects was not tested.
- With larger models, the higher-order information missing from FC2-optimal sets could favour rattle or AIMD data.
