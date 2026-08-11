# Real-material FC3/FC4 campaign audit

Generated from the machine-readable campaign archives by codes/validation/summarize_real_anharmonic_ifcs.py.

| Material | T (K) | atoms | parameters/rank | condition | FD FC2 | T-fit FC2 | T-fit FC2+3 | T-fit FC2+3+4 | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fcc Al | 300 | 27 | 106/106 | 312.7 | 0.102940 | 0.099546 | 0.027206 | 0.006508 | 0.999740 |
| rocksalt NaCl | 700 | 54 | 48/48 | 36.9 | 0.354316 | 0.331517 | 0.189062 | 0.098051 | 0.969066 |
| bcc W | 1500 | 27 | 126/126 | 269.4 | 0.298157 | 0.275658 | 0.086342 | 0.037296 | 0.999425 |

RMSE values are per Cartesian force component in eV/angstrom on 20 strictly held-out snapshots. FD FC2 is the conventional central finite-displacement phonopy model generated with matched QE settings; the other columns are temperature-fitted symfc models. Complete inversion pairs remain wholly inside either the 80-snapshot training set or the test set.

All campaigns passed the archive, finiteness, exact-pairing, symmetry-reduced full-rank, and acoustic-sum-rule checks. The higher-order workflow uses 100 canonical snapshots and is independent of the original few-configuration SOTC sampling claim.

Interpretive cautions:

- Al required the 5.0-angstrom FC2/FC3 cutoff selected by its held-out cutoff test; its denser-k-mesh force shift remains larger than the final model RMSE.
- NaCl has the largest canonical displacements and the weakest final held-out score; the model is useful evidence for higher-order force content but not a claim of converged transport properties.
- Transport-ready file formatting and tensor conventions are tested, but thermal conductivity, linewidths, and free energies were not computed in this campaign.
