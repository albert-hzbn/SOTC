#!/usr/bin/env python3
"""Generate the LaTeX tables of the JAP manuscript from the result archives.

Every number in the tables is read from
``validation_results/<material>/{primary_audit,robustness}/*.json`` so that
the manuscript cannot drift from the data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript" / "jap" / "tables"
MATERIALS = ["al", "ag", "au", "cu", "mo", "w", "nacl", "mgo", "pbte"]
ALL_MATERIALS = MATERIALS + ["si", "mg"]  # with the new diamond and hcp classes
LABELS = {"al": "Al", "ag": "Ag", "au": "Au", "cu": "Cu", "mo": "Mo",
          "w": "W", "nacl": "NaCl", "mgo": "MgO", "pbte": "PbTe", "si": "Si",
          "mg": "Mg"}
RUNS = {  # a (A), T_des (K), r_cut (A), N_iter, N_ens (Table I of the runs)
    "al": (4.05, 300, 4.5, 2, 5), "ag": (4.12, 1000, 4.5, 4, 5),
    "au": (4.13, 1200, 4.5, 2, 1), "cu": (3.67, 1200, 4.5, 4, 8),
    "mo": (3.16, 1500, 5.0, 2, 1), "w": (3.18, 1500, 5.0, 4, 3),
    "nacl": (5.73, 700, 4.3, 5, 3), "mgo": (4.26, 1500, 3.2, 4, 3),
    "pbte": (6.51, 700, 5.0, 4, 5),
}
# Peng et al., Acta Cryst. A 52, 456 (1996), supplementary Table 1, 293 K.
PENG_B_293 = {"al": 0.7806, "cu": 0.5505, "ag": 0.7341, "au": 0.6198,
              "mo": 0.2155, "w": 0.1596}


def load(material, kind):
    name = {"audit": ("primary_audit", "primary_audit.json"),
            "rob": ("robustness", "robustness_summary.json")}[kind]
    path = ROOT / "validation_results" / material / name[0] / name[1]
    return json.loads(path.read_text()) if path.exists() else None


def model(report, name):
    for row in report["models"]:
        if row["model"] == name:
            return row
    return None


def fmt(value, digits=2):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    return f"{value:.{digits}f}"


def runs_table():
    lines = [
        r"\begin{table}[t]",
        r"\caption{Correlator-matched SOTC runs (SOTC-C). $a$: lattice parameter; $n$: supercell atoms;"
        r" $N_{\rm iter}$, $N_{\rm ens}$: iterations and configurations per"
        r" iteration; $N_{\rm force}=N_{\rm iter}(N_{\rm ens}+1)$ QE calls;"
        r" $p$: parameters of the space-group adapted FC2 basis within"
        r" $r_{\rm cut}$ in the SOTC cell.}",
        r"\label{tab:runs}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{lcccccccc}",
        r" & $a$ (\AA) & $T_{\rm des}$ (K) & $n$ & $r_{\rm cut}$ (\AA) &"
        r" $N_{\rm iter}$ & $N_{\rm ens}$ & $N_{\rm force}$ & $p$\\",
        r"\hline",
    ]
    for m in MATERIALS:
        a, t, rc, ni, ne = RUNS[m]
        audit = load(m, "audit")
        n = audit["n_atoms"] if audit else "--"
        p = "--"
        if audit:
            design = audit["pooled"]["variants"]["space_group_symfc"].get("design")
            p = design["n_parameters"] if design else "--"
        lines.append(f"{LABELS[m]} & {a:.2f} & {t} & {n} & {rc:.1f} & {ni} & {ne}"
                     f" & {ni * (ne + 1)} & {p}\\\\")
    lines += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table}"]
    return "\n".join(lines)


def accuracy_table():
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Phonon-mesh RMSE (THz, $12^3$ mesh) of FC2 models for the"
        r" nine benchmarks. Columns 2--4 are measured against the converged"
        r" large-cell harmonic FD reference; columns 5--8 against the effective"
        r" FC2 fitted to 40 canonical snapshots at $T_{\rm des}$ in the SOTC cell."
        r" ``SOTC-C'' is the correlator-matched model (selected iteration, axial prior);"
        r" ``SOTC-D'' is the D-optimal antithetic design with the space-group"
        r" adapted fit at 8 force calls. The last column is the RMSE between"
        r" the harmonic and effective models (finite-temperature renormalization).}",
        r"\label{tab:accuracy}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{lccccccc}",
        r" & \multicolumn{3}{c}{vs converged harmonic FD} &"
        r" \multicolumn{3}{c}{vs same-cell effective FC2} & \\",
        r" & FD $3^3$ & FD SOTC cell & SOTC-C & SOTC-C & SOTC-C pooled & SOTC-D (8) &"
        r" harmonic--effective\\",
        r"\hline",
    ]
    for m in MATERIALS:
        rob = load(m, "rob")
        if rob is None:
            continue
        def r(name, key):
            row = model(rob, name)
            return row[key]["rmse_thz"] if row and key in row else None
        sotcd = None
        cs = rob.get("count_series", {}).get("symfc", {}).get("sotcd")
        if cs:
            sotcd = cs[rob["count_series"]["counts"].index(8)]["mesh_vs_effective"]
        lines.append(" & ".join([
            LABELS[m], fmt(r("fd_3x3x3", "mesh_vs_fd_large")),
            fmt(r("fd_same_cell", "mesh_vs_fd_large")),
            fmt(r("sotc_selected", "mesh_vs_fd_large")),
            fmt(r("sotc_selected", "mesh_vs_effective")),
            fmt(r("sotc_pooled", "mesh_vs_effective")), fmt(sotcd),
            fmt(r("fd_large", "mesh_vs_effective")),
        ]) + r"\\")
    for m in ("si", "mg"):  # new classes: no SOTC-C run
        path = ROOT / "validation_results" / m / "robustness" / "new_class_summary.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())["mesh_rmse_thz"]
        lines.append(" & ".join([
            LABELS[m], "--", fmt(r["fd_same_cell_vs_fd_large"]), "--", "--", "--",
            fmt(r["sotcd_8_vs_effective_iter0"]), fmt(r["effective_iter0_vs_fd_large"]),
        ]) + r"\\")
    lines += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table*}"]
    return "\n".join(lines)


def design_report(material):
    path = ROOT / "validation_results" / material / "robustness" / "design_comparison.json"
    return json.loads(path.read_text()) if path.exists() else None


def median_of(trials, key):
    return float(np.median([t[key] for t in trials]))


def bold_min(values, digits, cap=None):
    """Format a row group, bold the minimum; values above ``cap`` (singular,
    unidentified fits) are printed as >cap."""
    texts = [r"$>$" + f"{cap:g}" if cap is not None and v is not None and v > cap
             else fmt(v, digits) for v in values]
    finite = [v if v is not None else np.inf for v in values]
    best = int(np.argmin(finite))
    texts[best] = r"\textbf{" + texts[best] + "}"
    return texts


def matched_table(counts=(2, 8)):
    """Iteration 0: every design from the same-cell harmonic FD trial."""
    kinds = ["D", "C", "P", "Z", "R"]
    head = " & ".join(
        [r"\multicolumn{5}{c}{" + f"{c} calls" + "}" for c in counts])
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Matched comparison in the SOTC cell at equal numbers of"
        r" displaced force calls, all designed from the same-cell harmonic FD"
        r" trial and fitted in the supercell space-group basis. D: SOTC-D;"
        r" C: random canonical subsets; P: random canonical $\pm\mathbf u$ pairs"
        r" (medians over 20 draws); Z: special displacements with Walsh sign"
        r" patterns (ZG); R: thermal rattle (independent Gaussian atoms with the"
        r" canonical per-atom variance). Upper block: held-out force RMSE on 8"
        r" canonical test snapshots relative to that of the 40-snapshot"
        r" effective FC2. Lower block: mesh RMSE (THz) against an effective FC2"
        r" fitted to $40-n$ canonical snapshots disjoint from the scored design."
        r" The best entry is bold; the last row counts the materials in which D"
        r" is best. Entries $>$10~THz ($>$100 in force) are singular fits of"
        r" designs that do not identify the model.}",
        r"\label{tab:matched}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{l" + "c" * 5 * len(counts) + "}",
        " & " + head + r"\\",
        " & " + " & ".join([" & ".join(kinds)] * len(counts)) + r"\\",
        r"\hline",
    ]
    blocks = {"force": [], "mesh": []}
    wins = {(b, c): 0 for b in blocks for c in counts}
    for m in ALL_MATERIALS:
        rep = design_report(m)
        if rep is None:
            continue
        e = rep["targets"]["iter0"]["supercell"]
        floor = e["reference_held_out_force_rmse"]
        for key, block in (("held_out_force_rmse", "force"), ("mesh_vs_disjoint", "mesh")):
            cells = []
            for c in counts:
                c = str(c)
                vals = [e["designs"]["sotcd"][c][key],
                        median_of(e["canonical_unpaired"][c], key),
                        median_of(e["canonical_paired"][c], key),
                        e["designs"]["zg"][c][key], e["designs"]["rattle"][c][key]]
                if block == "force":
                    vals = [v / floor for v in vals]
                wins[(block, int(c))] += int(np.argmin(vals) == 0)
                cells += bold_min(vals, 3 if block == "force" else 2,
                                  cap=100 if block == "force" else 10)
            blocks[block].append(LABELS[m] + " & " + " & ".join(cells) + r"\\")
    n = len(blocks["mesh"])
    for block, title in (("force", "Held-out force RMSE / floor"),
                         ("mesh", "Mesh RMSE vs disjoint effective FC2 (THz)")):
        lines.append(r"\multicolumn{" + str(1 + 5 * len(counts)) + r"}{l}{" + title + r"}\\")
        lines += blocks[block]
        lines.append(r"D best & " + " & ".join(
            r"\multicolumn{5}{c}{" + f"{wins[(block, c)]}/{n}" + "}" for c in counts) + r"\\")
        lines.append(r"\hline")
    lines.pop()
    lines += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table*}"]
    return "\n".join(lines)


def selfconsistent_table(counts=(2, 8), basis="supercell"):
    """Iteration 1: designs from the updated trial, and the equal-cost test."""
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Updated trial (iteration 1, " + basis + r" basis). Mesh RMSE (THz)"
        r" against the iteration-1 effective FC2. Matched trial: SOTC-D designed"
        r" from the iteration-1 trial (D$_1$) versus the better of random"
        r" canonical subsets and pairs drawn from the same trial (C$_1$), scored"
        r" against the disjoint effective FC2. Equal cost: bootstrapped SOTC-D"
        r" (B; first pair from the harmonic trial, later pairs from the model"
        r" refitted to the first pair, no extra force calls) and"
        r" one-shot SOTC-D (D$_0$) versus canonical pairs from the harmonic trial"
        r" (C$_0$; a single sTDEP iteration), scored against the 40-snapshot iteration-1"
        r" effective FC2, which shares no configuration with them. B and D$_0$"
        r" coincide at 2 calls by construction. The best entry of each group is"
        r" bold.}",
        r"\label{tab:selfconsistent" + ("" if basis == "supercell" else "_" + basis) + "}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{l" + "ccccc" * len(counts) + "}",
        " & " + " & ".join(
            r"\multicolumn{2}{c}{matched, " + f"{c}" + r"} & \multicolumn{3}{c}{equal cost, "
            + f"{c}" + "}" for c in counts) + r"\\",
        " & " + " & ".join(["D$_1$ & C$_1$ & B & D$_0$ & C$_0$"] * len(counts)) + r"\\",
        r"\hline",
    ]
    wins = {(g, c): 0 for g in ("m", "e") for c in counts}
    n = 0
    for m in ALL_MATERIALS:
        rep = design_report(m)
        if rep is None or "iter1" not in rep["targets"]:
            continue
        n += 1
        e = rep["targets"]["iter1"][basis]
        cells = []
        for c in counts:
            c = str(c)
            matched = [e["designs"]["sotcd1"][c]["mesh_vs_disjoint"],
                       min(median_of(e["canonical_paired"][c], "mesh_vs_disjoint"),
                           median_of(e["canonical_unpaired"][c], "mesh_vs_disjoint"))]
            equal = [e["designs"]["boot"][c]["mesh_vs_full_reference"],
                     e["designs"]["sotcd"][c]["mesh_vs_full_reference"],
                     median_of(e["other_iteration_canonical"][c], "mesh_vs_full_reference")]
            wins[("m", int(c))] += int(np.argmin(matched) == 0)
            wins[("e", int(c))] += int(equal[0] < equal[2])
            cells += bold_min(matched, 2) + bold_min(equal, 2)
        lines.append(LABELS[m] + " & " + " & ".join(cells) + r"\\")
    lines.append(r"\hline")
    lines.append(r"D$_1<$C$_1$ / B$<$C$_0$ & " + " & ".join(
        r"\multicolumn{2}{c}{" + f"{wins[('m', c)]}/{n}" + r"} & \multicolumn{3}{c}{"
        + f"{wins[('e', c)]}/{n}" + "}" for c in counts) + r"\\")
    lines += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table*}"]
    return "\n".join(lines)


def calls_to_match(e, target, kind="canonical_paired", key="mesh_vs_disjoint"):
    for c in ("2", "4", "8", "16"):
        if median_of(e[kind][c], key) <= target:
            return c
    return r"$>$16"


def cost_table():
    """Measured QE cost per force call and the canonical budget that matches
    SOTC-D (matched iteration-1 trial, supercell basis)."""
    lines = [
        r"\begin{table}[t]",
        r"\caption{Cost. $t$: measured cost per force call in the SOTC cell"
        r" (core-hours, 32 cores, mean over all same-cell calls); $N_{\rm FD}$:"
        r" calls of the harmonic FD trial. Columns 4--5: number of random"
        r" canonical pairs (force calls) from the same trial needed to reach the"
        r" iteration-1 mesh error that SOTC-D attains with 2 and 8 calls.}",
        r"\label{tab:cost}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{lcccc}",
        r" & $t$ (core-h) & $N_{\rm FD}$ & match D$_1$(2) & match D$_1$(8)\\",
        r"\hline",
    ]
    for m in ALL_MATERIALS:
        rep = design_report(m)
        if rep is None or "iter1" not in rep["targets"]:
            continue
        base = ROOT / "validation_results" / m / "primary_cell"
        walls = [float(np.load(f)["wall_seconds"]) for f in
                 base.glob("*/state_*/result.npz")] + \
                [float(np.load(f)["wall_seconds"]) for f in
                 base.glob("*/*/state_*/result.npz")]
        n_fd = len(list((base / "fd").glob("**/result.npz"))) or len(
            list((base / "fd").glob("**/espresso.pwo")))
        e = rep["targets"]["iter1"]["supercell"]
        d = e["designs"]["sotcd1"]
        lines.append(" & ".join([
            LABELS[m], f"{32 * np.mean(walls) / 3600:.2f}", str(n_fd) if n_fd else "--",
            calls_to_match(e, d["2"]["mesh_vs_disjoint"]),
            calls_to_match(e, d["8"]["mesh_vs_disjoint"])]) + r"\\")
    lines += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table}"]
    return "\n".join(lines)


def basis_table():
    lines = [
        r"\begin{table}[t]",
        r"\caption{Supercell versus crystal space-group FC2 basis: number of"
        r" parameters $p$, alias-free crystal-basis cutoff $r_c$, held-out force"
        r" RMSE (eV/\AA) of the 40-snapshot iteration-1 effective FC2, and mesh"
        r" RMSE (THz) between the two effective FC2s.}",
        r"\label{tab:basis}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{lcccccc}",
        r" & $p_{\rm sc}$ & $p_{\rm cr}$ & $r_c$ (\AA) & $F_{\rm sc}$ & $F_{\rm cr}$"
        r" & $\Delta\omega$\\",
        r"\hline",
    ]
    for m in ALL_MATERIALS:
        rep = design_report(m)
        if rep is None or "iter1" not in rep["targets"]:
            continue
        t = rep["targets"]["iter1"]
        lines.append(" & ".join([
            LABELS[m], str(rep["n_parameters"]["supercell"]), str(rep["n_parameters"]["crystal"]),
            f"{rep['crystal_cutoff_ang']:.2f}",
            f"{t['supercell']['reference_held_out_force_rmse']:.3f}",
            f"{t['crystal']['reference_held_out_force_rmse']:.3f}",
            fmt(rep.get("basis_difference_thz"))]) + r"\\")
    lines += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table}"]
    return "\n".join(lines)


def fitter_table(count="8"):
    path = lambda m: ROOT / "validation_results" / m / "robustness" / "fitter_comparison.json"
    reports = {m: json.loads(path(m).read_text()) for m in ALL_MATERIALS if path(m).exists()}
    if not reports:
        return ""
    methods = next(iter(reports.values()))["fit_methods"]
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Regression frameworks (hiPhive cluster space with crystal"
        r" symmetry, trainstation fit methods) on the same DFT data, " + count +
        r" force calls. Entries: mesh RMSE (THz) against the disjoint effective"
        r" FC2 (supercell basis, so the entries include the basis difference"
        r" listed in the main text) for random canonical subsets (median of 10)"
        r" / SOTC-D.}",
        r"\label{tab:fitters}",
        r"\footnotesize",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{l" + "c" * len(reports) + "}",
        "Method & " + " & ".join(LABELS[m] for m in reports) + r"\\",
        r"\hline",
    ]
    for method in methods:
        cells = []
        for m, r in reports.items():
            trials = [t for t in r["canonical"][method][count] if "error" not in t]
            s = r["sotcd"][method][count]
            c = median_of(trials, "mesh_vs_disjoint") if trials else None
            cells.append(f"{fmt(c)}/{fmt(s.get('mesh_vs_disjoint'))}")
        lines.append(method + " & " + " & ".join(cells) + r"\\")
    lines += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table*}"]
    return "\n".join(lines)


SC_SURROGATES = ("al", "w", "nacl", "si", "mg")


def sscha_curve(r):
    """(median force calls, median mesh RMSE) per SSCHA population size; when
    several population sizes need the same number of calls, the best one."""
    groups = {}
    for x in r["results"].get("sscha", []):
        groups.setdefault(x["n_configs"], []).append(x)
    best = {}
    for g in groups.values():
        calls = int(np.median([x["force_calls"] for x in g]))
        err = float(np.median([x["mesh_rmse_thz"] for x in g]))
        best[calls] = min(err, best.get(calls, np.inf))
    return sorted(best.items())


def stdep_table(counts=(8, 16), basis="supercell"):
    """DFT forces, equal cost: bootstrapped SOTC-D against two-iteration
    sTDEP and one-shot canonical pairs, all from the harmonic trial."""
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Self-consistent protocols with DFT forces, equal cost. Mesh"
        r" RMSE (THz) against the 40-snapshot iteration-1 effective FC2 (" + basis +
        r" basis). B: bootstrapped SOTC-D. sTDEP: half of the calls as canonical"
        r" $\pm$ pairs from the harmonic trial, half as canonical $\pm$ pairs"
        r" from the model refitted to them, fitted to the second iteration only"
        r" (last) or to both (pool); median of three seeds. C$_0$: one-shot"
        r" canonical pairs from the harmonic trial (median of 20 subsets). No"
        r" configuration of any protocol enters the reference. The best entry"
        r" of each group is bold.}",
        r"\label{tab:stdep" + ("" if basis == "supercell" else "_" + basis) + "}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{l" + "cccc" * len(counts) + "}",
        " & " + " & ".join(r"\multicolumn{4}{c}{" + f"{c} calls" + "}" for c in counts) + r"\\",
        " & " + " & ".join(["B & sTDEP last & sTDEP pool & C$_0$"] * len(counts)) + r"\\",
        r"\hline",
    ]
    wins = {c: 0 for c in counts}
    n = 0
    for m in ALL_MATERIALS:
        rep = design_report(m)
        if rep is None or "iter1" not in rep["targets"]:
            continue
        e = rep["targets"]["iter1"][basis]
        if "stdep" not in e:
            continue
        n += 1
        cells = []
        for c in counts:
            k = str(c)
            vals = [e["designs"]["boot"][k]["mesh_vs_full_reference"],
                    median_of(e["stdep"]["stdep_last"][k], "mesh_vs_full_reference"),
                    median_of(e["stdep"]["stdep_pool"][k], "mesh_vs_full_reference"),
                    median_of(e["other_iteration_canonical"][k], "mesh_vs_full_reference")]
            wins[c] += int(np.argmin(vals) == 0)
            cells += bold_min(vals, 2)
        lines.append(LABELS[m] + " & " + " & ".join(cells) + r"\\")
    lines.append(r"\hline")
    lines.append("B best & " + " & ".join(
        r"\multicolumn{4}{c}{" + f"{wins[c]}/{n}" + "}" for c in counts) + r"\\")
    lines += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table*}"]
    return "\n".join(lines)


def sc_surrogate_table(budgets=("8", "16", "32", "64")):
    protocols = [("canonical_oneshot", "Canonical $\\pm$, one shot"),
                 ("sotcd_oneshot", "SOTC-D, one shot"),
                 ("stdep_last", "sTDEP (last iteration)"),
                 ("stdep_pool", "sTDEP (pooled)"),
                 ("sotcd_sc", "SOTC-D-SC")]
    data = {}
    for m in SC_SURROGATES:
        path = ROOT / "validation_results" / m / "surrogate_design" / "surrogate_self_consistent.json"
        if path.exists():
            data[m] = json.loads(path.read_text())
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Self-consistent protocols on the anharmonic surrogates."
        r" Median mesh RMSE (THz) against the self-consistent effective FC2"
        r" (= SCHA auxiliary FC2) at " + "/".join(budgets) + r" force calls (12"
        r" repetitions). SSCHA: python-sscha with the surrogate as calculator,"
        r" populations of $N=8$--128 configurations until convergence (5"
        r" repetitions), listed at the median total number of force calls (the better population size where two need the same number).}",
        r"\label{tab:sc_surrogate}",
        r"\begin{ruledtabular}", r"\footnotesize",
        r"\begin{tabular}{l" + "c" * len(data) + "}",
        "Protocol & " + " & ".join(LABELS[m] for m in data) + r"\\",
        r"\hline",
    ]
    for key, label in protocols:
        cells = [label]
        for m, r in data.items():
            rows = r["results"][key]
            cells.append("/".join(f"{np.median([x['mesh_rmse_thz'] for x in rows[b]]):.3f}"[1:]
                                  for b in budgets))
        lines.append(" & ".join(cells) + r"\\")
    cells, notes = ["SSCHA"], []
    for m, r in data.items():
        err = dict(sscha_curve(r))
        text = "/".join(f"{err[int(b)]:.3f}"[1:] if int(b) in err else "--" for b in budgets)
        extra = [f"{c} calls, {e:.3f}" for c, e in sorted(err.items()) if c > int(budgets[-1])
                 or (str(c) not in budgets and c > int(budgets[0]))]
        cells.append(text)
        if extra:
            notes.append(f"{LABELS[m]} " + "; ".join(extra))
    lines.append(" & ".join(cells) + r"\\")
    if notes:
        lines[1] = lines[1][:-1] + " Other SSCHA budgets: " + "; ".join(notes) + " THz.}"
    lines.append(r"\hline")
    lines.append("Harmonic vs self-consistent & " + " & ".join(
        f"{r['harmonic_vs_sc_mesh_thz']:.3f}" for r in data.values()) + r"\\")
    lines += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table*}"]
    return "\n".join(lines)


def audit_table():
    lines = [
        r"\begin{table}[t]",
        r"\caption{Identifiability and priors of the SOTC-C force data. Rank/$p$ and the"
        r" largest condition number over all iterations refer to the"
        r" unregularized space-group adapted FC2 design of each single"
        r" iteration. $f_{\rm unst}$ is the fraction of unstable"
        r" $12^3$-mesh modes, with the axial prior and without the axial"
        r" projection, in the worst iteration. $\Delta_\lambda$ and"
        r" $\Delta_r$ are the ranges (THz) of the mesh RMSE against the"
        r" converged FD reference over $\lambda\in[0,0.1]$ and"
        r" $r_{\rm cut}\pm0.5$~\AA\ for the pooled data.}",
        r"\label{tab:audit}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{lccccc}",
        r" & rank/$p$ & cond. & $f_{\rm unst}$ axial & $f_{\rm unst}$ no proj."
        r" & $\Delta_\lambda$, $\Delta_r$\\",
        r"\hline",
    ]
    for m in MATERIALS:
        audit = load(m, "audit")
        if audit is None:
            continue
        its = audit["iterations"]
        designs = [it["variants"]["space_group_symfc"]["design"] for it in its]
        rank_ok = all(d["full_rank"] for d in designs)
        p = designs[0]["n_parameters"]
        cond = max(d["normalized_design_condition_number"] for d in designs)
        f_pub = max(it["variants"]["published_prior"]["mesh"]["unstable_fraction"] for it in its)
        f_np = max(it["variants"]["no_projection_ridge_0.001"]["mesh"]["unstable_fraction"]
                   for it in its)
        v = audit["pooled"]["variants"]
        def rm(name):
            return v[name].get("dispersion_vs_fd", {}).get("rmse_thz")
        ridge = [rm(k) for k in v if k.startswith("central_ridge") or k == "published_prior"]
        cut = [rm(k) for k in v if k.startswith("central_cutoff")] + [rm("published_prior")]
        if None in ridge or None in cut:
            spread = "--"
        else:
            spread = f"{max(ridge) - min(ridge):.2f}, {max(cut) - min(cut):.2f}"
        lines.append(" & ".join([
            LABELS[m], f"{p if rank_ok else 'def.'}/{p}", f"{cond:.1f}",
            f"{f_pub:.3f}", f"{f_np:.3f}", spread,
        ]) + r"\\")
    lines += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table}"]
    return "\n".join(lines)


def thermo_table():
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Vibrational thermodynamics per atom at $T_{\rm des}$ and"
        r" displacement parameters. $S_{\rm vib}$ (J\,mol$^{-1}$K$^{-1}$),"
        r" $F_{\rm vib}$ (kJ\,mol$^{-1}$), and $T_D$ (K) from the converged"
        r" harmonic FD (FD), the same-cell effective FC2 (eff.), and the"
        r" SOTC-C model. $U_{\rm iso}(300\,{\rm K})$ (\AA$^2$) from the"
        r" converged harmonic model at the room-temperature (Al, Cu, Au) or"
        r" production lattice, compared with Peng \textit{et al.}\ \cite{ref55}"
        r" at 293~K.}",
        r"\label{tab:thermo}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{lccccccccccc}",
        r" & \multicolumn{3}{c}{$S_{\rm vib}$} & \multicolumn{3}{c}{$F_{\rm vib}$}"
        r" & \multicolumn{3}{c}{$T_D$} & \multicolumn{2}{c}{$U_{\rm iso}(300\,{\rm K})$}\\",
        r" & FD & eff. & SOTC-C & FD & eff. & SOTC-C & FD & eff. & SOTC-C & calc. & Ref.~\cite{ref55}\\",
        r"\hline",
    ]
    for m in MATERIALS:
        rob = load(m, "rob")
        if rob is None:
            continue
        def th(name, key):
            row = model(rob, name)
            return row["thermo_T_des"][key] if row else None
        uiso = rob["u_iso_ang2"]["values_by_atom"]
        u = None
        for key in ("fd_room_large", "fd_large"):
            if key in uiso:
                u = float(np.mean(uiso[key][0]))
                break
        peng = PENG_B_293.get(m)
        cells = [LABELS[m]]
        for key, digits in (("s_j_mol_atom_k", 1), ("fvib_kj_mol_atom", 1),
                            ("t_debye_spectral_k", 0)):
            cells += [fmt(th(n, key), digits) for n in
                      ("fd_large", "effective_reference_symfc_40", "sotc_selected")]
        cells += [fmt(u, 4), fmt(peng / (8 * np.pi ** 2), 4) if peng else "--"]
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table*}"]
    return "\n".join(lines)


def surrogate_table():
    kinds = [("canonical", "Canonical"), ("canonical_paired", r"Canonical $\pm$"),
             ("rademacher_random", "Exact ampl., random signs"),
             ("sotcd", "SOTC-D"), ("sotcdq", "Stratified ampl. + D-opt."),
             ("canonical_dopt", "Gaussian ampl. + D-opt.")]
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Surrogate design study (Al 300~K, W 1500~K, NaCl 700~K;"
        r" DFT-fitted FC2+FC3+FC4 surrogates in $3\times3\times3$ cells). Median"
        r" mesh RMSE (THz) against the surrogate effective FC2 over 30"
        r" repetitions, at 2/4/8/16 force calls.}",
        r"\label{tab:surrogate}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{lccc}",
        r"Design & Al & W & NaCl\\",
        r"\hline",
    ]
    data = {}
    for m in ("al", "w", "nacl"):
        path = ROOT / "validation_results" / m / "surrogate_design" / "surrogate_design_study.json"
        data[m] = json.loads(path.read_text()) if path.exists() else None
    for key, label in kinds:
        cells = [label]
        for m in ("al", "w", "nacl"):
            if data[m] is None:
                cells.append("--")
                continue
            rows = data[m]["results"][key]
            cells.append("/".join(
                f"{np.median([r['mesh_rmse_thz'] for r in rows[c]]):.3f}"
                for c in ("2", "4", "8", "16")))
        lines.append(" & ".join(cells) + r"\\")
    lines.append(r"\hline")
    lines.append("Harmonic vs effective & " + " & ".join(
        f"{data[m]['harmonic_vs_effective_mesh']['rmse_thz']:.3f}" if data[m] else "--"
        for m in ("al", "w", "nacl")) + r"\\")
    lines += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table*}"]
    return "\n".join(lines)


def iteration_table():
    lines = [
        r"\begin{longtable*}{lcccccc}",
        r"\caption{Individual iterations of the SOTC-C runs: configurations,"
        r" r.m.s.\ displacement, unstable mesh fraction and minimum frequency of"
        r" the axial-prior fit, and normalized condition number of the"
        r" space-group adapted design.\label{tab:iterations}}\\",
        r"\hline\hline",
        r"Material & iter. & $N_{\rm ens}$ & $u_{\rm rms}$ (\AA) & $f_{\rm unst}$"
        r" & $\omega_{\min}$ (THz) & cond.\\",
        r"\hline",
        r"\endfirsthead",
        r"\hline",
        r"Material & iter. & $N_{\rm ens}$ & $u_{\rm rms}$ (\AA) & $f_{\rm unst}$"
        r" & $\omega_{\min}$ (THz) & cond.\\",
        r"\hline",
        r"\endhead",
    ]
    for m in MATERIALS:
        audit = load(m, "audit")
        if audit is None:
            continue
        for it in audit["iterations"]:
            pub = it["variants"]["published_prior"]["mesh"]
            d = it["variants"]["space_group_symfc"]["design"]
            mark = "*" if it["iteration"] == audit["selected_iteration"] else ""
            lines.append(" & ".join([
                LABELS[m], f"{it['iteration']}{mark}", str(it["n_displaced"]),
                f"{it['rms_displacement_ang']:.3f}", f"{pub['unstable_fraction']:.3f}",
                f"{pub['min_frequency_thz']:.2f}",
                f"{d['normalized_design_condition_number']:.1f}",
            ]) + r"\\")
    lines += [r"\hline\hline", r"\end{longtable*}"]
    return "\n".join(lines)


def volume_table():
    lines = [
        r"\begin{table*}[t]",
        r"\caption{PBE equilibrium lattice parameters from Birch--Murnaghan fits"
        r" (primitive cells) and the effect of using them instead of the"
        r" production values on the harmonic $3\times3\times3$ FD model: mesh RMSE"
        r" and change in $S_{\rm vib}(T_{\rm des})$.}",
        r"\label{tab:volume}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{lcccccc}",
        r" & $a_{\rm prod}$ (\AA) & $a_{\rm PBE}$ (\AA) & $\Delta a/a$ (\%) &"
        r" $B_0$ (GPa) & RMSE (THz) & $\Delta S_{\rm vib}$\\",
        r"\hline",
    ]
    for m in MATERIALS:
        rob = load(m, "rob")
        if rob is None or "volume_sensitivity_3x3x3" not in rob:
            continue
        vs = rob["volume_sensitivity_3x3x3"]
        eos = vs.get("eos")
        if not eos:
            continue
        lines.append(" & ".join([
            LABELS[m], f"{eos['a_production_ang']:.3f}", f"{eos['a_pbe_ang']:.3f}",
            f"{100 * eos['relative_difference']:+.2f}", f"{eos['fit']['B0_gpa']:.0f}",
            f"{vs['mesh_pbe_vs_production']['rmse_thz']:.2f}",
            f"{vs['thermo_pbe']['s_j_mol_atom_k'] - vs['thermo_production']['s_j_mol_atom_k']:+.2f}",
        ]) + r"\\")
    lines += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table*}"]
    return "\n".join(lines)


def calls_to_accuracy_table(thresholds=(0.05, 0.03, 0.02)):
    """Force calls each self-consistent protocol needs on the surrogates to
    reach a given median mesh error against the self-consistent target."""
    protocols = [("canonical_oneshot", "Canonical $\\pm$, one shot"),
                 ("sotcd_oneshot", "SOTC-D, one shot"),
                 ("stdep_last", "sTDEP (last)"), ("stdep_pool", "sTDEP (pooled)"),
                 ("sotcd_sc", "SOTC-D-SC"), ("sscha", "SSCHA")]
    data = {}
    for m in SC_SURROGATES:
        path = ROOT / "validation_results" / m / "surrogate_design" / "surrogate_self_consistent.json"
        if path.exists():
            data[m] = json.loads(path.read_text())

    def curve(r, key):
        if key == "sscha":
            return [(float(c), e) for c, e in sscha_curve(r)]
        return sorted((float(b), float(np.median([x["mesh_rmse_thz"] for x in rows])))
                      for b, rows in r["results"][key].items())

    def first_reaching(points, threshold):
        # Smallest tested budget from which the median error stays below the
        # threshold (robust to a single lucky budget).
        for i, (calls, _) in enumerate(points):
            if all(e <= threshold for _, e in points[i:]):
                return f"{calls:.0f}"
        return "$>$" + f"{points[-1][0]:.0f}"

    lines = [
        r"\begin{table*}[t]",
        r"\caption{End-to-end cost on the anharmonic surrogates: force calls"
        r" needed by each complete protocol, started from the same harmonic"
        r" trial, to reach and keep a median mesh error of " +
        ", ".join(f"{t:.2f}" for t in thresholds) +
        r"~THz against the self-consistent effective FC2 (tested budgets"
        r" 4--64 calls; SSCHA 8--384).}",
        r"\label{tab:calls}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{l" + "c" * len(data) + "}",
        "Protocol & " + " & ".join(LABELS[m] for m in data) + r"\\",
        r"\hline",
    ]
    for key, label in protocols:
        cells = [label]
        for m, r in data.items():
            pts = curve(r, key)
            cells.append("/".join(first_reaching(pts, t) for t in thresholds) if pts else "--")
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table*}"]
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tab_runs.tex").write_text(runs_table() + "\n")
    (OUT / "tab_accuracy.tex").write_text(accuracy_table() + "\n")
    (OUT / "tab_matched.tex").write_text(matched_table() + "\n")
    (OUT / "tab_audit.tex").write_text(audit_table() + "\n")
    (OUT / "tab_thermo.tex").write_text(thermo_table() + "\n")
    (OUT / "tab_surrogate.tex").write_text(surrogate_table() + "\n")
    (OUT / "tab_iterations.tex").write_text(iteration_table() + "\n")
    (OUT / "tab_volume.tex").write_text(volume_table() + "\n")
    (OUT / "tab_selfconsistent.tex").write_text(selfconsistent_table() + "\n")
    (OUT / "tab_selfconsistent_crystal.tex").write_text(
        selfconsistent_table(basis="crystal") + "\n")
    (OUT / "tab_cost.tex").write_text(cost_table() + "\n")
    (OUT / "tab_basis.tex").write_text(basis_table() + "\n")
    (OUT / "tab_fitters.tex").write_text(fitter_table() + "\n")
    (OUT / "tab_sc_surrogate.tex").write_text(sc_surrogate_table() + "\n")
    (OUT / "tab_calls.tex").write_text(calls_to_accuracy_table() + "\n")
    (OUT / "tab_stdep.tex").write_text(stdep_table() + "\n")
    (OUT / "tab_stdep_crystal.tex").write_text(stdep_table(basis="crystal") + "\n")


if __name__ == "__main__":
    main()
