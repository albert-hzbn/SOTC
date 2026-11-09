#!/usr/bin/env python3
"""Workflow of SOTC-D (Fig. 1 of the JAP manuscript), every panel from the DFT
data of fcc Au at 1200 K in the 32-atom SOTC cell:

* (a) the harmonic trial (finite-displacement FC2 of the cell) and the wave
  vectors on the path that are commensurate with the cell;
* (b) the mode coordinates of the first SOTC-D configuration, recovered from its
  displacements, against those of a canonical configuration;
* (c) that configuration and its inverse;
* (d) the DFT forces of the first eight calls, split into the part odd and the
  part even under u -> -u, against the force -Phi u of the independent
  40-snapshot effective FC2;
* (e) the dispersion fitted to those eight calls against the trial and that
  effective FC2;
* (f) the error versus force calls from robustness_summary.json.

Writes manuscript/jap/figures/fig_workflow.pdf.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.ticker
import numpy as np
import phonopy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_designs import N_TRAIN, Fitter, datasets  # noqa: E402
from analyze_primary_archives import (  # noqa: E402
    PRIMARY, mesh_points, phonopy_frequencies, symfc_fc2_basis,
)
from plot_designs import band_path  # noqa: E402
from plot_style import AQUA, BLUE, GRAY, INK, MUTED, ORANGE, PAGE, RULE, plain_log, save  # noqa: E402
from sotc.optimal_design import thermal_modes  # noqa: E402

OUT = ROOT / "manuscript" / "jap" / "figures"
MATERIAL = "au"
N_CALLS = 8
ARROW_SCALE = 3.0  # exaggeration of the displacements in (c)
XS, WIDTH = (0.058, 0.392, 0.727), 0.238
Y_TOP, Y_BOTTOM, HEIGHT = 0.615, 0.082, 0.335
TRIAL_COLOR = "#a9a8a3"


def load():
    vr = ROOT / "validation_results" / MATERIAL
    spec = PRIMARY[MATERIAL]
    large = phonopy.load(vr / "fd_references" / "production_large" / "phonopy_harmonic.yaml")
    phonon = phonopy.load(vr / "primary_cell" / "fd" / "phonopy_harmonic.yaml")
    sup = phonon.supercell
    fit = Fitter(symfc_fc2_basis(np.asarray(sup.positions), np.asarray(sup.cell),
                                 list(sup.symbols), spec["cutoff"]))
    return spec, large, phonon, fit, datasets(MATERIAL)


def path_frequencies(phonon, large, q):
    ref = np.asarray(large.primitive.cell)
    return phonopy_frequencies(phonon, q @ ref @ np.linalg.inv(np.asarray(phonon.primitive.cell)))


def commensurate(phonon, large, q, x):
    """Points of the band path whose wave vector is commensurate with the cell."""
    cart = q @ np.linalg.inv(np.asarray(large.primitive.cell)).T
    n = cart @ np.asarray(phonon.supercell.cell).T
    return np.where(np.abs(n - np.rint(n)).max(axis=1) < 1e-6)[0]


def mode_coordinates(u, patterns):
    q, *_ = np.linalg.lstsq(patterns.reshape(len(patterns), -1).T, u.ravel(), rcond=None)
    return q


def band_axes(ax, ticks, names):
    for t in ticks[1:-1]:
        ax.axvline(t, color=RULE, lw=0.4, zorder=0)
    ax.set_xticks(ticks, names)
    ax.xaxis.set_tick_params(top=False)
    ax.set_xlim(0, ticks[-1])
    ax.set_ylim(0, None)
    ax.set_ylabel("frequency (THz)", labelpad=2)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))


def draw_configuration(ax, phonon, u, y0):
    """Oblique projection of the cell: ideal sites, displaced atoms, and arrows."""
    sup = phonon.supercell
    cell = np.asarray(sup.cell)
    frac = np.asarray(sup.scaled_positions) % 1.0
    frac[frac > 0.999] = 0.0
    pos = frac @ cell
    # view along the shortest principal axis of the cell, slightly tilted;
    # coordinates: page x, depth, page y
    centre = 0.5 * cell.sum(axis=0)
    corners_cart = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)],
                            float) @ cell - centre
    axes_cell = np.linalg.svd(corners_cart)[2]
    a, b = np.radians(14), np.radians(12)
    rot_z = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rot_x = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    frame = axes_cell[[0, 2, 1]]
    view = lambda r: (r - centre) @ frame.T @ rot_z.T @ rot_x.T  # noqa: E731
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
    edges = [(p, q) for p in range(8) for q in range(p + 1, 8)
             if np.abs(corners[p] - corners[q]).sum() == 1]
    box = view(corners @ cell)
    for p, q in edges:
        ax.plot(box[[p, q], 0], box[[p, q], 2] + y0, color=MUTED, lw=0.45, zorder=1)
    ideal, moved = view(pos), view(pos + ARROW_SCALE * u)
    depth = (ideal[:, 1] - ideal[:, 1].min()) / np.ptp(ideal[:, 1])
    for k in np.argsort(-ideal[:, 1]):
        shade = 1.0 - 0.6 * depth[k]
        z = 3 + (1 - depth[k])
        ax.scatter(ideal[k, 0], ideal[k, 2] + y0, s=4, facecolor="white", edgecolor=MUTED,
                   linewidth=0.3, zorder=z)
        ax.plot([ideal[k, 0], moved[k, 0]], [ideal[k, 2] + y0, moved[k, 2] + y0], color=ORANGE,
                lw=1.0, alpha=shade, solid_capstyle="round", zorder=z + 0.01)
        ax.scatter(moved[k, 0], moved[k, 2] + y0, s=8 + 9 * (1 - depth[k]), color=BLUE,
                   alpha=shade, edgecolor="white", linewidth=0.25, zorder=z + 0.02)
    return box


def main():
    spec, large, phonon, fit, data = load()
    sup = phonon.supercell
    masses = np.asarray(sup.masses)
    q, x, ticks, names = band_path(spec["bravais"], n_per=121)
    trial_fc = np.asarray(phonon.force_constants)
    trial = path_frequencies(phonon, large, q)

    fig = plt.figure(figsize=(PAGE, 4.15))
    axes = {"a": fig.add_axes([XS[0], Y_TOP, WIDTH, HEIGHT]),
            "b": fig.add_axes([XS[1], Y_TOP, WIDTH, HEIGHT]),
            "c": fig.add_axes([XS[2] - 0.02, Y_TOP - 0.06, WIDTH + 0.04, HEIGHT + 0.06]),
            "d": fig.add_axes([XS[2], Y_BOTTOM, WIDTH, HEIGHT]),
            "e": fig.add_axes([XS[1], Y_BOTTOM, WIDTH, HEIGHT]),
            "f": fig.add_axes([XS[0], Y_BOTTOM, WIDTH, HEIGHT])}
    titles = {"a": "(a) Harmonic trial", "b": "(b) Mode coordinates",
              "d": "(d) Forces of the pairs", "e": f"(e) Fit to {N_CALLS} calls",
              "f": "(f) Accuracy against cost"}
    for k, t in titles.items():
        axes[k].set_title(t, loc="left")

    # (a) trial dispersion and commensurate wave vectors
    ax = axes["a"]
    ax.plot(x, trial, color=INK, lw=0.9)
    idx = commensurate(phonon, large, q, x)
    for b in trial.T:
        ax.plot(x[idx], b[idx], "o", color=BLUE, ms=3.2, mec="white", mew=0.3, zorder=3,
                clip_on=False)
    ax.plot([], [], "o", color=BLUE, ms=3.2, mec="white", mew=0.3, label=r"commensurate $\mathbf{q}$")
    band_axes(ax, ticks, names)
    ax.set_ylim(0, 1.5 * trial.max())
    ax.legend(loc="upper left", borderaxespad=0.3)
    n_modes = 3 * len(sup.masses) - 3
    ax.text(0.035, 0.86, f"fcc Au, {spec['T']:.0f} K\n{len(sup.masses)}-atom cell, "
            f"$M={n_modes}$ modes", transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color=MUTED, linespacing=1.15)

    # (b) coordinates of real configurations against the mode frequency
    ax = axes["b"]
    frequencies, patterns, sigma = thermal_modes(trial_fc, masses, spec["T"])
    frequencies = np.asarray(frequencies)
    if len(frequencies) != len(sigma):
        frequencies = frequencies[np.abs(frequencies) > 1e-6]
    scale = 1e-10 * np.sqrt(1.66053907e-27)  # m kg^(1/2) -> Angstrom amu^(1/2)
    q_canon = mode_coordinates(data["canonical"][0][0], patterns) / scale
    q_sotc = mode_coordinates(data["sotcd"][0][0], patterns) / scale
    order = np.argsort(frequencies)
    w, s = frequencies[order], (sigma / scale)[order]
    ax.axhline(0, color=RULE, lw=0.4, zorder=0)
    ax.plot(w, s, color=MUTED, lw=0.6, ls=(0, (3, 2)), label=r"$\pm\sigma_m$")
    ax.plot(w, -s, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.plot(w, q_canon[order], "o", mfc="white", mec=GRAY, ms=2.6, mew=0.5, label="canonical")
    ax.plot(w, q_sotc[order], "o", color=BLUE, ms=2.0, label="SOTC-D")
    ax.set_xlabel(r"mode frequency $\omega_m$ (THz)", labelpad=1.5)
    ax.set_ylabel(r"$Q_m$ ($\sqrt{\mathrm{amu}}\,$Å)", labelpad=1)
    lim = 1.05 * np.abs(q_canon).max()
    ax.set_ylim(-lim, 1.5 * lim)
    ax.legend(loc="upper center", ncol=3, borderaxespad=0.2, columnspacing=0.8, handlelength=1.2,
              handletextpad=0.3)
    ax.text(0.97, 0.05, r"$\sigma_m^2=\dfrac{\hbar}{2\omega_m}\coth\dfrac{\hbar\omega_m}{2k_BT}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=MUTED)

    # (c) the configuration and its inverse
    ax = axes["c"]
    ax.axis("off")
    u = data["sotcd"][0]
    box = draw_configuration(ax, phonon, u[0], 0.0)
    wide, tall = np.ptp(box[:, 0]), np.ptp(box[:, 2])
    gap = 1.28 * tall
    draw_configuration(ax, phonon, -u[0], -gap)
    for y0, label in ((0.0, r"$+\mathbf{u}$"), (-gap, r"$-\mathbf{u}$")):
        ax.text(box[:, 0].min() - 0.02 * wide, box[:, 2].max() + y0 - 0.04 * tall, label,
                ha="left", va="top", fontsize=9)
    ax.set_xlim(box[:, 0].min() - 0.1 * wide, box[:, 0].max() + 0.1 * wide)
    ax.set_ylim(box[:, 2].min() - gap - 0.12 * tall, box[:, 2].max() + 0.12 * tall)
    ax.set_aspect("equal", adjustable="box", anchor="C")
    fig.text(XS[2], Y_TOP + HEIGHT + 0.0085, r"(c) Designed $\pm\mathbf{u}$ pair",
             fontsize=8.5, va="bottom")
    norms = np.linalg.norm(u[0], axis=1)
    ax.text(0.03, 0.0, f"rms $|u|$ {np.sqrt((norms ** 2).mean()):.2f} Å, "
            f"max {norms.max():.2f} Å\n(drawn ×{ARROW_SCALE:.0f})", transform=ax.transAxes,
            ha="left", va="top", fontsize=7, color=MUTED, linespacing=1.1)

    # (d) odd and even parts of the pair forces against the effective model
    ax = axes["d"]
    uu, ff = (v[:N_CALLS] for v in data["sotcd"])
    reference = fit(*[v[:N_TRAIN] for v in data["canonical"]])
    shaped = reference.reshape(len(sup.masses), len(sup.masses), 3, 3)
    model = -np.einsum("ijab,njb->nia", shaped, uu[::2])
    odd = 0.5 * (ff[::2] - ff[1::2])
    even = 0.5 * (ff[::2] + ff[1::2])
    span = 1.08 * np.abs(odd).max()
    ax.plot([-span, span], [-span, span], color=MUTED, lw=0.6, zorder=1)
    ax.axhline(0, color=RULE, lw=0.4, zorder=0)
    ax.plot(model.ravel(), even.ravel(), "o", color=ORANGE, ms=1.4, mew=0, alpha=0.8,
            label=r"even, $\frac{1}{2}(\mathbf{F}_{+}+\mathbf{F}_{-})$: removed", zorder=2)
    ax.plot(model.ravel(), odd.ravel(), "o", color=BLUE, ms=1.4, mew=0, alpha=0.8,
            label=r"odd, $\frac{1}{2}(\mathbf{F}_{+}-\mathbf{F}_{-})$: fitted", zorder=3)
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, 2.15 * span)
    ax.set_xlabel(r"$-\mathbf{\Phi}_{\rm eff}\mathbf{u}$ (eV/Å)", labelpad=1.5)
    ax.set_ylabel("DFT force (eV/Å)", labelpad=1)
    ax.legend(loc="upper left", borderaxespad=0.2, handletextpad=0.1, markerscale=2,
              handlelength=1.0)
    rms = lambda v: float(np.sqrt((v ** 2).mean()))  # noqa: E731
    ax.text(0.04, 0.70, f"rms: odd {rms(odd):.2f}, even {rms(even):.2f},\n"
            f"odd $-$ model {rms(odd - model):.2f} eV/Å", transform=ax.transAxes, ha="left",
            va="top", fontsize=7, color=MUTED, linespacing=1.15)

    # (e) fitted dispersion against the trial and the effective FC2
    ax = axes["e"]
    phonon.force_constants = reference
    effective = path_frequencies(phonon, large, q)
    phonon.force_constants = fit(uu, ff)
    fitted = path_frequencies(phonon, large, q)
    ax.plot(x, trial, color=TRIAL_COLOR, lw=1.3)
    ax.plot(x, effective, color=INK, lw=0.7)
    ax.plot(x, fitted, color=BLUE, lw=0.9, ls=(0, (3, 1.6)))
    # mesh RMSE against the effective FC2 (10^3 mesh, Gamma excluded)
    qm = mesh_points(10)
    qm = qm[np.linalg.norm(qm, axis=1) > 1e-8]
    freq = {}
    for name, fc in (("trial", trial_fc), ("eff", reference), ("fit", fit(uu, ff))):
        phonon.force_constants = fc
        freq[name] = np.sort(phonopy_frequencies(phonon, qm), axis=1)
    err = {k: float(np.sqrt(((freq[k] - freq["eff"]) ** 2).mean())) for k in ("trial", "fit")}
    ax.plot([], [], color=TRIAL_COLOR, lw=1.3, label=f"trial ({err['trial']:.2f} THz)")
    ax.plot([], [], color=INK, lw=0.7, label=r"$\mathbf{\Phi}_{\rm eff}$, 40 snapshots")
    ax.plot([], [], color=BLUE, lw=0.9, ls=(0, (3, 1.6)),
            label=f"SOTC-D, {N_CALLS} calls ({err['fit']:.2f} THz)")
    band_axes(ax, ticks, names)
    ax.set_ylim(0, 1.8 * effective.max())
    ax.legend(loc="upper left", borderaxespad=0.3, title=r"mesh RMSE vs $\mathbf{\Phi}_{\rm eff}$",
              title_fontsize=7, alignment="left")

    # (f) error against force calls
    ax = axes["f"]
    report = json.loads((ROOT / "validation_results" / MATERIAL / "robustness"
                         / "robustness_summary.json").read_text())["count_series"]
    counts = np.asarray(report["counts"])
    block = report["symfc"]
    for kind, color, marker, label in (("canonical_unpaired", GRAY, "s", "canonical"),
                                       ("canonical_paired", AQUA, "^", r"canonical $\pm$")):
        vals = np.asarray([[t["mesh_vs_disjoint_effective"] for t in trials]
                           for trials in block[kind]])
        ax.fill_between(counts, np.percentile(vals, 10, axis=1), np.percentile(vals, 90, axis=1),
                        color=color, alpha=0.16, lw=0)
        ax.plot(counts, np.median(vals, axis=1), color=color, marker=marker, ms=3, lw=1.0,
                label=label)
    sotcd = np.asarray([r["mesh_vs_disjoint_effective"] for r in block["sotcd"]])
    ax.plot(counts, sotcd, color=BLUE, marker="o", ms=3.5, lw=1.4, label="SOTC-D")
    # calls the paired canonical median needs to reach the 2-call SOTC-D error
    paired = np.median([[t["mesh_vs_disjoint_effective"] for t in trials]
                        for trials in block["canonical_paired"]], axis=1)
    target = sotcd[0]
    below = np.flatnonzero(paired <= target)
    if len(below) and below[0] > 0:
        k = below[0]
        t = (np.log(paired[k - 1]) - np.log(target)) / (np.log(paired[k - 1]) - np.log(paired[k]))
        n_eq = float(np.exp(np.log(counts[k - 1]) + t * (np.log(counts[k]) - np.log(counts[k - 1]))))
        ax.plot([counts[0], n_eq], [target, target], color=MUTED, lw=0.6, ls=":", zorder=1)
        ax.plot([n_eq, n_eq], [target, target * 0.9], color=MUTED, lw=0.6, ls=":", zorder=1)
        ax.text(2.06, 0.133, f"canonical $\\pm$ needs {n_eq:.1f}\ncalls for this error",
                ha="left", va="top", fontsize=7, color=MUTED, linespacing=1.1)
        print("paired canonical reaches the 2-call SOTC-D error after", n_eq, "calls")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    plain_log(ax)
    ax.set_ylim(0.082, None)
    ax.set_xticks(counts, [str(c) for c in counts])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("DFT force calls", labelpad=1.5)
    ax.set_ylabel("mesh RMSE (THz)", labelpad=1)
    ax.legend(loc="upper right", borderaxespad=0.3)

    # the cycle: (a)->(b)->(c), down to (d), (d)->(e)->(f), and (e) back to (a)
    def arrow(start, end, color=INK):
        fig.add_artist(FancyArrowPatch(start, end, transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=7, lw=0.7, color=color,
                                       shrinkA=0, shrinkB=0))

    mid_top, mid_bottom = Y_TOP + HEIGHT / 2, Y_BOTTOM + HEIGHT / 2
    arrow((XS[0] + WIDTH + 0.006, mid_top), (XS[1] - 0.064, mid_top))
    arrow((XS[1] + WIDTH + 0.006, mid_top), (XS[2] - 0.022, mid_top))
    arrow((XS[2] + WIDTH * 0.75, Y_TOP - 0.03), (XS[2] + WIDTH * 0.75, Y_BOTTOM + HEIGHT + 0.012))
    arrow((XS[2] - 0.066, mid_bottom), (XS[1] + WIDTH + 0.006, mid_bottom))
    arrow((XS[1] - 0.062, mid_bottom), (XS[0] + WIDTH + 0.006, mid_bottom))
    y_loop = Y_BOTTOM + HEIGHT + 0.066
    xa, xe = XS[0] + WIDTH / 2, XS[1] + WIDTH * 0.8
    fig.add_artist(plt.Line2D([xe, xe, xa], [Y_BOTTOM + HEIGHT + 0.012, y_loop, y_loop],
                              transform=fig.transFigure, color=ORANGE, lw=0.8))
    arrow((xa, y_loop), (xa, Y_TOP - 0.038), ORANGE)
    fig.text((xa + xe) / 2, y_loop - 0.006, "SOTC-D-SC: refit all forces, update the trial",
             ha="center", va="top", fontsize=7.5, style="italic")

    print("odd rms", np.sqrt((odd ** 2).mean()), "even rms", np.sqrt((even ** 2).mean()),
          "odd-model rms", np.sqrt(((odd - model) ** 2).mean()), "n commensurate", len(idx))
    OUT.mkdir(parents=True, exist_ok=True)
    save(fig, OUT, "fig_workflow")


if __name__ == "__main__":
    main()
