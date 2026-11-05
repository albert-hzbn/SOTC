"""Common figure style of the manuscript: serif text matching the journal font,
boxed axes with inward ticks, one validated categorical palette, and figures
drawn at their printed size (AIP reprint: column 3.37 in, page width 7.0 in)."""

from __future__ import annotations

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLUMN, PAGE = 3.37, 7.0
# Validated categorical slots (blue, orange, aqua, yellow) plus neutral gray.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK, MUTED, RULE = "#8a8985", "#0b0b0b", "#52514e", "#d6d5d0"

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "axes.titlepad": 3,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.5,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "lines.linewidth": 1.0, "lines.markersize": 3, "hatch.linewidth": 0.5,
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.columnspacing": 1.2,
    "legend.labelspacing": 0.25, "legend.handletextpad": 0.5,
    "pdf.fonttype": 42, "savefig.dpi": 300,
})


def panel(ax, letter, x=0.03, y=0.96):
    """Panel letter inside the axes."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, va="top", ha="left")


def plain_log(ax, subs=(1, 2, 5), axis="y"):
    """Log axis with plain decimal labels at 1, 2, 5 x 10^n."""
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=subs))
    target.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    target.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=range(1, 10)))
    target.set_minor_formatter(matplotlib.ticker.NullFormatter())


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
