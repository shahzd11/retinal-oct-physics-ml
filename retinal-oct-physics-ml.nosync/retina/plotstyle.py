"""Shared matplotlib style: validated categorical palette in fixed order, recessive axes."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# fixed entity -> colour mapping (colour follows the class, never its rank)
CLASS_COLOR = {"NORMAL": SERIES[0], "CSR": SERIES[1], "DR": SERIES[2], "MH": SERIES[3],
               "CNV": SERIES[4], "DME": SERIES[5], "DRUSEN": SERIES[6], "AMD": SERIES[7]}


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
        "axes.titlesize": 12, "font.size": 10, "axes.axisbelow": True,
    })


def hbar(ax, labels, values, color=SERIES[0], fmt="{:.3f}"):
    """Thin horizontal bars with direct value labels in text ink (not series colour)."""
    y = range(len(labels))
    ax.barh(list(y), values, color=color, height=0.6, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    span = max(abs(v) for v in values) if len(values) else 1
    # labels always sit just right of max(v, 0) so negative bars never collide with tick labels
    for i, v in enumerate(values):
        ax.text(max(v, 0) + span * 0.015, i, fmt.format(v), va="center", ha="left", color=INK2, fontsize=8)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(min(0, min(values)) - span * 0.02, max(values) + span * 0.18)
