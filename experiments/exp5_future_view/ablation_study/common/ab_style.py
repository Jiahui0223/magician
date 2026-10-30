"""Shared figure style of the EXP5 ablation (matplotlib; palette = the validated reference instance of the dataviz skill,
scorer colours continue the Task 2 figures: V2O light orange, V2M orange)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#9a9893", "#e4e3df", "#fcfcfb"
COL = {"V2O": "#f2a07b", "V2M": "#eb6834", "GTD": "#1baf7a", "Oracle": INK, "Random": MUTED}
LAB = {"V2O": "V2 Original (pred.)", "V2M": "V2-Mix (pred.)", "GTD": "GT depth + DistP1", "Oracle": "Oracle coverage gain",
       "Random": "Random (expected)"}
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
        "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE)
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a9a", "#d03b3b"])
KNOWN_GRAY, UNK_NOTNEW = "#c9c8c2", "#8f8e88"


def setup():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 9.5, "figure.facecolor": "white",
                         "axes.facecolor": "white", "axes.grid": False, "legend.frameon": False, "savefig.dpi": 160})


def clean(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
