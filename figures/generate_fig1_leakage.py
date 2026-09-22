"""Figure 1 — Leakage propagation diagram.

Two-panel diagram showing how TargetEncoder leakage propagates
through an unsupervised anomaly detection pipeline.

Panel A — CLEAN pipeline (OneHot/Frequency encoding)
Panel B — LEAKED pipeline (TargetEncoder)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

out = Path("figures/output")
out.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
fig.subplots_adjust(wspace=0.5)


def draw_box(ax, x, y, w, h, text, color="#3498db", fontsize=8, textcolor="white"):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor="black", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=textcolor)


def draw_arrow(ax, x1, y1, x2, y2, color="black", lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))


# ── Panel A — CLEAN PIPELINE ──
ax = ax1
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title("A. Clean Pipeline (OneHot/Frequency)", fontsize=10, fontweight="bold", pad=10)
ax.axis("off")

# Data flow boxes
draw_box(ax, 2, 8.5, 2.5, 1.0, "Raw Data\n(NIDS Records)", "#2c3e50")
draw_box(ax, 2, 6.5, 2.5, 1.0, "Feature\nExtraction", "#7f8c8d")

# Split into numeric and categorical
draw_box(ax, 1, 4.5, 1.8, 0.8, "Numeric\nFeatures", "#95a5a6")
draw_box(ax, 3.5, 4.5, 1.8, 0.8, "Categorical\nOneHot/Freq", "#27ae60")

draw_arrow(ax, 2, 8.0, 2, 7.1)
draw_arrow(ax, 2, 6.0, 1, 5.0)
draw_arrow(ax, 2, 6.0, 3.5, 5.0)

# Combine
draw_box(ax, 2.25, 3.0, 3.0, 0.8, "Combined Features\n(No Labels)", "#3498db")

draw_arrow(ax, 1, 4.1, 2.25, 3.5)
draw_arrow(ax, 3.5, 4.1, 2.25, 3.5)

# Detectors
draw_box(ax, 1, 1.5, 1.5, 0.8, "AE\n(unsupervised)", "#e74c3c")
draw_box(ax, 3.5, 1.5, 1.5, 0.8, "IF, LOF\n(unsupervised)", "#f39c12")

draw_arrow(ax, 2.25, 2.6, 1, 2.0)
draw_arrow(ax, 2.25, 2.6, 3.5, 2.0)

# Result
draw_box(ax, 2.25, 0.0, 3.0, 0.8, "AE: AUROC ≈ 0.40\n(harmful to ensemble)", "#c0392b")

# Annotation
ax.text(7, 5.0, "No label\ninformation\nleaks", fontsize=9, ha="center",
        color="#27ae60", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#d5f5e3", edgecolor="#27ae60"))
ax.text(7, 3.0, "EW < NL\n(AE removal helps)", fontsize=9, ha="center",
        color="#e74c3c", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fadbd8", edgecolor="#e74c3c"))

# ── Panel B — LEAKED PIPELINE ──
ax = ax2
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title("B. Leaked Pipeline (TargetEncoder)", fontsize=10, fontweight="bold", pad=10)
ax.axis("off")

# Data flow boxes
draw_box(ax, 2, 8.5, 2.5, 1.0, "Raw Data\n(NIDS Records)", "#2c3e50")
draw_box(ax, 2, 6.5, 2.5, 1.0, "Feature\nExtraction", "#7f8c8d")

# Split into numeric and categorical
draw_box(ax, 1, 4.5, 1.8, 0.8, "Numeric\nFeatures", "#95a5a6")
draw_box(ax, 3.5, 4.5, 1.8, 0.8, "Categorical\nTargetEncoder", "#e74c3c")

# Label leakage annotation
ax.text(3.5, 3.6, "Fits on y_train!\n(label leakage", fontsize=7, ha="center",
        color="#c0392b", fontweight="bold")

draw_arrow(ax, 2, 8.0, 2, 7.1)
draw_arrow(ax, 2, 6.0, 1, 5.0)
draw_arrow(ax, 2, 6.0, 3.5, 5.0)

# Combine
draw_box(ax, 2.25, 3.0, 3.0, 0.8, "Combined Features\n(Labels Embedded!)", "#e74c3c")

draw_arrow(ax, 1, 4.1, 2.25, 3.5)
draw_arrow(ax, 3.5, 4.1, 2.25, 3.5)

# Detectors
draw_box(ax, 1, 1.5, 1.5, 0.8, "AE\n(leaked features)", "#e74c3c")
draw_box(ax, 3.5, 1.5, 1.5, 0.8, "IF, LOF\n(unsupervised)", "#f39c12")

draw_arrow(ax, 2.25, 2.6, 1, 2.0)
draw_arrow(ax, 2.25, 2.6, 3.5, 2.0)

# Result
draw_box(ax, 2.25, 0.0, 3.0, 0.8, "AE: AUROC ≈ 0.65\n(accidentally helpful)", "#f39c12")

# Annotation
ax.text(7, 5.0, "Label info\npropagates to\nAE weights", fontsize=9, ha="center",
        color="#c0392b", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fadbd8", edgecolor="#c0392b"))
ax.text(7, 3.0, "EW > NL\n(AE retains)", fontsize=9, ha="center",
        color="#e74c3c", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fadbd8", edgecolor="#e74c3c"))

# Title
fig.suptitle("Figure 1: Label Leakage Propagation Through Categorical Encoding",
             fontsize=12, fontweight="bold", y=0.98)

# Save
fig.savefig(out / "fig1_leakage_propagation.png", dpi=600, bbox_inches="tight",
            facecolor="white", edgecolor="none")
fig.savefig(out / "fig1_leakage_propagation.svg", bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)

for ext in ["png", "svg"]:
    p = out / f"fig1_leakage_propagation.{ext}"
    size_kb = p.stat().st_size / 1024
    print(f"  {p.name}: {size_kb:.1f} KB")
print("Figure 1 generated")
