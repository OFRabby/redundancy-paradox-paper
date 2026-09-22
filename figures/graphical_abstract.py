"""Graphical abstract for the Redundancy Paradox paper.

Three horizontal panels:
  A — PROBLEM: LOO is used but unreliable
  B — VALIDATION: Ground-truth validation (2000 runs)
  C — PRACTICAL RULE: Use LOO as diagnostic, not selection rule

Output: 1200x600 px, 600 dpi PNG + SVG.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

out = Path("figures/output")
out.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 11,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.subplots_adjust(wspace=0.4)

# ── Panel A — PROBLEM ──
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title("A. PROBLEM", fontsize=12, fontweight="bold", pad=10)
ax.axis("off")

# Detectors
det_names = ["IF", "LOF", "OCSVM", "AE"]
det_y = [7.5, 5.5, 3.5, 1.5]
colors = ["#27ae60", "#e74c3c", "#f39c12", "#3498db"]

for i, (name, y, c) in enumerate(zip(det_names, det_y, colors)):
    rect = FancyBboxPatch((1, y - 0.6), 2, 1.2, boxstyle="round,pad=0.1",
                          facecolor=c, edgecolor="black", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(2, y, name, ha="center", va="center", fontsize=10, fontweight="bold", color="white")

# Ensemble box
ens_rect = FancyBboxPatch((5, 3), 3.5, 4, boxstyle="round,pad=0.2",
                          facecolor="#ecf0f1", edgecolor="#2c3e50", linewidth=2)
ax.add_patch(ens_rect)
ax.text(6.75, 6.2, "Rank-Average", ha="center", va="center", fontsize=9, fontweight="bold")
ax.text(6.75, 5.5, "Ensemble", ha="center", va="center", fontsize=9)

# LOO contribution arrows
for y, sign in [(7.5, "+"), (5.5, "−"), (3.5, "+"), (1.5, "?")]:
    ax.annotate("", xy=(5, y), xytext=(3.2, y),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5))
    ax.text(4.1, y + 0.35, f"C_i = {sign}", ha="center", fontsize=8,
            color="#27ae60" if sign == "+" else "#e74c3c" if sign == "−" else "#7f8c8d")

# LOO label
ax.text(6.75, 2.2, "LOO Contribution", ha="center", fontsize=8, style="italic", color="#7f8c8d")
ax.text(6.75, 1.2, "Which to keep?", ha="center", fontsize=9, fontweight="bold", color="#c0392b")

# Question mark
ax.text(8.5, 5, "?", fontsize=28, ha="center", va="center", color="#c0392b", fontweight="bold")

# ── Panel B — VALIDATION ──
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title("B. VALIDATION (2,000 runs)", fontsize=12, fontweight="bold", pad=10)
ax.axis("off")

# Confusion matrix heatmap
cm = np.array([
    [3048, 653, 8, 13],
    [1237, 609, 4, 11],
    [0, 0, 225, 0],
    [25, 29, 194, 3840],
])
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

im = ax.imshow(cm_pct, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
ax.set_xticks([0, 1, 2, 3])
ax.set_yticks([0, 1, 2, 3])
ax.set_xticklabels(["B", "R", "H", "N"], fontsize=9)
ax.set_yticklabels(["B", "R", "H", "N"], fontsize=9)
ax.set_xlabel("Predicted", fontsize=9)
ax.set_ylabel("True", fontsize=9)

# Annotate cells
labels = [["3048", "653", "8", "13"],
          ["1237", "609", "4", "11"],
          ["0", "0", "225", "0"],
          ["25", "29", "194", "3840"]]
for i in range(4):
    for j in range(4):
        color = "white" if cm_pct[i, j] > 50 else "black"
        ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=7, color=color)

# Key findings
ax.text(5, 8.5, "H recall = 1.000", fontsize=9, fontweight="bold", color="#27ae60")
ax.text(5, 7.5, "R recall = 0.327", fontsize=9, fontweight="bold", color="#e74c3c")
ax.text(5, 6.5, "Sign-conflict = 48%", fontsize=9, fontweight="bold", color="#f39c12")

# Color bar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("% of row", fontsize=8)

# ── Panel C — PRACTICAL RULE ──
ax = axes[2]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title("C. PRACTICAL RULE", fontsize=12, fontweight="bold", pad=10)
ax.axis("off")

# Flowchart nodes
nodes = {
    "compute": (5, 8.5, "Compute LOO\ncontributions", "#3498db"),
    "conflict": (5, 6.5, "Sign-conflict\n> 15%?", "#f39c12"),
    "epsilon": (2.5, 4.5, "Apply\nepsilon-VRG", "#e74c3c"),
    "ew": (7.5, 4.5, "Use Equal\nWeights", "#27ae60"),
    "result": (5, 2.5, "Reliable\nEnsemble", "#2c3e50"),
}

for key, (x, y, label, color) in nodes.items():
    w, h = (2.4, 1.2) if key != "conflict" else (2.4, 1.2)
    shape = "round" if key != "conflict" else "round"
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle=f"{shape},pad=0.15",
                          facecolor=color, edgecolor="black", linewidth=1.2, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=8, fontweight="bold", color="white")

# Arrows
ax.annotate("", xy=(5, 7.1), xytext=(5, 7.9),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
ax.annotate("", xy=(2.5, 5.1), xytext=(4, 6.5),
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5))
ax.annotate("", xy=(7.5, 5.1), xytext=(6, 6.5),
            arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.5))
ax.annotate("", xy=(5, 3.1), xytext=(2.5, 3.9),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
ax.annotate("", xy=(5, 3.1), xytext=(7.5, 3.9),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

# Labels on arrows
ax.text(3.2, 6.0, "YES", fontsize=8, color="#e74c3c", fontweight="bold")
ax.text(6.8, 6.0, "NO", fontsize=8, color="#27ae60", fontweight="bold")

# Key message
ax.text(5, 0.8, "LOO = diagnostic,\nnot selection rule",
        ha="center", fontsize=9, fontweight="bold", color="#c0392b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffeaa7", edgecolor="#c0392b"))

# Title
fig.suptitle("Fragile Diagnostics: Leave-One-Out Ensemble Selection",
             fontsize=14, fontweight="bold", y=0.98)

# Save
fig.savefig(out / "graphical_abstract.png", dpi=600, bbox_inches="tight",
            facecolor="white", edgecolor="none")
fig.savefig(out / "graphical_abstract.svg", bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)

# Verify
for ext in ["png", "svg"]:
    p = out / f"graphical_abstract.{ext}"
    size_kb = p.stat().st_size / 1024
    print(f"  {p.name}: {size_kb:.1f} KB")
print("Graphical abstract generated")
