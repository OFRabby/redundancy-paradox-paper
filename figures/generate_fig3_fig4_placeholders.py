"""Generate Figures 3 and 4 (or placeholders if Phase 3 data missing).

Figure 3: EW/NL/EV grouped bars per dataset (Phase 3 means)
Figure 4: Per-encoder EW vs NL (seed-fragility visualization)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

phase3_csv = Path("../research/kaggle_output/real_data_validation/phase3_output/aggregated_long.csv")

if phase3_csv.exists():
    df = pd.read_csv(phase3_csv)

    # ── Figure 3: EW/NL/EV grouped bars per dataset ──
    fig, ax = plt.subplots(figsize=(6, 4))
    datasets = df["dataset"].unique()
    methods = ["EW", "NL", "EV"]
    x = np.arange(len(datasets))
    width = 0.25

    for i, method in enumerate(methods):
        means = []
        stds = []
        for ds in datasets:
            subset = df[(df["dataset"] == ds) & (df["method"] == method)]
            means.append(subset["auroc_test"].mean())
            stds.append(subset["auroc_test"].std())
        ax.bar(x + i * width, means, width, yerr=stds, label=method, capsize=3)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Test AUROC")
    ax.set_title("Phase 3: Method Comparison per Dataset")
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets, rotation=15)
    ax.legend()
    ax.set_ylim(0.4, 1.0)

    fig.savefig(out / "fig3_method_comparison.png", dpi=600, bbox_inches="tight")
    fig.savefig(out / "fig3_method_comparison.svg", bbox_inches="tight")
    plt.close(fig)
    print("Figure 3 generated from real data")

    # ── Figure 4: Per-encoder EW vs NL (fragility) ──
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        subset = df[df["dataset"] == dataset]
        encoders = subset["encoder"].unique()

        for enc in encoders:
            enc_data = subset[subset["encoder"] == enc]
            seeds = enc_data["seed"].unique()
            ew_nl_deltas = []
            for seed in seeds:
                seed_data = enc_data[enc_data["seed"] == seed]
                ew_val = seed_data[seed_data["method"] == "EW"]["auroc_test"].values
                nl_val = seed_data[seed_data["method"] == "NL"]["auroc_test"].values
                if len(ew_val) > 0 and len(nl_val) > 0:
                    ew_nl_deltas.append(ew_val[0] - nl_val[0])

            if ew_nl_deltas:
                ax.scatter([enc] * len(ew_nl_deltas), ew_nl_deltas, alpha=0.5, s=20)

        ax.axhline(0, ls="--", color="gray", alpha=0.5)
        ax.set_title(dataset)
        ax.set_ylabel("EW − NL (AUROC)")

    fig.suptitle("Figure 4: Seed-Fragility (EW − NL per seed)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig4_seed_fragility.png", dpi=600, bbox_inches="tight")
    fig.savefig(out / "fig4_seed_fragility.svg", bbox_inches="tight")
    plt.close(fig)
    print("Figure 4 generated from real data")

else:
    # ── Placeholders ──
    for fig_num, title in [(3, "Method Comparison per Dataset"), (4, "Seed-Fragility (EW − NL)")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"Figure {fig_num}: {title}\n\nAwaiting Phase 3 data",
                ha="center", va="center", fontsize=14, color="#7f8c8d",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", edgecolor="#bdc3c7"))
        ax.axis("off")
        fig.savefig(out / f"fig{fig_num}_placeholder.png", dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure {fig_num} placeholder generated")

# Report file sizes
for p in sorted(out.glob("fig3*")) + sorted(out.glob("fig4*")):
    size_kb = p.stat().st_size / 1024
    print(f"  {p.name}: {size_kb:.1f} KB")
