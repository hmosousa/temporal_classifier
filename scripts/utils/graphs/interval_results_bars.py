import matplotlib.pyplot as plt
import numpy as np
from src.constants import IMGS_DIR

data = {
    "baselines": {
        "random": {"f1": 11.57, "p": 10.94, "r": 12.27},
        "majority": {"f1": 35.71, "p": 35.52, "r": 35.91},
        "UTTime": {"f1": 56.45, "p": 55.58, "r": 57.35},
        "Graph Staking": {"f1": 57.78, "p": 57.63, "r": 57.92},
        "TRelPro": {"f1": 58.48, "p": 58.80, "r": 58.27},
        "CATENA": {"f1": 61.9, "p": 62.6, "r": 61.3},
        "SP+ILP": {"f1": 67.2, "p": 69.1, "r": 65.5},
    },
    "Interval": {
        "SmolLM2-135M": {
            "Raw": {"f1": 62.85, "p": 62.82, "r": 62.87},
            "Inverse": {"f1": 64.93, "p": 64.89, "r": 64.97},
            "Closure": {"f1": 66.98, "p": 67.22, "r": 66.74},
            "Inverse & Closure": {"f1": 66.22, "p": 66.48, "r": 65.97},
        },
        "SmolLM2-360M": {
            "SmolLM2-360M": {
                "Raw": {"f1": 65.69, "p": 65.74, "r": 65.64},
                "Inverse": {"f1": 68.98, "p": 69.01, "r": 68.95},
                "Closure": {"f1": 65.54, "p": 65.67, "r": 65.41},
                "Inverse & Closure": {"f1": 67.36, "p": 67.22, "r": 67.51},
            },
        },
    },
    "IfP": {
        "IfP-135M": {
            "Raw": {"f1": 64.78, "p": 65.03, "r": 64.53},
            "Inverse": {"f1": 63.88, "p": 63.68, "r": 64.09},
            "Closure": {"f1": 63.97, "p": 64.07, "r": 63.87},
            "Inverse & Closure": {"f1": 64.01, "p": 64.37, "r": 63.65},
        },
        "IfP-360M": {
            "Raw": {"f1": 66.24, "p": 66.18, "r": 66.30},
            "Inverse": {"f1": 70.12, "p": 70.19, "r": 70.06},
            "Closure": {"f1": 67.91, "p": 68.08, "r": 67.73},
            "Inverse & Closure": {"f1": 69.28, "p": 69.39, "r": 69.17},
        },
    },
}

# Extract data for plotting
all_baseline_names = list(data["baselines"].keys())
plotted_baseline_names = [
    name for name in all_baseline_names if name not in ["random", "majority"]
]
baseline_f1 = [data["baselines"][model]["f1"] for model in plotted_baseline_names]

variants = ["Raw", "Inverse", "Closure", "Inverse & Closure"]
sizes = ["135M", "360M"]

# Extract Interval data
interval_data = {
    "135M": [data["Interval"]["SmolLM2-135M"][var]["f1"] for var in variants],
    "360M": [
        data["Interval"]["SmolLM2-360M"]["SmolLM2-360M"][var]["f1"] for var in variants
    ],
}

# Extract IfP data
ifp_data = {
    "135M": [data["IfP"]["IfP-135M"][var]["f1"] for var in variants],
    "360M": [data["IfP"]["IfP-360M"][var]["f1"] for var in variants],
}

# Plotting parameters
bar_width = 0.15

# Define color schemes
plotted_baseline_colors = [
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#7da0db",
]  # Colors for actual bars
legend_baseline_colors = [
    "#ffffff",
    "#ffffff",
] + plotted_baseline_colors  # Colors for legend (including random/majority)
ifp_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# Create figure with three subplots
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
    1, 5, figsize=(15, 5), sharey=True, width_ratios=[5, 4, 4, 4, 4]
)

# Plot baselines
x_baseline = np.arange(len(plotted_baseline_names))
bars = ax1.bar(x_baseline, baseline_f1, 1, color=plotted_baseline_colors)
# Add value labels inside each bar
for bar in bars:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height - 0.5,
        f"{height:.1f}",
        ha="center",
        va="top",
        color="white",
        zorder=2,
        fontsize=12,
        fontweight="bold",
    )
ax1.set_xlabel("Baselines")
ax1.set_xticks([])
ax1.set_ylabel("$F_a$ (%)")
ax1.tick_params(axis="y", colors="grey")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=6)

# Plot Interval-135M results
x = np.arange(1)  # Just for one size
for i, variant in enumerate(variants):
    offset = (i - 1.5) * bar_width
    values = [interval_data["135M"][i]]
    bars = ax2.bar(
        x + offset,
        values,
        bar_width,
        label=variant,
        color=ifp_colors[i],
    )
    # Add value labels inside each bar
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height - 0.5,
            f"{height:.1f}",
            ha="center",
            va="top",
            color="white",
            zorder=2,
            fontsize=12,
            fontweight="bold",
        )
ax2.set_xticks(x)
ax2.set_xticklabels(["Interval 135M"])
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=6)
ax2.tick_params(axis="y", length=0)

# Plot Interval-360M results
for i, variant in enumerate(variants):
    offset = (i - 1.5) * bar_width
    values = [interval_data["360M"][i]]
    bars = ax3.bar(
        x + offset,
        values,
        bar_width,
        label=variant,
        color=ifp_colors[i],
    )
    # Add value labels inside each bar
    for bar in bars:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height - 0.5,
            f"{height:.1f}",
            ha="center",
            va="top",
            color="white",
            zorder=2,
            fontsize=12,
            fontweight="bold",
        )
ax3.set_xticks([])
ax3.set_xlabel("Interval 360M")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["left"].set_visible(False)
ax3.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=6)
ax3.tick_params(axis="y", length=0)

# Plot IfP-135M results
for i, variant in enumerate(variants):
    offset = (i - 1.5) * bar_width
    values = [ifp_data["135M"][i]]
    bars = ax4.bar(
        x + offset,
        values,
        bar_width,
        label=variant,
        color=ifp_colors[i],
    )
    # Add value labels inside each bar
    for bar in bars:
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height - 0.5,
            f"{height:.1f}",
            ha="center",
            va="top",
            color="white",
            zorder=2,
            fontsize=12,
            fontweight="bold",
        )
ax4.set_xticks([])
ax4.set_xlabel("IfP-135M")
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.spines["left"].set_visible(False)
ax4.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=6)
ax4.tick_params(axis="y", length=0)

# Plot IfP-360M results
for i, variant in enumerate(variants):
    offset = (i - 1.5) * bar_width
    values = [ifp_data["360M"][i]]
    bars = ax5.bar(
        x + offset,
        values,
        bar_width,
        label=variant,
        color=ifp_colors[i],
    )
    # Add value labels inside each bar
    for bar in bars:
        height = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            height - 0.5,
            f"{height:.1f}",
            ha="center",
            va="top",
            color="white",
            zorder=2,
            fontsize=12,
            fontweight="bold",
        )
ax5.set_xticks([])
ax5.set_xlabel("IfP-360M")
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)
ax5.spines["left"].set_visible(False)
ax5.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=6)
ax5.tick_params(axis="y", length=0)
ax1.set_ylim(52, 72)

# Create baseline legend handles and labels manually
baseline_handles = [
    plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_baseline_colors
]
baseline_labels = [
    f"Random ({data['baselines']['random']['f1']:.1f} %)",
    f"Majority ({data['baselines']['majority']['f1']:.1f} %)",
] + plotted_baseline_names

# Get variant handles and labels
variant_handles, variant_labels = ax2.get_legend_handles_labels()

# Combine both sets of handles and labels
all_handles = baseline_handles + variant_handles
all_labels = baseline_labels + variant_labels

# Add combined legend to the figure
fig.legend(
    all_handles, all_labels, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=6
)

# Adjust layout and display
plt.tight_layout()
plt.savefig(
    IMGS_DIR / "interval_results_bars.pdf", format="pdf", bbox_inches="tight", dpi=600
)
