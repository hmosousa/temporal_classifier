import json

import matplotlib.pyplot as plt
import numpy as np

from src.constants import IMGS_DIR, RESULTS_DIR


MODEL_ORDER = [
    "random",
    "majority",
    "smol-135-0dd0da37",
    "smol-135-a-191329ff",
    "smol-135-c-3ed00d05",
    "smol-135-ac-a4eaad65",
    "smol-360-89128df1",
    "smol-360-a-4a820490",
    "smol-360-c-6af17138",
    "smol-360-ac-b19ae776",
]

MODEL_TO_LEGEND = {
    "random": "Random",
    "majority": "Majority",
    "smol-135-0dd0da37": "R",
    "smol-135-a-191329ff": "I",
    "smol-135-c-3ed00d05": "C",
    "smol-135-ac-a4eaad65": "IC",
    "smol-360-89128df1": "R",
    "smol-360-a-4a820490": "I",
    "smol-360-c-6af17138": "C",
    "smol-360-ac-b19ae776": "IC",
}

MODELS = ["135M", "360M"]


def create_subplot(ax, results_data, metric, models, configs, width):
    x = np.arange(len(models))

    # Plot bars for each configuration
    for idx, config in enumerate(configs):
        values = [
            results_data["135"][config][metric],
            results_data["360"][config][metric],
        ]

        # Calculate error bar lengths
        yerr_lower = [
            results_data["135"][config][metric]
            - results_data["135"][config]["confidence"][metric]["lower"],
            results_data["360"][config][metric]
            - results_data["360"][config]["confidence"][metric]["lower"],
        ]
        yerr_upper = [
            results_data["135"][config]["confidence"][metric]["upper"]
            - results_data["135"][config][metric],
            results_data["360"][config]["confidence"][metric]["upper"]
            - results_data["360"][config][metric],
        ]
        yerr = [yerr_lower, yerr_upper]

        offset = width * (idx - len(configs) / 2 + 0.5)
        bars = ax.bar(x + offset, values, width, label=config)

        # Add value labels inside each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(
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

        # Error bars
        ax.errorbar(
            x + offset,
            values,
            yerr=yerr,
            fmt="none",
            color="black",
            capsize=5,
            capthick=2,
            elinewidth=0.5,
            alpha=0.5,
            zorder=4,
        )

    # Customize the subplot
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", colors="grey")


def main():
    results = json.load(open(RESULTS_DIR / "point" / "results.json"))
    model_results = {r["model"]: r for r in results["point_tempeval"]}

    # Reorganize data by model size and configuration
    results_data = {
        "135": {
            "Raw": model_results["smol-135-0dd0da37"],
            "Inverse": model_results["smol-135-a-191329ff"],
            "Closure": model_results["smol-135-c-3ed00d05"],
            "Inverse & Closure": model_results["smol-135-ac-a4eaad65"],
        },
        "360": {
            "Raw": model_results["smol-360-89128df1"],
            "Inverse": model_results["smol-360-a-4a820490"],
            "Closure": model_results["smol-360-c-6af17138"],
            "Inverse & Closure": model_results["smol-360-ac-b19ae776"],
        },
    }

    configs = ["Raw", "Inverse", "Closure", "Inverse & Closure"]
    width = 0.2

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), height_ratios=[1, 1])

    # Create accuracy subplot
    create_subplot(ax1, results_data, "accuracy", MODELS, configs, width)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(72, 83.9)

    # Create F1-score subplot
    create_subplot(ax2, results_data, "f1-score", MODELS, configs, width)
    ax2.set_ylabel("Macro $F_1$ (%)")
    ax2.set_ylim(60, 75.5)
    # Add legend at the bottom
    ax2.legend(bbox_to_anchor=(0.5, -0.05), loc="upper center", ncol=4)

    plt.tight_layout()
    plt.savefig(
        IMGS_DIR / "point_barplots.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=600,
    )


if __name__ == "__main__":
    main()
