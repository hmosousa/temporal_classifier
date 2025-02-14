import json
from typing import Literal

import fire

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


def main(metric: Literal["f1-score", "accuracy", "precision", "recall"] = "accuracy"):
    results = json.load(open(RESULTS_DIR / "point" / "results.json"))
    model_results = {r["model"]: r for r in results["point_tempeval"]}

    # Reorganize data by model size and configuration
    model_135 = {
        "Raw": model_results["smol-135-0dd0da37"],
        "Inverse": model_results["smol-135-a-191329ff"],
        "Closure": model_results["smol-135-c-3ed00d05"],
        "Inverse & Closure": model_results["smol-135-ac-a4eaad65"],
    }

    model_360 = {
        "Raw": model_results["smol-360-89128df1"],
        "Inverse": model_results["smol-360-a-4a820490"],
        "Closure": model_results["smol-360-c-6af17138"],
        "Inverse & Closure": model_results["smol-360-ac-b19ae776"],
    }

    configs = ["Raw", "Inverse", "Closure", "Inverse & Closure"]
    x = np.arange(len(MODELS))  # [0, 1] for 135M and 360M
    width = 0.2  # Width of the bars

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot bars for each configuration
    for idx, config in enumerate(configs):
        values = [
            model_135[config][metric],
            model_360[config][metric],
        ]

        # Calculate error bar lengths
        yerr_lower = [
            model_135[config][metric]
            - model_135[config]["confidence"][metric]["lower"],
            model_360[config][metric]
            - model_360[config]["confidence"][metric]["lower"],
        ]
        yerr_upper = [
            model_135[config]["confidence"][metric]["upper"]
            - model_135[config][metric],
            model_360[config]["confidence"][metric]["upper"]
            - model_360[config][metric],
        ]
        yerr = [yerr_lower, yerr_upper]

        offset = width * (idx - len(configs) / 2 + 0.5)
        bars = ax.bar(x + offset, values, width, label=config)

        # Add value labels inside each bar with lower z-order
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
            )  # Lower z-order for text

        # Error bars with higher z-order to appear on top
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
        )  # Higher z-order for error bars

    # Customize the plot
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.legend(bbox_to_anchor=(0.5, -0.05), loc="upper center", ncol=4)

    ax.set_ylim(72, 85)

    # Add grid for better readability
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=6)

    # Remove top, right, and left spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Make y-axis ticks grey
    ax.tick_params(axis="y", colors="grey")

    plt.tight_layout()
    plt.savefig(
        IMGS_DIR / f"results_bars_{metric}.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=600,
    )


if __name__ == "__main__":
    fire.Fire(main)
