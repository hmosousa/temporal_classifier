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


def main(metric: Literal["f1-score", "accuracy", "precision", "recall"] = "f1-score"):
    results = json.load(open(RESULTS_DIR / "point" / "results.json"))
    model_results = results["point_tempeval"]

    # Reorganize data by model size and configuration
    model_135 = {
        "R": model_results["smol-135-0dd0da37"],
        "I": model_results["smol-135-a-191329ff"],
        "C": model_results["smol-135-c-3ed00d05"],
        "IC": model_results["smol-135-ac-a4eaad65"],
    }

    model_360 = {
        "R": model_results["smol-360-89128df1"],
        "I": model_results["smol-360-a-4a820490"],
        "C": model_results["smol-360-c-6af17138"],
        "IC": model_results["smol-360-ac-b19ae776"],
    }

    configs = ["R", "I", "C", "IC"]
    x = np.arange(len(MODELS))  # [0, 1] for 135M and 360M
    width = 0.15  # Width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars for each configuration
    for idx, config in enumerate(configs):
        values = [
            model_135[config][0],
            model_360[config][0],
        ]  # Using first benchmark for example
        offset = width * (idx - len(configs) / 2 + 0.5)
        ax.bar(x + offset, values, width, label=config)

    # Customize the plot
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"{metric.capitalize()} by Model Size and Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.legend()

    # Add grid for better readability
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(IMGS_DIR / f"results_bars_{metric}.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    fire.Fire(main)
