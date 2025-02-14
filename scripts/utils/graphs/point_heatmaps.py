import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.constants import IMGS_DIR, RESULTS_DIR

MODEL_ORDER = [
    "smol-135-0dd0da37",
    "smol-135-a-191329ff",
    "smol-135-c-3ed00d05",
    "smol-135-ac-a4eaad65",
    "smol-360-89128df1",
    "smol-360-a-4a820490",
    "smol-360-c-6af17138",
    "smol-360-ac-b19ae776",
]

MODEL_NAME_TO_DATASET = {
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
DATASETS = ["R", "I", "C", "IC"]

MODEL_NAME_TO_MODEL = {
    "smol-135-0dd0da37": "135M",
    "smol-135-a-191329ff": "135M",
    "smol-135-c-3ed00d05": "135M",
    "smol-135-ac-a4eaad65": "135M",
    "smol-360-89128df1": "360M",
    "smol-360-a-4a820490": "360M",
    "smol-360-c-6af17138": "360M",
    "smol-360-ac-b19ae776": "360M",
}

TYPES = ["ss", "se", "es", "ee"]
LABELS = ["<", ">", "="]


def create_point_heatmaps():
    # Initialize data storage
    data = {
        type_: {label: np.zeros((len(DATASETS), len(MODELS))) for label in LABELS}
        for type_ in TYPES
    }

    # Collect data
    for model_name in MODEL_ORDER:
        model_results = json.load(
            open(RESULTS_DIR / "point" / "point_tempeval" / f"{model_name}.json")
        )
        dataset = MODEL_NAME_TO_DATASET[model_name]
        model = MODEL_NAME_TO_MODEL[model_name]

        dataset_idx = DATASETS.index(dataset)
        model_idx = MODELS.index(model)

        for type_ in TYPES:
            per_label = model_results["pre_type"][type_]["per_label"]
            for label in LABELS:
                f1_score = per_label[label]["f1-score"]
                data[type_][label][dataset_idx, model_idx] = f1_score

    # Create plot
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))

    # Plot heatmaps
    for i, type_ in enumerate(TYPES):
        for j, label in enumerate(LABELS):
            ax = axes[i, j]
            sns.heatmap(
                data[type_][label],
                annot=True,
                fmt=".2f",
                cmap="RdYlBu",
                vmin=0,
                vmax=1,
                xticklabels=MODELS,
                yticklabels=DATASETS,
                ax=ax,
            )

            ax.set_title(f"Type: {type_}, Label: {label}")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = create_point_heatmaps()
    plt.savefig(IMGS_DIR / "point_heatmaps.png")
    plt.close()
