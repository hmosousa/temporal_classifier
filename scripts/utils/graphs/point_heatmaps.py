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
TYPE_LABELS = {
    "ss": "Start-Start",
    "se": "Start-End",
    "es": "End-Start",
    "ee": "End-End",
}

LABELS = ["<", ">", "="]


def create_point_heatmaps():
    # Initialize data storage
    data = {
        type_: {
            label: {
                "f1": np.zeros((len(DATASETS), len(MODELS))),
            }
            for label in LABELS
        }
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
                support = per_label[label]["support"]
                data[type_][label]["f1"][dataset_idx, model_idx] = f1_score
                data[type_][label]["support"] = support

    # Create plot with transposed layout: 3 rows (labels) x 4 columns (types)
    fig, axes = plt.subplots(len(LABELS), len(TYPES), figsize=(8, 12))

    # Plot heatmaps
    for i, label in enumerate(LABELS):
        for j, type_ in enumerate(TYPES):
            ax = axes[i, j]

            # Create annotation text with F1 scores
            f1_values = data[type_][label]["f1"]
            annotations = np.array(
                [f"{f1*100:.1f}" for f1 in f1_values.flatten()]
            ).reshape(f1_values.shape)

            sns.heatmap(
                data[type_][label]["f1"] * 100,
                annot=annotations,
                fmt="",
                cmap="RdYlBu",
                vmin=0,
                vmax=100,
                xticklabels=MODELS if i == len(LABELS) - 1 else False,
                yticklabels=DATASETS if j == len(TYPES) - 1 else False,
                ax=ax,
                cbar=False,
                annot_kws={"weight": "bold", "size": 10},
                linewidths=0.5,
                linecolor="white",
            )

            # Add total support in the middle of the heatmap
            total_support = int(data[type_][label]["support"])
            ax.text(
                0.5,
                0.5,
                f"n={total_support}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
            )

            # Only show type in the top row
            if i == 0:
                ax.set_title(TYPE_LABELS[type_], fontsize=18)
            else:
                ax.set_title("")

            # Only show label on the left side
            if j == 0:
                ax.set_ylabel(
                    label.upper(), fontsize=18, rotation=0, ha="right", va="center"
                )
            else:
                ax.set_ylabel("")

            # Set x and y label text only for the rightmost and bottom plots
            if j == len(TYPES) - 1:
                ax.yaxis.set_ticks_position("right")
                ax.set_ylabel("")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = create_point_heatmaps()
    plt.savefig(IMGS_DIR / "point_heatmaps.pdf", format="pdf", dpi=600)
    plt.close()
