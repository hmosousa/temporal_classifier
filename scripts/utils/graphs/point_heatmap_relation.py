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
RELATIONS = ["<", ">", "="]

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


def create_relation_heatmaps():
    # Initialize data storage for each model size
    data = {model: np.zeros((len(RELATIONS), len(DATASETS))) for model in MODELS}

    # Collect data
    for model_name in MODEL_ORDER:
        model_results = json.load(
            open(RESULTS_DIR / "point" / "point_tempeval" / f"{model_name}.json")
        )
        dataset = MODEL_NAME_TO_DATASET[model_name]
        model = MODEL_NAME_TO_MODEL[model_name]

        dataset_idx = DATASETS.index(dataset)

        # Average F1 scores across all types for each relation
        for relation_idx, relation in enumerate(RELATIONS):
            f1_scores = []
            for type_ in ["ss", "se", "es", "ee"]:
                f1_score = model_results["pre_type"][type_]["per_label"][relation][
                    "f1-score"
                ]
                f1_scores.append(f1_score)
            data[model][relation_idx, dataset_idx] = np.mean(f1_scores)

    # Create plot with 1 row and 2 columns (one for each model size)
    fig, axes = plt.subplots(1, 2, figsize=(6, 2.8))

    # Plot heatmaps
    for idx, model in enumerate(MODELS):
        ax = axes[idx]

        # Create annotation text with F1 scores
        f1_values = data[model]
        annotations = np.array([f"{f1*100:.1f}" for f1 in f1_values.flatten()]).reshape(
            f1_values.shape
        )

        sns.heatmap(
            data[model] * 100,
            annot=annotations,
            fmt="",
            cmap="RdYlBu",
            vmin=0,
            vmax=100,
            xticklabels=DATASETS,
            yticklabels=RELATIONS if idx == 0 else False,
            ax=ax,
            cbar=False,
            annot_kws={"size": 10},
            linewidths=0.5,
            linecolor="white",
        )

        ax.set_yticklabels(RELATIONS if idx == 0 else [], rotation=0)

        ax.set_title(f"{model} Model", fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = create_relation_heatmaps()
    plt.savefig(IMGS_DIR / "point_relation_heatmaps.pdf", format="pdf", dpi=600)
    plt.close()
