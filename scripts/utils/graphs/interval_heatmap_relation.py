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
    "smol-135-interval-1b7d11c1",
    "smol-135-interval-a-6ba0463e",
    "smol-135-interval-c-7a430df7",
    "smol-135-interval-ca-86f6ae17",
    "smol-360-interval-575aff8f",
    "smol-360-interval-a-04fbd03a",
    "smol-360-interval-c-6bd44a78",
    "smol-360-interval-ca-8bd7cf03",
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
    "smol-135-interval-1b7d11c1": "R",
    "smol-135-interval-a-6ba0463e": "I",
    "smol-135-interval-c-7a430df7": "C",
    "smol-135-interval-ca-86f6ae17": "IC",
    "smol-360-interval-575aff8f": "R",
    "smol-360-interval-a-04fbd03a": "I",
    "smol-360-interval-c-6bd44a78": "C",
    "smol-360-interval-ca-8bd7cf03": "IC",
}

MODELS = ["135M", "360M"]
DATASETS = ["R", "I", "C", "IC"]
RELATIONS = [
    "BEFORE",
    "AFTER",
    "SIMULTANEOUS",
    "INCLUDES",
    "IS_INCLUDED",
    "IBEFORE",
    "IAFTER",
    "ENDS",
    "ENDED_BY",
    "BEGINS",
    "BEGUN_BY",
    # "OVERLAPS",
    # "OVERLAPPED_BY",
]

MODEL_NAME_TO_MODEL = {
    "smol-135-0dd0da37": "IfP-135M",
    "smol-135-a-191329ff": "IfP-135M",
    "smol-135-c-3ed00d05": "IfP-135M",
    "smol-135-ac-a4eaad65": "IfP-135M",
    "smol-360-89128df1": "IfP-360M",
    "smol-360-a-4a820490": "IfP-360M",
    "smol-360-c-6af17138": "IfP-360M",
    "smol-360-ac-b19ae776": "IfP-360M",
    "smol-135-interval-1b7d11c1": "I-135M",
    "smol-135-interval-a-6ba0463e": "I-135M",
    "smol-135-interval-c-7a430df7": "I-135M",
    "smol-135-interval-ca-86f6ae17": "I-135M",
    "smol-360-interval-575aff8f": "I-360M",
    "smol-360-interval-a-04fbd03a": "I-360M",
    "smol-360-interval-c-6bd44a78": "I-360M",
    "smol-360-interval-ca-8bd7cf03": "I-360M",
}


MODEL_NAME_MAP = {
    "135M Point": "IfP-135M",
    "135M Interval": "Interval 135M",
    "360M Point": "IfP-360M",
    "360M Interval": "Interval 360M",
}


def create_relation_heatmaps():
    # Initialize data storage for point and interval models
    data = {
        "135M-point": np.zeros((len(RELATIONS), len(DATASETS))),
        "360M-point": np.zeros((len(RELATIONS), len(DATASETS))),
        "135M-interval": np.zeros((len(RELATIONS), len(DATASETS))),
        "360M-interval": np.zeros((len(RELATIONS), len(DATASETS))),
    }

    # Store support values for the first model (smol-135-0dd0da37)
    support_values = []

    # Collect data
    for model_name in MODEL_ORDER:
        is_interval = "interval" in model_name
        model = MODEL_NAME_TO_MODEL[model_name]
        dataset = MODEL_NAME_TO_DATASET[model_name]

        # Load appropriate results file
        model_results = json.load(
            open(
                RESULTS_DIR
                / "interval"
                / "interval_tempeval"
                / "most_likely"
                / f"{model_name}.json"
            )
        )

        # Store the first model's results for support values
        if model_name == "smol-135-0dd0da37":
            support_values = [
                str(int(model_results.get(rel, {}).get("support", 0)))
                for rel in RELATIONS
            ]

        dataset_idx = DATASETS.index(dataset)

        # Determine the correct key for the data dictionary
        model_size = "135M" if "135" in model_name else "360M"
        approach = "interval" if is_interval else "point"
        data_key = f"{model_size}-{approach}"

        # Average F1 scores across all types for each relation
        for relation_idx, relation in enumerate(RELATIONS):
            data[data_key][relation_idx, dataset_idx] = model_results.get(
                relation, {}
            ).get("f1-score", 0)

    # Create plot with 1 row and 4 columns
    fig, axes = plt.subplots(
        1, 4, figsize=(15, 6)
    )  # Increased width to accommodate support numbers

    # Plot heatmaps
    plot_configs = [
        ("135M", "point"),
        ("135M", "interval"),
        ("360M", "point"),
        ("360M", "interval"),
    ]

    for idx, (model, approach) in enumerate(plot_configs):
        key = f"{model}-{approach}"
        ax = axes[idx]

        # Create annotation text with F1 scores
        f1_values = data[key]
        annotations = np.array([f"{f1*100:.1f}" for f1 in f1_values.flatten()]).reshape(
            f1_values.shape
        )

        sns.heatmap(
            data[key] * 100,
            annot=annotations,
            fmt="",
            cmap="RdYlBu",
            vmin=0,
            vmax=100,
            xticklabels=DATASETS,
            yticklabels=RELATIONS if idx == 0 else False,
            ax=ax,
            cbar=False,
            annot_kws={"size": 8},
            linewidths=0.5,
            linecolor="white",
        )

        ax.set_yticklabels(RELATIONS if idx == 0 else [], rotation=0)
        ax.set_title(MODEL_NAME_MAP[f"{model} {approach.capitalize()}"], fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Add support numbers to the right of the last heatmap
        if idx == 3:
            for i, support in enumerate(support_values):
                ax.text(
                    len(DATASETS) + 0.2,
                    i + 0.5,
                    f"n={support}",
                    va="center",
                    fontsize=8,
                )

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = create_relation_heatmaps()
    plt.savefig(IMGS_DIR / "interval_relation_heatmaps.pdf", format="pdf", dpi=600)
    plt.close()
    print(f"Saved to {IMGS_DIR / 'interval_relation_heatmaps.pdf'}")
