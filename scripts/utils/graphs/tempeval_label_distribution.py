import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from src.constants import CACHE_DIR, IMGS_DIR
from src.data import augment_dataset, load_dataset


GREEN = "#56b355"
ORANGE = "#ff983d"
DARK_BLUE = "#2e6e98"
BLUE = "#4c92c3"
LIGHT_BLUE = "#64c8eb"


def create_distribution_subplot(
    ax,
    dataset_type,
    title,
    legend=False,
    xlabel=False,
):
    cachepath = CACHE_DIR / "graphs"
    cachepath.mkdir(exist_ok=True)
    cache_file = cachepath / f"{dataset_type}_label_counts.json"

    # Try to load from cache first
    if cache_file.exists():
        with open(cache_file, "r") as f:
            counts = json.load(f)
            train_closure_counter = Counter(counts["closure"])
            train_labels_counter = Counter(counts["raw"])
            augmented_labels_counter = Counter(counts["augmented"])
            augmented_closure_counter = Counter(counts["augmented_closure"])
            valid_labels_counter = Counter(counts["valid"])
            test_labels_counter = Counter(counts["test"])
    else:
        trainset_closure = load_dataset(
            f"{dataset_type}_tempeval", "train", closure=True
        )
        trainset = load_dataset(f"{dataset_type}_tempeval", "train")
        trainset_augmented = augment_dataset(trainset)
        trainset_augmented_closure = augment_dataset(trainset_closure)
        validset = load_dataset(f"{dataset_type}_tempeval", "valid")
        testset = load_dataset(f"{dataset_type}_tempeval", "test")

        train_closure_labels = trainset_closure["label"]
        train_closure_counter = Counter(train_closure_labels)
        train_labels = trainset["label"]
        train_labels_counter = Counter(train_labels)
        augmented_labels = trainset_augmented["label"]
        augmented_labels_counter = Counter(augmented_labels)
        augmented_closure_labels = trainset_augmented_closure["label"]
        augmented_closure_counter = Counter(augmented_closure_labels)
        valid_labels = validset["label"]
        valid_labels_counter = Counter(valid_labels)
        test_labels = testset["label"]
        test_labels_counter = Counter(test_labels)

        # Save counts to cache
        counts = {
            "closure": dict(train_closure_counter),
            "raw": dict(train_labels_counter),
            "augmented": dict(augmented_labels_counter),
            "augmented_closure": dict(augmented_closure_counter),
            "valid": dict(valid_labels_counter),
            "test": dict(test_labels_counter),
        }
        with open(cache_file, "w") as f:
            json.dump(counts, f)

    label_order = sorted(
        train_labels_counter.keys(), key=lambda x: train_labels_counter.get(x, 0)
    )

    # Create positions for the bars
    y_pos = np.arange(len(label_order)) * 1.5  # Increased spacing between label groups
    width = 0.22  # Width of each bar

    # Plot bars with adjusted positions
    test_counts = [test_labels_counter.get(label, 0) for label in label_order]
    valid_counts = [valid_labels_counter.get(label, 0) for label in label_order]
    train_counts = [train_labels_counter.get(label, 0) for label in label_order]
    augmented_counts = [augmented_labels_counter.get(label, 0) for label in label_order]
    closure_counts = [train_closure_counter.get(label, 0) for label in label_order]
    augmented_closure_counts = [
        augmented_closure_counter.get(label, 0) for label in label_order
    ]

    ax.barh(y_pos - width * 2.5, test_counts, width, label="Test", color=GREEN)
    ax.barh(y_pos - width * 1.5, valid_counts, width, label="Valid", color=ORANGE)
    ax.barh(y_pos - width * 0.5, train_counts, width, label="Raw", color=LIGHT_BLUE)
    ax.barh(y_pos + width * 0.5, augmented_counts, width, label="Inverse", color=BLUE)
    ax.barh(
        y_pos + width * 1.5, closure_counts, width, label="Closure", color=DARK_BLUE
    )
    ax.barh(
        y_pos + width * 2.5,
        augmented_closure_counts,
        width,
        label="Inverse & Closure",
        color="#1d4e6f",
    )

    # Update text positions to match new bar positions
    for i, (test, valid, train, augmented, closure, aug_closure) in enumerate(
        zip(
            test_counts,
            valid_counts,
            train_counts,
            augmented_counts,
            closure_counts,
            augmented_closure_counts,
        )
    ):
        ax.text(
            test,
            y_pos[i] - width * 2.5,
            f"{test:_}".replace("_", " ") + " ",
            va="center",
            ha="right",
            fontsize=4,
            color="white",
            weight="bold",
        )
        ax.text(
            valid,
            y_pos[i] - width * 1.5,
            f"{valid:_}".replace("_", " ") + " ",
            va="center",
            ha="right",
            fontsize=4,
            color="white",
            weight="bold",
        )
        ax.text(
            train,
            y_pos[i] - width * 0.5,
            f"{train:_}".replace("_", " ") + " ",
            va="center",
            ha="right",
            fontsize=4,
            color="white",
            weight="bold",
        )
        ax.text(
            augmented,
            y_pos[i] + width * 0.5,
            f"{augmented:_}".replace("_", " ") + " ",
            va="center",
            ha="right",
            fontsize=4,
            color="white",
            weight="bold",
        )
        ax.text(
            closure,
            y_pos[i] + width * 1.5,
            f"{closure:_}".replace("_", " ") + " ",
            va="center",
            ha="right",
            fontsize=4,
            color="white",
            weight="bold",
        )
        ax.text(
            aug_closure,
            y_pos[i] + width * 2.5,
            f"{aug_closure:_}".replace("_", " ") + " ",
            va="center",
            ha="right",
            fontsize=4,
            color="white",
            weight="bold",
        )

    # Set the y-tick labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label_order, fontsize=8)

    ax.set_title(title, fontsize=10)
    ax.set_xscale("log")

    # set right axis to invisible
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    if xlabel:
        ax.set_xlabel("Count (log scale)", fontsize=8, color="grey")


# Create figure with two vertically stacked subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 8), sharex=True, height_ratios=[11, 3])

# Create the interval relations distribution plot
create_distribution_subplot(ax1, "interval", "Interval Relations", legend=False)

# Create the point relations distribution plot
create_distribution_subplot(ax2, "point", "Point Relations", xlabel=True)

# Get handles and labels from the bottom subplot only
handles, labels = ax2.get_legend_handles_labels()

# Add legend with only the bottom subplot items
fig.legend(
    handles=handles,
    labels=labels,
    bbox_to_anchor=(0.5, -0.01),
    ncols=3,
    loc="center",
    borderaxespad=0.0,
    fontsize=8,
)

plt.tight_layout()
plt.savefig(
    IMGS_DIR / "tempeval_labels_distribution.pdf",
    format="pdf",
    dpi=600,
    bbox_inches="tight",
)
