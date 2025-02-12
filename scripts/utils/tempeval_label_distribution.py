from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from src.constants import IMGS_DIR
from src.data import load_dataset, augment_dataset


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
    trainset_closure = load_dataset(f"{dataset_type}_tempeval", "train", closure=True)
    trainset = load_dataset(f"{dataset_type}_tempeval", "train")
    trainset_augmented = augment_dataset(trainset)
    validset = load_dataset(f"{dataset_type}_tempeval", "valid")
    testset = load_dataset(f"{dataset_type}_tempeval", "test")

    train_closure_labels = trainset_closure["label"]
    train_closure_counter = Counter(train_closure_labels)
    train_labels = trainset["label"]
    train_labels_counter = Counter(train_labels)
    augmented_labels = trainset_augmented["label"]
    augmented_labels_counter = Counter(augmented_labels)
    valid_labels = validset["label"]
    valid_labels_counter = Counter(valid_labels)
    test_labels = testset["label"]
    test_labels_counter = Counter(test_labels)

    label_order = sorted(
        train_labels_counter.keys(), key=lambda x: train_labels_counter.get(x, 0)
    )

    # Create positions for the bars
    y_pos = np.arange(len(label_order))
    width = 0.18  # Reduced width to accommodate 5 bars

    # Plot bars and store counts for labels
    test_counts = [test_labels_counter.get(label, 0) for label in label_order]
    valid_counts = [valid_labels_counter.get(label, 0) for label in label_order]
    train_counts = [train_labels_counter.get(label, 0) for label in label_order]
    augmented_counts = [augmented_labels_counter.get(label, 0) for label in label_order]
    closure_counts = [train_closure_counter.get(label, 0) for label in label_order]

    ax.barh(y_pos + width*2, closure_counts, width, label="Closure", color=DARK_BLUE)
    ax.barh(y_pos + width, augmented_counts, width, label="Inverse", color=BLUE)
    ax.barh(y_pos, train_counts, width, label="Train", color=LIGHT_BLUE)
    ax.barh(y_pos - width, valid_counts, width, label="Valid", color=ORANGE)
    ax.barh(y_pos - width*2, test_counts, width, label="Test", color=GREEN)

    # Add value labels at the end of each bar
    for i, (test, valid, augmented, train, closure) in enumerate(
        zip(test_counts, valid_counts, augmented_counts, train_counts, closure_counts)
    ):
        ax.text(
            test, y_pos[i] - width*2, f"{test:,} ", va="center", ha="right", fontsize=6, color="white", weight="bold"
        )
        ax.text(
            valid, y_pos[i] - width, f"{valid:,} ", va="center", ha="right", fontsize=6, color="white" , weight="bold"
        )
        ax.text(
            augmented, y_pos[i] + width, f"{augmented:,} ", va="center", ha="right", fontsize=6, color="white" , weight="bold"
        )
        ax.text(
            train, y_pos[i], f"{train:,} ", va="center", ha="right", fontsize=6, color="white" , weight="bold"
        )   
        ax.text(
            closure, y_pos[i] + width*2, f"{closure:,} ", va="center", ha="right", fontsize=6, color="white" , weight="bold"
        )

    # Set the y-tick labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label_order)

    ax.set_title(title)
    ax.set_xscale("log")

    # set right axis to invisible
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    if legend:
        ax.legend(
            bbox_to_anchor=(0.8, -0.2),
            ncols=3,
            borderaxespad=0.0,
        )
        
    if xlabel:
        ax.set_xlabel("Count (log scale)")


# Create figure with two vertically stacked subplots
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(5, 11), sharex=True, height_ratios=[11, 3]
)

# Create the interval relations distribution plot
create_distribution_subplot(ax1, "interval", "Interval Relations", )

# Create the point relations distribution plot
create_distribution_subplot(ax2, "point", "Point Relations", xlabel=True, legend=True)


plt.tight_layout()
plt.savefig(IMGS_DIR / "tempeval_labels_distribution.pdf", format="pdf", dpi=600, bbox_inches="tight")
