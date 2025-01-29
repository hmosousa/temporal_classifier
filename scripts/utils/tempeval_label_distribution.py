from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from src.constants import IMGS_DIR
from src.data import load_dataset


def create_distribution_subplot(
    ax,
    dataset_type,
    title,
    legend=False,
):
    trainset = load_dataset(f"{dataset_type}_tempeval", "train")
    validset = load_dataset(f"{dataset_type}_tempeval", "valid")
    testset = load_dataset(f"{dataset_type}_tempeval", "test")

    train_labels = trainset["label"]
    train_labels_counter = Counter(train_labels)

    valid_labels = validset["label"]
    valid_labels_counter = Counter(valid_labels)

    test_labels = testset["label"]
    test_labels_counter = Counter(test_labels)

    label_order = sorted(
        train_labels_counter.keys(), key=lambda x: train_labels_counter[x]
    )

    # Create positions for the bars
    y_pos = np.arange(len(label_order))
    width = 0.25  # Width of each bar

    # Plot bars and store counts for labels
    test_counts = [test_labels_counter[label] for label in label_order]
    valid_counts = [valid_labels_counter[label] for label in label_order]
    train_counts = [train_labels_counter[label] for label in label_order]

    ax.barh(y_pos + width, train_counts, width, label="Train", alpha=0.8)
    ax.barh(y_pos, valid_counts, width, label="Valid", alpha=0.8)
    ax.barh(y_pos - width, test_counts, width, label="Test", alpha=0.8)

    # Add value labels at the end of each bar
    for i, (test, valid, train) in enumerate(
        zip(test_counts, valid_counts, train_counts)
    ):
        ax.text(
            test, y_pos[i] - width, f"{test:,} ", va="center", ha="right", fontsize=8
        )
        ax.text(valid, y_pos[i], f"{valid:,} ", va="center", ha="right", fontsize=8)
        ax.text(
            train, y_pos[i] + width, f"{train:,} ", va="center", ha="right", fontsize=8
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
            bbox_to_anchor=(0.75, -0.2),
            ncols=3,
            borderaxespad=0.0,
        )
        ax.set_xlabel("Count (log scale)")


# Create figure with two vertically stacked subplots
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(7, 10), sharex=True, height_ratios=[11, 3]
)

# Create the interval relations distribution plot
create_distribution_subplot(ax1, "interval", "Interval Relations")

# Create the point relations distribution plot
create_distribution_subplot(ax2, "point", "Point Relations", legend=True)


plt.tight_layout()
plt.savefig(IMGS_DIR / "tempeval_labels_distribution.png", dpi=600, bbox_inches="tight")
