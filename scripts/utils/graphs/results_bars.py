import json
from typing import Literal

import fire

import matplotlib.pyplot as plt
import numpy as np

from src.constants import IMGS_DIR, RESULTS_DIR


MODEL_ORDER = [
    "random",
    "majority",
    "smol-135-tq",
    "smol-135-tq-closure",
    "smol-135-tq-augment",
    "smol-135-tq-synthetic",
    "smol-135-tq-closure-augment",
    "smol-135-tq-closure-synthetic",
    "smol-135-tq-augment-synthetic",
    "smol-135-tq-closure-augment-synthetic",
]

MODEL_TO_LEGEND = {
    "random": "Random",
    "majority": "Majority",
    "smol-135-tq": "Raw",
    "smol-135-tq-closure": "Closure",
    "smol-135-tq-augment": "Augment",
    "smol-135-tq-synthetic": "Synthetic",
    "smol-135-tq-closure-augment": "Closure + Augment",
    "smol-135-tq-closure-synthetic": "Closure + Synthetic",
    "smol-135-tq-augment-synthetic": "Augment + Synthetic",
    "smol-135-tq-closure-augment-synthetic": "Closure + Augment + Synthetic",
}


BENCHMAKS_NAME_MAP = {
    "timeset": "TimeSet",
    "matres": "MATRES",
    "temporal_questions_closure": r"Temporal Questions$_{closure}$",
    "temporal_questions_raw": r"Temporal Questions$_{raw}$",
}


def main(metric: Literal["f1-score", "accuracy", "precision", "recall"] = "f1-score"):
    results = json.load(open(RESULTS_DIR / "point" / "results.json"))

    benchmarks = []
    model_results = {}
    model_confidence = {}  # New dictionary to store confidence intervals
    for benchmark, content in results.items():
        benchmarks.append(benchmark)
        for model in content:
            model_name = model["model"]
            if model_name not in model_results:
                model_results[model_name] = []
                model_confidence[model_name] = []
            model_results[model_name].append(model[metric])
            # Store confidence intervals as (lower_error, upper_error)
            confidence = model["confidence"][metric]
            model_confidence[model_name].append(
                (
                    model[metric] - confidence["lower"],  # lower error
                    confidence["upper"] - model[metric],  # upper error
                )
            )

    n_models = len(model_results) - 2  # exclude random and majority
    xs = np.arange(len(benchmarks))  # the label locations
    full_width = 0.8
    width = full_width / n_models
    multiplier = 0

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot bars
    for attribute in MODEL_ORDER:
        measurement = model_results[attribute]
        confidence = model_confidence[attribute]
        offset = width * multiplier

        if attribute in ["random", "majority"]:
            continue

        ax.barh(
            xs - offset,
            measurement,
            width,
            label=MODEL_TO_LEGEND[attribute],
        )

        # Add error bars
        error_positions = xs - offset
        lower_errors = [conf[0] for conf in confidence]
        upper_errors = [conf[1] for conf in confidence]
        ax.errorbar(
            measurement,
            error_positions,
            xerr=[lower_errors, upper_errors],
            fmt="none",
            color="black",
            capsize=3,
            capthick=1,
            elinewidth=1,
        )

        multiplier += 1

    # Plot lines
    for attribute in MODEL_ORDER:
        measurement = model_results[attribute]
        offset = width * multiplier

        if attribute == "random":
            for idx, (m, x) in enumerate(zip(measurement, xs)):
                if idx == 0:
                    label = "Random"
                else:
                    label = None
                ymin = x - full_width + width / 2
                ymax = x + width / 2

                ax.plot(
                    [m, m],
                    [ymin, ymax],
                    color="black",
                    linestyle="dotted",
                    linewidth=1,
                    label=label,
                )
        elif attribute == "majority":
            for idx, (m, x) in enumerate(zip(measurement, xs)):
                if idx == 0:
                    label = "Majority"
                else:
                    label = None
                ymin = x - full_width + width / 2
                ymax = x + width / 2
                ax.plot(
                    [m, m],
                    [ymin, ymax],
                    color="black",
                    linestyle="--",
                    linewidth=1,
                    label=label,
                )
        else:
            continue

    # Add some text for labels, title and custom x-axis tick labels, etc.
    if metric == "f1-score":
        ax.set_xlabel("F$_1$-Score")
        ax.set_title("F$_1$-Score by model")
    elif metric == "accuracy":
        ax.set_xlabel("Accuracy")
        ax.set_title("Accuracy by model")
    elif metric == "precision":
        ax.set_xlabel("Precision")
        ax.set_title("Precision by model")
    elif metric == "recall":
        ax.set_xlabel("Recall")
        ax.set_title("Recall by model")
    else:
        raise ValueError(f"Metric {metric} not supported")

    ax.set_yticks(
        ticks=xs - (full_width - width) / 2,
        labels=[BENCHMAKS_NAME_MAP[bench] for bench in benchmarks],
        rotation=90,
        va="center",
    )
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(IMGS_DIR / f"results_bars_{metric}.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    fire.Fire(main)
