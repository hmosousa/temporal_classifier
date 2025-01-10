"""Aggregate results from the results folder to a single results.json file."""

import json
from typing import Literal

from src.constants import RESULTS_DIR


def main(relation_type: Literal["point", "interval"] = "interval"):
    """Aggregate results from the results folder to a single results.json file."""
    results_dir = RESULTS_DIR / relation_type

    results = {}
    for file in results_dir.rglob("*.json"):
        if file.name == "results.json":
            # skip the results.json file that we are creating
            continue

        result = {}
        model = file.stem
        benchmark = file.parent.name
        content = json.load(file.open())

        result["model"] = model

        if "closure" in file.stem:
            result["closure"] = True
        else:
            result["closure"] = False

        if "augment" in file.stem:
            result["augmented"] = True
        else:
            result["augmented"] = False

        if "synthetic" in file.stem:
            result["synthetic"] = True
        else:
            result["synthetic"] = False

        if "accuracy" in content:
            result["accuracy"] = round(content["accuracy"] * 100, 2)
        else:
            result["accuracy"] = None

        if "micro avg" in content:
            result["micro avg f1-score"] = round(
                content["micro avg"]["f1-score"] * 100, 2
            )
        else:
            result["micro avg f1-score"] = None

        result["precision"] = round(content["weighted avg"]["precision"] * 100, 2)
        result["recall"] = round(content["weighted avg"]["recall"] * 100, 2)
        result["f1-score"] = round(content["weighted avg"]["f1-score"] * 100, 2)
        # result["other"] = content

        if benchmark not in results:
            results[benchmark] = []
        results[benchmark].append(result)

    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
