"""Aggregate results from the results folder to a single results.json file."""

import json

from src.constants import RESULTS_DIR


def main():
    """Aggregate results from the results folder to a single results.json file."""
    results = {}
    for file in RESULTS_DIR.rglob("*.json"):
        if file.name == "results.json":
            # skip the results.json file that we are creating
            continue

        result = {}
        if file.stem in ["majority", "random"]:
            model = file.stem
            benchmark = file.parent.name
            content = json.load(file.open())
            result["model"] = model
            result["raw"] = True
            result["augmented"] = False
            result["synthetic"] = False
        else:
            *model, train_dataset, augmented = file.stem.split("-")
            model = "-".join(model)
            benchmark = file.parent.name
            content = json.load(file.open())
            result["model"] = model

            if train_dataset in ["temporal_questions", "all_temporal_questions"]:
                result["raw"] = True
            else:
                result["raw"] = False

            if augmented == "True":
                result["augmented"] = True
            else:
                result["augmented"] = False

            if train_dataset == "temporal_questions":
                result["synthetic"] = False
            else:
                result["synthetic"] = True

        result["accuracy"] = round(content["accuracy"] * 100, 2)
        result["precision"] = round(content["macro avg"]["precision"] * 100, 2)
        result["recall"] = round(content["macro avg"]["recall"] * 100, 2)
        result["f1-score"] = round(content["macro avg"]["f1-score"] * 100, 2)
        # result["other"] = content

        if benchmark not in results:
            results[benchmark] = []
        results[benchmark].append(result)

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
