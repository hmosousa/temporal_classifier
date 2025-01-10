import json
from typing import Literal

import fire
import pandas as pd

from src.constants import RESULTS_DIR


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


def main(relation_type: Literal["point", "interval"] = "interval"):
    results = json.load(open(RESULTS_DIR / relation_type / "results.json"))

    for benchmark, models in results.items():
        print(benchmark)
        df = pd.DataFrame(models)
        df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)

        # Replace True with checkmark
        df["augmented"] = df["augmented"].apply(lambda x: "✅" if x else "❌")
        df["closure"] = df["closure"].apply(lambda x: "✅" if x else "❌")
        df["synthetic"] = df["synthetic"].apply(lambda x: "✅" if x else "❌")

        df.sort_values(by=["model"], inplace=True)
        print(df.to_markdown(index=False))
        print()


if __name__ == "__main__":
    fire.Fire(main)
