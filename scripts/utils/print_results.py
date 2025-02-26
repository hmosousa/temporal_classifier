import json
from typing import Literal

import fire
import pandas as pd

from src.constants import RESULTS_DIR


def main(relation_type: Literal["point", "interval"] = "point"):
    results = json.load(open(RESULTS_DIR / relation_type / "results.json"))

    for benchmark, models in results.items():
        print(benchmark)
        df = pd.DataFrame(models)

        # Replace True with checkmark
        df["augmented"] = df["augmented"].apply(lambda x: "✅" if x else "❌")
        df["closure"] = df["closure"].apply(lambda x: "✅" if x else "❌")

        df.drop(columns=["confidence"], inplace=True)

        df.sort_values(by=["model"], inplace=True)
        print(df.to_markdown(index=False))
        print()


if __name__ == "__main__":
    fire.Fire(main)
