import json

import pandas as pd

from src.constants import RESULTS_DIR


MODEL_ORDER = ["random", "majority", "SmolLM2-135M", "SmolLM2-360M", "SmolLM2-1.7B"]


def main():
    results = json.load(open(RESULTS_DIR / "results.json"))

    for benchmark, models in results.items():
        print(benchmark)
        df = pd.DataFrame(models)
        df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
        df.sort_values(by=["model"], inplace=True)
        print(df.to_markdown(index=False))
        print()


if __name__ == "__main__":
    main()
