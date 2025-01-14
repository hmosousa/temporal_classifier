from typing import Literal

import datasets

_RELATION_MAP = {
    "BEFORE": "<",
    "AFTER": ">",
    "COEX": "=",
}


def load_timeset(split: Literal["valid", "test"] = "test", **kwargs):
    if split == "valid":
        split = "validation"

    dataset = datasets.load_dataset(
        "kimihiroh/timeset", "pairwise", trust_remote_code=True, split=split
    )

    def process_example(example):
        source = example["arg1"]
        target = example["arg2"]

        context = (
            example["context"][: source["start"]]
            + f"<start_source>{source['mention']}</start_source>"
        )
        context += (
            example["context"][source["end"] : target["start"]]
            + f"<start_target>{target['mention']}</start_target>"
        )
        context += example["context"][target["end"] :]

        return {"text": context, "label": _RELATION_MAP[example["relation"]]}

    processed_dataset = dataset.map(process_example)
    processed_dataset = processed_dataset.remove_columns(dataset.column_names)
    return processed_dataset
