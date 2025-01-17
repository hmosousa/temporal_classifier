from typing import Literal

import datasets

from src.data.utils import POINT_EXPECTED_TAGS

_RELATION_MAP = {
    "BEFORE": "<",
    "AFTER": ">",
    "COEX": "=",
}


def load_timeset(split: Literal["valid", "test"] = "test", **kwargs):
    """Timeset is a point relation dataset."""
    if split == "valid":
        split = "validation"

    dataset = datasets.load_dataset(
        "kimihiroh/timeset", "pairwise", trust_remote_code=True, split=split
    )

    def process_example(example):
        source = example["arg1"]
        target = example["arg2"]

        text = (
            example["context"][: source["start"]]
            + f"<start_source>{source['mention']}</start_source>"
        )
        text += (
            example["context"][source["end"] : target["start"]]
            + f"<start_target>{target['mention']}</start_target>"
        )
        text += example["context"][target["end"] :]
        text = text.replace("\n", "").strip()

        tag_count = sum(1 for tag in POINT_EXPECTED_TAGS if tag in text)
        if tag_count != 4:
            return None

        return {
            "doc": example["filename"],
            "text": text,
            "label": _RELATION_MAP[example["relation"]],
        }

    processed_dataset = dataset.map(process_example)
    processed_dataset = processed_dataset.remove_columns(dataset.column_names)
    return processed_dataset
