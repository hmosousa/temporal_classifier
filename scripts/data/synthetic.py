"""Generate synthetic data."""

import logging

import datasets
from src.base import ENDPOINT_TYPES, RELATIONS

from src.data.utils import get_entity_mapping
from src.model.gemini import GeminiAPI


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def add_relation_type(example: dict) -> dict:
    entity_map = get_entity_mapping(example["text"])
    for key, value in entity_map.items():
        if "source" in key:
            example["source_text"] = value
            example["source_type"] = key.split("_")[0]
        elif "target" in key:
            example["target_text"] = value
            example["target_type"] = key.split("_")[0]
    return example


def main():
    source_dataset = datasets.concatenate_datasets(
        [
            datasets.load_dataset(
                "hugosousa/TemporalQuestions", "default", split="train"
            ),
            datasets.load_dataset(
                "hugosousa/TemporalQuestions", "default", split="valid"
            ),
        ]
    )

    source_dataset = source_dataset.map(add_relation_type)
    pairs = [
        tuple(sorted([example["source_text"], example["target_text"]]))
        for example in source_dataset
    ]
    pairs = list(set(pairs))

    num_examples_to_generate = len(pairs) * len(RELATIONS) * len(ENDPOINT_TYPES) ** 2
    logging.info(f"Generating {num_examples_to_generate} examples")

    estimated_time = num_examples_to_generate // GeminiAPI.QUOTA_LIMIT_PER_MINUTE
    logging.info(
        f"Estimated time: {estimated_time} minutes aka {estimated_time // 60} hours"
    )

    # model = GeminiAPI()


if __name__ == "__main__":
    main()
