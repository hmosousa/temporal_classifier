"""Generate synthetic data."""

import logging
import random

import datasets

from src.base import ENDPOINT_TYPES, RELATIONS
from src.constants import CACHE_DIR
from src.data.utils import get_entity_mapping
from src.prompts import GenerationPrompter
from tqdm import tqdm

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


def generate_prompts(dataset: datasets.Dataset) -> list[str]:
    dataset = dataset.map(add_relation_type)
    pairs = [
        tuple(sorted([example["source_text"], example["target_text"]]))
        for example in dataset
    ]
    pairs = list(set(pairs))

    prompter = GenerationPrompter()

    examples_datasets = {
        (source_type, target_type, relation): dataset.filter(
            lambda x: x["source_type"] == source_type
            and x["target_type"] == target_type
            and x["label"] == relation
        )
        for relation in RELATIONS
        for source_type in ENDPOINT_TYPES
        for target_type in ENDPOINT_TYPES
    }

    # log the number of examples for each relation, source_type, target_type
    for relation in RELATIONS:
        for source_type in ENDPOINT_TYPES:
            for target_type in ENDPOINT_TYPES:
                examples = examples_datasets[(source_type, target_type, relation)]
                logging.info(
                    f"Number of examples for {relation} {source_type} {target_type}: {len(examples)}"
                )

    # create a progress bar
    pb = tqdm(total=len(pairs) * len(RELATIONS) * len(ENDPOINT_TYPES) ** 2)
    prompts = []
    for pair in pairs:
        for relation in RELATIONS:
            for source_type in ENDPOINT_TYPES:
                for target_type in ENDPOINT_TYPES:
                    examples = examples_datasets[(source_type, target_type, relation)]

                    # sample 10 examples
                    n_examples = 10
                    if len(examples) < n_examples:
                        n_examples = len(examples)
                    ids = random.sample(range(len(examples)), n_examples)
                    examples = examples.select(ids)

                    example = {
                        "source_text": pair[0],
                        "source_type": source_type,
                        "target_text": pair[1],
                        "target_type": target_type,
                        "label": relation,
                    }
                    prompt = prompter(example, examples)
                    prompts.append(prompt)
                    pb.update(1)
    return prompts


def main():
    dataset = datasets.concatenate_datasets(
        [
            datasets.load_dataset(
                "hugosousa/TemporalQuestions", "default", split="train"
            ),
            datasets.load_dataset(
                "hugosousa/TemporalQuestions", "default", split="valid"
            ),
        ]
    )

    output_dir = CACHE_DIR / "synthetic" / "prompts"
    if not output_dir.exists():
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        prompts = generate_prompts(dataset)
        prompts_dataset = datasets.Dataset.from_dict({"prompt": prompts})
        prompts_dataset.save_to_disk(output_dir)
    else:
        prompts_dataset = datasets.load_from_disk(output_dir)

    # num_examples_to_generate = len(prompts)
    # logging.info(f"Generating {num_examples_to_generate} examples")

    # model = GeminiAPI()
    # estimated_time = num_examples_to_generate // model.QUOTA_LIMIT_PER_MINUTE
    # logging.info(
    #     f"Estimated time: {estimated_time} minutes aka {estimated_time // 60} hours"
    # )

    # logging.info("Generating answers")
    # answers = model.generate(prompts)

    # dataset = datasets.Dataset.from_list(answers)


if __name__ == "__main__":
    main()
