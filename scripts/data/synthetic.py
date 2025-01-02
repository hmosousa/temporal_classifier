"""Generate synthetic data."""

import asyncio
import logging
import random

import datasets

from src.base import ENDPOINT_TYPES, RELATIONS
from src.constants import CACHE_DIR, DATA_DIR
from src.data.utils import get_entity_mapping
from src.model.gemini import GeminiAPI
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
                few_shot_examples = examples_datasets[
                    (source_type, target_type, relation)
                ]
                logging.info(
                    f"Number of examples for {relation} {source_type} {target_type}: {len(few_shot_examples)}"
                )

    # create a progress bar
    pb = tqdm(total=len(pairs) * len(RELATIONS) * len(ENDPOINT_TYPES) ** 2)
    examples = []
    for pair in pairs:
        for relation in RELATIONS:
            for source_type in ENDPOINT_TYPES:
                for target_type in ENDPOINT_TYPES:
                    few_shot_examples = examples_datasets[
                        (source_type, target_type, relation)
                    ]

                    # sample 10 examples
                    n_examples = 10
                    if len(few_shot_examples) < n_examples:
                        n_examples = len(few_shot_examples)
                    ids = random.sample(range(len(few_shot_examples)), n_examples)
                    few_shot_examples = few_shot_examples.select(ids)

                    example = {
                        "source_text": pair[0],
                        "source_type": source_type,
                        "target_text": pair[1],
                        "target_type": target_type,
                        "label": relation,
                    }
                    example["prompt"] = prompter(example, few_shot_examples)
                    examples.append(example)
                    pb.update(1)
    return examples


async def main(use_cache: bool = True):
    # Generate all the prompts
    output_dir = CACHE_DIR / "synthetic" / "prompts"
    if not output_dir.exists() or not use_cache:
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
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        prompts = generate_prompts(dataset)
        prompts_dataset = datasets.Dataset.from_list(prompts)
        prompts_dataset.save_to_disk(output_dir)
    else:
        prompts_dataset = datasets.load_from_disk(output_dir)

    num_examples_to_generate = len(prompts_dataset)
    logging.info(f"Generating {num_examples_to_generate} examples")

    # Call the Gemini API to generate the answers
    raw_output_dir = DATA_DIR / "synthetic" / "raw"
    if not raw_output_dir.exists() or not use_cache:
        raw_output_dir.mkdir(parents=True, exist_ok=True)

        model = GeminiAPI()
        estimated_time = num_examples_to_generate // model.QUOTA_LIMIT_PER_MINUTE
        logging.info(
            f"Estimated time: {estimated_time} minutes aka {estimated_time // 60} hours"
        )

        logging.info("Generating answers")
        answers = await model(
            prompts_dataset["prompt"], CACHE_DIR / "synthetic" / "gemini" / "answers"
        )

        prompt_answers = prompts_dataset.add_column("answer", answers)
        prompt_answers.save_to_disk(raw_output_dir)
    else:
        prompt_answers = datasets.load_from_disk(raw_output_dir)

    # Verify the answers

    # TODO: Remove this
    # infos = prompt_answers["prompt"]
    # data = datasets.Dataset.from_list(infos)
    # data = data.add_column("answer", prompt_answers["answer"])
    # prompt_answers = data

    # def check_answer(example: dict) -> bool:
    #     if example["source_type"] == "start" and example["target_type"] == "start":
    #         if "<start_source>" in example["answer"] and "</start_source>" in example["answer"] and "<start_target>" in example["answer"] and "</start_target>" in example["answer"]:
    #             return True
    #     elif example["source_type"] == "end" and example["target_type"] == "end":
    #         if "<end_source>" in example["answer"] and "</end_source>" in example["answer"] and "<end_target>" in example["answer"] and "</end_target>" in example["answer"]:
    #             return True
    #     elif example["source_type"] == "start" and example["target_type"] == "end":
    #         if "<start_source>" in example["answer"] and "</start_source>" in example["answer"] and "<end_target>" in example["answer"] and "</end_target>" in example["answer"]:
    #             return True
    #     elif example["source_type"] == "end" and example["target_type"] == "start":
    #         if "<end_source>" in example["answer"] and "</end_source>" in example["answer"] and "<start_target>" in example["answer"] and "</start_target>" in example["answer"]:
    #             return True
    #     return False

    # prompt_answers = prompt_answers.filter(check_answer)
    # prompt_answers.save_to_disk(DATA_DIR / "synthetic" / "clean")
    # prompt_answers.push_to_hub("hugosousa/tmp")


if __name__ == "__main__":
    asyncio.run(main())
