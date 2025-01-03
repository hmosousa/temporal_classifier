import random
import re
from collections import Counter

import datasets

from src.base import INVERT_RELATION


def balance_dataset_classes(dataset: datasets.Dataset, column: str) -> datasets.Dataset:
    # Count the number of samples per class
    class_counts = Counter(dataset[column])

    if len(class_counts) <= 1:
        return dataset

    max_count = max(class_counts.values())

    label_datasets = []
    for ref_label in class_counts:
        label_dataset = dataset.filter(lambda x: x[column] == ref_label)
        oversampled_indices = random.choices(range(len(label_dataset)), k=max_count)
        label_dataset = label_dataset.select(oversampled_indices)
        label_datasets.append(label_dataset)
    balanced_dataset = datasets.concatenate_datasets(label_datasets)

    return balanced_dataset


def get_entity_mapping(context: str) -> dict:
    """Get entity id to entity text mapping from context.

    context (str): Context tagged with the entities. Example:
    ```
    The <e1>New York Times</e1> is a newspaper.
    ```

    Returns:
        dict: Entity id to entity text mapping.

    Example:
    ```
    {"e1": "New York Times"}
    ```
    """

    pattern = re.compile(r"<(.*?)>(.*?)</\1>")
    matches = pattern.findall(context)
    return {id: text for id, text in matches}


def drop_context_tags(context: str, exceptions: list[str] = []) -> str:
    pattern = re.compile(r"<(.*?)>(.*?)</\1>")
    matches = pattern.findall(context)
    for id, text in matches:
        if id not in exceptions:
            context = context.replace(f"<{id}>{text}</{id}>", text)
    return context


def augment_dataset(dataset: datasets.Dataset) -> datasets.Dataset:
    """Augment the dataset by replacing the source with the target relation."""

    def augment_row(example: dict) -> dict:
        # Use regex to swap _source and _target in a single operation
        example["text"] = re.sub(
            r"(_source>|_target>)",
            lambda m: "_target>" if m.group() == "_source>" else "_source>",
            example["text"],
        )
        example["label"] = INVERT_RELATION[example["label"]]
        return example

    mirror_dataset = dataset.map(augment_row)
    augmented_dataset = datasets.concatenate_datasets([dataset, mirror_dataset])
    return augmented_dataset
