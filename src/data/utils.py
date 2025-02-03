import copy
import random
import re
from collections import Counter

import datasets
from tieval.base import Document
from tieval.entities import Timex

from tieval.links import TLink

from src.base import INVERT_POINT_RELATION
from src.constants import NEW_TOKENS

INTERVAL_EXPECTED_TAGS = ["<source>", "</source>", "<target>", "</target>"]
POINT_EXPECTED_TAGS = NEW_TOKENS


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
        example["label"] = INVERT_POINT_RELATION[example["label"]]
        return example

    mirror_dataset = dataset.map(augment_row)
    augmented_dataset = datasets.concatenate_datasets([dataset, mirror_dataset])
    return augmented_dataset


def add_tags(text: str, entities: list, dct: Timex = None) -> str:
    """Add tags to the text."""

    context = ""
    if dct and dct in entities:
        context = f"Documents creation time: <{dct.id}>{dct.text}</{dct.id}>\n"
        entities.remove(dct)
    else:
        context = f"Documents creation time: {dct.text}\n"

    entities = sorted(list(entities), key=lambda x: x.offsets[0])

    e_prev = 0
    for entity in entities:
        s, e = entity.offsets
        context += text[e_prev:s]
        context += f"<{entity.id}>{entity.text}</{entity.id}>"
        e_prev = e
    context += text[e:]
    return context


def get_tlink_context(doc: Document, tlink: TLink, just_sentences: bool = False):
    """Get the context of a tlink. The context are the sentences that contain the entities of the tlink."""
    entities_map = {ent.id: ent for ent in list(doc.entities) + [doc.dct]}

    if (
        tlink.source.id not in entities_map
        or tlink.target.id not in entities_map
        or tlink.source.id == tlink.target.id
    ):
        return

    has_dct = False
    if tlink.source.is_dct:
        entities = [tlink.target]
        has_dct = True
    elif tlink.target.is_dct:
        entities = [tlink.source]
        has_dct = True
    else:
        entities = [tlink.source, tlink.target]

    if just_sentences:
        offsets = [idx for ent in entities for idx in ent.offsets]

        min_offset = min(offsets)
        max_offset = max(offsets)

        # Get the sentences that contain the entities
        sentences = []
        min_sent_offset = None
        for sent in doc.sentences:
            s_sent, e_sent = sent.offsets
            if (
                s_sent <= min_offset <= e_sent
                or min_offset <= s_sent <= e_sent <= max_offset
                or s_sent <= max_offset <= e_sent
            ):
                sentences.append(str(sent))
                if min_sent_offset is None or s_sent < min_sent_offset:
                    min_sent_offset = s_sent
        context = " ".join(sentences)

        # Update entity offsets of the current context
        for idx, ent in enumerate(entities):
            ent_ = copy.deepcopy(ent)
            s_ent, e_ent = ent.offsets
            ent_.offsets = [s_ent - min_sent_offset, e_ent - min_sent_offset]
            entities[idx] = ent_
    else:
        context = doc.text

    if has_dct:
        context = add_tags(context, entities + [doc.dct], doc.dct)
    else:
        context = add_tags(context, entities, doc.dct)
    return context


def fix_closure_tlinks(tlinks: list[TLink], entity_map: dict):
    """Fix the closure of the tlinks."""
    # Temporal closure changes the entities from Entity to str.
    # This is a workaround to get the entities back.
    tlinks2drop = []
    for idx, tlink in enumerate(tlinks):
        if isinstance(tlink.source, str):
            if tlink.source in entity_map:
                tlink.source = entity_map[tlink.source]
            else:
                tlinks2drop.append(idx)
                continue

        if isinstance(tlink.target, str):
            if tlink.target in entity_map:
                tlink.target = entity_map[tlink.target]
            else:
                tlinks2drop.append(idx)
                continue

    for idx in sorted(tlinks2drop, reverse=True):
        tlinks.pop(idx)

    tlinks = set(tlinks)

    return tlinks
