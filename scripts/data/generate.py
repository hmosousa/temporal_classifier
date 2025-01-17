"""Generate Temporal Questions dataset from TemporalEval3 corpus."""

import logging
from collections import Counter
from pathlib import Path

import datasets
import fire
import tieval.datasets
from sklearn.model_selection import train_test_split

from src.base import Timeline
from src.constants import HF_TOKEN
from src.data.utils import get_tlink_context
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"


def doc2questions(doc, closure: bool, just_sentences: bool = False):
    entities_map = {ent.id: ent for ent in doc.entities + [doc.dct]}
    if closure:
        tlinks = list(doc.temporal_closure)

        # Temporal closure changes the entities from Entity to str.
        # This is a workaround to get the entities back.
        tlinks2drop = []
        for idx, tlink in enumerate(tlinks):
            if isinstance(tlink.source, str):
                if tlink.source in entities_map:
                    tlink.source = entities_map[tlink.source]
                else:
                    tlinks2drop.append(idx)
                    continue

            if isinstance(tlink.target, str):
                if tlink.target in entities_map:
                    tlink.target = entities_map[tlink.target]
                else:
                    tlinks2drop.append(idx)
                    continue

        for idx in sorted(tlinks2drop, reverse=True):
            tlinks.pop(idx)

        tlinks = set(tlinks)
    else:
        tlinks = doc.tlinks

    samples = []
    for tlink in tlinks:
        if (
            tlink.source.id not in entities_map
            or tlink.target.id not in entities_map
            or tlink.source.id == tlink.target.id
        ):
            continue

        context = get_tlink_context(doc, tlink, just_sentences=just_sentences)

        tlink.source = tlink.source.id
        tlink.target = tlink.target.id
        timeline = Timeline(tlinks=[tlink]).to_dict()

        samples.append(
            {
                "id": f"{doc.name}",
                "context": context,
                "source": tlink.source,
                "target": tlink.target,
                "timeline": timeline["relations"],
            }
        )
    return samples


def transform_corpus(documents, closure: bool, just_sentences: bool = False):
    # Transform the documents into pair-wise contexts
    # Each tlink has its own context
    samples = []
    for doc in tqdm(documents):
        samples += doc2questions(doc, closure=closure, just_sentences=just_sentences)

    # Transform the contexts to have the special tokens
    examples = []
    for sample in samples:
        src_tags = f"<{sample['source']}>", f"</{sample['source']}>"
        tgt_tags = f"<{sample['target']}>", f"</{sample['target']}>"
        for relation in sample["timeline"]:
            # Skip classification of self-relations (ex: start A -> end A)
            src_id = relation["source"].split(" ")[1]
            tgt_id = relation["target"].split(" ")[1]
            if src_id == tgt_id:
                continue

            if relation["source"].startswith("start"):
                new_src_tags = "<start_source>", "</start_source>"
            else:
                new_src_tags = "<end_source>", "</end_source>"

            if relation["target"].startswith("start"):
                new_tgt_tags = "<start_target>", "</start_target>"
            else:
                new_tgt_tags = "<end_target>", "</end_target>"

            context = (
                sample["context"]
                .replace(src_tags[0], new_src_tags[0])
                .replace(src_tags[1], new_src_tags[1])
                .replace(tgt_tags[0], new_tgt_tags[0])
                .replace(tgt_tags[1], new_tgt_tags[1])
            )

            examples.append(
                {
                    "text": context,
                    "label": relation["type"],
                }
            )

    return examples


def validate_dataset(examples: list[dict]):
    """Drop any relation that appears more than once.
    Most likely a mistake in the dataset."""
    text_counter = Counter([example["text"] for example in examples])
    duplicates = [text for text, count in text_counter.items() if count > 1]
    examples = [example for example in examples if example["text"] not in duplicates]
    return examples


def drop_long_texts(examples: list[dict]):
    """Drop texts that are longer than 512 words."""
    return [example for example in examples if len(example["text"].split()) <= 512]


def main(
    dataset_name: str = "tempeval_3",
    n_valid_samples: int = 5_000,
    closure: bool = False,
    just_sentences: bool = False,
):
    """Generate TemporalQuestions dataset from TemporalEval3 corpus.

    Args:
        dataset_name: Name of the dataset to use.
        n_valid_samples: Number of samples to use for validation.
        closure: Whether to compute temporal closure or not.
        just_sentences: Whether to use just the sentences that contain the temporal entities as context or not.
    """
    corpus = tieval.datasets.read(dataset_name)

    test_examples = transform_corpus(
        corpus.test, closure=closure, just_sentences=just_sentences
    )
    dev_examples = transform_corpus(
        corpus.train, closure=closure, just_sentences=just_sentences
    )

    if closure:
        test_examples = validate_dataset(test_examples)
    dev_examples = validate_dataset(dev_examples)

    # Stratified split into train and validation
    train_examples, valid_examples = train_test_split(
        dev_examples,
        test_size=n_valid_samples,
        random_state=42,
        stratify=[example["label"] for example in dev_examples],
        shuffle=True,
    )

    logging.info("Pushing to hub")
    trainset = datasets.Dataset.from_list(train_examples)
    validset = datasets.Dataset.from_list(valid_examples)
    testset = datasets.Dataset.from_list(test_examples)

    config = "closure" if closure else "raw"
    trainset.push_to_hub(
        "hugosousa/TemporalQuestions", config_name=config, split="train", token=HF_TOKEN
    )
    validset.push_to_hub(
        "hugosousa/TemporalQuestions", config_name=config, split="valid", token=HF_TOKEN
    )
    testset.push_to_hub(
        "hugosousa/TemporalQuestions", config_name=config, split="test", token=HF_TOKEN
    )


if __name__ == "__main__":
    fire.Fire(main)
