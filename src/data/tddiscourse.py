from typing import Literal

import datasets
import tieval.datasets

from src.base import tlink_to_point_relations
from src.data.utils import (
    get_tlink_context,
    INTERVAL_EXPECTED_TAGS,
    POINT_EXPECTED_TAGS,
)


# Train, valid, test split from https://github.com/aakanksha19/TDDiscourse
TRAIN_DOCS = [
    "APW19980227.0468",
    "APW19980213.1320",
    "CNN19980222.1130.0084",
    "ABC19980108.1830.0711",
    "PRI19980205.2000.1890",
    "PRI19980205.2000.1998",
    "ABC19980120.1830.0957",
    "APW19980213.1380",
    "ea980120.1830.0071",
    "CNN19980227.2130.0067",
    "ea980120.1830.0456",
    "ABC19980304.1830.1636",
    "APW19980219.0476",
    "APW19980213.1310",
    "PRI19980213.2000.0313",
    "AP900816-0139",
    "PRI19980121.2000.2591",
    "NYT19980206.0466",
    "NYT19980206.0460",
    "AP900815-0044",
    "ABC19980114.1830.0611",
]

VALID_DOCS = [
    "APW19980227.0487",
    "PRI19980216.2000.0170",
    "ed980111.1130.0089",
    "CNN19980223.1130.0960",
]

TEST_DOCS = [
    "APW19980418.0210",
    "CNN19980213.2130.0155",
    "APW19980227.0494",
    "CNN19980126.1600.1104",
    "APW19980308.0201",
    "NYT19980402.0453",
    "APW19980227.0489",
    "PRI19980306.2000.1675",
    "PRI19980115.2000.0186",
]


def load_interval_tddiscourse(
    split: Literal["train", "valid", "test"],
    **kwargs,
) -> datasets.Dataset:
    """Load TDDiscourse dataset."""
    corpus = tieval.datasets.read("tddiscourse")

    if split == "train":
        docs = [doc for doc in corpus.train if doc.name in TRAIN_DOCS]
    elif split == "valid":
        docs = [doc for doc in corpus.train if doc.name in VALID_DOCS]
    elif split == "test":
        docs = [doc for doc in corpus.test if doc.name in TEST_DOCS]
    else:
        raise ValueError(f"Invalid split: {split}")

    examples = []
    for doc in docs:
        tlinks = doc.tlinks
        for tlink in tlinks:
            if tlink.source_id == tlink.target_id:
                continue

            context = get_tlink_context(doc, tlink)
            srcid = tlink.source.id
            tgtid = tlink.target.id
            text = (
                context.replace(f"<{srcid}>", "<source>")
                .replace(f"</{srcid}>", "</source>")
                .replace(f"<{tgtid}>", "<target>")
                .replace(f"</{tgtid}>", "</target>")
            )

            tag_count = sum(1 for tag in INTERVAL_EXPECTED_TAGS if tag in text)
            if tag_count != 4:
                continue

            text = text.replace("\n", " ").strip()
            if "(PROFILE" in text:
                text, _ = text.split("(PROFILE")
                text = text.strip()
            examples.append(
                {"doc": doc.name, "text": text, "label": tlink.relation.interval}
            )

    return datasets.Dataset.from_list(examples)


def load_point_tddiscourse(
    split: Literal["train", "valid", "test"],
    **kwargs,
) -> datasets.Dataset:
    """Load TDDiscourse dataset."""
    corpus = tieval.datasets.read("tddiscourse")

    if split == "train":
        docs = [doc for doc in corpus.train if doc.name in TRAIN_DOCS]
    elif split == "valid":
        docs = [doc for doc in corpus.train if doc.name in VALID_DOCS]
    elif split == "test":
        docs = [doc for doc in corpus.test if doc.name in TEST_DOCS]
    else:
        raise ValueError(f"Invalid split: {split}")

    examples = []
    for doc in docs:
        tlinks = doc.tlinks
        for tlink in tlinks:
            context = get_tlink_context(doc, tlink)
            relations = tlink_to_point_relations(tlink)
            for relation in relations:
                if relation.source_id == relation.target_id:
                    continue

                if relation.source_endpoint == "start":
                    new_src_tags = "<start_source>", "</start_source>"
                else:
                    new_src_tags = "<end_source>", "</end_source>"

                if relation.target_endpoint == "start":
                    new_tgt_tags = "<start_target>", "</start_target>"
                else:
                    new_tgt_tags = "<end_target>", "</end_target>"

                text = (
                    context.replace(f"<{relation.source_id}>", new_src_tags[0])
                    .replace(f"</{relation.source_id}>", new_src_tags[1])
                    .replace(f"<{relation.target_id}>", new_tgt_tags[0])
                    .replace(f"</{relation.target_id}>", new_tgt_tags[1])
                )

                tag_count = sum(1 for tag in POINT_EXPECTED_TAGS if tag in text)
                if tag_count != 4:
                    continue

                text = text.replace("\n", " ").strip()
                if "(PROFILE" in text:
                    text, _ = text.split("(PROFILE")
                    text = text.strip()
                examples.append({"doc": doc.name, "text": text, "label": relation.type})

    return datasets.Dataset.from_list(examples)
