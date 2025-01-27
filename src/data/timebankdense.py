from typing import Literal

import datasets
import tieval.datasets

from src.base import tlink_to_point_relations

from src.data.utils import (
    get_tlink_context,
    INTERVAL_EXPECTED_TAGS,
    POINT_EXPECTED_TAGS,
)

# Train, valid, test split from https://github.com/muk343/TimeBank-dense
TRAIN_DOCS = [
    "ABC19980108.1830.0711",
    "ABC19980114.1830.0611",
    "ABC19980120.1830.0957",
    "ABC19980304.1830.1636",
    "AP900815-0044",
    "AP900816-0139",
    "APW19980213.1310",
    "APW19980213.1320",
    "APW19980213.1380",
    "APW19980219.0476",
    "APW19980227.0468",
    "APW19980227.0476",
    "CNN19980222.1130.0084",
    "CNN19980227.2130.0067",
    "NYT19980206.0460",
    "NYT19980206.0466",
    "PRI19980121.2000.2591",
    "PRI19980205.2000.1890",
    "PRI19980205.2000.1998",
    "PRI19980213.2000.0313",
    "ea980120.1830.0071",
    "ea980120.1830.0456",
]

VALID_DOCS = [
    "APW19980227.0487",
    "CNN19980223.1130.0960",
    "NYT19980212.0019",
    "PRI19980216.2000.0170",
    "ed980111.1130.0089",
]

TEST_DOCS = [
    "APW19980227.0489",
    "APW19980227.0494",
    "APW19980308.0201",
    "APW19980418.0210",
    "CNN19980126.1600.1104",
    "CNN19980213.2130.0155",
    "NYT19980402.0453",
    "PRI19980115.2000.0186",
    "PRI19980306.2000.1675",
]


def load_interval_timebank_dense(
    split: Literal["train", "valid", "test"],
    **kwargs,
) -> datasets.Dataset:
    """Load TimeBank Dense dataset."""
    corpus = tieval.datasets.read("timebank_dense")

    if split == "train":
        docs = [doc for doc in corpus.documents if doc.name in TRAIN_DOCS]
    elif split == "valid":
        docs = [doc for doc in corpus.documents if doc.name in VALID_DOCS]
    elif split == "test":
        docs = [doc for doc in corpus.documents if doc.name in TEST_DOCS]
    else:
        raise ValueError(f"Invalid split: {split}")

    examples = []
    for doc in docs:
        for tlink in set(doc.tlinks):
            if tlink.relation.interval == "VAGUE":
                continue

            if tlink.source.id == tlink.target.id:
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
            examples.append(
                {"doc": doc.name, "text": text, "label": tlink.relation.interval}
            )
    return datasets.Dataset.from_list(examples)


def load_point_timebank_dense(
    split: Literal["train", "valid", "test"],
    **kwargs,
) -> datasets.Dataset:
    """Load TimeBank Dense dataset."""
    corpus = tieval.datasets.read("timebank_dense")

    if split == "train":
        docs = [doc for doc in corpus.documents if doc.name in TRAIN_DOCS]
    elif split == "valid":
        docs = [doc for doc in corpus.documents if doc.name in VALID_DOCS]
    elif split == "test":
        docs = [doc for doc in corpus.documents if doc.name in TEST_DOCS]
    else:
        raise ValueError(f"Invalid split: {split}")

    examples = []
    for doc in docs:
        tlinks = doc.tlinks
        for tlink in tlinks:
            context = get_tlink_context(doc, tlink)

            if tlink.relation.interval == "VAGUE":
                continue

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
                examples.append({"doc": doc.name, "text": text, "label": relation.type})
    return datasets.Dataset.from_list(examples)
