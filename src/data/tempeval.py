from typing import Literal

import datasets
import tieval.datasets
from tieval.links import TLink

from src.base import PointRelation, tlink_to_point_relations
from src.data.utils import (
    get_tlink_context,
    INTERVAL_EXPECTED_TAGS,
    POINT_EXPECTED_TAGS,
)

CLOSURE_DOCS_TO_DROP = [
    "NYT20000424.0319",
    "wsj_1025",
    "wsj_0266",
    "NYT19990312.0271",
    "wsj_0918",
    "wsj_0661",
    "ABC19980108.1830.0711",
    "APW19980306.1001",
    "wsj_0026",
    "wsj_0027",
    "wsj_0159",
    "wsj_0172",
    "wsj_1039",
    "wsj_0584",
    "wsj_0675",
    "CNN19980222.1130.0084",
    "CNN19980223.1130.0960",
    "wsj_0568",
    "wsj_0541",
]


def load_interval_tempeval(
    split: Literal["train", "valid", "test"],
    closure: bool = False,
    **kwargs,
) -> datasets.Dataset:
    """Load TempEval-3 dataset."""
    corpus = tieval.datasets.read("tempeval_3")

    if split in ["train", "valid"]:
        n_train_docs = int(len(corpus.train) * 0.8)
        if split == "train":
            docs = corpus.train[:n_train_docs]
        else:
            docs = corpus.train[n_train_docs:]
    elif split == "test":
        docs = corpus.test
    else:
        raise ValueError(f"Invalid split: {split}")

    examples = []
    for doc in docs:
        if closure:
            tlinks = doc.temporal_closure
        else:
            tlinks = doc.tlinks

        for tlink in tlinks:
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


def load_point_tempeval(
    split: Literal["train", "valid", "test"],
    closure: bool = False,
    **kwargs,
) -> datasets.Dataset:
    """Load TempEval-3 dataset."""
    corpus = tieval.datasets.read("tempeval_3")

    if split in ["train", "valid"]:
        n_train_docs = int(len(corpus.train) * 0.8)
        if split == "train":
            docs = corpus.train[:n_train_docs]
        else:
            docs = corpus.train[n_train_docs:]
    elif split == "test":
        docs = corpus.test
    else:
        raise ValueError(f"Invalid split: {split}")

    # Compile all point relations
    point_relations = []
    for doc in docs:
        if closure:
            entities = doc.entities + [doc.dct]
            entity_map = {entity.id: entity for entity in entities}
            relations = doc.point_temporal_closure
            for relation in relations:
                if relation["source"].startswith("s"):
                    relation["source"] = "start " + relation["source"][1:]
                elif relation["source"].startswith("e"):
                    relation["source"] = "end " + relation["source"][1:]
                if relation["target"].startswith("s"):
                    relation["target"] = "start " + relation["target"][1:]
                elif relation["target"].startswith("e"):
                    relation["target"] = "end " + relation["target"][1:]
                relation["type"] = relation.pop("relation")
                relation = PointRelation(**relation)

                source_entity = entity_map[relation.source_id]
                target_entity = entity_map[relation.target_id]
                if source_entity == target_entity:
                    continue

                tlink = TLink(
                    source=source_entity,
                    target=target_entity,
                    relation="None",
                )
                context = get_tlink_context(doc, tlink)
                point_relations.append((context, relation))
            else:
                for tlink in doc.tlinks:
                    context = get_tlink_context(doc, tlink)
                    relations = tlink_to_point_relations(tlink)
                    for relation in relations:
                        if relation.source_id == relation.target_id:
                            continue
                        point_relations.append((context, relation))

    # Change the context tags according to the point relation
    examples = []
    for context, relation in point_relations:
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
