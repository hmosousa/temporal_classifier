from typing import Any, Dict, Tuple

from torch.utils.data import Dataset

from src.data.matres import load_matres
from src.data.synthetic import load_synthetic

from src.data.tddiscourse import load_interval_tddiscourse
from src.data.tempeval import load_tempeval
from src.data.temporal_questions import load_temporal_questions
from src.data.timebankdense import load_timebank_dense
from src.data.timeset import load_timeset
from src.data.utils import augment_dataset, balance_dataset_classes

DATASETS = {
    "timeset": load_timeset,
    "temporal_questions": load_temporal_questions,
    "synthetic": load_synthetic,
    "matres": load_matres,
    "interval_tddiscourse": load_interval_tddiscourse,
    "tempeval": load_tempeval,
    "timebank_dense": load_timebank_dense,
}


def load_dataset(
    dataset_name: str, split: str, **kwargs: Dict[str, Any]
) -> Tuple[Dataset, Dataset]:
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} not found. Valid datasets are: {list(DATASETS.keys())}"
        )
    return DATASETS[dataset_name](split=split, **kwargs)


__all__ = [
    "load_dataset",
    "load_timeset",
    "load_temporal_questions",
    "load_synthetic_temporal_questions",
    "load_all_temporal_questions",
    "load_matres",
    "load_interval_tddiscourse",
    "load_tempeval",
    "load_timebank_dense",
    "balance_dataset_classes",
    "augment_dataset",
]
