from typing import Any, Dict, Tuple

from torch.utils.data import Dataset

from src.data.matres import load_matres

from src.data.meantime import load_interval_meantime, load_point_meantime
from src.data.synthetic import load_synthetic
from src.data.tddiscourse import load_interval_tddiscourse, load_point_tddiscourse
from src.data.tempeval import load_interval_tempeval, load_point_tempeval
from src.data.temporal_contexts import load_temporal_contexts
from src.data.timebankdense import (
    load_interval_timebank_dense,
    load_point_timebank_dense,
)
from src.data.timeset import load_timeset
from src.data.utils import augment_dataset, balance_dataset_classes

DATASETS = {
    "timeset": load_timeset,
    "temporal_contexts": load_temporal_contexts,
    "synthetic": load_synthetic,
    "matres": load_matres,
    "interval_tddiscourse": load_interval_tddiscourse,
    "point_tddiscourse": load_point_tddiscourse,
    "interval_tempeval": load_interval_tempeval,
    "point_tempeval": load_point_tempeval,
    "interval_timebank_dense": load_interval_timebank_dense,
    "point_timebank_dense": load_point_timebank_dense,
    "interval_meantime": load_interval_meantime,
    "point_meantime": load_point_meantime,
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
    "load_temporal_contexts",
    "load_synthetic",
    "load_matres",
    "load_interval_tddiscourse",
    "load_interval_tempeval",
    "load_point_tempeval",
    "load_interval_timebank_dense",
    "load_point_timebank_dense",
    "load_interval_meantime",
    "load_point_meantime",
    "balance_dataset_classes",
    "augment_dataset",
]
