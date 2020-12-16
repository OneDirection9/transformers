from __future__ import absolute_import, division, print_function

import itertools
import logging
from typing import List, Optional

import numpy as np
import torch

from transformers.config import configurable
from transformers.utils.comm import get_world_size
from transformers.utils.env import seed_all_rng
from .common import DatasetFromList, MapDataset
from .datasets import DATASET_REGISTER
from .processors import Processor, build_processor
from .samplers import TrainingSampler


class DatasetMapper(object):
    def __init__(self, cfg, is_train):
        pass

    def __call__(self, item):
        return item


def get_dataset_dicts(dataset_names, processor: Optional[Processor] = None) -> List[dict]:
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    assert len(dataset_names)

    dataset_dicts = [DATASET_REGISTER.get(name)() for name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset `{}` is empty!".format(dataset_name)

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))
    if processor is not None:
        dataset_dicts = processor(dataset_dicts)
    assert len(dataset_dicts), "No valid data found in {}".format(dataset_names)
    return dataset_dicts


def build_batch_data_loader(dataset, sampler, total_batch_size, num_workers=0):
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )

    batch_size = total_batch_size // world_size
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=True
    )  # drop_last so the batch always have the same size
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
    )


def _train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    if dataset is None:
        processor = build_processor(cfg)
        dataset = get_dataset_dicts(cfg.DATASETS.TRAIN, processor)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        if sampler_name == "TrainingSampler":
            sampler = TrainingSampler(len(dataset))
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": dataset,
        "mapper": mapper,
        "sampler": sampler,
        "total_batch_size": cfg.SOLVER.NUM_PER_BATCH,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }


@configurable(from_config=_train_loader_from_config)
def build_train_loader(dataset, *, mapper, sampler=None, total_batch_size, num_workers=0):
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    return build_batch_data_loader(dataset, sampler, total_batch_size, num_workers)


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)
