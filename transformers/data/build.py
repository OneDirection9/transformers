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
from .dataset_mapper import DatasetMapper
from .datasets import DATASET_REGISTRY
from .processors import Processor, build_processor
from .samplers import TrainingSampler


def get_dataset_dicts(dataset_names, processor: Optional[Processor] = None) -> List[dict]:
    """
    Load and prepare dataset dicts for natural language tasks.

    Args:
        dataset_names (str or list[str]):  A dataset name or a list of dataset names.
        processor (Processor, optional): Callable object to process all examples

    Returns:
        list[dict]: A list of dicts.
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    assert len(dataset_names)

    dataset_dicts = [DATASET_REGISTRY.get(name)() for name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset `{}` is empty!".format(dataset_name)

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))
    if processor is not None:
        dataset_dicts = processor(dataset_dicts)
    assert len(dataset_dicts), "No valid data found in {}".format(dataset_names)
    return dataset_dicts


def build_batch_data_loader(dataset, sampler, total_batch_size, num_workers=0):
    """
    Build a batched dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Map-style PyTorch dataset. Can be indexed.
        sampler (torch.utils.data.Sampler): A sampler taht produces indices.
        total_batch_size: See :func:`build_train_loader`.
        num_workers: See :func:`build_train_loader`.

    Returns:
        iterable[list]: Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
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
    """
    Build a train loader with some default features.

    Args:
        dataset (list or torch.utils.data.Dataset): A list of dataset dicts, or a map-style PyTorch
            dataset. They can be obtained by using :func:`DATASET_REGISTRY.get` or
            :func:`get_dataset_dicts`.
        mapper (callable): A callable which takes a sample (dict) from dataset and returns the
            format to be consumed by the model.
        sampler (torch.utils.data.Sampler or None): A sampler that produces indices to be applied on
            ``dataset``.
        total_batch_size (int): Total batch size across all workers. Batching simply puts data into
            a list.
        num_workers (int): Number of parallel data loading workers.

    Returns:
        torch.utils.data.DataLoader: A dataloader. Each output from it is a
            ``list[mapped_element]`` of length ``total_batch_size / num_workers``,
            where ``mapped_element`` is produced by the ``mapper``.
    """
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
