# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from yelp_recommendation.yelprcs.data.dataset_mapper import build_yelp_mapper
import numpy as np
import torch.utils.data
import os

from .common import DatasetFromList, MapDataset
from .samplers import InferenceSampler, TrainingSampler
from ..utils import seed_all_rng
import json

import logging
import random
from datetime import datetime
import torch
from tqdm import tqdm

"""
This file contains the default logic to build a dataloader for training or testing.
"""

__all__ = [
    "get_yelp_dataset_dicts",
    "build_batch_data_loader",
    "build_yelp_train_loader",
    "build_yelp_test_loader",
    "seed_all_rng",
]

def get_yelp_dataset_dicts(
    cfg, is_train=True
):
    yelp_json_root = os.path.join(cfg.DATA_ROOT, cfg.DATA_JSON)
    split_id = os.path.join(cfg.DATA_ROOT, 'train.txt' if is_train else 'test.txt') if cfg.IS_PREPROCESSED is True else None
    dataset_dicts = None

    with open(yelp_json_root, 'r') as yelp_json_file:
        if cfg.IS_PREPROCESSED is False:
            dataset_dicts = [json.loads(json_object) for json_object in tqdm(yelp_json_file.readlines(), desc='read yelp json file')]
        else:
            dataset_dicts = json.load(yelp_json_file)
            review_dicts = dataset_dicts['reviews']
            #hash_to_idx = dataset_dicts['id_to_instance']
            with open(split_id, 'r') as split_id_file:
                dataset_dicts['reviews'] = [review_dicts[int(idx)] for idx in split_id_file.readlines()]

    return dataset_dicts



def build_batch_data_loader(
    dataset, sampler, total_batch_size, *, num_workers=0
):
    """
    Build a batched dataloader for training.

    Args:
        dataset (torch.utils.data.Dataset): map-style PyTorch dataset. Can be indexed.
        sampler (torch.utils.data.sampler.Sampler): a sampler that produces indices
        total_batch_size (int): total batch size across GPUs.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    #world_size = get_world_size()
    world_size = 1
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



def build_yelp_train_loader(cfg, mapper=None):
    """
    A data loader is created by the following steps:

    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Coordinate a random shuffle order shared among all processes (all GPUs)
    3. Each process spawn another few workers to process the dicts. Each worker will:
       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will yield.

    Args:
        cfg (CfgNode): the config
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            By default it will be ``DatasetMapper(cfg, True)``.

    Returns:
        an infinite iterator of training data
    """
    dataset_dicts = get_yelp_dataset_dicts(
        cfg, is_train=True
    )
    dataset = DatasetFromList(dataset_dicts['reviews'], copy=False)

    if mapper is None:
        mapper = build_yelp_mapper(cfg, dataset_dicts['statistics'])
    dataset = MapDataset(dataset, mapper)

    sampler = TrainingSampler(len(dataset))

    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.REVIEW_PER_BATCH,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


def build_yelp_test_loader(cfg, mapper):
    dataset_dicts = get_yelp_dataset_dicts(
        cfg, is_train=False
    )

    dataset = DatasetFromList(dataset_dicts)

    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    #sampler = TrainingSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, cfg.SOLVER.REVIEW_PER_BATCH, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)


