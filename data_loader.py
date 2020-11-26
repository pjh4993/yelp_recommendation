import os
import sys
import torch
import torch.utils.data
from torch.utils.data.sampler import Sampler
import json
from fvcore.common.config import CfgNode as CN
import copy
import logging
import numpy as np
import pickle
import random
import torch.utils.data as data
import cloudpickle

class yelpDataMapper:
    def __init__(self, cfg, attrib_dict = None):
        self.is_train = cfg.IS_TRAIN
        self.attrib_dict = attrib_dict
        pass

    def __call__(self, dataset_dict):
        if self.attrib_dict is not None:
            dataset_dict = {k : v for k, v in dataset_dict.items() if k in attrib_dict}
        
        if not self.is_train:
            dataset_dict.pop("stars", None)

        return dataset_dict

class PicklableWrapper(object):
    """
    Wrap an object to make it more picklable, note that it uses
    heavy weight serialization libraries that are slower than pickle.
    It's best to use it only on closures (which are usually not picklable).

    This is a simplified version of
    https://github.com/joblib/joblib/blob/master/joblib/externals/loky/cloudpickle_wrapper.py
    """

    def __init__(self, obj):
        self._obj = obj

    def __reduce__(self):
        s = cloudpickle.dumps(self._obj)
        return cloudpickle.loads, (s,)

    def __call__(self, *args, **kwargs):
        return self._obj(*args, **kwargs)

    def __getattr__(self, attr):
        # Ensure that the wrapped object can be used seamlessly as the previous object.
        if attr not in ["_obj"]:
            return getattr(self._obj, attr)
        return getattr(self, attr)

class MapDataset(data.Dataset):
    """
    Map a function over the elements in a dataset.

    Args:
        dataset: a dataset where map function is applied.
        map_func: a callable which maps the element in dataset. map_func is
            responsible for error handling, when error happens, it needs to
            return None so the MapDataset will randomly use other
            elements from the dataset.
    """

    def __init__(self, dataset, map_func):
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

        self._rng = random.Random(42)
        self._fallback_candidates = set(range(len(dataset)))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)

        while True:
            data = self._map_func(self._dataset[cur_idx])
            if data is not None:
                self._fallback_candidates.add(cur_idx)
                return data

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                        idx, retry_count
                    )
                )

class DatasetFromList(data.Dataset):
    """
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    """

    def __init__(self, lst: list, copy: bool = True, serialize: bool = True):
        """
        Args:
            lst (list): a list which contains elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
            serialize (bool): whether to hold memory using serialized objects, when
                enabled, data loader workers can use shared RAM from master
                process instead of making a copy.
        """
        self._lst = lst
        self._copy = copy
        self._serialize = serialize

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            logger = logging.getLogger(__name__)
            logger.info(
                "Serializing {} elements to byte tensors and concatenating them all ...".format(
                    len(self._lst)
                )
            )
            self._lst = [_serialize(x) for x in self._lst]
            self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
            self._addr = np.cumsum(self._addr)
            self._lst = np.concatenate(self._lst)
            logger.info("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024 ** 2))

    def __len__(self):
        if self._serialize:
            return len(self._addr)
        else:
            return len(self._lst)

    def __getitem__(self, idx):
        if self._serialize:
            start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
            end_addr = self._addr[idx].item()
        elif self._copy:
            bytes = memoryview(self._lst[start_addr:end_addr])
            return pickle.loads(bytes)
            return copy.deepcopy(self._lst[idx])
        else:
            return self._lst[idx]

def get_yelp_dataset_dicts(cfg):
    yelp_json_root = cfg.INPUT.YELP_JSON_ROOT
    yelp_dataset_list = None
    try:
        with open(yelp_json_root) as yelp_json_file:
            yelp_dataset_list = [json.loads(object_line) for object_line in yelp_json_file.readlines()]
    except OSError as err:
        print("OS error: {0}".format(err))
    except ValueError:
        print("Could not convert data to an integer.")
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
    return yelp_dataset_list

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
        #worker_init_fn=worker_init_reset_seed,
    )

def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch

if __name__ == '__main__':
    yelp_json_root = "/media/pjh3974/datasets/yelp/data.json"
    cfg = CN()
    cfg.IS_TRAIN = True
    cfg.INPUT = CN()
    cfg.INPUT.YELP_JSON_ROOT = yelp_json_root
    cfg.INPUT.BATCH_SIZE = 16
    attrib_dict = ["user_id", "business_id", "stars"]

    yelp_dataset_dicts = get_yelp_dataset_dicts(cfg)

    yelp_dataset = DatasetFromList(yelp_dataset_dicts, copy=False)

    yelp_mapper = yelpDataMapper(cfg, attrib_dict)

    yelp_dataset = MapDataset(yelp_dataset, yelp_mapper)

    yelp_training_sampler = Sampler(len(yelp_dataset))

    yelp_data_loader = build_batch_data_loader(yelp_dataset, yelp_training_sampler, cfg.INPUT.BATCH_SIZE)

    data_loader_iter = iter(yelp_data_loader)

    data = next(data_loader_iter)