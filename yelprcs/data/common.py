import copy
import logging
import numpy as np
import pickle
import random
import torch.utils.data as data
from collections import defaultdict

from ..utils import PicklableWrapper

__all__ = ["MapDataset", "DatasetFromList"]


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
        self._fallback_candidates = [set(range(x)) for x in dataset.__len__()]

    def __len__(self):
        return self._dataset.__len__()

    def __getitem__(self, idx):
        retry_count = 0

        while True:
            if isinstance(idx, int):
                data = self._map_func(self._dataset.__getitem__(0, idx))
            else:
                data = [self._map_func(self._dataset.__getitem__(row, x)) for row, x in enumerate(idx.tolist())]
            """
            if data is not None:
                for row, x in enumerate(idx):
                    self._fallback_candidates[row].add(x)
            """
            if data is not None:
                return data

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            """
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                        idx, retry_count
                    )
                )
            """


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
            lst_list = []
            addr_list = []
            for i in range(len(lst)):
                curr_lst = [_serialize(x) for x in self._lst[i]]
                curr_addr = np.asarray([len(x) for x in curr_lst], dtype=np.int64)
                curr_addr = np.cumsum(curr_addr)
                curr_lst = np.concatenate(curr_lst)
                logger.info("Serialized dataset takes {:.2f} MiB".format(len(curr_lst) / 1024 ** 2))
                lst_list.append(curr_lst)
                addr_list.append(curr_addr)

            self._lst = lst_list
            self._addr = addr_list


    def __len__(self):
        if self._serialize:
            return [len(x) for x in self._addr]
        else:
            return [len(x) for x in self._lst]

    def __getitem__(self, row, idx):
        if self._serialize:
            start_addr = 0 if idx == 0 else self._addr[row][idx - 1].item()
            end_addr = self._addr[row][idx].item()
            bytes = memoryview(self._lst[row][start_addr:end_addr])
            return pickle.loads(bytes)
        elif self._copy:
            return copy.deepcopy(self._lst[row][idx])
        else:
            return self._lst[row][idx]

class StarGroupedDataset(data.IterableDataset):

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self._buckets = [[] for _ in range(5)]
        # Can add support for more aspect ratio groups, but doesn't seem useful

    def __iter__(self):
        for d in self.dataset:
            star = d[0][1]
            bucket = self._buckets[star-1]
            bucket.extend(d)
            min_len = min([len(x) for x in self._buckets])
            if min_len == self.batch_size:
                out = []
                for x in self._buckets:
                    out.extend(x[:self.batch_size])
                    x = x[self.batch_size:]
                yield out
                del bucket[:]