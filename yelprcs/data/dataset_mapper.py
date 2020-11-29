# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
class DatasetMapper:

    def __init__(
        self,
        cfg,
    ):
        return NotImplementedError

    def __call__(self, dataset_dict):
        return NotImplementedError
