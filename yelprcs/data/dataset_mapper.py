# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
class DatasetMapper:

    def __init__(
        self,
        cfg,
    ):
        return NotImplementedError

    def __call__(self, dataset_dict):
        return NotImplementedError

class YelpPreprocessMapper(DatasetMapper):
    def __init__(self, cfg, hook):
        super().__init__(cfg)
        self.is_train = cfg.IS_TRAIN
        self.attrib_dict = cfg.DATASET_MAPPER.ATTRIB_DICT
        self.hook = hook

    def __call__(self, dataset_dict):
        dataset_dict = {k : v for k,v in dataset_dict.items() if k in self.attrib_dict}
        if self.hook is not None:
            self.hook(dataset_dict, self.attrib_dict)
        return dataset_dict

class YelpMeanTableMapper(DatasetMapper):
    def __init__(self, cfg, mean_tables):
        super().__init__(cfg)
        self.is_train = cfg.IS_TRAIN
        self.mean_tables = mean_tables
    
    def __call__(self, dataset_dict):
        user_id = dataset_dict['user_id']
        business_id = dataset_dict['business_id']
        dataset_dict['whole_mean'] = self.mean_tables['whole_mean']
        dataset_dict['user_mean'] = self.mean_tables['user_mean'][user_id]
        dataset_dict['business_mean'] = self.mean_tables['user_mean'][business_id]

        return dataset_dict

def build_yelp_mapper(cfg, mean_tables=None):
    return YelpMeanTableMapper(cfg, mean_tables)