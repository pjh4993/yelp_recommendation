from ..yelprcs.data import build_yelp_test_loader, DatasetMapper
from ..yelprcs.config import setup
import argparse
import sys
import numpy as np
from tqdm import tqdm
import os
import json
import torch
from collections import defaultdict


def argument_parser(epilog=None):
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --config-file configs/Base-config.yaml --opts
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    return parser


class YelpPreprocessMapper(DatasetMapper):
    def __init__(self, cfg, hook):
        super().__init__(cfg)
        self.is_train = cfg.IS_TRAIN
        self.attrib_dict = cfg.DATASET_MAPPER.ATTRIB_DICT
        self.hook = hook
        self.memory = {'whole_mean':0.0,
                       'hash_to_int': {},
                       'id_to_instance': {},
                       'reviews':[]}
        for attrib in self.attrib_dict:
            if attrib == 'stars':
                continue
            self.memory['hash_to_int'][attrib] = {}
            self.memory['id_to_instance'][attrib] = defaultdict(list)


    def __call__(self, dataset_dict, idx):
        dataset_dict = {k : v for k,v in dataset_dict.items() if k in self.attrib_dict}
        self.hook(dataset_dict, idx, self.attrib_dict, self.memory)
        print('idx: ', idx)
        print(self.memory)
        return dataset_dict

def change_hash_to_int(dataset_dict, idx, attrib_names, memory):
    for attrib in attrib_names:
        if attrib == 'stars':
            memory['whole_mean'] += dataset_dict['stars']
            continue
        hash_id = dataset_dict[attrib]
        if hash_id not in memory['hash_to_int'][attrib]:
            _id = len(memory['hash_to_int'][attrib])
            memory['hash_to_int'][attrib][hash_id] = _id

        _id = memory['hash_to_int'][attrib][hash_id]
        dataset_dict[attrib] = _id

        memory['id_to_instance'][attrib][_id].append(idx)
    memory['reviews'].append(dataset_dict)

"""
def change_hash_to_int_id(dataset_dicts, attrib_names, split_attrib):
    attrib_idx_alloc = {}
    train_split = []
    valid_split = []
    len_instances = []
    for attrib in attrib_names:
        attrib_id = np.array([x[attrib] for x in dataset_dicts])
        unique_attrib_id = np.unique(attrib_id)
        attrib_idx_alloc[attrib]={hash_id: int(int_id) for hash_id, int_id in zip(unique_attrib_id,np.arange(len(unique_attrib_id)))}

        if attrib == split_attrib:
            for unique_id in unique_attrib_id:
                instances = (attrib_id == unique_id).nonzero()[0]
                np.random.shuffle(instances)
                num_train = int(0.7 * len(instances))
                if num_train == 0:
                    num_train += 1
                train_split.extend(instances[:num_train])
                valid_split.extend(instances[num_train:])
                len_instances.append(len(instances))

    print(np.array(len_instances).mean())
    for _id, x in enumerate(dataset_dicts):
        for k, v in attrib_idx_alloc.items():
            x[k] = v[x[k]]
        dataset_dicts[_id] = x

    return dataset_dicts, attrib_idx_alloc , {'train': train_split, 'valid':valid_split}
"""

def main(args):
    cfg = setup(args)
    preprocess_mapper = YelpPreprocessMapper(cfg, change_hash_to_int)
    data_loader = build_yelp_test_loader(cfg, preprocess_mapper)

    processed_items = []
    for idx, inputs in enumerate(tqdm(data_loader, desc='only extract attrib in dataset')):
        print('idx: ', idx)
        print(data_loader.memory)
        break

    #split_idx = change_hash_to_int_id(processed_items, ['user_id', 'business_id'], 'user_id')

    processed_dataset = preprocess_mapper.memory
    print(processed_dataset)
    hash_to_idx = processed_dataset['hash_to_int']

    with open(os.path.join(cfg.DATA_ROOT, 'processed_yelp_dataset_review.json'),'w') as fp:
        json.dump(processed_dataset, fp)

    for k, v in hash_to_idx.items():
        print(k, len(v))

    """
    for k,v in split_idx.items():
        print(k, len(v))
        with open(os.path.join(cfg.DATA_ROOT, k+'.txt'), 'w') as sp:
            v = [str(x) for x in v]
            sp.write('\n'.join(v))
    """


if __name__ == '__main__':
    args = argument_parser().parse_args()
    main(args)
