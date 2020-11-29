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



    def __call__(self, dataset_dict):
        dataset_dict = {k : v for k,v in dataset_dict.items() if k in self.attrib_dict}
        if self.hook is not None:
            self.hook(dataset_dict, self.attrib_dict)
        return dataset_dict

def change_hash_to_int(dataset_dicts, idx, attrib_names, memory):
    for dataset_dict in dataset_dicts:
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

def split_dataset(processed_dataset, split_attrib):
    train_split = []
    valid_split = []

    instance_per_attrib = processed_dataset['id_to_instance'][split_attrib]
    for _id, instances in instance_per_attrib.items():
        np.random.shuffle(instances)
        num_train = int(0.5 * len(instances))
        if num_train == 0:
            num_train += 1
        train_split.extend(instances[:num_train])
        valid_split.extend(instances[num_train:])

    return {'train': train_split, 'valid':valid_split}

def main(args):
    cfg = setup(args)
    preprocess_mapper = YelpPreprocessMapper(cfg, None)
    data_loader = build_yelp_test_loader(cfg, preprocess_mapper)

    processed_dataset = {'whole_mean':0.0,
                    'hash_to_int': {},
                    'id_to_instance': {},
                    'reviews':[]}
    for attrib in preprocess_mapper.attrib_dict:
        if attrib == 'stars':
            continue
        processed_dataset['hash_to_int'][attrib] = {}
        processed_dataset['id_to_instance'][attrib] = defaultdict(list)

    for idx, inputs in enumerate(tqdm(data_loader, desc='only extract attrib in dataset')):
        change_hash_to_int(inputs, idx, preprocess_mapper.attrib_dict, processed_dataset)

    split_idx = split_dataset(processed_dataset, 'user_id')

    hash_to_idx = processed_dataset['hash_to_int']

    with open(os.path.join(cfg.DATA_ROOT, 'processed_yelp_dataset_review.json'),'w') as fp:
        json.dump(processed_dataset, fp)

    for k, v in hash_to_idx.items():
        print(k, len(v))

    for k,v in split_idx.items():
        print(k, len(v))
        with open(os.path.join(cfg.DATA_ROOT, k+'.txt'), 'w') as sp:
            v = [str(x) for x in v]
            sp.write('\n'.join(v))

    print(len(split_idx['train']) / (len(split_idx['train']) + len(split_idx['valid'])))

if __name__ == '__main__':
    args = argument_parser().parse_args()
    main(args)
