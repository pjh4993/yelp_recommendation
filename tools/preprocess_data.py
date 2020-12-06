from ..yelprcs.data import build_yelp_preprocess_loader, DatasetMapper
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

def change_hash_to_int(dataset_dicts, attrib_names, memory):
    for d_id, dataset_dict in enumerate(dataset_dicts):
        for attrib in attrib_names:
            if attrib == 'stars':
                continue
            hash_id = dataset_dict[attrib]
            if hash_id not in memory['hash_to_int'][attrib]:
                _id = len(memory['hash_to_int'][attrib])
                memory['hash_to_int'][attrib][hash_id] = _id

            _id = memory['hash_to_int'][attrib][hash_id]
            dataset_dict[attrib] = _id

            memory['id_to_instance'][attrib][_id].append(d_id)
        memory['reviews'].append(dataset_dict)
    return memory

def split_dataset(processed_dataset, split_attrib):
    train_split = []
    valid_split = []
    instance_per_attrib = processed_dataset['id_to_instance'][split_attrib]
    for _id, instances in tqdm(instance_per_attrib.items(), desc='split'):
        np.random.shuffle(instances)
        num_train = int(0.7 * len(instances))
        if num_train == 0:
            num_train += 1
        train_split.extend(instances[:num_train])
        valid_split.extend(instances[num_train:])

    return {'train': train_split, 'valid':valid_split}

def analysis_dataset(processed_dataset, split):
    reviews = processed_dataset['reviews']
    reviews = [reviews[id] for id in split]
    user_id_size = len(processed_dataset['hash_to_int']['user_id'])
    business_id_size = len(processed_dataset['hash_to_int']['business_id'])

    user_mean_table = np.zeros((user_id_size,2))
    business_mean_table = np.zeros((business_id_size,2))
    whole_mean = 0
    for review in tqdm(reviews, desc='calculating mean table'):
        user_id = review['user_id']
        business_id = review['business_id']
        user_mean_table[user_id, 0] += review['stars']
        user_mean_table[user_id, 1] += 1
        business_mean_table[business_id, 0] += review['stars']
        business_mean_table[business_id, 1] += 1
        whole_mean += review['stars']

    whole_mean = whole_mean / len(reviews)
    user_mean = user_mean_table[:,0] / (user_mean_table[:,1] + 1e-5) - whole_mean
    business_mean = business_mean_table[:,0] / (business_mean_table[:,1] + 1e-5)- whole_mean
    return {'whole_mean' : whole_mean, 'user_mean' : user_mean.tolist(), 'business_mean' : business_mean.tolist(), 
            'user_num': user_mean_table[:,1].tolist(), 'business_num': business_mean_table[:,1].tolist()}

def main(args):
    cfg = setup(args)
    preprocess_mapper = YelpPreprocessMapper(cfg, None)
    data_loader = build_yelp_preprocess_loader(cfg, preprocess_mapper)

    processed_dataset = {
                    'hash_to_int': {},
                    'id_to_instance': {},
                    'reviews':[],
    }
    for attrib in preprocess_mapper.attrib_dict:
        if attrib == 'stars':
            continue
        processed_dataset['hash_to_int'][attrib] = {}
        processed_dataset['id_to_instance'][attrib] = defaultdict(list)
    
    processed_items = []

    for idx, inputs in enumerate(tqdm(data_loader, desc='only extract attrib in dataset')):
        processed_items.extend(inputs)

    processed_dataset = change_hash_to_int(processed_items, ['user_id', 'business_id'], processed_dataset)

    split_idx = split_dataset(processed_dataset, 'user_id')

    processed_dataset['statistics'] = analysis_dataset(processed_dataset, split_idx['train'])

    with open(os.path.join(cfg.DATA_ROOT, 'processed_yelp_dataset_review.json'),'w') as fp:
        json.dump(processed_dataset, fp)

    for k,v in split_idx.items():
        print(k, len(v))
        with open(os.path.join(cfg.DATA_ROOT, k+'.txt'), 'w') as sp:
            v = [str(x) for x in v]
            sp.write('\n'.join(v))

    print(len(split_idx['train']) / (len(split_idx['train']) + len(split_idx['valid'])))

if __name__ == '__main__':
    args = argument_parser().parse_args()
    main(args)
