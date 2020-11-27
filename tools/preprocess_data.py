from ..yelprcs.data import build_yelp_test_loader, DatasetMapper
from ..yelprcs.config import setup
import argparse
import sys
import numpy as np


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
        dataset_dict = {k : v for k,v in dataset_dict if k in self.attrib_dict}
        return dataset_dict

def change_hash_to_int_id(dataset_dicts, attrib_names):
    attrib_idx_alloc = {}
    for attrib in attrib_names:
        attrib_id = [x[attrib] for x in dataset_dicts]
        unique_attrib_id = np.unique(np.array(attrib_id))
        attrib_idx_alloc[attrib]={hash_id: int_id for hash_id, int_id in zip(unique_attrib_id,np.arange(len(unique_attrib_id)))}
        
    for _id, x in enumerate(dataset_dicts):
        for k, v in attrib_idx_alloc.items():
            x[k] = v[x[k]]
        dataset_dicts[_id] = x

    return dataset_dicts

def main(args):
    cfg = setup(args)
    preprocess_mapper = YelpPreprocessMapper(cfg, None)
    data_loader = build_yelp_test_loader(cfg, preprocess_mapper)

    processed_items = []
    for idx, inputs in enumerate(data_loader):
        processed_items.append(inputs)

    print(len(processed_items))
    change_hash_to_int_id(processed_items, ['user_id', 'business_id'])







if __name__ == '__main__':
    args = argument_parser().parse_args()
    main(args)