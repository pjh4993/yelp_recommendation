from ..yelprcs.config import get_cfg, default_setup, default_argument_parser
from ..yelprcs.config.dist import launch
from ..yelprcs.checkpoint import YelpCheckpointer
from ..yelprcs.trainer import YelpTrainer

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = YelpTrainer.build_model(cfg)
        YelpCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = YelpTrainer.test(cfg, model)
        return res

    trainer = YelpTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        args=(args,),
    )
