from ..yelprcs.config import setup, default_argument_parser
from ..yelprcs.config.dist import launch
from ..yelprcs.checkpoint import YelpCheckpointer
from ..yelprcs.trainer import YelpTrainer



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
