import os
from ..evaluation import YelpEvaluator, inference_on_dataset
from ..utils import setup_logger, comm
from ..checkpoint import YelpCheckpointer
from .train_loop import SimpleTrainer
from ..data import build_yelp_train_loader, build_yelp_test_loader
from .hooks import LRScheduler, PeriodicCheckpointer, EvalHook, PeriodicWriter
from ..model import build_model
from ..solver import build_optimizer, build_lr_scheduler
import logging
from torch.nn.parallel import DistributedDataParallel
 
class YelpTrainer(SimpleTrainer):

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        logger = logging.getLogger("yelprcs")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        super().__init__(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = YelpCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        #self.max_iter = cfg.SOLVER.MAX_ITER
        self.max_iter = int(len(data_loader.dataset) / cfg.SOLVER.REVIEW_PER_BATCH) * 5
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True`, and last checkpoint exists, resume from it, load all checkpointables
        (eg. optimizer and scheduler) and update iteration counter from it. ``cfg.MODEL.WEIGHTS``
        will not be used.

        Otherwise, load the model specified by the config (skip all checkpointables) and start from
        the first iteration.

        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            LRScheduler(self.optimizer, self.scheduler),
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(PeriodicWriter(period=20))
        return ret

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_iter, self.max_iter)

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:

        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)

    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    def build_train_loader(cls, cfg):

        return build_yelp_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg):
        return build_yelp_test_loader(cfg)

    @classmethod
    def build_evaluator(cls, cfg, statistics=None ,distributed=False,output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return YelpEvaluator(cfg, statistics, distributed=distributed, output_folder=output_folder)

    @classmethod
    def test(cls, cfg, model):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        data_loader, statistics = cls.build_test_loader(cfg)
        evaluator = cls.build_evaluator(cfg, statistics)
        result = inference_on_dataset(model, data_loader, evaluator)

        if comm.is_main_process():
            assert isinstance(
                result, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                result
            )
            #print_csv_format(result)

        if len(result) == 1:
            result = list(result.values())[0]
        return result