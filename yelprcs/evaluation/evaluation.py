import logging
from contextlib import contextmanager
from ..utils.logger import log_every_n_seconds
import torch
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

class YelpEvaluator:
    @classmethod
    def __init__(self, cfg, statistics, distributed, output_folder):
        self.output_dir = output_folder
        self.distributed = distributed
        self.statistics = statistics
        self.rmse = []
        self.user_num = []
        self.business_num = []
        user_num_thrs = [0] + cfg.TEST.USER_NUM_THRS + [1e5]
        item_num_thrs = [0] + cfg.TEST.ITEM_NUM_THRS + [1e5]

        self.user_num_thrs = [[0, 1e5]]
        self.item_num_thrs = [[0, 1e5]]

        for i in range(1,len(user_num_thrs)-1):
            self.user_num_thrs.append([user_num_thrs[i], user_num_thrs[i+1]])
        for i in range(1,len(item_num_thrs)-1):
            self.item_num_thrs.append([item_num_thrs[i], item_num_thrs[i+1]])


    def evaluate(self):
        self.rmse = np.array(self.rmse)
        self.user_num = np.array(self.user_num)
        self.item_num = np.array(self.business_num)
        result = []
        for ithr in self.item_num_thrs:
            ithr_result = ["ithr: [{:3d}, {:3d})".format(int(ithr[0]), int(ithr[1]))]
            for uthr in self.user_num_thrs:
                user_mask = np.logical_and(self.user_num >= uthr[0], self.user_num < uthr[1])
                item_mask = np.logical_and(self.item_num >= ithr[0], self.item_num < ithr[1])
                idx_mask = np.logical_and(user_mask, item_mask)

                ithr_result.append((np.sqrt(np.mean(self.rmse[idx_mask])), "{:2f}".format(idx_mask.sum() / len(idx_mask))))
            result.append(ithr_result)
        
        header = []
        for uthr in self.user_num_thrs:
            header.append("uthr: ({:3d}, {:3d})".format(int(uthr[0]), int(uthr[1])))

        return {'table': result, 'header': header} ,tabulate(result, header, tablefmt="github")

    def process(self, input, output):
        stars = np.array([x['stars'] for x in input])
        self.user_num.extend([self.statistics['user_num'][x['user_id']] for x in input])
        self.business_num.extend([self.statistics['business_num'][x['business_id']] for x in input])
        self.rmse.extend(((stars - output) ** 2).tolist())


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

def inference_on_dataset(model, data_loader, evaluator):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    #num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length

    num_warmup = min(5, total - 1)
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(tqdm(data_loader, desc="processing input")):
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            evaluator.process(inputs, outputs)

            if idx >= num_warmup * 2:
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}.".format(
                        idx + 1, total
                    ),
                    n=5,
                )

    results, table = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    print(table)
    return results

