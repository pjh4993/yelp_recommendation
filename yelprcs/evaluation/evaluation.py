import logging
from contextlib import contextmanager
from ..utils.logger import log_every_n_seconds
import torch
import numpy as np

class YelpEvaluator:
    @classmethod
    def __init__(self, cfg, distributed, output_folder):
        self.output_dir = output_folder
        self.distributed = distributed
        self.rmse = []

    def evaluate(self):
        return {'rmse': np.mean(self.rmse)}

    def processs(self, input, output):
        self.rmse.append(np.sqrt((input['stars'] - output) ** 2).item())


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
        for idx, inputs in enumerate(data_loader):
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
            break

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

