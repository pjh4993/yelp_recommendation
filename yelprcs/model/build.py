from ..utils import Registry
import torch

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = ""

def build_model(cfg):
    model_name = cfg.MODEL.NAME
    model = MODEL_REGISTRY.get(model_name)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model
