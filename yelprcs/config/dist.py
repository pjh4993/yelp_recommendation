# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ..utils import comm

__all__ = ["launch"]

def launch(main_func, num_gpus_per_machine, args=()):
    """
    Launch multi-gpu training.
    This function must be called on all machines involved in the training.
    It will spawn child processes (defined by ``num_gpus_per_machine``) on machine.

    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_gpus_per_machine (int): number of GPUs per machine
        args (tuple): arguments passed to main_func
    """
    world_size = num_gpus_per_machine
    if world_size > 1:
        mp.spawn(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(main_func, world_size, num_gpus_per_machine, args),
            daemon=False,
        )
    else:
        main_func(*args)


def _distributed_worker(
    local_rank, main_func, world_size, num_gpus_per_machine, args
):
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    global_rank = local_rank
    try:
        dist.init_process_group(
            backend="NCCL", world_size=world_size, rank=global_rank
        )
    except Exception as e:
        raise e
    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    # Setup the local process group (which contains ranks within the same machine)
    assert comm._LOCAL_PROCESS_GROUP is None

    ranks_on_machine = list(range(0, num_gpus_per_machine))
    pg = dist.new_group(ranks_on_machine)
    comm._LOCAL_PROCESS_GROUP = pg

    main_func(*args)
