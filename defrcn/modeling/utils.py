import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def concat_all_gathered(tensor):
    """gather and concat tensor from all GPUs"""
    gathered = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, tensor)
    output = torch.cat(gathered, dim=0)
    return output

@torch.no_grad()
def select_all_gather(tensor, idx):
    """
    args:
        idx (LongTensor), 0s and 1s.
    Performs all_gather operation on the provided tensors sliced by idx.
    """
    world_size = torch.distributed.get_world_size()

    tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)

    idx_gather = [torch.ones_like(idx) for _ in range(world_size)]
    torch.distributed.all_gather(idx_gather, idx, async_op=False)
    idx_gather = torch.cat(idx_gather , dim=0)
    keep = torch.where(idx_gather)
    return output[keep]


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)