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

def apply_random_block_mask(x: torch.Tensor, block_h: int = 3, block_w: int = 3, noise_level: float = 1.0):
    """
    x: (R, C, H, W)
    returns:
        masked_x: Input with blocks filled with noise (simulating occlusion)
        mask: Binary mask (1 for valid, 0 for occluded) for loss computation
    """
    R, C, H, W = x.shape
    mask = x.new_ones((R, 1, H, W))
    
    # Calculate start positions
    max_h_start = max(1, H - block_h + 1)
    max_w_start = max(1, W - block_w + 1)
    h_starts = torch.randint(0, max_h_start, (R,), device=x.device)
    w_starts = torch.randint(0, max_w_start, (R,), device=x.device)
    
    # Create a clone for the masked input
    masked_x = x.clone()
    
    for i in range(R):
        hs = int(h_starts[i].item())
        ws = int(w_starts[i].item())
        he = min(H, hs + block_h)
        we = min(W, ws + block_w)
        
        # Update the binary mask for loss tracking
        mask[i, :, hs:he, ws:we] = 0.0
        
        # SUGGESTION APPLIED HERE:
        # Instead of zeroing out (masked_x[...] = 0), fill with noise.
        # This simulates X_occ (occluder features) described in Eq 8.
        noise = torch.randn((C, he-hs, we-ws), device=x.device) * noise_level
        masked_x[i, :, hs:he, ws:we] = noise

    return masked_x, mask


def sample_indices(total: int, ratio: float, max_samples: Optional[int] = None, device=None):
    k = int(total * ratio)
    if max_samples is not None:
        k = min(k, int(max_samples))
    k = max(1, k)
    return torch.randperm(total, device=device)[:k]