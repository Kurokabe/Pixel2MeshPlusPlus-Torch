"""
Helper functions that have not yet been implemented in pytorch
"""

import torch


def recursive_detach(t):
    if isinstance(t, torch.Tensor):
        return t.detach()
    elif isinstance(t, list):
        return [recursive_detach(x) for x in t]
    elif isinstance(t, dict):
        return {k: recursive_detach(v) for k, v in t.items()}
    else:
        return t


def batch_mm(matrix, batch):
    """
    https://github.com/pytorch/pytorch/issues/14489
    """
    # TODO: accelerate this with batch operations
    # with torch.cuda.amp.autocast(enabled=False):
    out = [matrix.mm(b) for b in batch]
    return torch.stack(out, dim=0)


def dot(x, y, sparse=False):
    """Wrapper for torch.matmul (sparse vs dense)."""
    if sparse:
        # cast = False
        # if y.dtype == torch.float16:
        #     y = y.to(torch.float32)
        #     cast = True
        # result = batch_mm(x, y)
        # if cast:
        #     result = result.to(torch.float16)
        # return result
        return batch_mm(x, y)
    else:
        return torch.matmul(x, y)
