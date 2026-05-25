import numpy as np
import torch


def maybe_to_torch(data):
    if isinstance(data, list):
        return [maybe_to_torch(item) if not isinstance(item, torch.Tensor) else item for item in data]
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float()
    return torch.as_tensor(data).float()


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        return [item.cuda(gpu_id, non_blocking=non_blocking) for item in data]
    return data.cuda(gpu_id, non_blocking=non_blocking)
