def mean_tensor(tensor, axes, keepdim=False):
    axes = sorted(axes, reverse=True)
    for axis in axes:
        tensor = tensor.mean(dim=axis, keepdim=keepdim)
    return tensor
