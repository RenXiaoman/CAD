import torch.nn.functional as F


def softmax_helper(x):
    return F.softmax(x, 1)
