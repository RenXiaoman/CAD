#!/usr/bin/env python3

import torch
from torch import nn

from models.ResGNet import ResGNet, VNet


class ResGNetTwoChannelLogits(nn.Module):
    """Wrap VNet(ResGNet) sigmoid output as two-channel logits."""

    def __init__(self):
        super().__init__()
        self.model = VNet(ResGNet)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        foreground_prob = self.model(x).clamp(1e-6, 1.0 - 1e-6)
        foreground_logit = torch.logit(foreground_prob)
        background_logit = torch.logit(1.0 - foreground_prob)
        return torch.cat([background_logit, foreground_logit], dim=1)


def build_resgnet_two_channel_logits() -> nn.Module:
    return ResGNetTwoChannelLogits()
