from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels, hidden_channels = 128) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(channels, hidden_channels, kernel_size=3, padding_mode='zeros', padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding_mode='zeros', padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)

    def forward(self, x):
        skip = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + skip
        out = F.relu(out)
        return out

class HexNet(nn.Module):
    def __init__(self, n_res_block = 20) -> None:
        super().__init__()

        self.input_block = ResBlock(1)
        self.blocks = nn.ModuleList([ResBlock(128) for _ in range(n_res_block-1)])

        # Policy
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 11 * 11, 11 * 11)
        # Value
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 11 * 11, 128)
        self.value_fc2 = nn.Linear(128, 1)


    def forward(self, x) ->  Tuple[torch.Tensor, torch.Tensor]:
        # x (batch, 1, 11, 11)
        x = self.input_block(x)

        for block in self.blocks:
            x = block(x)

        # Policy
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        # Value
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = F.tanh(self.value_fc2(value))

        return policy, value
