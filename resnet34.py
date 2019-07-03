# ResNet-34 implementation based on publication https://arxiv.org/abs/1512.03385

from torch import nn

import torch
from resnet_blocks import ResNetBlock

class ResNet34(nn.Module):

    def __init__(self, class_num):
        super(ResNet34, self).__init__()

        self.class_num = class_num

        self.conv2_blocks = 3
        self.conv3_blocks = 4
        self.conv4_blocks = 6
        self.conv5_blocks = 3

        self.conv1 = nn.Sequential()

        self.conv1.add_module(
            'conv1_1',

            nn.Conv2d(
                in_channels = 3,
                out_channels = 64,
                kernel_size = 7,
                stride = 2,
                padding = 3
            )
        )
        self.conv1.add_module(
            'conv1_bn',
            nn.BatchNorm2d(64)
        )
        self.conv1.add_module(
            'conv1_relu',
            nn.ReLU()
        )

        current_channels = 64

        #act map should be image of size 64 x 112x112
        self.conv2 = nn.Sequential()
        self.conv2.add_module(
            'conv2_max_pool',
            nn.MaxPool2d(
                kernel_size = 3,
                stride = 2,
                padding = 1
            )
        )

        #act map should be image of size 64 x 56 x 56
        for block_idx in range(self.conv2_blocks):
            self.conv2.add_module(
                f'conv2_{block_idx+1}',
                ResNetBlock(
                    current_channels
                )
            )

        #act map should be image of size 128 x 28 x 28
        self.conv3 = nn.Sequential()
        for block_idx in range(self.conv3_blocks):
            is_downsampling_block = block_idx == 0
            self.conv3.add_module(
                f'conv3_{block_idx+1}',
                ResNetBlock(
                    current_channels,
                    is_downsampling_block = is_downsampling_block
                )
            )
            if is_downsampling_block:
                current_channels *= 2

        #act map should be image of size 256 x 14 x 14
        self.conv4 = nn.Sequential()
        for block_idx in range(self.conv4_blocks):
            is_downsampling_block = block_idx == 0
            self.conv4.add_module(
                f'conv4_{block_idx+1}',
                ResNetBlock(
                    current_channels,
                    is_downsampling_block = is_downsampling_block)
            )
            if is_downsampling_block:
                current_channels *= 2

        #act map should be image of size 512 x 7 x 7
        self.conv5 = nn.Sequential()
        for block_idx in range(self.conv5_blocks):
            is_downsampling_block = block_idx == 0
            self.conv5.add_module(
                f'conv5_{block_idx+1}',
                ResNetBlock(
                    current_channels,
                    is_downsampling_block = is_downsampling_block)
            )
            if is_downsampling_block:
                current_channels *= 2

        #act map should be image of size 512 x 7 x 7
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fully_connected = nn.Linear(current_channels, self.class_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avg_pool(x)

        x = torch.squeeze(x, dim = 3)
        x = torch.squeeze(x, dim = 2)

        x = self.fully_connected(x)
        x = self.softmax(x)

        return x
