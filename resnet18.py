# ResNet-18 implementation based on publication https://arxiv.org/abs/1512.03385

import torch
from torch import nn

from resnet_blocks import ResNetBlock

class ResNet18(nn.Module):

    def __init__(self, class_num):
        super(ResNet18, self).__init__()

        self.class_num = class_num

        self.conv2_blocks = 2
        self.conv3_blocks = 2
        self.conv4_blocks = 2
        self.conv5_blocks = 2

        self.conv1 = nn.Sequential()
        self.conv1.add_module(
            'conv2_1',
            nn.Conv2d(
                in_channels = 3,
                out_channels = 64,
                kernel_size = 7,
                stride = 2,
                padding = 3
            )
        )

        current_channels = 64

        #act map should be of size 64 x 112x112
        self.conv2 = nn.Sequential()
        self.conv2.add_module(
            'conv2_max_pool',
            nn.MaxPool2d(
                kernel_size = 3,
                stride = 2,
                padding = 1
            )
        )

        #act map should be of size 64 x 56 x 56
        for block_idx in range(self.conv2_blocks):
            self.conv2.add_module(
                f'conv2_{block_idx+1}',
                ResNetBlock(
                    current_channels
                )
            )

        #act map should be of size 128 x 28 x 28
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

        #act map should be of size 256 x 14 x 14
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

        #act map should be of size 512 x 7 x 7
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

        #act map should be of size 512 x 7 x 7
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fully_connected = nn.Linear(current_channels, self.class_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        print(f"x shape before resnet is {x.shape}")
        x = self.conv1(x)
        print(f"x shape after conv1 {x.shape}")
        x = self.conv2(x)
        print(f"x shape after conv2 {x.shape}")
        x = self.conv3(x)
        print(f"x shape after conv3 {x.shape}")
        x = self.conv4(x)
        print(f"x shape after conv4 {x.shape}")
        x = self.conv5(x)
        print(f"x shape after conv5 {x.shape}")

        x = self.avg_pool(x)

        print(f"x shape after avg_pooling {x.shape}")

        x = torch.squeeze(x, dim = 3)
        x = torch.squeeze(x, dim = 2)

        x = self.fully_connected(x)
        x = self.softmax(x)

        print(f"x shape after fully connected {x.shape}")

        return x
