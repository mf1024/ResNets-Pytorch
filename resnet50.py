# ResNet-50 implementation based on publication https://arxiv.org/abs/1512.03385

from torch import nn
import torch

import numpy as np
import matplotlib.pyplot as plt

from resnet_blocks import ResNetBottleneckBlock

class ResNet50(nn.Module):

    def __init__(self, class_num):
        super(ResNet50, self).__init__()

        self.class_num = class_num

        self.conv2_blocks = 3
        self.conv3_blocks = 4
        self.conv4_blocks = 6
        self.conv5_blocks = 3

        self.conv1 = nn.Sequential()
        self.conv1.add_module(
            'conv2_1',
            nn.Conv2d(
                in_channels = 3,
                out_channels = 256,
                kernel_size = 7,
                stride = 2,
                padding = 3
            )
        )
        current_channels = 256

        #activation map should of size 256 x 112x112
        self.conv2 = nn.Sequential()
        self.conv2.add_module(
            'conv2_max_pool',
            nn.MaxPool2d(
                kernel_size = 3,
                stride = 2,
                padding = 1
            )
        )

        #activation map of size 256 x 56x56
        for block_idx in range(self.conv2_blocks):
            self.conv2.add_module(
                f'conv2_{block_idx+1}',
                ResNetBottleneckBlock(
                    current_channels
                )
            )

        #activation map of size 512 x 28x28
        self.conv3 = nn.Sequential()
        for block_idx in range(self.conv3_blocks):
            is_downsampling_block = block_idx == 0
            self.conv3.add_module(
                f'conv3_{block_idx+1}',
                ResNetBottleneckBlock(
                    current_channels,
                    is_downsampling_block = is_downsampling_block
                )
            )
            if is_downsampling_block:
                current_channels *= 2

        #activation map should be of size 1024 x 14x14
        self.conv4 = nn.Sequential()
        for block_idx in range(self.conv4_blocks):
            is_downsampling_block = block_idx == 0
            self.conv4.add_module(
                f'conv4_{block_idx+1}',
                ResNetBottleneckBlock(
                    current_channels,
                    is_downsampling_block = is_downsampling_block)
            )
            if is_downsampling_block:
                current_channels *= 2


        #activation map should be of size 2048 x 7x7
        self.conv5 = nn.Sequential()
        for block_idx in range(self.conv5_blocks):
            is_downsampling_block = block_idx == 0
            self.conv5.add_module(
                f'conv5_{block_idx+1}',
                ResNetBottleneckBlock(
                    current_channels,
                    is_downsampling_block = is_downsampling_block)
            )
            if is_downsampling_block:
                current_channels *= 2


        #activation map should be of size 2048 x 7x7
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


def plot_results(image_batch, predictions, truth, image_name = "plot"):

    plot_columns = 4
    plot_rows = image_batch.shape[0] // plot_columns

    img_count = plot_columns * plot_rows

    fig, axes = plt.subplots(plot_rows, plot_columns, figsize=(plot_columns * 4, plot_rows * 3))
    plt.subplots_adjust(hspace = 0.4)

    for img_idx in range(img_count):
        row = img_idx // plot_columns
        col = img_idx % plot_columns

        img = image_batch[img_idx]
        img = np.transpose(img,(1,2,0))

        axes[row, col].imshow(img)

        predicted_class = dataset_test.get_class_name(predictions[img_idx])
        actual_class = dataset_test.get_class_name(truth[img_idx])
        axes[row, col].set_title(f"Predicted class {predicted_class} \n but actually {actual_class}")

    plt.savefig(f"{image_name}.jpg")
    plt.close()

