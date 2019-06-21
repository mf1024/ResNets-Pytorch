
# Implementation based on publication https://arxiv.org/abs/1512.03385

from torch import nn
import torchvision
from torchvision import transforms, utils

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

import time
import random

from imagenet_dataset import ImageNetDataset



class ResNetBlock(nn.Module):

    # We default to 2 weight layers per block as in paper
    def __init__(self, in_channels_block, is_downsampling_block = False):
        super(ResNetBlock, self).__init__()

        self.in_channels_block = in_channels_block
        self.out_channels_block = in_channels_block
        self.is_downsampling_block = is_downsampling_block

        self.layer_2_stride = 1

        if self.is_downsampling_block:
            self.out_channels_block *= 2
            self.layer_2_stride = 2

            self.projection_shortcut = nn.Conv2d(
                in_channels = self.in_channels_block,
                out_channels = self.out_channels_block,
                kernel_size = 1,
                stride = 2,
                padding = 0
            )

        self.conv_layer_1 = nn.Conv2d(
            in_channels = self.in_channels_block,
            out_channels = self.in_channels_block,
            kernel_size = 3,
            stride = 1,
            padding = 1)
        self.batch_norm_1 = nn.BatchNorm2d(in_channels_block)

        self.conv_layer_2 = nn.Conv2d(
            in_channels = self.in_channels_block,
            out_channels = self.out_channels_block,
            kernel_size = 3,
            stride = self.layer_2_stride,
            padding = 1)
        self.batch_norm_2 = nn.BatchNorm2d(self.out_channels_block)

    def forward(self,x):

        identity = x # Double check if the copy os stored

        if self.is_downsampling_block:
            identity = self.projection_shortcut(identity)

        x = self.conv_layer_1.forward(x)
        x = self.batch_norm_1(x)
        x = nn.functional.relu(x)

        x = self.conv_layer_2.forward(x)
        x = self.batch_norm_2(x)
        x = x + identity

        x = nn.functional.relu(x)

        return x


class ResNetBottleneckBlock(nn.Module):

    def __init__(self, in_channels_block, is_downsampling_block = False):
        super(ResNetBottleneckBlock, self).__init__()

        self.is_downsampling_block = is_downsampling_block
        self.in_channels_block = in_channels_block
        self.bottleneck_channels = in_channels_block // 4
        self.out_channels_block = self.bottleneck_channels * 4
        self.layer_3_stride = 1

        if self.is_downsampling_block:
            self.out_channels_block *= 2
            self.layer_3_stride = 2

            self.projection_shortcut = nn.Conv2d(
                in_channels = self.in_channels_block,
                out_channels = self.out_channels_block,
                kernel_size = 1,
                stride = 2,
                padding = 0
            )

        self.conv_layer_1 = nn.Conv2d(
            in_channels=self.in_channels_block,
            out_channels=self.bottleneck_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.batch_norm_1 = nn.BatchNorm2d(self.bottleneck_channels)

        self.conv_layer_2 = nn.Conv2d(
            in_channels = self.bottleneck_channels,
            out_channels = self.bottleneck_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )
        self.batch_norm_2 = nn.BatchNorm2d(self.bottleneck_channels)

        self.conv_layer_3 = nn.Conv2d(
            in_channels = self.bottleneck_channels,
            out_channels = self.out_channels_block,
            kernel_size = 1,
            stride = self.layer_3_stride,
            padding = 0
        )
        self.batch_norm_3 = nn.BatchNorm2d(self.in_channels_block)


    def forward(self,x):

        identity = x

        if self.is_downsampling_block:
            identity = self.projection_shortcut(identity)

        x = self.conv_layer_1(x)
        x = self.batch_norm_1(x)
        x = nn.functional.relu(x)

        x = self.conv_layer_2(x)
        x = self.batch_norm_2(x)
        x = nn.functional.relu(x)

        x = self.conv_layer_3(x)
        x = x + identity

        x = nn.functional.relu(x)

        return x


class ResNet(nn.Module):

    def __init__(self, class_num, is_bottleneck_resnet = False):
        super(ResNet, self).__init__()

        print("INIT!")
        self.class_num = class_num
        self.is_bottleneck_resnet = is_bottleneck_resnet

        if self.is_bottleneck_resnet:
            self.BlockClass = ResNetBottleneckBlock
        else:
            self.BlockClass = ResNetBlock

        self.conv2_blocks = 3
        self.conv3_blocks = 4
        self.conv4_blocks = 6
        self.conv5_blocks = 3

        #If ResNet-50 then input from 3x3 max pool should be 256 chnannels
        #If ResNet-34 then input from 3x3 max pool should be 64 channesl

        if self.is_bottleneck_resnet:
            self.conv1_out_channels = 256
        else:
            self.conv1_out_channels = 64


        self.conv1 = nn.Sequential()
        self.conv1.add_module(
            'conv2_1',
            nn.Conv2d(
                in_channels = 3,
                out_channels = self.conv1_out_channels,
                kernel_size = 7,
                stride = 2,
                padding = 3
            )
        )

        current_channels = self.conv1_out_channels

        #should be image of size (256 or 64) x 112x112
        self.conv2 = nn.Sequential()
        self.conv2.add_module(
            'conv2_max_pool',
            nn.MaxPool2d(
                kernel_size = 3,
                stride = 2,
                padding = 1
            )
        )

        #should be image of size (256 or 64) x 56 x 56
        for block_idx in range(self.conv2_blocks):
            is_last_block = block_idx == self.conv2_blocks - 1
            self.conv2.add_module(
                f'conv2_{block_idx+1}',
                self.BlockClass(
                    current_channels,
                    is_downsampling_block = is_last_block
                )
            )

        #should be image of size (512 or 128) x 28 x 28
        current_channels *= 2
        self.conv3 = nn.Sequential()
        for block_idx in range(self.conv3_blocks):
            is_last_block = block_idx == self.conv3_blocks - 1
            self.conv3.add_module(
                f'conv3_{block_idx+1}',
                self.BlockClass(
                    current_channels,
                    is_downsampling_block = is_last_block
                )
            )

        #should be image of size (1024 or 256) x 14 x 14
        current_channels *= 2
        self.conv4 = nn.Sequential()
        for block_idx in range(self.conv4_blocks):
            is_last_block = block_idx == self.conv4_blocks - 1
            self.conv4.add_module(
                f'conv4_{block_idx+1}',
                self.BlockClass(
                    current_channels,
                    is_downsampling_block = is_last_block)
            )

        #should be image of size (2048 or 512) x 7 x 7
        current_channels *= 2
        self.conv5 = nn.Sequential()
        for block_idx in range(self.conv5_blocks):
            self.conv5.add_module(
                f'conv5_{block_idx+1}',
                self.BlockClass(
                    current_channels
                )
            )

        #should be image of size (2048 or 512) x 7 x 7
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fully_connected = nn.Linear(current_channels, self.class_num)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        print(f"x_shape{x.shape}")

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


#TODO: Training
#TODO: Testing
#TODO: SGD

model_resnet50 = ResNet(class_num = 10, is_bottleneck_resnet = True)

data_path = "/Users/martinsf/data/images_1/imagenet_images/"
random_seed = int(time.time())
dataset_train = ImageNetDataset(data_path,is_train = True, random_seed=random_seed, num_classes = 10)

# data_loader_train = DataLoader(dataset_train, BATCH_SIZE, shuffle = True)
#
# import matplotlib.pyplot as plt
#
# for x in data_loader_train:
#
#     print("blaa")
#
#     print(x["image"].shape)
#     y_prim = model_resnet50.forward(x["image"])
#
#     print(torch.sum(y_prim, dim=1))
#     print(f"y_prim {y_prim}")
#
#     for i in range(BATCH_SIZE):
#         img = x['image'][i].numpy()
#         plt.title(x['class_name'][i])
#         plt.imshow(np.transpose(img,(1,2,0)))
#         plt.show()
#
#     break


#Training



NUM_CLASSES = dataset_train.get_class_num()
print(f'num_classes {NUM_CLASSES}')
NUM_EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

model_resnet50 = model_resnet50.to("cpu")
optimizer = torch.optim.Adam(params = model_resnet50.parameters(), lr = LEARNING_RATE)

data_loader_train = DataLoader(dataset_train, BATCH_SIZE, shuffle = True)


for epoch in range(NUM_EPOCHS):

    print(f"Starting epoch {epoch}")

    for batch in data_loader_train:
        x = batch['image']
        y = batch['cls']

        y_one_hot = torch.zeros(BATCH_SIZE, NUM_CLASSES)
        y_one_hot = y_one_hot.scatter_(1, y.unsqueeze(dim=1), 1)

        labels = batch['class_name']

        y_prim = model_resnet50.forward(x)

        loss = torch.sum(-y_one_hot * torch.log(y_prim))
        print(f"loss is {loss}")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

