
# Implementation based on publication https://arxiv.org/abs/1512.03385

from torch import nn
import torchvision
from torchvision import transforms, utils

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

import os

import time
import random

from imagenet_dataset import get_imagenet_datasets

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

            self.projection_batch_norm = nn.BatchNorm2d(self.out_channels_block)

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
            identity = self.projection_batch_norm(identity)

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

            self.projection_batch_norm = nn.BatchNorm2d(self.out_channels_block)

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
            identity = self.projection_batch_norm(identity)

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

NUM_CLASSES = None
NUM_CLASSES = 10

data_path = "/home/martin/ai/ImageNet-datasets-downloader/images_4/imagenet_images"
dataset_train, dataset_test = get_imagenet_datasets(data_path, num_classes = NUM_CLASSES)

if NUM_CLASSES == None:
    NUM_CLASSES = dataset_train.get_number_of_classes()

NUM_TRAIN_SAMPLES = dataset_train.get_number_of_samples()
NUM_TEST_SAMPLES= dataset_test.get_number_of_samples()

print(f"train_samples  {NUM_TRAIN_SAMPLES} test_samples {NUM_TEST_SAMPLES}")

print(f'num_classes {NUM_CLASSES}')
NUM_EPOCHS = 10000
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
#DEVICE = 'cuda'
DEVICE = 'cpu'

model_resnet50 = ResNet(class_num = NUM_CLASSES, is_bottleneck_resnet = True).to(DEVICE)
optimizer = torch.optim.Adam(params = model_resnet50.parameters(), lr = LEARNING_RATE)

def layers_debug(optim):
    layer_count = 0
    for var_name in optim.state_dict():
        if len(optim.state_dict()[var_name].shape)>1:
            layer_count += 1
            print(f"{var_name}\t\t{optim.state_dict()[var_name].shape}")
    print(layer_count)


layers_debug(model_resnet50)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"number of resnet50 params {count_parameters(model_resnet50)}")

trained_models_path = "./trained_models"
last_model_path = os.path.join(trained_models_path, "last.pt")
best_model_path = os.path.join(trained_models_path, "best.pt")

print(last_model_path)

best_test_acc = 0

data_loader_train = DataLoader(dataset_train, BATCH_SIZE, shuffle = True)
data_loader_test = DataLoader(dataset_test, BATCH_SIZE, shuffle = True)

for epoch in range(NUM_EPOCHS):

    print(f"Starting epoch {epoch}")

    #TRAINING

    model_resnet50 = model_resnet50.train()

    epoch_train_losses = []
    epoch_train_true_positives = 0

    for batch_idx, batch in enumerate(data_loader_train):

        x = batch['image'].to(DEVICE)
        y = batch['cls'].to(DEVICE)

        y_one_hot = torch.zeros(x.shape[0], NUM_CLASSES).to(DEVICE)
        y_one_hot = y_one_hot.scatter_(1, y.unsqueeze(dim=1), 1)

        labels = batch['class_name']

        y_prim = model_resnet50.forward(x)

        loss = torch.sum(-y_one_hot * torch.log(y_prim))
        epoch_train_losses.append(loss.detach().to('cpu').numpy())

        epoch_train_true_positives += torch.sum(y_prim.argmax(dim=1) == y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    epoch_accuracy = float(epoch_train_true_positives) / float(NUM_TRAIN_SAMPLES)
    print(f"true_positives {epoch_train_true_positives} from {NUM_TRAIN_SAMPLES} samples")
    print(f"Epoch {epoch} train mean loss is {np.mean(epoch_train_losses)}")
    print(f"Epoch {epoch} train accuracy is {epoch_accuracy*100} %")

    #TEST
    with torch.no_grad():

        model_resnet50 = model_resnet50.eval()

        epoch_test_losses = []
        epoch_test_true_positives = 0

        for batch_idx, batch in enumerate(data_loader_test):

            x = batch['image'].to(DEVICE)
            y = batch['cls'].to(DEVICE)

            y_one_hot = torch.zeros(x.shape[0], NUM_CLASSES).to(DEVICE)
            y_one_hot = y_one_hot.scatter_(1, y.unsqueeze(dim=1), 1)

            labels = batch['class_name']

            y_prim = model_resnet50.forward(x)
            loss = torch.sum(-y_one_hot * torch.log(y_prim))
            epoch_test_losses.append(loss.detach().to('cpu').numpy())

            epoch_test_true_positives += torch.sum(y_prim.argmax(dim=1) == y)

            if batch_idx == 0:
                plot_results(x.detach().to('cpu').numpy(),
                             y_prim.argmax(dim=1).detach().to('cpu').numpy(),
                             y.detach().to('cpu').numpy(),
                             image_name=f"epoch_{epoch}")


        epoch_test_accuracy = float(epoch_test_true_positives) / float(NUM_TEST_SAMPLES)
        print(f"true_positives {epoch_test_true_positives} from {NUM_TEST_SAMPLES} samples")
        print(f"Epoch {epoch} test mean loss is {np.mean(epoch_test_losses)}")
        print(f"Epoch {epoch} test accuracy is {epoch_test_accuracy * 100}")

        torch.save(model_resnet50, last_model_path)
        if epoch_test_accuracy > best_test_accuracy:
            best_test_accuracy = epoch_test_accuracy
            torch.save(model_resnet50, best_model_path)

    #TODO: Try SGD

