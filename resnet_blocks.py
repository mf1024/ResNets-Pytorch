from torch import nn

class ResNetBlock(nn.Module):

    def __init__(self, in_channels_block, is_downsampling_block = False):
        super(ResNetBlock, self).__init__()

        self.in_channels_block = in_channels_block
        self.out_channels_block = in_channels_block
        self.is_downsampling_block = is_downsampling_block

        self.layer_1_stride = 1

        if self.is_downsampling_block:
            self.out_channels_block *= 2
            self.layer_1_stride = 2

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
            out_channels = self.out_channels_block,
            kernel_size = 3,
            stride = self.layer_1_stride,
            padding = 1)
        self.batch_norm_1 = nn.BatchNorm2d(self.out_channels_block)
        self.relu_1 = nn.ReLU()

        self.conv_layer_2 = nn.Conv2d(
            in_channels = self.out_channels_block,
            out_channels = self.out_channels_block,
            kernel_size = 3,
            stride = 1,
            padding = 1)
        self.batch_norm_2 = nn.BatchNorm2d(self.out_channels_block)
        self.relu_res = nn.ReLU()

    def forward(self,x):

        identity = x

        if self.is_downsampling_block:
            identity = self.projection_shortcut.forward(identity)
            identity = self.projection_batch_norm.forward(identity)

        x = self.conv_layer_1.forward(x)
        x = self.batch_norm_1.forward(x)
        x = self.relu_1.forward(x)

        x = self.conv_layer_2.forward(x)
        x = self.batch_norm_2.forward(x)
        x = x + identity

        x = self.relu_res.forward(x)

        return x


class ResNetBottleneckBlock(nn.Module):

    def __init__(self, in_channels_block, is_downsampling_block = False):
        super(ResNetBottleneckBlock, self).__init__()

        self.is_downsampling_block = is_downsampling_block
        self.in_channels_block = in_channels_block
        self.bottleneck_channels = in_channels_block // 4
        self.out_channels_block = self.bottleneck_channels * 4
        self.layer_1_stride = 1

        if self.is_downsampling_block:
            self.bottleneck_channels *= 2
            self.out_channels_block *= 2
            self.layer_1_stride = 2

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
            stride=self.layer_1_stride,
            padding=0)
        self.batch_norm_1 = nn.BatchNorm2d(self.bottleneck_channels)
        self.relu_1 = nn.ReLU()

        self.conv_layer_2 = nn.Conv2d(
            in_channels = self.bottleneck_channels,
            out_channels = self.bottleneck_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )
        self.batch_norm_2 = nn.BatchNorm2d(self.bottleneck_channels)
        self.relu_2 = nn.ReLU()

        self.conv_layer_3 = nn.Conv2d(
            in_channels = self.bottleneck_channels,
            out_channels = self.out_channels_block,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )
        self.batch_norm_3 = nn.BatchNorm2d(self.out_channels_block)
        self.relu_res = nn.ReLU()


    def forward(self,x):

        identity = x

        if self.is_downsampling_block:
            identity = self.projection_shortcut.forward(identity)
            identity = self.projection_batch_norm.forward(identity)

        x = self.conv_layer_1.forward(x)
        x = self.batch_norm_1.forward(x)
        x = self.relu_1.forward(x)

        x = self.conv_layer_2.forward(x)
        x = self.batch_norm_2.forward(x)
        x = self.relu_2.forward(x)

        x = self.conv_layer_3.forward(x)
        x = x + identity
        x = self.relu_res.forward(x)

        return x


class ResNetBlock_v2(nn.Module):

    def __init__(self, in_channels_block, is_downsampling_block = False):
        super(ResNetBlock_v2, self).__init__()

        self.in_channels_block = in_channels_block
        self.out_channels_block = in_channels_block
        self.is_downsampling_block = is_downsampling_block

        self.layer_1_stride = 1

        if self.is_downsampling_block:
            self.out_channels_block *= 2
            self.layer_1_stride = 2

            self.projection_shortcut = nn.Conv2d(
                in_channels = self.in_channels_block,
                out_channels = self.out_channels_block,
                kernel_size = 1,
                stride = 2,
                padding = 0
            )

        self.batch_norm_1 = nn.BatchNorm2d(self.in_channels_block)
        self.relu_1 = nn.ReLU()
        self.conv_layer_1 = nn.Conv2d(
            in_channels = self.in_channels_block,
            out_channels = self.out_channels_block,
            kernel_size = 3,
            stride = self.layer_1_stride,
            padding = 1)

        self.batch_norm_2 = nn.BatchNorm2d(self.out_channels_block)
        self.relu_2 = nn.ReLU()
        self.conv_layer_2 = nn.Conv2d(
            in_channels = self.out_channels_block,
            out_channels = self.out_channels_block,
            kernel_size = 3,
            stride = 1,
            padding = 1)

    def forward(self,x):

        identity = x

        if self.is_downsampling_block:
            identity = self.projection_shortcut.forward(identity)

        x = self.batch_norm_1.forward(x)
        x = self.relu_1.forward(x)
        x = self.conv_layer_1.forward(x)

        x = self.batch_norm_2.forward(x)
        x = self.relu_2.forward(x)
        x = self.conv_layer_2.forward(x)

        x = x + identity

        return x


class ResNetBottleneckBlock_v2(nn.Module):

    def __init__(self, in_channels_block, is_downsampling_block = False):
        super(ResNetBottleneckBlock_v2, self).__init__()

        self.is_downsampling_block = is_downsampling_block
        self.in_channels_block = in_channels_block
        self.bottleneck_channels = in_channels_block // 4
        self.out_channels_block = self.bottleneck_channels * 4
        self.layer_1_stride = 1

        if self.is_downsampling_block:
            self.bottleneck_channels *= 2
            self.out_channels_block *= 2
            self.layer_1_stride = 2

            self.projection_shortcut = nn.Conv2d(
                in_channels = self.in_channels_block,
                out_channels = self.out_channels_block,
                kernel_size = 1,
                stride = 2,
                padding = 0
            )


        self.batch_norm_1 = nn.BatchNorm2d(self.in_channels_block)
        self.relu_1 = nn.ReLU()
        self.conv_layer_1 = nn.Conv2d(
            in_channels=self.in_channels_block,
            out_channels=self.bottleneck_channels,
            kernel_size=1,
            stride=self.layer_1_stride,
            padding=0)

        self.batch_norm_2 = nn.BatchNorm2d(self.bottleneck_channels)
        self.relu_2 = nn.ReLU()
        self.conv_layer_2 = nn.Conv2d(
            in_channels = self.bottleneck_channels,
            out_channels = self.bottleneck_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )

        self.batch_norm_3 = nn.BatchNorm2d(self.bottleneck_channels)
        self.relu_3 = nn.ReLU()
        self.conv_layer_3 = nn.Conv2d(
            in_channels = self.bottleneck_channels,
            out_channels = self.out_channels_block,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )

    def forward(self,x):

        identity = x

        if self.is_downsampling_block:
            identity = self.projection_shortcut.forward(identity)

        x = self.batch_norm_1.forward(x)
        x = self.relu_1.forward(x)
        x = self.conv_layer_1.forward(x)

        x = self.batch_norm_2.forward(x)
        x = self.relu_2.forward(x)
        x = self.conv_layer_2.forward(x)

        x = self.batch_norm_3.forward(x)
        x = self.relu_3.forward(x)
        x = self.conv_layer_3.forward(x)

        x = x + identity

        return x
