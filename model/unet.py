import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# 代码接口如下，请完善模型
########################################################################################################################
class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        filters = [32, 64, 128, 256, 512]

        # downsampling
        self.conv1 = UnetConv2D(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = UnetConv2D(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = UnetConv2D(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = UnetConv2D(filters[2], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center = UnetConv2D(filters[3], filters[4])

        # upsampling
        self.up_concat4 = UnetUp2D(filters[4], filters[3])
        self.up_concat3 = UnetUp2D(filters[3], filters[2])
        self.up_concat2 = UnetUp2D(filters[2], filters[1])
        self.up_concat1 = UnetUp2D(filters[1], filters[0])

        self.final = nn.Conv2d(filters[0], self.out_channels, kernel_size=1)
        # dropout防止过拟合
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        center = self.dropout(center)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)
        if self.out_channels > 2:
            final = F.softmax(final, dim=1)  # 多类分割任务使用 softmax
        else:
            final = torch.sigmoid(final)  # 二类分割任务使用 sigmoid

        return final


class UnetConv2D(nn.Module):
    def __init__(self, in_size, out_size):
        super(UnetConv2D, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
                                   nn.InstanceNorm2d(out_size),
                                   nn.LeakyReLU(), )
        self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
                                   nn.InstanceNorm2d(out_size),
                                   nn.LeakyReLU(), )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetUp2D(nn.Module):
    def __init__(self, in_size, out_size):
        super(UnetUp2D, self).__init__()
        self.conv = UnetConv2D(in_size + out_size, out_size)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = [offset // 2, offset // 2, 0, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))
