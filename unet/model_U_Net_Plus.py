# Tool: PyCharm
# coding: utf-8
"""=========================================
# Project_Name: light_seg
# Author: WenDong
# Date: 2022/4/8 15:46
# Function: 
# Description: 
========================================="""

import torch
from torch import nn
# from model.Attention_block.block_CBAM import ChannelGate
import torch.nn.functional as F
from unet.cbam import CBAM


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=True),
            # nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class d_up(nn.Module):
    def __init__(self, ch_in, ch_out, scale_factor):
        super(d_up, self).__init__()
        self.scale_factor = scale_factor
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=self.scale_factor),
            nn.Conv2d(self.ch_in, self.ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=ch_in, bias=False),  # 深度卷积&逐点卷积
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=False),

            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),  # 从上层网络nn.Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存，不用多存储其他变量。

            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, groups=ch_out, bias=False),  # 深度卷积&逐点卷积
            nn.BatchNorm2d(ch_out),
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0, bias=False),

            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio, pool_types=['avg']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.reduction_ratio = reduction_ratio
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(self.gate_channels, self.gate_channels // 4),
            nn.ReLU(),
            nn.Linear(self.gate_channels // 4, self.gate_channels)
        )
        self.pool_types = pool_types
        self.sig = nn.Sigmoid()

    def forward(self, x):
        channel_att_sum = None
        # print(x.shape)
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = self.sig(x)
        print(scale.shape)

        # print('c_out', out.shape)
        return scale


class SeparableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               in_channels,
                               kernel_size,
                               stride,
                               padding,
                               dilation,
                               groups=1,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels,
                                   out_channels,
                                   1,
                                   1,
                                   0,
                                   1,
                                   1,
                                   bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class MFM(nn.Module):
    def __init__(self, img_ch, out_ch):
        super(MFM, self).__init__()
        self.rate = 3
        if self.rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = 3
            self.conv1 = SeparableConv2d(out_ch, out_ch, 3, 1, 1)
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.relu1 = nn.ReLU(inplace=True)
        self.atrous_convolution = SeparableConv2d(img_ch, out_ch, kernel_size, 1, padding, self.rate)
        self.conv_1 = nn.Conv2d(out_ch, out_ch, kernel_size=(1,1))
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self._init_weight()

    def forward(self, x):
        x1 = self.atrous_convolution(x)
        x1 = self.bn(x1)
        x1 = self.relu(x1)

        if self.rate != 1:
            x2 = self.conv1(x1)
            x2 = self.bn1(x2)
            x2 = self.relu1(x2)
            x1 = x1 + x2
        return x1

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# U_Net
class U_Net_PLUS(nn.Module):
    def __init__(self, img_ch=3, out_ch=1):
        super(U_Net_PLUS, self).__init__()
        self.img_ch = img_ch
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = MFM(img_ch=self.img_ch, out_ch=64)
        self.Conv2 = MFM(img_ch=64, out_ch=128)
        self.Conv3 = MFM(img_ch=128, out_ch=256)
        self.Conv4 = MFM(img_ch=256, out_ch=512)
        self.Conv5 = MFM(img_ch=512, out_ch=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = MFM(img_ch=1024, out_ch=512)
        self.d5_up = d_up(ch_in=512, ch_out=1, scale_factor=8)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = MFM(img_ch=512, out_ch=256)
        self.d4_up = d_up(ch_in=256, ch_out=1, scale_factor=4)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = MFM(img_ch=256, out_ch=128)
        self.d3_up = d_up(ch_in=128, ch_out=1, scale_factor=2)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = MFM(img_ch=128, out_ch=64)
        self.d2_up = d_up(ch_in=64, ch_out=1, scale_factor=1)

        self.Conv_1x1 = nn.Conv2d(64, out_ch, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)


        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        return d1, d2


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AttNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.cbam_64 = CBAM(n_channels)
        self.cbam_up = OutConv(n_channels, n_classes)
    
    def forward(self, x):
        _, cbam_64 = self.cbam_64(x)
        attn_score = torch.sigmoid(self.cbam_up(cbam_64))
        return attn_score


class Edge(nn.Module):
    def __init__(self, img_ch=3, out_ch=8):
        super(Edge, self).__init__()
        self.img_ch = img_ch

        self.Conv1_1 = MFM(self.img_ch, out_ch)
    def forward(self, x):
        x1_1 = self.Conv1_1(x)
        return x1_1


if __name__ == '__main__':
    input = torch.randn(4, 3, 224, 224).cuda()
    # change = torch.randn(4, 4, 224, 224).cuda()
    net = U_Net_PLUS(img_ch=3, out_ch=1).cuda()
    y = net(input)
    print(y.shape)
    # net = WHY(ch_in=8, ch_out=8)

    # from thop import profile
    # #
    # flops, params = profile(net, inputs=(input,))
    # print('Total params: %.3fM' % (params / 1000000.0))
    # print('Total flops: %.3fM' % (flops / 1000000.0))
    # Total params: 25.580M    channel = 9
    # Total flops: 188988.048M
    # Total params: 25.580M    channel = 3
    # Total flops: 47490.520M
