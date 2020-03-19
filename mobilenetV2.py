import json
import pickle
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from lib.nn import SynchronizedBatchNorm2d


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class DepthNetRoboeye(nn.Module):
    def __init__(self):
        super(DepthNetRoboeye, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.convConc = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.convU1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.convU11 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.convU2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.convU22 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.convU3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.convU33 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.convU4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.convU44 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.convU5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.convU55 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.convUd6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.convUd66 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.convR4 = nn.Conv2d(1280, 64, kernel_size=3, padding=1)
        self.convR44 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.convR3 = nn.Conv2d(192, 64, kernel_size=3, padding=1)
        self.convR33 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.convR2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.convR22 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.convR1 = nn.Conv2d(96, 32, kernel_size=3, padding=1)
        self.convR11 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.convF = nn.Conv2d(32, 2, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)
        self.bnU1 = nn.BatchNorm2d(32)
        self.bnU11 = nn.BatchNorm2d(32)
        self.bnU2 = nn.BatchNorm2d(32)
        self.bnU22 = nn.BatchNorm2d(64)
        self.bnU3 = nn.BatchNorm2d(64)
        self.bnU33 = nn.BatchNorm2d(128)
        self.bnU4 = nn.BatchNorm2d(128)
        self.bnU44 = nn.BatchNorm2d(256)
        self.bnU5 = nn.BatchNorm2d(256)
        self.bnU55 = nn.BatchNorm2d(512)
        self.bnUd6 = nn.BatchNorm2d(512)
        self.bnUd66 = nn.BatchNorm2d(512)
        self.bnR4 = nn.BatchNorm2d(1280)
        self.bnR44 = nn.BatchNorm2d(64)
        self.bnR3 = nn.BatchNorm2d(192)
        self.bnR33 = nn.BatchNorm2d(64)
        self.bnR2 = nn.BatchNorm2d(128)
        self.bnR22 = nn.BatchNorm2d(64)
        self.bnR1 = nn.BatchNorm2d(96)
        self.bnR11 = nn.BatchNorm2d(32)

        self.dropOut = nn.Dropout(p=0.2)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, left, right):
        _, _, height, width = left.shape
        # Layer-1 [32, 160, 320]
        conv1_l = self.dropOut(F.relu(self.bn1(self.conv1(left))))
        conv1_r = self.dropOut(F.relu(self.bn1(self.conv1(right))))

        # Layer-2 [64, 80, 160]
        conv2_l = self.dropOut(F.relu(self.bn2(self.conv2(conv1_l))))
        conv2_r = self.dropOut(F.relu(self.bn2(self.conv2(conv1_r))))

        # Layer-3 [128, 40, 80]
        conv3_l = self.dropOut(F.relu(self.bn3(self.conv3(conv2_l))))
        conv3_r = self.dropOut(F.relu(self.bn3(self.conv3(conv2_r))))

        # Layer , Concatenate
        conv_ldi = self.convConc(conv3_l)
        conv_rdi = self.convConc(conv3_r)
        concat1 = torch.cat([conv_ldi, conv_rdi], 1)  # [256, 40, 80]

        # layer-4 Upsample [128, 80, 160]
        conv4 = nn.functional.interpolate(
            concat1, [height // 4, width // 4], mode="nearest"
        )
        conv4 = self.dropOut(F.relu(self.bn4(self.conv4(conv4))))

        # layer-5 Upsample [64, 160, 320]
        up_conv3 = nn.functional.interpolate(
            conv4, [height // 2, width // 2], mode="nearest"
        )
        up_conv3 = self.dropOut(F.relu(self.bn5(self.conv5(up_conv3))))

        # layer-6 Upsample [32, 160, 320]
        up_conv3 = nn.functional.interpolate(
            up_conv3, [height, width], mode="nearest"
        )
        up_conv3 = self.dropOut(F.relu(self.bn6(self.conv6(up_conv3))))

        # u-net LAYER -1 # [32, 160, 320]
        conv1 = self.convU11(
            self.dropOut(
                F.relu(self.bnU11(self.convU1(F.relu(self.bnU1(up_conv3)))))
            )
        )
        pool1 = self.pool(conv1)
        # u-net LAYER -2 # [64, 80, 160]
        conv2 = self.convU22(
            self.dropOut(
                F.relu(self.bnU22(self.convU2(F.relu(self.bnU2(pool1)))))
            )
        )
        pool2 = self.pool(conv2)
        # u-net LAYER -3 # [128, 40, 80]
        conv3 = self.convU33(
            self.dropOut(
                F.relu(self.bnU33(self.convU3(F.relu(self.bnU3(pool2)))))
            )
        )
        pool3 = self.pool(conv3)
        # u-net LAYER -4 # [256, 20, 40]
        conv4 = self.convU44(
            self.dropOut(
                F.relu(self.bnU44(self.convU4(F.relu(self.bnU4(pool3)))))
            )
        )
        pool4 = self.pool(conv4)
        # u-net LAYER -5 # [512, 20, 40]
        conv5 = self.convU55(
            self.dropOut(
                F.relu(self.bnU55(self.convU5(F.relu(self.bnU5(pool4)))))
            )
        )

        # 6th layer dilation
        conv6 = self.convUd66(
            self.dropOut(
                F.relu(self.bnUd66(self.convUd6(F.relu(self.bnUd6(conv5)))))
            )
        )
        conv7 = self.convUd66(
            self.dropOut(
                F.relu(self.bnUd66(self.convUd6(F.relu(self.bnUd6(conv6)))))
            )
        )
        conv8 = self.convUd66(
            self.dropOut(
                F.relu(self.bnUd66(self.convUd6(F.relu(self.bnUd6(conv7)))))
            )
        )

        # =====================
        # Decoding
        # =====================
        # 5th layer
        merged = torch.cat([conv8, conv5], 1)

        # 4th layer
        merged = torch.cat(
            [
                nn.functional.interpolate(
                    merged, [height // 8, width // 8], mode="nearest"
                ),
                conv4,
            ],
            1,
        )
        up_conv4 = self.convR44(
            self.dropOut(
                F.relu(self.bnR44(self.convR4(F.relu(self.bnR4(merged)))))
            )
        )

        # 3rd layer
        merged = torch.cat(
            [
                nn.functional.interpolate(
                    up_conv4, [height // 4, width // 4], mode="nearest"
                ),
                conv3,
            ],
            1,
        )
        up_conv3 = self.convR33(
            self.dropOut(
                F.relu(self.bnR33(self.convR3(F.relu(self.bnR3(merged)))))
            )
        )

        # 2nd layer
        merged = torch.cat(
            [
                nn.functional.interpolate(
                    up_conv3, [height // 2, width // 2], mode="nearest"
                ),
                conv2,
            ],
            1,
        )
        up_conv2 = self.convR22(
            self.dropOut(
                F.relu(self.bnR22(self.convR2(F.relu(self.bnR2(merged)))))
            )
        )

        # 2nd layer
        merged = torch.cat(
            [
                nn.functional.interpolate(
                    up_conv2, [height, width], mode="nearest"
                ),
                conv1,
            ],
            1,
        )
        up_conv1 = self.convR11(
            self.dropOut(
                F.relu(self.bnR11(self.convR1(F.relu(self.bnR1(merged)))))
            )
        )

        # Final layer Upsample [2, 320, 640]
        convF = self.sigmoid(self.convF(up_conv1))

        return convF


class DepthNetSmall(nn.Module):
    def __init__(self):
        super(DepthNetSmall, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.convConc = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.dropOut = nn.Dropout(p=0.2)

        # self.conv1_2 = nn.Conv2d(256, 256+128, kernel_size=3, stride=1, padding=1)
        # self.conv2_3 = nn.Conv2d(256+128, 256, kernel_size=3, stride=1, padding=1)
        # self.conv3_4 = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)

        self.conv1_2 = nn.Conv2d(
            256, 256 + 32, kernel_size=3, stride=1, padding=1
        )
        self.conv2_3 = nn.Conv2d(
            256 + 32, 256 + 64, kernel_size=3, stride=1, padding=1
        )
        self.conv3_4 = nn.Conv2d(
            256 + 64, 2, kernel_size=3, stride=1, padding=1
        )

        # this should be last
        self.apply(weights_init)

    def forward(self, left, right):
        _, _, height, width = left.shape
        # Layer
        conv1_l = self.dropOut(F.relu(self.bn1(self.conv1(left))))
        conv1_r = self.dropOut(F.relu(self.bn1(self.conv1(right))))

        # Layer
        conv2_l = self.dropOut(F.relu(self.bn2(self.conv2(conv1_l))))
        conv2_r = self.dropOut(F.relu(self.bn2(self.conv2(conv1_r))))

        # Layer
        conv3_l = self.dropOut(F.relu(self.bn3(self.conv3(conv2_l))))
        conv3_r = self.dropOut(F.relu(self.bn3(self.conv3(conv2_r))))

        # Layer , Concatenate
        conv_ldi = self.convConc(conv3_l)
        conv_rdi = self.convConc(conv3_r)
        concat1 = torch.cat([conv_ldi, conv_rdi], 1)

        up_conv = nn.functional.interpolate(
            concat1,
            [height // 4, width // 4],
            mode="bilinear",
            align_corners=True,
        )
        up_conv = F.relu(self.dropOut(self.conv1_2(up_conv)))
        up_conv = nn.functional.interpolate(
            up_conv,
            [height // 2, width // 2],
            mode="bilinear",
            align_corners=True,
        )
        up_conv = F.relu(self.dropOut(self.conv2_3(up_conv)))
        up_conv = nn.functional.interpolate(
            up_conv, [height, width], mode="bilinear", align_corners=True
        )
        up_conv = self.dropOut(up_conv)
        up_conv = self.conv3_4(up_conv)
        final_out = nn.Sigmoid()(up_conv)
        return final_out


class DepthNetSmallSkip(nn.Module):
    def __init__(self):
        super(DepthNetSmallSkip, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.convConc = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.dropOut = nn.Dropout(p=0.2)

        self.conv1_2 = nn.Conv2d(
            128 * 2 + 64, 128 * 2, kernel_size=3, stride=1, padding=1
        )
        self.conv2_3 = nn.Conv2d(
            128 * 2 + 64, 128, kernel_size=3, stride=1, padding=1
        )
        self.conv3_4 = nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1)

        self.skip_conv_1 = nn.Conv2d(
            128, 64, kernel_size=3, stride=1, padding=1
        )
        self.skip_conv_2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1
        )

        # this should be last
        self.apply(weights_init)

    def forward(self, left, right):
        _, _, height, width = left.shape
        # Layer
        conv1_l = self.dropOut(F.relu(self.bn1(self.conv1(left))))
        conv1_r = self.dropOut(F.relu(self.bn1(self.conv1(right))))
        # B,32,239,319

        # Layer
        conv2_l = self.dropOut(F.relu(self.bn2(self.conv2(conv1_l))))
        conv2_r = self.dropOut(F.relu(self.bn2(self.conv2(conv1_r))))
        # 2,64,119,159

        # Layer
        conv3_l = self.dropOut(F.relu(self.bn3(self.conv3(conv2_l))))
        conv3_r = self.dropOut(F.relu(self.bn3(self.conv3(conv2_r))))

        # Layer , Concatenate
        conv_ldi = F.relu(self.convConc(conv3_l))
        conv_rdi = F.relu(self.convConc(conv3_r))
        concat1 = torch.cat([conv_ldi, conv_rdi], 1)
        # B,256,60,80

        skip_1 = torch.cat([conv2_l, conv2_r], 1)
        skip_1 = nn.functional.interpolate(
            skip_1,
            [height // 4, width // 4],
            mode="bilinear",
            align_corners=True,
        )
        skip_1 = self.skip_conv_1(skip_1)
        # B,64,120,160

        skip_2 = torch.cat([conv1_l, conv1_r], 1)
        skip_2 = nn.functional.interpolate(
            skip_2,
            [height // 2, width // 2],
            mode="bilinear",
            align_corners=True,
        )
        skip_2 = self.skip_conv_2(skip_2)
        # # B,64,240,320

        up_conv = nn.functional.interpolate(
            concat1,
            [height // 4, width // 4],
            mode="bilinear",
            align_corners=True,
        )
        up_conv = torch.cat([up_conv, skip_1], 1)
        up_conv = F.relu(self.dropOut(self.conv1_2(up_conv)))
        # B,256,120,160

        up_conv = nn.functional.interpolate(
            up_conv,
            [height // 2, width // 2],
            mode="bilinear",
            align_corners=True,
        )
        up_conv = torch.cat([up_conv, skip_2], 1)
        up_conv = F.relu(self.dropOut(self.conv2_3(up_conv)))

        up_conv = nn.functional.interpolate(
            up_conv, [height, width], mode="bilinear", align_corners=True
        )
        up_conv = self.conv3_4(up_conv)

        final_out = nn.Sigmoid()(up_conv)
        return final_out


class DepthNetSmallSkip2(nn.Module):
    def __init__(self, hyperparameters_dict):
        super(DepthNetSmallSkip2, self).__init__()
        h_d = hyperparameters_dict  # hyperparameters dictionary

        l1 = h_d["c1"]
        self.conv1 = nn.Conv2d(
            l1["in"], l1["out"], kernel_size=l1["k"], stride=2, padding=1
        )
        self.bn1 = nn.BatchNorm2d(l1["out"])
        l2 = h_d["c2"]
        self.conv2 = nn.Conv2d(
            l2["in"], l2["out"], kernel_size=l2["k"], stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(l2["out"])
        l3 = h_d["c3"]
        self.conv3 = nn.Conv2d(
            l3["in"], l3["out"], kernel_size=l3["k"], stride=2, padding=1
        )
        self.bn3 = nn.BatchNorm2d(l3["out"])

        l4 = h_d["c_merge"]
        self.convConc = nn.Conv2d(
            l4["in"], l4["out"], kernel_size=l4["k"], stride=2, padding=1
        )
        l5 = h_d["c_skip_1"]
        self.skip_conv_1 = nn.Conv2d(
            l5["in"], l5["out"], kernel_size=l5["k"], stride=1, padding=1
        )
        l6 = h_d["c_skip_2"]
        self.skip_conv_2 = nn.Conv2d(
            l6["in"], l6["out"], kernel_size=l6["k"], stride=1, padding=1
        )

        l7 = h_d["c5"]
        self.conv1_2 = nn.Conv2d(
            l7["in"], l7["out"], kernel_size=l7["k"], stride=1, padding=1
        )
        l8 = h_d["c6"]
        self.conv2_3 = nn.Conv2d(
            l8["in"], l8["out"], kernel_size=l8["k"], stride=1, padding=1
        )
        l9 = h_d["c7"]
        self.conv3_4 = nn.Conv2d(
            l9["in"], l9["out"], kernel_size=l9["k"], stride=1, padding=1
        )

        self.dropOut = nn.Dropout(p=h_d["dropout"])

        # this should be last
        self.apply(weights_init)

    def forward(self, left, right):
        _, _, height, width = left.shape
        # Layer
        conv1_l = self.dropOut(F.relu(self.bn1(self.conv1(left))))
        conv1_r = self.dropOut(F.relu(self.bn1(self.conv1(right))))
        # B,32,239,319

        # Layer
        conv2_l = self.dropOut(F.relu(self.bn2(self.conv2(conv1_l))))
        conv2_r = self.dropOut(F.relu(self.bn2(self.conv2(conv1_r))))
        # 2,64,119,159

        # Layer
        conv3_l = self.dropOut(F.relu(self.bn3(self.conv3(conv2_l))))
        conv3_r = self.dropOut(F.relu(self.bn3(self.conv3(conv2_r))))

        # Layer , Concatenate
        conv_ldi = F.relu(self.convConc(conv3_l))
        conv_rdi = F.relu(self.convConc(conv3_r))
        concat1 = torch.cat([conv_ldi, conv_rdi], 1)
        # B,256,60,80

        skip_1 = torch.cat([conv2_l, conv2_r], 1)
        skip_1 = nn.functional.interpolate(
            skip_1,
            [height // 4, width // 4],
            mode="bilinear",
            align_corners=True,
        )
        skip_1 = F.relu(self.skip_conv_1(skip_1))
        # B,64,120,160

        skip_2 = torch.cat([conv1_l, conv1_r], 1)
        skip_2 = nn.functional.interpolate(
            skip_2,
            [height // 2, width // 2],
            mode="bilinear",
            align_corners=True,
        )
        skip_2 = F.relu(self.skip_conv_2(skip_2))
        # # B,64,240,320

        up_conv = nn.functional.interpolate(
            concat1,
            [height // 4, width // 4],
            mode="bilinear",
            align_corners=True,
        )
        up_conv = torch.cat([up_conv, skip_1], 1)
        up_conv = F.relu(self.dropOut(self.conv1_2(up_conv)))
        # B,256,120,160

        up_conv = nn.functional.interpolate(
            up_conv,
            [height // 2, width // 2],
            mode="bilinear",
            align_corners=True,
        )
        up_conv = torch.cat([up_conv, skip_2], 1)
        up_conv = F.relu(self.dropOut(self.conv2_3(up_conv)))

        up_conv = nn.functional.interpolate(
            up_conv, [height, width], mode="bilinear", align_corners=True
        )
        up_conv = self.conv3_4(up_conv)

        final_out = nn.Sigmoid()(up_conv)
        return final_out


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        SynchronizedBatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        SynchronizedBatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(
        self, in_planes, out_planes, kernel_size=3, stride=1, groups=1
    ):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend(
            [
                # dw
                ConvBNReLU(
                    hidden_dim, hidden_dim, stride=stride, groups=hidden_dim
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def make_inverted_residual(features, input_channel, ir_setting, width_mult):
    for t, c, n, s in ir_setting:
        output_channel = int(c * width_mult)
        for i in range(n):
            if i == 0:
                features.append(
                    InvertedResidual(
                        input_channel, output_channel, s, expand_ratio=t
                    )
                )
            else:
                features.append(
                    InvertedResidual(
                        input_channel, output_channel, 1, expand_ratio=t
                    )
                )
            input_channel = output_channel
    return features, output_channel


class AutoMobileDepthNet(nn.Module):
    def __init__(
        self,
        input_channel,
        last_channel,
        encoder_block_settings,
        skip_outs,
        decoder_outs,
        width_mult=1.0,
        dropout_prob=0.2,
        image_channels=3,
    ):
        super(AutoMobileDepthNet, self).__init__()
        # blocks : InvertedResidual

        # break into blocks
        interverted_residual_setting_1 = [encoder_block_settings[0]]
        c1 = interverted_residual_setting_1[0][1]

        interverted_residual_setting_2 = [encoder_block_settings[1]]
        c2 = interverted_residual_setting_2[0][1]

        interverted_residual_setting_3 = [encoder_block_settings[2]]
        c3 = interverted_residual_setting_3[0][1]

        interverted_residual_setting_4 = encoder_block_settings[3:]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = (
            int(last_channel * width_mult)
            if width_mult > 1.0
            else last_channel
        )
        # building inverted residual blocks
        # block 1
        self.features_1 = [conv_bn(image_channels, input_channel, 2)]
        self.features_1, input_channel = make_inverted_residual(
            self.features_1,
            input_channel,
            interverted_residual_setting_1,
            width_mult,
        )
        # block 2
        self.features_2, input_channel = make_inverted_residual(
            [], input_channel, interverted_residual_setting_2, width_mult
        )
        # block 3
        self.features_3, input_channel = make_inverted_residual(
            [], input_channel, interverted_residual_setting_3, width_mult
        )
        # blocks 4 - 8
        self.features_4, input_channel = make_inverted_residual(
            [], input_channel, interverted_residual_setting_4, width_mult
        )
        # building last several layers
        # make it nn.Sequential
        self.features_1 = nn.Sequential(*self.features_1)
        self.features_2 = nn.Sequential(*self.features_2)
        self.features_3 = nn.Sequential(*self.features_3)
        # append last channel
        self.features_4.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features_4 = nn.Sequential(*self.features_4)

        # skip_outs = [c1*2, c2*2, c3*2]
        # skip blocks
        skip_1_out = skip_outs[0]
        self.conv_skip_block_1 = ConvBNReLU(c1 * 2, skip_1_out)  # out=32
        skip_2_out = skip_outs[1]
        self.conv_skip_block_2 = ConvBNReLU(c2 * 2, skip_2_out)  # out=48
        skip_3_out = skip_outs[2]
        self.conv_skip_block_3 = ConvBNReLU(c3 * 2, skip_3_out)  # out=64

        # decoder blocks
        d_out_1 = decoder_outs[0]
        self.convbnrelu0_1 = ConvBNReLU(last_channel * 2, d_out_1)
        d_out_2 = decoder_outs[1]
        self.convbnrelu1_2 = ConvBNReLU(
            d_out_1 + skip_2_out + skip_3_out, d_out_2
        )
        d_out_3 = decoder_outs[2]
        self.convbnrelu2_3 = ConvBNReLU(d_out_2 + skip_1_out, d_out_3)
        self.conv3_4 = nn.Conv2d(
            d_out_3, 2, kernel_size=3, stride=1, padding=1
        )

        self.dropOut = nn.Dropout(p=dropout_prob)
        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def forward(self, x, y):
        _, _, height, width = x.shape
        # extract features
        x_1 = self.features_1(x)
        y_1 = self.features_1(y)
        x_2 = self.features_2(x_1)
        y_2 = self.features_2(y_1)
        x_3 = self.features_3(x_2)
        y_3 = self.features_3(y_2)
        x_4 = self.features_4(x_3)
        y_4 = self.features_4(y_3)
        xy = torch.cat([x_4, y_4], 1)
        xy = self.convbnrelu0_1(xy)  # combined features

        # process skip blocks
        skip_block_1 = torch.cat([x_1, y_1], 1)  # 32,240,320
        skip_block_1 = self.dropOut(self.conv_skip_block_1(skip_block_1))
        skip_block_2 = torch.cat([x_2, y_2], 1)  # 48,120,160
        skip_block_2 = self.dropOut(self.conv_skip_block_2(skip_block_2))
        skip_block_3 = torch.cat([x_3, y_3], 1)  # 32,60,80
        skip_block_3 = self.dropOut(self.conv_skip_block_3(skip_block_3))
        skip_block_3 = nn.functional.interpolate(
            skip_block_3,
            [height // 4, width // 4],
            mode="bilinear",
            align_corners=True,
        )

        # 128,120,160
        xy = nn.functional.interpolate(
            xy, [height // 4, width // 4], mode="bilinear", align_corners=True
        )
        xy = torch.cat([xy, skip_block_2, skip_block_3], 1)
        xy = self.dropOut(self.convbnrelu1_2(xy))

        xy = nn.functional.interpolate(
            xy, [height // 2, width // 2], mode="bilinear", align_corners=True
        )
        xy = torch.cat([xy, skip_block_1], 1)
        xy = self.dropOut(self.convbnrelu2_3(xy))

        xy = nn.functional.interpolate(
            xy, [height, width], mode="bilinear", align_corners=True
        )
        xy = self.conv3_4(xy)
        xy = self.sigmoid(xy)
        return xy

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def auto_mobile_depthnet_1(image_channels=3):
    # best options from network search
    # model_options = [32, 136,
    #                 [[1, 16, 1, 1],
    #                 [2, 64, 1, 2],
    #                 [2, 32, 1, 2],
    #                 [2, 80, 1, 2],
    #                 [6, 96, 3, 1],
    #                 [6, 128, 2, 2],
    #                 [6, 256, 1, 1]],
    #                 [224, 160, 80],
    #                 [128, 128, 64]]
    # 2nd best option from network search
    model_options = [
        16,
        80,
        [
            [1, 16, 1, 1],
            [2, 64, 1, 2],
            [2, 32, 1, 2],
            [2, 80, 1, 2],
            [6, 96, 3, 1],
            [6, 128, 2, 2],
            [6, 256, 1, 1],
        ],
        [224, 160, 80],
        [128, 128, 64],
    ]
    model = AutoMobileDepthNet(*model_options, image_channels=image_channels)
    return model


if __name__ == "__main__":
    from torchsummary import summary

    model = auto_mobile_depthnet_1()

    channels = 1
    summary(model, (channels, 480, 640), device="cpu")
    exit(0)
    from thop import profile

    fake_input = torch.randn(1, 1, 480, 640)
    flops, params = profile(
        model, inputs=(fake_input, fake_input,), verbose=False
    )
    print("flops: {}, params: {}".format(flops, params))
