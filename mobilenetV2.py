import json
import pickle
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from lib.nn import SynchronizedBatchNorm2d


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
    # an example
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
