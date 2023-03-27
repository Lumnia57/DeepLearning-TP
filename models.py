#!/usr/bin/env python3
"""
    This script belongs to the lab work on semantic segmenation of the
    deep learning lectures https://github.com/jeremyfix/deeplearning-lectures
    Copyright (C) 2022 Jeremy Fix

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# Standard imports
import sys
import logging

# External imports
import torch
import torch.nn as nn
import torchvision.models
import deepcs.display

available_models = ["fcn_resnet50", "UNet"]


class TorchvisionModel(nn.Module):
    def __init__(self, modelname, input_size, num_classes, pretrained, in_channels):
        super().__init__()
        if pretrained:
            logging.info("Loading a pretrained model")
        else:
            logging.info("Loading a model with random init")
        exec(
            f"self.model = torchvision.models.segmentation.{modelname}(pretrained={pretrained}, pretrained_backbone={pretrained})"
        )
        old_conv1 = self.model.backbone.conv1
        self.model.backbone.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.stride,
        )
        old_head = self.model.classifier
        last_conv = list(old_head.children())[-1]
        self.model.classifier = nn.Sequential(
            *list(old_head.children())[:-1],
            nn.Conv2d(
                in_channels=last_conv.in_channels,
                out_channels=num_classes,
                kernel_size=last_conv.kernel_size,
                stride=last_conv.stride,
                padding=last_conv.stride,
            ),
        )

    def forward(self, x):
        return self.model(x)["out"]


def fcn_resnet50(input_size, num_classes):
    return TorchvisionModel("fcn_resnet50", input_size, num_classes, True, 3)


class UNetConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        # vvvvvvvvv
        # CODE HERE
        kernel_size = 3
        self.block1 = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride=1, padding=(kernel_size - 1) // 2), 
            nn.ReLU(), 
            nn.BatchNorm2d(cout)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(cout, cout, kernel_size, stride=1, padding=(kernel_size - 1) // 2), 
            nn.ReLU(), 
            nn.BatchNorm2d(cout)
        )
        self.block3 = nn.MaxPool2d(2, stride=2)
        # ^^^^^^^^^

    def forward(self, inputs):
        # vvvvvvvvv
        # CODE HERE
        #features = sortie de block2
        #outputs = sortie de block3
        features = self.block2.forward(self.block1.forward(inputs))
        outputs = self.block3.forward(features)
        # ^^^^^^^^^
        return outputs, features


class UNetEncoder(nn.Module):
    def __init__(self, cin, num_blocks, base_chan=64):
        super().__init__()
        # Note: use ModuleList to correctly register
        #       the modules it contains rather than plain list
        #  e.g. with plain list, the model.parameters() do not
        #       return the internal parameters of the modules contained
        #       in the list
        self.blocks = nn.ModuleList()
        self.cout = base_chan
        for i in range(num_blocks):
            self.blocks.append(UNetConvBlock(cin, self.cout))
            # Prepare the parameters for the next layer
            cin = self.cout
            self.cout *= 2

        # Add the last encoding layer
        # which outputs 32 * 2*num_blocks channels
        # vvvvvvvvv
        kernel_size = 3
        # CODE HERE
        self.last_block = nn.Sequential(
            nn.Conv2d(cin, self.cout, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.ReLU(), 
            nn.BatchNorm2d(self.cout)
        )
        # ^^^^^^^^^

    def forward(self, inputs):
        # While iterating through the stages of the encoder
        # we keep a pointer to the outputs of "block2"
        # which will be latter used by the decoder
        prev_outputs, lfeatures = inputs, []
        for b in self.blocks:
            outb, featb = b(prev_outputs)
            # Keep track of the encoder features
            # to be given to the decoder pathway
            lfeatures.append(featb)
            # Prepare the input for the next block
            prev_outputs = outb
        outputs = self.last_block(prev_outputs)
        # Here :
        # outputs is the output tensor of the last encoding layer
        # lfeatures is the output features of the num_blocks blocks
        return outputs, lfeatures


class UNetUpConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        # vvvvvvvvv
        # CODE HERE
        kernel_size = 3
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(cin, cout, kernel_size, stride=1, padding=(kernel_size - 1) // 2), 
            nn.ReLU(),
            nn.BatchNorm2d(cout)
        )
        self.convblock = nn.Sequential(
            nn.Conv2d(2 * cout, cout, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.BatchNorm2d(cout), 
            nn.Conv2d(cout, cout, kernel_size, stride=1, padding=(kernel_size - 1) // 2), 
            nn.ReLU(), 
            nn.BatchNorm2d(cout)
        )
        # ^^^^^^^^^

    def forward(self, inputs, encoder_features):
        # vvvvvvvvv
        # CODE HERE
        upconv_features = self.upconv(inputs)
        concat_features = torch.cat((upconv_features, encoder_features), dim=1)
        outputs = self.convblock(concat_features)
        # ^^^^^^^^^
        return outputs


class UNetDecoder(nn.Module):
    def __init__(self, cin, num_blocks, num_classes):
        super().__init__()
        # vvvvvvvvv
        # CODE HERE
        kernel_size = 3
        self.first_block = nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size, stride=1, padding=(kernel_size - 1) // 2), 
            nn.ReLU(), 
            nn.BatchNorm2d(cin)
        )
        # ^^^^^^^^^

        # Note: use ModuleList to correctly register
        #       the modules it contains rather than plain list
        #  e.g. with plain list, the model.parameters() do not
        #       return the internal parameters of the modules contained
        #       in the list
        self.blocks = nn.ModuleList()
        cout = cin // 2
        for i in range(num_blocks):
            self.blocks.append(UNetUpConvBlock(cin, cout))
            # Prepare the parameters for the next layer
            cin = cout
            cout = cout // 2

        # Add the last encoding layer
        # vvvvvvvvv
        # CODE HERE
        self.last_conv = nn.Conv2d(cin, num_classes, 1, stride = 1, padding=0)
        # ^^^^^^^^^

    def forward(self, encoder_outputs, encoder_features):
        outputs = self.first_block(encoder_outputs)
        for b, enc_features in zip(self.blocks, encoder_features[::-1]):
            outputs = b(outputs, enc_features)
        outputs = self.last_conv(outputs)
        return outputs


class UNet(nn.Module):
    def __init__(self, num_classes, num_blocks=4, cin=3):
        super().__init__()
        self.encoder = UNetEncoder(cin, num_blocks)
        encoder_cout = self.encoder.cout
        self.decoder = UNetDecoder(encoder_cout, num_blocks, num_classes)

    def forward(self, inputs):
        # inputs is B, cin, H, W
        encoder_outputs, encoder_features = self.encoder(inputs)
        # encoder outputs is B, 32*(2**num_blocks), H/2^num_blocks, W/2^num_blocks
        # encoder_features is a list of num_blocks tensors
        outputs = self.decoder(encoder_outputs, encoder_features)
        # outputs is B, num_classes, H, W
        return outputs


def build_model(model_name, num_classes):
    if model_name not in available_models:
        raise RuntimeError(f"Unavailable model {model_name}")
    exec(f"m = {model_name}(num_classes)")
    return locals()["m"]


def main():
    m = build_model("UNet", 14)
    # vvvvvvvvv
    # CODE HERE
    dummy = torch.zeros(64, 3, 128, 128)
    print(f'Dimension of the input : {dummy.shape}')
    res = m.forward(dummy)
    print(f'Dimension of the output : {res.shape}')
    # ^^^^^^^^^


if __name__ == "__main__":
    main()