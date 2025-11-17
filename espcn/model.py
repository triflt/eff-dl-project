# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math

import torch
from torch import nn, Tensor
import config

__all__ = [
    "ESPCN",
    "espcn_x2", "espcn_x3", "espcn_x4", "espcn_x8",
]


class ESPCN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int,
            upscale_factor: int,
    ) -> None:
        super(ESPCN, self).__init__()
        hidden_channels = channels // 2
        out_channels = int(out_channels * (upscale_factor ** 2))

        # Feature mapping
        feature_layers = [
            nn.Conv2d(in_channels, channels, (5, 5), (1, 1), (2, 2)),
            nn.Tanh(),
            nn.Conv2d(channels, hidden_channels, (3, 3), (1, 1), (1, 1)),
            nn.Tanh(),
        ]

        # Sub-pixel convolution layer
        subpixel_layers = [
            nn.Conv2d(hidden_channels, out_channels, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
        ]

        if getattr(config, "qat_enabled", False):
            method = getattr(config, "qat_method", "lsq")
            bits = getattr(config, "qat_bits", 8)
            quant_act = getattr(config, "qat_quantize_activations", True)
            if method == "lsq":
                from lsq.quant import QAConv2d as QAConv2d_LSQ
                from lsq.quant import QuantAct as QuantAct_LSQ
                feature_layers = _quantize_layers(feature_layers, QAConv2d_LSQ, QuantAct_LSQ, bits, quant_act)
                subpixel_layers = _quantize_layers(subpixel_layers, QAConv2d_LSQ, QuantAct_LSQ, bits, quant_act, insert_act_before_pixelshuffle=True)
            elif method == "pact":
                from pact.quant import QAConv2d as QAConv2d_PACT
                from pact.quant import QuantAct as QuantAct_PACT
                feature_layers = _quantize_layers(feature_layers, QAConv2d_PACT, QuantAct_PACT, bits, quant_act)
                subpixel_layers = _quantize_layers(subpixel_layers, QAConv2d_PACT, QuantAct_PACT, bits, quant_act, insert_act_before_pixelshuffle=True)
            elif method == "apot":
                from apot.quant import QAConv2d as QAConv2d_APoT
                from apot.quant import QuantAct as QuantAct_APoT
                feature_layers = _quantize_layers(feature_layers, QAConv2d_APoT, QuantAct_APoT, bits, quant_act)
                subpixel_layers = _quantize_layers(subpixel_layers, QAConv2d_APoT, QuantAct_APoT, bits, quant_act, insert_act_before_pixelshuffle=True)
            elif method == "efficientqat":
                from efficientqat.quant import QAConv2d as QAConv2d_EfficientQAT
                from efficientqat.quant import QuantAct as QuantAct_EfficientQAT
                feature_layers = _quantize_layers(feature_layers, QAConv2d_EfficientQAT, QuantAct_EfficientQAT, bits, quant_act)
                subpixel_layers = _quantize_layers(subpixel_layers, QAConv2d_EfficientQAT, QuantAct_EfficientQAT, bits, quant_act, insert_act_before_pixelshuffle=True)
            else:
                raise ValueError(f"Unsupported QAT method: {method}")

        self.feature_maps = nn.Sequential(*feature_layers)
        self.sub_pixel = nn.Sequential(*subpixel_layers)

        # Initial model weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.in_channels == 32:
                    nn.init.normal_(module.weight.data,
                                    0.0,
                                    0.001)
                    nn.init.zeros_(module.bias.data)
                else:
                    nn.init.normal_(module.weight.data,
                                    0.0,
                                    math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                    nn.init.zeros_(module.bias.data)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.feature_maps(x)
        x = self.sub_pixel(x)

        x = torch.clamp_(x, 0.0, 1.0)

        return x


def _quantize_layers(layers, QAConv2dCls, QuantActCls, bits: int, quantize_activation: bool, insert_act_before_pixelshuffle: bool = False):
    """Replace Conv2d with QAConv2d and insert activation quantization after Tanh (and before PixelShuffle if requested)."""
    new_layers = []
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Conv2d):
            qconv = QAConv2dCls(layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride, layer.padding, layer.bias is not None, bit=bits, quantize_activation=False)
            # copy weights
            with torch.no_grad():
                qconv.conv.weight.copy_(layer.weight)
                if layer.bias is not None:
                    qconv.conv.bias.copy_(layer.bias)
            new_layers.append(qconv)
        elif isinstance(layer, nn.Tanh):
            new_layers.append(layer)
            if quantize_activation:
                new_layers.append(QuantActCls(bits))
        elif isinstance(layer, nn.PixelShuffle):
            if insert_act_before_pixelshuffle and quantize_activation:
                new_layers.append(QuantActCls(bits))
            new_layers.append(layer)
        else:
            new_layers.append(layer)
    return new_layers

def espcn_x2(**kwargs) -> ESPCN:
    model = ESPCN(upscale_factor=2, **kwargs)

    return model


def espcn_x3(**kwargs) -> ESPCN:
    model = ESPCN(upscale_factor=3, **kwargs)

    return model


def espcn_x4(**kwargs) -> ESPCN:
    model = ESPCN(upscale_factor=4, **kwargs)

    return model


def espcn_x8(**kwargs) -> ESPCN:
    model = ESPCN(upscale_factor=8, **kwargs)

    return model