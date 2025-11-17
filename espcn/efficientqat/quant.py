"""
EfficientQAT Quantization for ESPCN Conv2d layers.
Adapted from the LSTM implementation.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_MIN = 1e-4


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator for round()."""
    return (x.round() - x).detach() + x


def clamp_ste(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """Straight-through estimator for clamp()."""
    return (x.clamp(min_val, max_val) - x).detach() + x


class WeightQuantizer(nn.Module):
    """
    Simplified LSQ-style weight quantizer for EfficientQAT Conv2d.
    Per-tensor symmetric quantization.
    """

    def __init__(self, bit: int = 8) -> None:
        super().__init__()
        self.bit = bit
        self.qmin = -(2 ** (bit - 1))
        self.qmax = (2 ** (bit - 1)) - 1
        self.s = nn.Parameter(torch.ones(1))
        self._initialized = False
        
        self.thd_pos = self.qmax
        self.thd_neg = self.qmin

    def init_from(self, x: torch.Tensor) -> None:
        if self._initialized:
            return
        with torch.no_grad():
            max_val = x.detach().abs().mean() * 2.0
            max_val = max_val.clamp(min=CLIP_MIN)
            self.s.copy_(max_val / self.qmax)
        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bit >= 32:
            return x
        if not self._initialized:
            self.init_from(x)
        
        # LSQ gradient scaling
        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = (self.s * s_grad_scale - self.s.detach() * s_grad_scale).detach() + self.s
        s_scale = s_scale.to(x.device, x.dtype)
        
        x_scaled = x / s_scale
        x_scaled = torch.clamp(x_scaled, self.thd_neg, self.thd_pos)
        x_int = round_ste(x_scaled)
        x_dequant = x_int * s_scale
        
        return x_dequant

    @torch.no_grad()
    def quantize_to_int(
        self,
        x: torch.Tensor,
        dtype: torch.dtype = torch.int8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns integer tensors suitable for integer inference."""
        if not self._initialized:
            self.init_from(x)
        
        scale = torch.clamp(self.s.detach(), min=CLIP_MIN).to(x.device)
        x_scaled = x / scale
        x_scaled = torch.clamp(x_scaled, self.thd_neg, self.thd_pos)
        x_int = torch.round(x_scaled).to(dtype)
        return x_int, scale


class ActivationQuantizer(nn.Module):
    """LSQ-style activation quantizer for EfficientQAT (adapted for Conv2d)."""

    def __init__(self, bit: int, signed: bool = False) -> None:
        super().__init__()
        if bit < 2:
            raise ValueError("Activation quantizer expects >=2 bits.")
        self.bit = bit
        self.signed = signed
        if signed:
            self.qmin = -(2 ** (bit - 1))
            self.qmax = (2 ** (bit - 1)) - 1
        else:
            self.qmin = 0
            self.qmax = (2**bit) - 1

        self.s = nn.Parameter(torch.ones(1))
        self.initialized = False

        self.thd_pos = self.qmax
        self.thd_neg = self.qmin

    def _init_from(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            max_val = x.detach().abs().mean() * 2.0
            max_val = max_val.clamp(min=CLIP_MIN)
            self.s.copy_(max_val / max(abs(self.qmax), 1))
        self.initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            self._init_from(x)

        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = (self.s * s_grad_scale - self.s.detach() * s_grad_scale).detach() + self.s
        s_scale = torch.clamp(s_scale, min=CLIP_MIN).to(x.device, x.dtype)

        x_scaled = x / s_scale
        x_scaled = torch.clamp(x_scaled, self.thd_neg, self.thd_pos)
        q = round_ste(x_scaled)
        y = q * s_scale
        return y

    @torch.no_grad()
    def quantize_to_int(
        self,
        x: torch.Tensor,
        dtype: torch.dtype = torch.int8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized:
            self._init_from(x)

        scale = torch.clamp(self.s.detach(), min=CLIP_MIN).to(x.device)
        x_scaled = x / scale
        x_scaled = torch.clamp(x_scaled, self.thd_neg, self.thd_pos)
        x_int = torch.round(x_scaled).to(dtype)
        return x_int, scale


class QuantAct(nn.Module):
    """Activation fake-quant using EfficientQAT quantizer."""
    def __init__(self, bit: int, signed: bool = True):
        super().__init__()
        self.quant = ActivationQuantizer(bit, signed=signed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant(x)


class QAConv2d(nn.Module):
    """Conv2d with EfficientQAT fake-quant on activations and weights."""
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        bit: int = 8,
        quantize_activation: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        self.quantize_activation = quantize_activation
        
        if quantize_activation:
            self.quant_a = ActivationQuantizer(bit, signed=True)  # For Tanh
        
        self.quant_w = WeightQuantizer(bit=bit)
        # Initialize weight quantizer
        self.quant_w.init_from(self.conv.weight.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantize_activation:
            x = self.quant_a(x)
        
        w = self.quant_w(self.conv.weight)
        
        return F.conv2d(x, w, self.conv.bias, self.conv.stride, 
                       self.conv.padding, self.conv.dilation, self.conv.groups)


class Conv2dInt(nn.Module):
    """Integer Conv2d backend for EfficientQAT QAT models."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        w_scale: torch.Tensor = None,
        act_quantizer: ActivationQuantizer = None,
        device: str = "cpu",
    ):
        super().__init__()
        if device != "cpu":
            raise ValueError("Conv2dInt currently supports CPU execution only.")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        
        # Store weight scale
        self.register_buffer("w_scale", w_scale.float() if w_scale is not None else torch.tensor(1.0))
        
        # Weight and bias as int8
        self.register_buffer("weight_int", torch.zeros(out_channels, in_channels // groups, *self.kernel_size, dtype=torch.int8))
        self.register_buffer("bias", torch.zeros(out_channels, dtype=torch.float32))
        
        # Activation quantizer
        self.act_quantizer = copy.deepcopy(act_quantizer) if act_quantizer is not None else None
        if self.act_quantizer is not None:
            self.act_quantizer.eval()
            for p in self.act_quantizer.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cpu":
            x = x.cpu()
        
        # Quantize activation
        if self.act_quantizer is not None:
            x_int, act_scale = self.act_quantizer.quantize_to_int(x, dtype=torch.int8)
        else:
            # No quantizer - just pass through without quantization
            act_scale = torch.tensor(1.0, device=x.device)
            x_int = x  # Don't quantize if no quantizer!
        
        # INT8 convolution with proper scale handling
        weight_fp32 = self.weight_int.float() * self.w_scale.view(-1, 1, 1, 1)
        if self.act_quantizer is not None:
            x_fp32 = x_int.float() * act_scale.view(1, 1, 1, 1)
        else:
            x_fp32 = x_int  # Already FP32
        
        out = F.conv2d(
            x_fp32,
            weight_fp32,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return out

    @classmethod
    def from_qat(cls, qat_conv: QAConv2d, int_dtype: torch.dtype = torch.int8) -> "Conv2dInt":
        """Convert QAConv2d (EfficientQAT) to Conv2dInt for real INT8 inference."""
        if not isinstance(qat_conv, QAConv2d):
            raise TypeError("Expected QAConv2d module.")
        
        # Extract quantized weights
        with torch.no_grad():
            weight_int, weight_scale = qat_conv.quant_w.quantize_to_int(
                qat_conv.conv.weight.data,
                dtype=int_dtype,
            )
        
        # Create Conv2dInt instance with copied quantizer
        act_quantizer = copy.deepcopy(qat_conv.quant_a) if qat_conv.quantize_activation else None
        
        conv_int = cls(
            in_channels=qat_conv.conv.in_channels,
            out_channels=qat_conv.conv.out_channels,
            kernel_size=qat_conv.conv.kernel_size,
            stride=qat_conv.conv.stride,
            padding=qat_conv.conv.padding,
            dilation=qat_conv.conv.dilation,
            groups=qat_conv.conv.groups,
            w_scale=weight_scale,
            act_quantizer=act_quantizer,
            device="cpu",
        )
        
        # Copy quantized weights and bias
        conv_int.weight_int.copy_(weight_int)
        if qat_conv.conv.bias is not None:
            conv_int.bias.copy_(qat_conv.conv.bias.detach())
        
        return conv_int

