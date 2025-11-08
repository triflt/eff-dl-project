import torch
import torch.nn as nn
import torch.nn.functional as F


class LSQQuantizer(nn.Module):
    """Learned Step Size Quantizer (per-tensor), dequantized fake-quant output."""
    def __init__(self, bit: int):
        super().__init__()
        self.bit = bit
        self.thd_neg = -(2 ** (bit - 1))
        self.thd_pos = 2 ** (bit - 1) - 1
        self.s = nn.Parameter(torch.ones(1))
        self._initialized = False

    @torch.no_grad()
    def init_from(self, x: torch.Tensor) -> None:
        """Initialize scale from tensor statistics as in LSQ: s = 2 * E|x| / sqrt(Qp)."""
        qpos = float(self.thd_pos)
        s_init = 2.0 * x.detach().to(torch.float32).abs().mean() / (qpos ** 0.5 + 1e-12)
        # Avoid zero/NaN (use small constant to be FX-trace friendly)
        s_init = torch.clamp(s_init, min=1e-8)
        self.s.copy_(s_init)
        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bit >= 32:
            return x
        if not self._initialized:
            self.init_from(x)
        # scale grad as in LSQ
        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = (self.s * s_grad_scale - self.s.detach() * s_grad_scale).detach() + self.s
        s_scale = s_scale.to(x.device, x.dtype)
        x = x / s_scale
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        # STE round
        y = torch.round(x)
        y = (y - x).detach() + x
        # dequantize
        y = y * s_scale
        return y


class QuantAct(nn.Module):
    """Activation fake-quant using LSQ quantizer (unsigned if specified)."""
    def __init__(self, bit: int):
        super().__init__()
        self.quant = LSQQuantizer(bit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant(x)


class QAConv2d(nn.Module):
    """Conv2d with LSQ fake-quant on activations (pre) and weights."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size, stride=1, padding=0, bias=True, bit: int = 8, quantize_activation: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        self.quant_w = LSQQuantizer(bit)
        self.quantize_activation = quantize_activation
        if quantize_activation:
            self.quant_a = LSQQuantizer(bit)
        # Initialize weight scale from weights
        self.quant_w.init_from(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantize_activation:
            x = self.quant_a(x)
        w = self.quant_w(self.conv.weight)
        return F.conv2d(x, w, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)

