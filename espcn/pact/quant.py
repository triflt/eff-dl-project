import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricUniformWeightQuantizer(nn.Module):
    """Symmetric uniform fake-quant for weights (canonical baseline with PACT for activations)."""
    def __init__(self, n_bits: int = 8, signed: bool = True):
        super().__init__()
        assert n_bits >= 2
        self.n_bits = n_bits
        self.signed = signed
        if signed:
            self.qmax = (2 ** (n_bits - 1)) - 1
            self.qmin = -(2 ** (n_bits - 1))
        else:
            self.qmax = (2 ** n_bits) - 1
            self.qmin = 0

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        max_abs = w.detach().abs().amax()
        scale = max_abs / float(self.qmax)
        eps = torch.finfo(w.dtype).eps
        scale = torch.clamp(scale, min=eps).to(dtype=w.dtype, device=w.device)
        y = torch.clamp(w / scale, self.qmin, self.qmax).round() * scale
        return y


class PACTQuantizer(nn.Module):
    """PACT quantizer that learns clipping alpha. Returns dequantized fake-quant output."""
    def __init__(self, n_bits: int = 8, signed: bool = False):
        super().__init__()
        assert n_bits >= 2
        self.n_bits = n_bits
        self.signed = signed
        if signed:
            self.qmax = (2 ** (n_bits - 1)) - 1
            self.qmin = -(2 ** (n_bits - 1))
        else:
            self.qmax = (2 ** n_bits) - 1
            self.qmin = 0
        # single learnable alpha (per-tensor)
        self.alpha = nn.Parameter(torch.tensor(6.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # constrain alpha > 0
        alpha = torch.nn.functional.softplus(self.alpha).to(dtype=x.dtype, device=x.device)
        if self.signed:
            x = torch.clamp(x, min=-alpha, max=alpha)
        else:
            zero = torch.zeros((), dtype=x.dtype, device=x.device)
            x = torch.clamp(x, min=zero, max=alpha)
        scale = alpha / self.qmax
        eps = torch.finfo(x.dtype).eps
        scale = torch.clamp(scale, min=eps)
        # fake-quant dequantized
        y = torch.round(x / scale) * scale
        return y


class QuantAct(nn.Module):
    """Activation fake-quant using PACT (unsigned by default)."""
    def __init__(self, bit: int):
        super().__init__()
        # For ESPCN with Tanh activations, use signed activations
        self.quant = PACTQuantizer(bit, signed=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant(x)


class QAConv2d(nn.Module):
    """Conv2d with PACT fake-quant on activations (pre) and on weights (signed)."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size, stride=1, padding=0, bias=True, bit: int = 8, quantize_activation: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        # Canonical PACT: activations via PACT, weights via symmetric uniform
        self.quant_w = SymmetricUniformWeightQuantizer(bit, signed=True)
        self.quantize_activation = quantize_activation
        if quantize_activation:
            self.quant_a = PACTQuantizer(bit, signed=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantize_activation:
            x = self.quant_a(x)
        w = self.quant_w(self.conv.weight)
        return F.conv2d(x, w, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)

