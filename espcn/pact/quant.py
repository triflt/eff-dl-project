import copy
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
    
    @torch.no_grad()
    def quantize_to_int(self, w: torch.Tensor, dtype: torch.dtype = torch.int8):
        """Directly quantize tensor to integer."""
        max_abs = w.detach().abs().amax()
        scale = max_abs / float(self.qmax)
        eps = torch.finfo(w.dtype).eps
        scale = torch.clamp(scale, min=eps).to(dtype=w.dtype, device=w.device)
        w_int = torch.clamp(w / scale, self.qmin, self.qmax).round().to(dtype)
        return w_int, scale


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
    
    @torch.no_grad()
    def quantize_to_int(self, x: torch.Tensor, dtype: torch.dtype = torch.int8):
        """Directly quantize tensor to integer."""
        alpha = torch.nn.functional.softplus(self.alpha).to(dtype=x.dtype, device=x.device)
        if self.signed:
            x_clipped = torch.clamp(x, min=-alpha, max=alpha)
        else:
            zero = torch.zeros((), dtype=x.dtype, device=x.device)
            x_clipped = torch.clamp(x, min=zero, max=alpha)
        scale = alpha / self.qmax
        eps = torch.finfo(x.dtype).eps
        scale = torch.clamp(scale, min=eps)
        x_int = torch.round(x_clipped / scale).to(dtype)
        return x_int, scale


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


class Conv2dInt(nn.Module):
    """
    Integer Conv2d backend for PACT QAT models.
    Uses INT8 quantization for efficient inference on CPU.
    """
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
        act_quantizer: PACTQuantizer = None,
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
        
        # Activation quantizer (PACT)
        self.act_quantizer = act_quantizer
        if self.act_quantizer is not None:
            self.act_quantizer.eval()
            for p in self.act_quantizer.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cpu":
            x = x.cpu()
        
        # Quantize activation using direct quantization
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
        """
        Convert a QAConv2d (PACT QAT) module to Conv2dInt for real INT8 inference.
        """
        if not isinstance(qat_conv, QAConv2d):
            raise TypeError("Expected QAConv2d module.")
        
        # Extract quantized weights using quantize_to_int
        with torch.no_grad():
            w_int, w_scale = qat_conv.quant_w.quantize_to_int(
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
            w_scale=w_scale,
            act_quantizer=act_quantizer,
            device="cpu",
        )
        
        # Copy quantized weights and bias
        conv_int.weight_int.copy_(w_int)
        if qat_conv.conv.bias is not None:
            conv_int.bias.copy_(qat_conv.conv.bias.detach())
        
        return conv_int

