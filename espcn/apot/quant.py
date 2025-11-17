"""
APoT (Additive Powers-of-Two) Quantization for ESPCN Conv2d layers.
Adapted from the LSTM implementation.
"""

import copy
import math
from itertools import combinations_with_replacement

import torch
import torch.nn as nn
import torch.nn.functional as F


def _choose_num_terms(bit: int, requested: int | None = None) -> int:
    """
    Determine how many additive terms are required so that the APoT grid
    contains at least 2**bit distinct values.
    """
    if requested is not None:
        return max(1, int(requested))

    choices_len = bit + 1  # zero + one power-of-two per "bit"
    target_levels = 2 ** bit
    for k in range(1, bit + 1):
        combos = math.comb(choices_len + k - 1, k)
        if combos >= target_levels:
            return k
    return bit


class Quantizer(nn.Module):
    """
    Additive Powers-of-Two (APoT) quantizer.

    The quantizer keeps a learnable clipping value alpha (optionally per-channel),
    clamps the input to [-alpha, alpha] (or [0, alpha] if unsigned), and projects
    the normalized value onto a non-uniform grid composed of sums of power-of-two
    values. Straight-through estimation (STE) is used so the gradients follow the
    unclipped signal.
    """

    def __init__(
        self,
        bit: int,
        signed: bool = True,
        num_terms: int | None = None,
        init_alpha: float = 2.5,
        per_channel: bool = False,
        channel_dim: int = 0,
    ) -> None:
        super().__init__()
        if bit < 2:
            raise ValueError("APoT quantizer expects bit width >= 2.")
        self.bit = int(bit)
        self.signed = bool(signed)
        self.per_channel = bool(per_channel)
        self.channel_dim = int(channel_dim)

        self.num_terms = _choose_num_terms(self.bit, num_terms)
        # Largest contribution we allow is 2^{-shift_start}; ensure k * contribution <= 1.
        self.shift_start = 0 if self.num_terms <= 1 else math.ceil(math.log2(self.num_terms))
        self.scale_denominator = 2 ** (self.shift_start + self.bit - 1)

        # Build base choices (0 plus `bit` different power-of-two magnitudes).
        float_choices = [0.0]
        int_choices: list[int] = [0]
        for idx in range(1, self.bit + 1):
            shift = self.shift_start + idx - 1
            float_choices.append(2.0 ** (-shift))
            int_choices.append(1 << (self.bit - idx))
        self.register_buffer("choice_values", torch.tensor(float_choices, dtype=torch.float32))
        self._choice_int = int_choices

        positive_codebook, combo_table = self._build_positive_codebook()
        self.register_buffer("positive_codebook", positive_codebook)
        self.register_buffer("combo_table", combo_table)

        # Alpha initialisation (PACT-style lazy init for per-channel usage).
        self.per_channel = per_channel
        self._init_alpha_value = float(init_alpha)
        if per_channel:
            self.alpha = None
            self._alpha_initialized = False
        else:
            self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
            self._alpha_initialized = True

        self.register_buffer("eps", torch.tensor(1e-6, dtype=torch.float32))

        if self.signed:
            self.int_qmax = (2 ** (self.bit - 1)) - 1
            self.int_qmin = -(2 ** (self.bit - 1))
        else:
            self.int_qmax = (2 ** self.bit) - 1
            self.int_qmin = 0

    def _build_positive_codebook(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Enumerate all distinct sums of `num_terms` base choices (with replacement).
        Returns:
            values: ascending tensor of non-negative magnitudes
            combos: tensor of shape (len(values), num_terms) storing the indices
                    of the base choices that formed each value.
        """
        combos_iter = combinations_with_replacement(range(len(self.choice_values)), self.num_terms)
        seen: dict[int, tuple[int, ...]] = {}
        for combo in combos_iter:
            total_int = sum(self._choice_int[idx] for idx in combo)
            if total_int not in seen:
                seen[total_int] = combo

        sorted_items = sorted(seen.items())
        levels = torch.tensor(
            [total / self.scale_denominator for total, _ in sorted_items],
            dtype=torch.float32,
        )
        combos = torch.tensor(
            [combo for _, combo in sorted_items],
            dtype=torch.long,
        )
        return levels, combos

    def _lazy_init_alpha(self, x: torch.Tensor) -> None:
        if not self.per_channel or self._alpha_initialized:
            return

        num_channels = x.size(self.channel_dim)
        alpha_shape = [1] * x.dim()
        alpha_shape[self.channel_dim] = num_channels
        alpha_init = torch.full(
            alpha_shape,
            self._init_alpha_value,
            dtype=x.dtype,
            device=x.device,
        )
        self.alpha = nn.Parameter(alpha_init)
        self._alpha_initialized = True

    def _positive_bucketize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Find the nearest codebook entry index for every magnitude in `values`.
        """
        codebook = self.positive_codebook.to(values.device, values.dtype)
        flat = values.reshape(-1)
        idx = torch.bucketize(flat, codebook)
        idx_hi = idx.clamp(max=codebook.numel() - 1)
        idx_lo = torch.clamp(idx - 1, min=0)
        hi_vals = codebook[idx_hi]
        lo_vals = codebook[idx_lo]
        choose_hi = (flat - lo_vals) > (hi_vals - flat)
        nearest = torch.where(choose_hi, idx_hi, idx_lo)
        return nearest.view(values.shape)

    def forward(
        self,
        x: torch.Tensor,
        return_indices: bool = False,
    ):
        """
        Args:
            x: input tensor
            return_indices: if True, also returns (indices, sign) tensors

        Returns:
            q: fake-quantized tensor (normalized to the APoT grid)
            scale: tensor containing the learned alpha(s)
            (optional) indices: indices into `positive_codebook`
            (optional) sign: tensor with {-1, 0, 1} values (only when signed)
        """
        self._lazy_init_alpha(x)
        if self.alpha is None:
            raise RuntimeError("Alpha parameters are not initialized.")

        alpha = F.softplus(self.alpha).to(x.device, x.dtype)
        scale = torch.clamp(alpha, min=self.eps.to(alpha.dtype))

        if self.signed:
            x_clipped = torch.clamp(x, min=-scale, max=scale)
        else:
            zero = torch.zeros_like(scale)
            x_clipped = torch.clamp(x, min=zero, max=scale)

        norm = x_clipped / scale
        if not self.signed:
            norm = torch.clamp(norm, min=0.0)

        if self.signed:
            sign = torch.sign(norm)
            magnitude = norm.abs()
        else:
            sign = torch.ones_like(norm)
            magnitude = norm

        pos_indices = self._positive_bucketize(magnitude)
        pos_codebook = self.positive_codebook.to(x.device, x.dtype)
        mag_quant = torch.take(pos_codebook, pos_indices.view(-1)).view_as(magnitude)
        norm_quant = sign * mag_quant

        # Straight-through estimator
        norm_out = norm + (norm_quant - norm).detach()

        if return_indices:
            sign_tensor = torch.sign(x_clipped.detach()) if self.signed else torch.ones_like(x_clipped.detach())
            return norm_out, scale.detach(), pos_indices.detach(), sign_tensor
        return norm_out, scale.detach()

    def quantize_to_int(
        self,
        x: torch.Tensor,
        dtype: torch.dtype = torch.int8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns integer tensors suitable for integer GEMM.
        """
        norm_out, scale = self.forward(x)
        qmax = float(self.int_qmax)
        qmin = float(self.int_qmin)
        q = torch.round(norm_out * qmax).clamp_(qmin, qmax).detach()
        return q.to(dtype), (scale / qmax).detach()


class QuantAct(nn.Module):
    """Activation fake-quant using APoT quantizer."""
    def __init__(self, bit: int, signed: bool = True):
        super().__init__()
        self.quant = Quantizer(bit=bit, signed=signed, per_channel=False, init_alpha=6.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, scale = self.quant(x)
        return q * scale


class QAConv2d(nn.Module):
    """Conv2d with APoT fake-quant on activations (pre) and weights."""
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
        num_terms: int | None = None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        self.quantize_activation = quantize_activation
        
        if quantize_activation:
            self.quant_a = Quantizer(
                bit=bit,
                signed=True,  # For Tanh activations
                num_terms=num_terms,
                init_alpha=6.0,
                per_channel=False,
            )
        
        self.quant_w = Quantizer(
            bit=bit,
            signed=True,
            num_terms=num_terms,
            init_alpha=2.0,
            per_channel=False,  # Can be True for per-channel quantization
            channel_dim=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantize_activation:
            act_q, act_scale = self.quant_a(x)
            x = act_q * act_scale
        
        weight_q, weight_scale = self.quant_w(self.conv.weight)
        w = weight_q * weight_scale
        
        return F.conv2d(x, w, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)


class Conv2dInt(nn.Module):
    """
    Integer Conv2d backend for APoT QAT models.
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
        act_quantizer: Quantizer = None,
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
        Convert a QAConv2d (APoT QAT) module to Conv2dInt for real INT8 inference.
        """
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

