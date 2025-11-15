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


class Quantizer(nn.Module):
    """
    EfficientQAT-style uniform affine (per-group) quantizer.

    The implementation follows the reference repo:
    https://github.com/OpenGVLab/EfficientQAT (quantize/quantizer.py)
    but exposes a slightly higher-level API that matches the other
    quantization modules in this project.
    """

    def __init__(
        self,
        n_bits: int = 4,
        group_size: int | None = None,
        init_tensor: torch.Tensor | None = None,
        enable: bool = True,
        signed: bool = True,
    ) -> None:
        super().__init__()
        if not (2 <= n_bits <= 16):
            raise ValueError("Quantizer supports between 2 and 16 bits.")
        self.n_bits = int(n_bits)
        self.group_size = group_size
        self.enable = enable
        self.signed = bool(signed)
        if self.signed:
            self.qmin = -(2 ** (self.n_bits - 1))
            self.qmax = (2 ** (self.n_bits - 1)) - 1
        else:
            self.qmin = 0
            self.qmax = (1 << self.n_bits) - 1

        self.scale: nn.Parameter | None = None
        self.zero_point: nn.Parameter | None = None
        self._effective_group_size: int | None = None

        if init_tensor is not None:
            self._init_from_tensor(init_tensor.detach())

    def _init_from_tensor(self, tensor: torch.Tensor) -> None:
        last_dim = tensor.shape[-1]
        group_size = self.group_size if self.group_size not in (None, -1) else last_dim
        if last_dim % group_size != 0:
            raise ValueError(f"last dim ({last_dim}) must be divisible by group_size ({group_size}).")
        self._effective_group_size = group_size

        reshaped = tensor.reshape(-1, group_size)
        if self.signed:
            max_abs = reshaped.abs().amax(dim=1, keepdim=True)
            scale = max_abs / max(self.qmax, 1)
            scale = torch.clamp(scale, min=CLIP_MIN, max=1e4)
            zero_point = torch.zeros_like(scale)
        else:
            xmin = reshaped.amin(dim=1, keepdim=True)
            xmax = reshaped.amax(dim=1, keepdim=True)
            rng = xmax - xmin
            rng = torch.clamp(rng, min=CLIP_MIN)

            scale = rng / (self.qmax - self.qmin)
            scale = torch.clamp(scale, min=CLIP_MIN, max=1e4)
            zero_point = -(xmin / scale)
            zero_point = torch.clamp(zero_point, min=-1e4, max=1e4)

        self.scale = nn.Parameter(scale)
        self.zero_point = nn.Parameter(zero_point, requires_grad=not self.signed)

    def _check_ready(self, tensor: torch.Tensor) -> None:
        if self.scale is None or self.zero_point is None:
            self._init_from_tensor(tensor.detach())
        if self._effective_group_size is None:
            raise RuntimeError("Quantizer has not been initialised properly.")

    def _reshape(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
        orig_shape = tensor.shape
        reshaped = tensor.reshape(-1, self._effective_group_size)
        return reshaped, orig_shape

    def forward(
        self,
        tensor: torch.Tensor,
        return_params: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.enable:
            if return_params:
                return tensor, None, None
            return tensor

        self._check_ready(tensor)
        reshaped, orig_shape = self._reshape(tensor)
        scale = clamp_ste(self.scale, CLIP_MIN, 1e4)
        if self.signed:
            x_int = round_ste(reshaped / scale)
            x_int = clamp_ste(x_int, self.qmin, self.qmax)
            dequant = x_int * scale
        else:
            zp = clamp_ste(round_ste(self.zero_point), self.qmin, self.qmax)
            x_int = round_ste(reshaped / scale)
            x_int = x_int + zp
            x_int = clamp_ste(x_int, self.qmin, self.qmax)
            dequant = (x_int - zp) * scale
        dequant = dequant.reshape(orig_shape)

        if return_params:
            zp_ret = None
            if not self.signed:
                zp_ret = zp
            return dequant, scale, zp_ret
        return dequant

    @torch.no_grad()
    def quantize_to_int(
        self,
        tensor: torch.Tensor,
        dtype: torch.dtype = torch.int8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            q_int: integer tensor representing round(weight / scale)
            scale: per-group scale values, shape (num_groups,)

        This helper only supports the common case where group_size equals
        the full input dimension (per-row scaling). LinearInt.from_qat
        checks this constraint before exporting a module.
        """
        if not self.enable:
            raise RuntimeError("Cannot export integer weights when quantization is disabled.")

        self._check_ready(tensor)
        reshaped, orig_shape = self._reshape(tensor.detach())
        scale = torch.clamp(self.scale.detach(), min=CLIP_MIN, max=1e4)

        if self.signed:
            q = torch.round(reshaped / scale).clamp_(self.qmin, self.qmax)
            q_int = q.to(dtype).reshape(orig_shape)
            per_group_scale = scale.squeeze(-1)
            return q_int, per_group_scale

        zp = torch.clamp(torch.round(self.zero_point.detach()), self.qmin, self.qmax)

        centered = torch.round(reshaped / scale)
        min_centered = self.qmin - zp
        max_centered = self.qmax - zp
        centered = torch.max(centered, min_centered)
        centered = torch.min(centered, max_centered)

        q_int = centered.to(dtype).reshape(orig_shape)
        per_group_scale = scale.squeeze(-1)
        return q_int, per_group_scale


class ActivationQuantizer(nn.Module):
    """
    Lightweight LSQ-style activation quantizer so that the EfficientQAT
    layers expose the same activation fake-quant API as the other backends.
    """

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
        self.scale = nn.Parameter(torch.ones(1))
        self.initialized = False

    def _init_from(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            if self.signed:
                max_val = x.detach().abs().mean()
            else:
                max_val = x.detach().abs().mean()
            max_val = max_val.clamp(min=CLIP_MIN)
            scale = max_val * 2.0 / (self.qmax ** 0.5)
            self.scale.copy_(scale)
        self.initialized = True

    def grad_scale(self, tensor: torch.Tensor, scale: float) -> torch.Tensor:
        y_grad = tensor * scale
        return (tensor - y_grad).detach() + y_grad

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized:
            self._init_from(x)

        s_grad_scale = 1.0 / ((self.qmax * x.numel()) ** 0.5)
        scale = self.grad_scale(self.scale, s_grad_scale)
        scale = torch.clamp(scale, min=CLIP_MIN).to(x.device)

        x_scaled = x / scale
        x_scaled = torch.clamp(x_scaled, self.qmin, self.qmax)
        q = round_ste(x_scaled)
        return q, scale.detach()

    @torch.no_grad()
    def quantize_to_int(
        self,
        x: torch.Tensor,
        dtype: torch.dtype = torch.int8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized:
            self._init_from(x)

        scale = torch.clamp(self.scale.detach(), min=CLIP_MIN).to(x.device)
        x_scaled = x / scale
        x_scaled = torch.clamp(x_scaled, self.qmin, self.qmax)
        x_int = torch.round(x_scaled).to(dtype)
        return x_int, scale


class QuantLinear(nn.Module):
    """
    Linear layer that leverages EfficientQAT's uniform affine weight quantizer.
    Activation quantization is optional (LSQ-style) to preserve the familiar API.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bit: int,
        group_size: int | None = None,
        use_act_quant: bool = True,
        act_signed: bool = False,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=True)
        self.use_act_quant = use_act_quant

        if use_act_quant:
            self.act_quant = ActivationQuantizer(bit, signed=act_signed)
        else:
            self.act_quant = None

        self.weight_quant = Quantizer(
            n_bits=bit,
            group_size=group_size,
            init_tensor=self.fc.weight,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_act = 1.0
        if self.use_act_quant and self.act_quant is not None:
            x, scale_act = self.act_quant(x)

        weight_q = self.weight_quant(self.fc.weight)
        return F.linear(
            x * scale_act,
            weight_q,
            bias=self.fc.bias,
        )

    def set_quant_state(self, weight_quant: bool = True) -> None:
        self.weight_quant.enable = weight_quant

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bit: int,
        **kwargs,
    ) -> "QuantLinear":
        qa = cls(
            linear.in_features,
            linear.out_features,
            bit,
            **kwargs,
        )
        qa.to(linear.weight.device)
        with torch.no_grad():
            qa.fc.weight.copy_(linear.weight)
            if linear.bias is not None:
                qa.fc.bias.copy_(linear.bias)
        qa.weight_quant._init_from_tensor(qa.fc.weight)
        return qa


class LinearInt(nn.Linear):
    """
    Integer backend that exports QuantLinear into a torch._int_mm friendly module.

    NOTE: exporting currently assumes per-row weight scales (group_size equals the
    input feature size). This matches the usage in this project and keeps the
    implementation lightweight.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        w_scale: torch.Tensor,
        act_quant: ActivationQuantizer,
        int_dtype: torch.dtype,
    ) -> None:
        super().__init__(in_features, out_features, bias=True)
        self.int_dtype = int_dtype

        self.weight.requires_grad = False
        self.bias.requires_grad = False

        w_scale = w_scale.detach().to(torch.float32)
        w_scale = w_scale.view(1, -1)
        self.register_buffer("w_scale", w_scale)

        self.quantizer_act = copy.deepcopy(act_quant)
        if self.quantizer_act is None:
            raise RuntimeError("Activation quantizer is required for integer export.")
        self.quantizer_act.eval()
        for param in self.quantizer_act.parameters():
            param.requires_grad = False

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        act_int, act_scale = self.quantizer_act.quantize_to_int(
            input_x,
            dtype=self.int_dtype,
        )
        act_int = act_int.to(self.int_dtype).contiguous()
        weight_int = self.weight.to(self.int_dtype).contiguous()
        q_out = torch._int_mm(act_int, weight_int.T).to(torch.int32)

        scale_total = act_scale.to(torch.float32).view(-1, 1) * self.w_scale
        out = q_out.to(torch.float32) * scale_total
        if self.bias is not None:
            out = out + self.bias
        return out

    @classmethod
    def from_qat(
        cls,
        quantized_fc: QuantLinear,
        int_dtype: torch.dtype,
    ) -> "LinearInt":
        weight_int, weight_scale = quantized_fc.weight_quant.quantize_to_int(
            quantized_fc.fc.weight.data,
            dtype=int_dtype,
        )

        act_quant = copy.deepcopy(quantized_fc.act_quant)

        linear_int = cls(
            quantized_fc.fc.in_features,
            quantized_fc.fc.out_features,
            w_scale=weight_scale,
            act_quant=act_quant,
            int_dtype=int_dtype,
        )
        linear_int.weight.data = weight_int.to(int_dtype)
        linear_int.bias.data = quantized_fc.fc.bias.detach().to(linear_int.bias.dtype)
        return linear_int
