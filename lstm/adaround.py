import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class Quantizer(nn.Module):
    """
    Minimal AdaRound-style quantizer.
    - Learns rounding offsets (alpha) while keeping scale deterministic by default.
    - Returns (x_quantized, scale).
    """
    def __init__(self, bit=8, per_channel=False, ch_axis=0, symmetric=True):
        super().__init__()
        assert bit >= 2
        self.bit = bit
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.symmetric = symmetric

        # Learned rounding parameters; lazily initialized to match input shape
        self.alpha = None  # shape like x (or broadcastable)
        # Constants from AdaRound paper for “soft” → “hard” transition
        self.register_buffer("gamma", torch.tensor(-0.1))
        self.register_buffer("zeta",  torch.tensor(1.1))

        qmin = -(2 ** (bit - 1)) if symmetric else 0
        qmax = (2 ** (bit - 1) - 1) if symmetric else (2 ** bit - 1)
        self.register_buffer("qmin", torch.tensor(qmin, dtype=torch.float32))
        self.register_buffer("qmax", torch.tensor(qmax, dtype=torch.float32))

    def _init_alpha(self, x, scale):
        # Initialize alpha so initial rounding is close to standard .round()
        with torch.no_grad():
            x_s = x / scale
            frac = x_s - torch.floor(x_s)
            # inverse of sigmoid clipped into (0,1)
            p = torch.clamp(frac, 1e-4, 1 - 1e-4)
            # map [0,1] to (gamma,zeta) then to alpha space
            u = (p - self.gamma) / (self.zeta - self.gamma)
            u = torch.clamp(u, 1e-4, 1 - 1e-4)
            alpha = torch.log(u) - torch.log(1 - u)
        return nn.Parameter(alpha)

    def _compute_scale(self, x):
        # simple max-based scale (can be replaced with learned or percentile-based)
        if self.per_channel:
            # move quant axis to dim 0, take max over the rest
            perm = list(range(x.dim()))
            perm[0], perm[self.ch_axis] = perm[self.ch_axis], perm[0]
            x_perm = x.permute(perm)
            # per-channel max
            maxv = x_perm.abs().reshape(x_perm.shape[0], -1).max(dim=1).values
            s = maxv / self.qmax if self.symmetric else maxv / self.qmax
            s = torch.clamp(s, min=1e-8)
            # reshape back to broadcastable shape
            shape = [1] * x.dim()
            shape[self.ch_axis] = -1
            return s.view(*shape)
        else:
            maxv = x.abs().max()
            s = maxv / self.qmax if self.symmetric else maxv / self.qmax
            return torch.clamp(s, min=1e-8)

    def _round_offset(self, hard: bool) -> torch.Tensor:
        s = torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma
        s = torch.clamp(s, 0.0, 1.0)
        if hard:
            return (s >= 0.5).to(s.dtype)
        return s

    def forward(self, x, hard: bool = False, return_int: bool = False):
        """
        Returns:
            x_q: quantized tensor (same shape as x)
            scale: scaling factor tensor (per-tensor or per-channel; broadcastable to x)
        """
        scale = self._compute_scale(x).detach()  # keep scale fixed by default
        if self.alpha is None or self.alpha.shape != x.shape:
            # Learn a rounding param per-element (simple & effective); for efficiency,
            # you can reduce to per-channel later if desired.
            self.alpha = self._init_alpha(x.detach(), scale).to(x.device)

        # Soft-to-hard rounding as in AdaRound:
        # r = clamp(sigmoid(alpha)*(zeta-gamma)+gamma, 0, 1)
        # x_int = floor(x/scale) + r, then clamp to [qmin, qmax]
        x_s = x / scale
        x_floor = torch.floor(x_s)
        r = self._round_offset(hard=hard)

        x_int = x_floor + r
        qmin = self.qmin.to(x_int.device)
        qmax = self.qmax.to(x_int.device)
        x_int = torch.clamp(x_int, qmin, qmax)

        if return_int:
            return x_int, scale

        x_q = x_int * scale
        return x_q, scale


class QuantLinear(nn.Module):
    """
    Linear layer that quantizes its WEIGHTS with AdaRoundQuantizer.
    Forward returns ONLY the output (no scales), as requested.
    """
    def __init__(self, in_features, out_features, bit=8, 
                 per_channel=True, symmetric=True):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=True)
        # Per-output-channel quantization is standard for linear/conv weights
        self.quantizer = Quantizer(
            bit=bit, per_channel=per_channel, ch_axis=0, symmetric=symmetric
        )

    def forward(self, x):
        w_q, _ = self.quantizer(self.fc.weight)
        # Return ONLY the linear output (drop scales)
        return F.linear(x, w_q, self.fc.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, bit: int) -> "QuantLinear":
        qa = cls(
            linear.in_features,
            linear.out_features,
            bit,
        )
        qa.to(linear.weight.device)
        with torch.no_grad():
            qa.fc.weight.copy_(linear.weight)
            qa.fc.bias.copy_(linear.bias)
        return qa


class LinearInt(nn.Linear):
    def __init__(self, in_features, out_features, w_scale, int_dtype):
        super().__init__(in_features, out_features, bias=True)
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.int_dtype = int_dtype

        if w_scale.dim() == 2:
            w_scale = w_scale.T
        self.register_buffer("w_scale", w_scale)
        self.quantizer_act = Quantizer(
            torch.iinfo(int_dtype).bits,
            per_channel=False,
        )

    def forward(self, input_x, act_scale=None):
        if act_scale is None:
            act_q, act_scale = self.quantize_input(input_x)
            act_q = act_q.to(self.int_dtype)
        else:
            act_q = input_x

        q_out = torch._int_mm(act_q, self.weight.T)
        return q_out * (act_scale * self.w_scale) + self.bias

    def quantize_input(self, input_x):
        return self.quantizer_act(input_x, hard=True, return_int=True)

    @classmethod
    def from_qat(cls, quantized_fc: QuantLinear, int_dtype: torch.dtype) -> "LinearInt":
        in_features = quantized_fc.fc.in_features
        out_features = quantized_fc.fc.out_features
        weight_q, weight_scale = quantized_fc.quantizer(
            quantized_fc.fc.weight.data, hard=True, return_int=True
        )
        linear_int = cls(in_features, out_features, weight_scale, int_dtype)
        linear_int.weight.data = weight_q.to(int_dtype)
        linear_int.bias.data = copy.deepcopy(quantized_fc.fc.bias.data.detach())
        return linear_int
