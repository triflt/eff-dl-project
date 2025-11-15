import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class Quantizer(nn.Module):
    """
    PACT (Parameterized Clipping Activation) quantizer.

    - Learns a clipping parameter alpha.
    - Uniform fake-quantization with STE (straight-through estimator).
    - Works for unsigned (typical for activations) or signed (useful for weights).

    forward(x) -> (y, scale):
        y: dequantized (fake-quantized) tensor
        scale: scalar/tensor scale used during quantization (detached)
    """
    def __init__(
        self,
        n_bits: int = 8,
        signed: bool = False,         # PACT is usually for activations (unsigned). Set True for weights if you like.
        init_alpha: float = 6.0,      # Common init for activations; for weights, you might init to weight.max().item()
        per_channel: bool = False,    # For weights you may want per-out-channel scales. For activations keep False.
        channel_dim: int = 0,         # Channel axis if per_channel=True (e.g., out_features for weights)
    ):
        super().__init__()
        assert n_bits >= 2, "Use at least 2 bits."
        self.n_bits = n_bits
        self.signed = signed
        self.per_channel = per_channel
        self.channel_dim = channel_dim

        # Alpha is learnable; shape depends on per_channel
        if per_channel:
            # alpha will be initialized later once we see the tensor shape (lazy init)
            self.alpha = None
            self._alpha_initialized = False
            self._saved_shape = None
            self._init_alpha_value = float(init_alpha)
        else:
            self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
            self._alpha_initialized = True

        # Precompute integer ranges
        if signed:
            self.qmax = (2 ** (n_bits - 1)) - 1
            self.qmin = -(2 ** (n_bits - 1))
        else:
            self.qmax = (2 ** n_bits) - 1
            self.qmin = 0

    def _lazy_init_alpha(self, x: torch.Tensor):
        if not self.per_channel or self._alpha_initialized:
            return

        # Create per-channel alpha parameter with size equal to x.size(channel_dim)
        num_channels = x.size(self.channel_dim)
        alpha_shape = [1] * x.dim()
        alpha_shape[self.channel_dim] = num_channels
        alpha = torch.full(alpha_shape, self._init_alpha_value, device=x.device, dtype=x.dtype)
        self.alpha = nn.Parameter(alpha).to(x.device)
        self._alpha_initialized = True

    def forward(self, x: torch.Tensor):
        """
        Returns:
            y (torch.Tensor): fake-quantized (dequantized) tensor
            scale (torch.Tensor): scale used, detached (shape matches alpha broadcasting)
        """
        self._lazy_init_alpha(x)

        # Ensure alpha is positive (PACT typically constrains alpha >= 0)
        # Use softplus to keep it > 0 while still learnable.
        alpha = F.softplus(self.alpha) if self.alpha is not None else None

        if self.signed:
            # Clip to [-alpha, alpha]
            x_clipped = torch.clamp(x, min=-alpha, max=alpha)
            # Scale: alpha maps to qmax (symmetric about 0)
            scale = alpha / self.qmax
        else:
            # Clip to [0, alpha]
            x_clipped = torch.clamp(x, min=torch.zeros_like(alpha), max=alpha)
            # Scale: alpha maps to qmax with zero-point 0
            scale = alpha / self.qmax

        # Guard against tiny/zero scale
        eps = torch.finfo(x.dtype).eps
        scale_safe = torch.clamp(scale, min=eps)

        # Fake-quantization with straight-through gradients (return values before scaling back)
        scaled = x_clipped / scale_safe
        q = torch.round(scaled).clamp_(self.qmin, self.qmax)
        y = scaled + (q - scaled).detach()

        return y, scale_safe.detach()


class QuantLinear(nn.Module):
    """
    Linear layer that can use PACT quantization for inputs and/or weights.

    Returns forward:
        out, scales_dict
        where scales_dict contains keys 'act_scale' and/or 'weight_scale' if applicable.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bit: int,
        use_act_quant: bool = True,
        act_init_alpha: float = 6.0,
        weight_per_channel: bool = True,
    ):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=True)

        self.use_act_quant = use_act_quant

        if use_act_quant:
            self.act_quant = Quantizer(
                n_bits=bit,
                signed=False,
                init_alpha=act_init_alpha,
                per_channel=False,
            )

        # Per-output-channel (i.e., along out_features) often works best for weights
        self.weight_quant = Quantizer(
            n_bits=bit,
            signed=True,
            init_alpha=2.0,                  # a small start for weights; will adapt
            per_channel=weight_per_channel,
            channel_dim=0,                   # out_features dimension in [out, in]
        )

    def _quantize_weights(self):
        W = self.fc.weight
        Wq, sw = self.weight_quant(W)
        return Wq, sw

    def forward(self, x: torch.Tensor):

        scale_act = 1.0
        if self.use_act_quant:
            x, scale_act = self.act_quant(x)

        Wq, scale_w = self._quantize_weights()

        return F.linear(
            x * scale_act,
            Wq * scale_w,
            bias=self.fc.bias,
        )

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
        qa.weight_quant._lazy_init_alpha(qa.fc.weight)
        return qa


class LinearInt(nn.Linear):
    def __init__(self, in_features, out_features, w_scale, q_a, int_dtype):
        super().__init__(in_features, out_features, bias=True)
        self.int_dtype = int_dtype

        self.weight.requires_grad = False
        self.bias.requires_grad = False

        if w_scale.dim() == 0:
            w_scale = w_scale.reshape(1, 1)
        elif w_scale.dim() == 1:
            w_scale = w_scale.unsqueeze(0)
        else:
            w_scale = w_scale.reshape(1, -1)
        self.register_buffer("w_scale", w_scale.float())

        self.quantizer_act = None
        if q_a is not None:
            self.quantizer_act = Quantizer(
                torch.iinfo(int_dtype).bits,
                signed=q_a.signed,
                per_channel=q_a.per_channel,
                channel_dim=q_a.channel_dim,
            )
            alpha = q_a.alpha.detach().clone()
            self.quantizer_act.alpha = nn.Parameter(alpha, requires_grad=False)

    def forward(self, input_x, act_scale=None):
        if act_scale is None:
            act_q, act_scale = self.quantize_input(input_x)
            act_q = act_q.to(self.int_dtype)
        else:
            act_q = input_x
        
        q_out = torch._int_mm(act_q, self.weight.T)
        return q_out * (act_scale * self.w_scale) + self.bias

    def quantize_input(self, input_x):
        return self.quantizer_act(input_x)

    @classmethod
    def from_qat(cls, quantized_fc: QuantLinear, int_dtype: torch.dtype) -> "LinearInt":
        in_features = quantized_fc.fc.in_features
        out_features = quantized_fc.fc.out_features
        weight_q, weight_scale = quantized_fc.weight_quant(quantized_fc.fc.weight.data)
        act_quant = quantized_fc.act_quant if getattr(quantized_fc, "use_act_quant", False) else None
        linear_int = cls(in_features, out_features, weight_scale, act_quant, int_dtype)
        linear_int.weight.data = weight_q.to(int_dtype)
        linear_int.bias.data = copy.deepcopy(quantized_fc.fc.bias.detach())
        return linear_int
