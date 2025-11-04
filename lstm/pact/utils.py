import torch
import torch.nn as nn
import torch.nn.functional as F


class PACTQuantizer(nn.Module):
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
        self.alpha = nn.Parameter(alpha)
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
            x_clipped = torch.clamp(x, min=0.0, max=alpha)
            # Scale: alpha maps to qmax with zero-point 0
            scale = alpha / self.qmax

        # Guard against tiny/zero scale
        eps = torch.finfo(x.dtype).eps
        scale_safe = torch.clamp(scale, min=eps)

        # Fake-quantization with STE
        y = torch.round(x_clipped / scale_safe) * scale_safe

        # Detach scale from graph for returning
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
        bias: bool = True,
        # Activation quantizer (PACT, typically unsigned)
        act_bits: int = 8,
        use_act_quant: bool = True,
        act_init_alpha: float = 6.0,
        # Weight quantizer (PACT-style symmetric)
        weight_bits: int = 8,
        use_weight_quant: bool = True,
        weight_per_channel: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.use_act_quant = use_act_quant
        self.use_weight_quant = use_weight_quant

        if use_act_quant:
            self.act_quant = PACTQuantizer(
                n_bits=act_bits,
                signed=False,
                init_alpha=act_init_alpha,
                per_channel=False,
            )

        if use_weight_quant:
            # Per-output-channel (i.e., along out_features) often works best for weights
            self.weight_quant = PACTQuantizer(
                n_bits=weight_bits,
                signed=True,
                init_alpha=2.0,                  # a small start for weights; will adapt
                per_channel=weight_per_channel,
                channel_dim=0,                   # out_features dimension in [out, in]
            )

    def _quantize_weights(self):
        W = self.linear.weight
        Wq, sw = self.weight_quant(W)
        return Wq, sw

    def forward(self, x: torch.Tensor):
        scales = {}

        if self.use_act_quant:
            x, sa = self.act_quant(x)
            scales["act_scale"] = sa

        if self.use_weight_quant:
            Wq, sw = self._quantize_weights()
            scales["weight_scale"] = sw
        else:
            Wq = self.linear.weight

        b = self.linear.bias
        out = F.linear(x, Wq, b)
        return out, scales
