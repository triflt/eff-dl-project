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
            # Using the integer domain scaled by 2^{shift_start + bit - 1}
            # simplifies deduplication when building the codebook.
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
                    of the base choices that formed each value (useful for later
                    decomposition / debugging).
        """
        combos_iter = combinations_with_replacement(range(len(self.choice_values)), self.num_terms)
        seen: dict[int, tuple[int, ...]] = {}
        for combo in combos_iter:
            total_int = sum(self._choice_int[idx] for idx in combo)
            # Keep the first representation for determinism.
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
            return_indices: if True, also returns (indices, sign) tensors that
                            encode the chosen positive codebook entry and sign.

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

        # Straight-through estimator: gradients follow the unclipped path.
        norm_out = norm + (norm_quant - norm).detach()

        if return_indices:
            sign_tensor = torch.sign(x_clipped.detach()) if self.signed else torch.ones_like(x_clipped.detach())
            return norm_out, scale.detach(), pos_indices.detach(), sign_tensor
        return norm_out, scale.detach()

    def quantize_to_int(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience helper that returns integer tensors suitable for integer GEMM.

        Returns:
            x_int: tensor of dtype `dtype` within [int_qmin, int_qmax]
            eff_scale: floating scale so that dequant(x_int) = x_int * eff_scale
        """
        norm_out, scale = self.forward(x)
        qmax = float(self.int_qmax)
        qmin = float(self.int_qmin)
        q = torch.round(norm_out * qmax).clamp_(qmin, qmax).detach()
        return q, (scale / qmax).detach()


class QuantLinear(nn.Module):
    """
    Linear layer backed by APoT quantizers for both activations and weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bit: int,
        num_terms: int | None = None,
        use_act_quant: bool = True,
        act_signed: bool = False,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=True)
        self.use_act_quant = use_act_quant

        if use_act_quant:
            self.act_quant = Quantizer(
                bit=bit,
                signed=act_signed,
                num_terms=num_terms,
                init_alpha=6.0,
                per_channel=False,
            )
        else:
            self.act_quant = None

        self.weight_quant = Quantizer(
            bit=bit,
            signed=True,
            num_terms=num_terms,
            init_alpha=2.0,
            per_channel=True,
            channel_dim=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_act_quant and self.act_quant is not None:
            act_q, act_scale = self.act_quant(x)
            x = act_q * act_scale

        weight_q, weight_scale = self.weight_quant(self.fc.weight)
        w = weight_q * weight_scale
        return F.linear(x, w, self.fc.bias)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bit: int,
        **kwargs,
    ) -> "QuantLinear":
        qa = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bit=bit,
            **kwargs,
        )
        qa.to(linear.weight.device)
        with torch.no_grad():
            qa.fc.weight.copy_(linear.weight)
            if linear.bias is not None:
                qa.fc.bias.copy_(linear.bias)
        return qa


class LinearInt(nn.Linear):
    """
    Integer GEMM backend that pairs with `QuantLinear`.

    - Stores integer weights (e.g., int8) produced by APoT quantization.
    - Reuses the activation quantizer to turn runtime inputs into integers.
    - Performs matmul via `torch._int_mm` and rescales the output back to float.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        w_scale: torch.Tensor,
        act_quant: Quantizer,
        int_dtype: torch.dtype = torch.int8,
    ) -> None:
        super().__init__(in_features, out_features, bias=True)
        self.int_dtype = int_dtype
        self.weight.requires_grad = False
        self.bias.requires_grad = False

        w_scale = w_scale.detach().to(torch.float32)
        if w_scale.numel() == 1:
            w_scale = w_scale.view(1, 1).repeat(1, out_features)
        else:
            w_scale = w_scale.view(-1)
            if w_scale.numel() != out_features:
                raise ValueError("Weight scale size must match out_features.")
            w_scale = w_scale.view(1, out_features)
        self.register_buffer("w_scale", w_scale)
        self.quantizer_act = act_quant
        if self.quantizer_act is None:
            raise RuntimeError("APoT LinearInt requires an activation quantizer.")
        self.quantizer_act.eval()
        for param in self.quantizer_act.parameters():
            param.requires_grad = False

    def forward(self, input_x, act_scale=None):
        if act_scale is None:
            act_q, act_scale = self.quantize_input(input_x)
            act_q = act_q.to(self.int_dtype)
        else:
            act_q = input_x
        
        q_out = torch._int_mm(act_q, self.weight.T)
        return q_out * (act_scale * self.w_scale) + self.bias

    def quantize_input(self, input_x):
        return self.quantizer_act.quantize_to_int(input_x)

    @classmethod
    def from_qat(
        cls,
        quantized_fc: QuantLinear,
        int_dtype: torch.dtype,
    ) -> "LinearInt":
        weight_int, weight_scale = quantized_fc.weight_quant.quantize_to_int(
            quantized_fc.fc.weight.data,
        )
        act_quant = copy.deepcopy(quantized_fc.act_quant)

        linear_int = cls(
            quantized_fc.fc.in_features,
            quantized_fc.fc.out_features,
            w_scale=weight_scale,
            act_quant=act_quant,
            int_dtype=int_dtype,
        )

        linear_int.weight.data = weight_int.to(linear_int.int_dtype)
        linear_int.bias.data = copy.deepcopy(quantized_fc.fc.bias.detach())
        return linear_int
