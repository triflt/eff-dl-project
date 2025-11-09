import torch
import torch.nn as nn
import torch.nn.functional as F


class Quantizer(nn.Module):
    def __init__(self, bit, only_positive=False, per_tensor=False):
        super().__init__()
        self.bit = bit
        self.only_positive = only_positive
        self.per_tensor = per_tensor
        if only_positive:
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            self.thd_neg = -(2 ** (bit - 1))
            self.thd_pos = 2 ** (bit - 1) - 1
        self.s = nn.Parameter(torch.ones(1))
        self.init_from_called = False

    def init_from(self, x):
        self.init_from_called = True
        if self.per_tensor:
            self.s = nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))
        else:
            self.s = nn.Parameter(
                x.detach().abs().mean(dim=1, keepdim=True) * 2 / (self.thd_pos ** 0.5)
            )

    def grad_scale(self, x, scale):
        y = x
        y_grad = x * scale
        return (y - y_grad).detach() + y_grad

    def round_pass(self, x):
        y = x.round()
        y_grad = x
        return (y - y_grad).detach() + y_grad

    def forward(self, x):
        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)

        s_scale = self.grad_scale(self.s, s_grad_scale).to(x.device)

        x = x / s_scale
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = self.round_pass(x)

        return x, s_scale


class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bit: int, only_positive_activations: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=True)
        self.quantizer_act = Quantizer(bit, only_positive=only_positive_activations, per_tensor=True)
        self.quantizer_weight = Quantizer(bit, per_tensor=True)
        self.quantizer_weight.init_from(self.fc.weight)

    def forward(self, input_x):
        # if not self.quantizer_weight.init_from_called:
        #     self.quantizer_weight.init_from(self.fc.weight)
        # if not self.quantizer_act.init_from_called:
        #     self.quantizer_act.init_from(input_x)

        weight_q, weight_scale = self.quantizer_weight(self.fc.weight)
        act_q, act_scale = self.quantizer_act(input_x)

        return F.linear(
            act_q * act_scale,
            weight_q * weight_scale,
            bias=self.fc.bias,
        )

    @classmethod
    def from_linear(cls, linear: nn.Linear, bit: int, only_positive_activations: bool = False) -> "QuantLinear":
        qa = cls(
            linear.in_features,
            linear.out_features,
            bit,
            only_positive_activations,
        )
        qa.to(linear.weight.device)
        with torch.no_grad():
            qa.fc.weight.copy_(linear.weight)
            qa.fc.bias.copy_(linear.bias)
        qa.quantizer_weight.init_from(qa.fc.weight)
        return qa


class LinearInt(nn.Linear):
    def __init__(self, in_features, out_features, w_scale, q_a, int_dtype, device='cpu'):
        super().__init__(in_features, out_features, bias=True, device=device)
        self.weight.requires_grad = False
        self.weight.data = self.weight.data.to(int_dtype)
        self.bias.requires_grad = False
        self.bias.data = self.bias.data.to(int_dtype)
        self.int_dtype = int_dtype

        self.w_scale = w_scale.to(device)
        if self.w_scale.dim() == 2:
            self.w_scale = self.w_scale.T
        self.quantizer_act = Quantizer(
            torch.iinfo(int_dtype).bits,
            only_positive=q_a.only_positive,
            per_tensor=True,
        ).to(device)
        self.quantizer_act.s = nn.Parameter(q_a.s.to(device))

    def forward(self, input_x):
        act_q, act_scale = self.quantizer_act(input_x)
        q_out = torch._int_mm(act_q.to(self.int_dtype), self.weight.T) + self.bias
        return q_out * (act_scale * self.w_scale)

    @classmethod
    def from_qat(cls, quantized_fc: QuantLinear, int_dtype: torch.dtype) -> "LinearInt":
        in_features = quantized_fc.in_features
        out_features = quantized_fc.out_features
        weight_q, weight_scale = quantized_fc.quantizer_weight(quantized_fc.fc.weight.data)
        linear_int = cls(in_features, out_features, weight_scale, quantized_fc.quantizer_act, int_dtype)
        linear_int.weight.data = weight_q.to(int_dtype)
        linear_int.bias.data = quantized_fc.fc.bias.data.to(int_dtype)
        return linear_int
