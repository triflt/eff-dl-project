import copy
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaRoundQuantizer(nn.Module):
    """AdaRound-style weight quantizer with learnable rounding offsets."""
    def __init__(self, bit: int = 8, per_channel: bool = True, ch_axis: int = 0, symmetric: bool = True):
        super().__init__()
        assert bit >= 2
        self.bit = bit
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.symmetric = symmetric

        self.alpha = None  # lazily initialized to weight shape
        self.register_buffer("gamma", torch.tensor(-0.1))
        self.register_buffer("zeta", torch.tensor(1.1))

        qmin = -(2 ** (bit - 1)) if symmetric else 0
        qmax = (2 ** (bit - 1) - 1) if symmetric else (2 ** bit - 1)
        self.register_buffer("qmin", torch.tensor(float(qmin)))
        self.register_buffer("qmax", torch.tensor(float(qmax)))

    @torch.no_grad()
    def _compute_scale(self, w: torch.Tensor) -> torch.Tensor:
        if self.per_channel:
            perm = list(range(w.dim()))
            perm[0], perm[self.ch_axis] = perm[self.ch_axis], perm[0]
            w_perm = w.permute(perm)
            maxv = w_perm.abs().reshape(w_perm.shape[0], -1).max(dim=1).values
            s = maxv / self.qmax
            s = torch.clamp(s, min=1e-8)
            shape = [1] * w.dim()
            shape[self.ch_axis] = -1
            return s.view(*shape).to(w.dtype).to(w.device)
        else:
            maxv = w.abs().max()
            s = maxv / self.qmax
            return torch.clamp(s, min=1e-8).to(w.dtype).to(w.device)

    @torch.no_grad()
    def _init_alpha(self, w: torch.Tensor, scale: torch.Tensor) -> nn.Parameter:
        w_s = w / scale
        frac = w_s - torch.floor(w_s)
        p = torch.clamp(frac, 1e-4, 1 - 1e-4)
        u = (p - self.gamma) / (self.zeta - self.gamma)
        u = torch.clamp(u, 1e-4, 1 - 1e-4)
        alpha = torch.log(u) - torch.log(1 - u)
        return nn.Parameter(alpha)

    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self._compute_scale(w).detach()
        if self.alpha is None or self.alpha.shape != w.shape:
            self.alpha = self._init_alpha(w.detach(), scale).to(w.device)
        w_s = w / scale
        w_floor = torch.floor(w_s)
        s = torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma
        r = torch.clamp(s, 0, 1)
        w_int = torch.clamp(w_floor + r, self.qmin.item(), self.qmax.item())
        w_q = w_int * scale
        return w_q, scale


class AdaRoundConv2d(nn.Module):
    """Conv2d wrapper that quantizes weights with AdaRound; activations remain float."""
    def __init__(self, conv: nn.Conv2d, bit: int = 8, per_channel: bool = True, symmetric: bool = True):
        super().__init__()
        self.conv = conv
        self.quantizer = AdaRoundQuantizer(bit=bit, per_channel=per_channel, ch_axis=0, symmetric=symmetric)

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d, bit: int = 8) -> "AdaRoundConv2d":
        new_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                             conv.stride, conv.padding, conv.dilation, conv.groups, bias=(conv.bias is not None))
        new_conv.to(conv.weight.device)
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight)
            if conv.bias is not None:
                new_conv.bias.copy_(conv.bias)
        return cls(new_conv, bit=bit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q, s = self.quantizer(self.conv.weight)
        return F.conv2d(x, w_q, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)


def wrap_model_adaround(model: nn.Module, bit: int = 8) -> nn.Module:
    """Replace Conv2d layers with AdaRoundConv2d wrappers (weights quantizable via learned rounding)."""
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Conv2d):
            setattr(model, name, AdaRoundConv2d.from_conv2d(module, bit=bit))
        else:
            wrap_model_adaround(module, bit=bit)
    return model


def calibrate_adaround(model_fp32: nn.Module,
                       calib_loader: Iterable,
                       device: torch.device,
                       steps: int = 200,
                       bit: int = 8,
                       lr: float = 1e-2) -> nn.Module:
    """
    Layer-wise PTQ finetune: learn AdaRound alphas to match FP32 outputs.
    Returns a float model with quantized weights baked in.
    """
    model_fp32 = copy.deepcopy(model_fp32).to(device).eval()
    # Build quantizable copy
    model_q = copy.deepcopy(model_fp32)
    model_q = wrap_model_adaround(model_q, bit=bit).to(device).train()

    # Freeze all except AdaRound alphas
    for p in model_q.parameters():
        p.requires_grad = False
    alpha_params = []
    for m in model_q.modules():
        if isinstance(m, AdaRoundConv2d) and m.quantizer.alpha is not None:
            m.quantizer.alpha.requires_grad = True
            alpha_params.append(m.quantizer.alpha)
    # Alpha may be lazily initialized on first pass; create optimizer on-the-fly later if empty
    optimizer = torch.optim.Adam(alpha_params, lr=lr) if alpha_params else None
    loss_fn = nn.MSELoss()

    it = 0
    for batch in calib_loader:
        if it >= steps:
            break
        gt = batch["gt"].to(device)
        lr_img = batch["lr"].to(device)
        with torch.no_grad():
            ref = model_fp32(lr_img)
        out = model_q(lr_img)
        loss = loss_fn(out, ref)
        if optimizer is None:
            # Now quantizer alphas should be initialized; collect them
            alpha_params = []
            for m in model_q.modules():
                if isinstance(m, AdaRoundConv2d) and m.quantizer.alpha is not None:
                    m.quantizer.alpha.requires_grad = True
                    alpha_params.append(m.quantizer.alpha)
            optimizer = torch.optim.Adam(alpha_params, lr=lr)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        it += 1

    # Bake quantized weights into plain Conv2d and return a float model ready for CPU inference
    model_baked = copy.deepcopy(model_q).eval()
    for module in model_baked.modules():
        if isinstance(module, AdaRoundConv2d):
            with torch.no_grad():
                w_q, s = module.quantizer(module.conv.weight)
                module.conv.weight.copy_(w_q)
            # replace wrapper by inner conv
    def strip(module: nn.Module) -> nn.Module:
        for name, child in list(module.named_children()):
            if isinstance(child, AdaRoundConv2d):
                setattr(module, name, child.conv)
            else:
                strip(child)
        return module
    model_baked = strip(model_baked)
    return model_baked

