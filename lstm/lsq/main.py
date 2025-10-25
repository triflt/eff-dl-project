# lsq_mlp_mnist.py
import math
import time
import copy
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchao.quantization import Int8WeightOnlyConfig, quantize_
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# -----------------------------
# Utilities for LSQ (no custom autograd.Function)
# -----------------------------

def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    # Pass-through value but scale its gradient by `scale`
    return (x - x.detach()) * scale + x.detach()

def ste_round(x: torch.Tensor) -> torch.Tensor:
    # Straight-through estimator for rounding
    return (x - x.detach()) + x.detach().round()


# -----------------------------
# LSQLinear: quantizes activations (inputs) and weights
# -----------------------------

class LSQLinear(nn.Module):
    """
    LSQ (Learned Step Size Quantization) Linear layer.
    - Symmetric, signed per-tensor quantization for W and A
    - Uses ReLU outside this module (as requested, "always use relu")
    - No custom autograd.Function; uses STE via detach-tricks
    """
    def __init__(self, in_features, out_features, bias=True, n_bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Learnable step sizes. Initialize later (lazy) for activations; weights at module init.
        self.w_step = nn.Parameter(torch.tensor(0.0))
        self.a_step = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.register_buffer("init_done_w", torch.tensor(0, dtype=torch.uint8))
        self.register_buffer("init_done_a", torch.tensor(0, dtype=torch.uint8))

        # Quant bounds (signed)
        self.Qn = -(2 ** (n_bits - 1))
        self.Qp = (2 ** (n_bits - 1)) - 1

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def _init_w_step(self):
        # LSQ paper: s_w = 2 * E(|w|) / sqrt(Qp)
        # (signed symmetric)
        if self.weight.numel() == 0:
            val = 1e-8
        else:
            val = 2 * self.weight.abs().mean() / math.sqrt(self.Qp)
            val = float(val.clamp(min=1e-8))
        self.w_step.copy_(torch.tensor(val))
        self.init_done_w.fill_(1)

    @torch.no_grad()
    def _init_a_step(self, x: torch.Tensor):
        # LSQ paper: s_a = 2 * E(|x|) / sqrt(Qp)
        ax = x.detach()
        if ax.numel() == 0:
            val = 1e-8
        else:
            val = 2 * ax.abs().mean() / math.sqrt(self.Qp)
            val = float(val.clamp(min=1e-8))
        self.a_step.copy_(torch.tensor(val))
        self.init_done_a.fill_(1)

    def _quantize(self, x: torch.Tensor, s: torch.Tensor, Qn: int, Qp: int, grad_scale_factor: float) -> torch.Tensor:
        # Scale the gradient flowing into s
        s_scaled = grad_scale(s, grad_scale_factor)

        # Normalize, clamp to Q-range, round with STE, de-quantize
        x_div_s = x / s_scaled
        x_hat = torch.clamp(x_div_s, Qn, Qp)
        x_bar = ste_round(x_hat)
        x_q = x_bar * s_scaled
        return x_q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.init_done_w.item() == 0:
            self._init_w_step()

        if self.training and self.init_done_a.item() == 0:
            # lazy init using first batch stats
            self._init_a_step(x)

        # Gradient scales per LSQ (per-tensor):
        # g_w = 1/sqrt(N * Qp), N = number of elements in weight tensor
        # g_a = 1/sqrt(N * Qp), N = number of elements in current activation tensor
        g_w = 1.0 / math.sqrt(self.weight.numel() * self.Qp)
        # guard a_step init for eval (if eval before train)
        a_step = self.a_step if self.init_done_a.item() == 1 else torch.tensor(1.0, device=x.device)

        g_a = 1.0 / math.sqrt(x.numel() * max(1, self.Qp))

        # Quantize weights and input activations (signed symmetric)
        w_q = self._quantize(self.weight, self.w_step, self.Qn, self.Qp, g_w)
        a_q = self._quantize(x, a_step, self.Qn, self.Qp, g_a)

        return F.linear(a_q, w_q, self.bias)

    # Helpers for exporting to a float Linear (dequantized weights)
    @torch.no_grad()
    def export_dequantized(self):
        # Return (W_dequant, b)
        if self.init_done_w.item() == 0:
            self._init_w_step()
        s_w = self.w_step
        w_div = self.weight / s_w
        w_clamped = torch.clamp(w_div, self.Qn, self.Qp)
        w_int = w_clamped.round()
        w_deq = w_int * s_w
        b = self.bias.clone() if self.bias is not None else None
        return w_deq, b


# -----------------------------
# MLP with LSQLinear + ReLU
# -----------------------------

class LSQMLP(nn.Module):
    def __init__(self, n_bits=8):
        super().__init__()
        self.net = nn.Sequential(
            LSQLinear(4, 16, n_bits=n_bits),
            nn.ReLU(),
            LSQLinear(16, 3, n_bits=n_bits),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def to_float_mlp(self):
        # Build a float MLP (nn.Linear) with the same topology, dequantized weights from LSQ
        modules = []
        i = 0
        children = list(self.net.children())
        while i < len(children):
            m = children[i]
            if isinstance(m, LSQLinear):
                w, b = m.export_dequantized()
                lin = nn.Linear(m.in_features, m.out_features, bias=m.use_bias)
                lin.weight.copy_(w)
                if b is not None:
                    lin.bias.copy_(b)
                modules.append(lin)
                # optional ReLU after it
                if i + 1 < len(children) and isinstance(children[i+1], nn.ReLU):
                    modules.append(nn.ReLU(inplace=True))
                    i += 2
                else:
                    i += 1
            else:
                # just in case (should be only ReLU here)
                modules.append(m)
                i += 1

        return nn.Sequential(*modules)


# -----------------------------
# Simple training & eval on MNIST
# -----------------------------

class IrisDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def train_lsq_mlp(epochs=3, batch_size=16, lr=1e-3, n_bits=8, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X.astype('float32'), y, train_size=0.7)
    train_ds = IrisDataset(X_train, y_train)
    test_ds = IrisDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    model = LSQMLP(n_bits).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for ep in range(1, epochs+1):
        total, correct, running = 0, 0, 0.0
        t0 = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running += loss.item() * xb.size(0)
            total += xb.size(0)
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
        dt = time.time() - t0
        print(f"Epoch {ep}/{epochs} | loss {running/total:.4f} | acc {correct/total:.4f} | time {dt:.1f}s")

    # Validation (LSQ model on chosen device)
    lsq_acc = evaluate(model, test_loader, device=device)
    print(f"[LSQ model] Test accuracy: {lsq_acc:.4f}")

    return model, (test_loader, device)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device=None, cpu_only=False):
    was_training = model.training
    model.eval()
    if cpu_only:
        model_cpu = model.to("cpu")
        device = torch.device("cpu")
        m = model_cpu
    else:
        if device is None:
            device = next(model.parameters()).device
        m = model

    total, correct = 0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = m(xb)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)

    if was_training:
        model.train()
    return correct / total


# -----------------------------
# Inference export & dynamic quantization (CPU)
# -----------------------------

def export_and_quantize_dynamic(lsq_model: LSQMLP):
    # Convert LSQ -> float MLP with nn.Linear + ReLU
    float_mlp = lsq_model.to_float_mlp()
    float_mlp.eval()
    float_mlp_cpu = float_mlp.to("cpu")

    # Apply dynamic quantization on Linear for CPU inference speedup
    qdyn = copy.deepcopy(float_mlp_cpu)
    quantize_(
        qdyn,
        Int8WeightOnlyConfig()
    )
    return float_mlp_cpu, qdyn


@torch.no_grad()
def benchmark_inference(models: List[nn.Module], loader: DataLoader, repeats=3, warmup=1, label: List[str]=None):
    # Force CPU benchmarking (dynamic quant is CPU only)
    data = list(loader)  # materialize
    results = []
    for idx, m in enumerate(models):
        # Warmup
        for _ in range(warmup):
            for xb, _ in data:
                m(xb.cpu())
        # Timed
        t0 = time.time()
        for _ in range(repeats):
            for xb, _ in data:
                _ = m(xb.cpu())
        dt = time.time() - t0
        # compute images/sec
        n_images = repeats * sum(xb.size(0) for xb, _ in data)
        ips = n_images / dt
        tag = label[idx] if label else f"model_{idx}"
        results.append((tag, ips, dt))
    return results


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    # Hyperparameters: kept modest for a quick demo
    lsq_model, (test_loader, device) = train_lsq_mlp(
        epochs=3,           # bump to 5-10 for better accuracy
        batch_size=16,
        lr=1e-3,
        n_bits=8,
        seed=123,
    )

    # Evaluate LSQ model (on its training device)
    lsq_acc = evaluate(lsq_model, test_loader, device=device)
    print(f"\n[LSQ train-time model] Test accuracy: {lsq_acc:.4f}")

    # Export to float MLP (nn.Linear) and dynamic-quantize for CPU inference
    float_mlp_cpu, dq_model = export_and_quantize_dynamic(lsq_model)

    # Evaluate accuracy on CPU (float vs dynamic-quantized)
    float_acc = evaluate(float_mlp_cpu, test_loader, cpu_only=True)
    dq_acc    = evaluate(dq_model,       test_loader, cpu_only=True)
    print(f"[Float exported]   Test accuracy (CPU): {float_acc:.4f}")
    print(f"[Dynamic quant]    Test accuracy (CPU): {dq_acc:.4f}")

    # Benchmark CPU inference throughput
    bench = benchmark_inference(
        [float_mlp_cpu, dq_model],
        test_loader,
        repeats=5,
        warmup=1,
        label=["Float-CPU", "DynQuant-CPU"]
    )
    for tag, ips, dt in bench:
        print(f"[{tag}] ~{ips:.1f} images/sec (total {dt:.2f}s)")

    # Reminder:
    # - Speedup/accuracy will vary by CPU (AVX2/AVX512/VNNI), PyTorch build, and batch sizes.
