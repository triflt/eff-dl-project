import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


def get_mnist(batch_size, train=True):
    return torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]),
        ),
        batch_size=batch_size,
        shuffle=True,
    )


def train(model, epoch, loss_fn, optimizer, train_loader, use_cuda, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)

        loss = loss_fn(output, target)

        loss.backward()

        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                f'Train Epoch: {epoch} '
                f'[{batch_idx * len(data)}/{len(train_loader.dataset)} '
                f'({100. * batch_idx / len(train_loader):.0f}%)]'
                f'\tLoss: {loss.item():.6f}'
            )


def test(model, loss_fn, optimizer, test_loader, use_cuda):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        #data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += loss_fn(output, target).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print(
        f'\nTest set: Average loss: {test_loss:.4f}, '
        f'Accuracy: {correct}/{len(test_loader.dataset)} '
        f'({acc:.0f}%)\n'
    )

    return acc


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
        # s = (x.max() - x.min()) / (self.thd_pos - self.thd_neg)
        # self.s = nn.Parameter(s)
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


class QALinear(nn.Module):
    def __init__(self, in_features, out_features, bit, only_positive_activations=False):
        super().__init__()
        self.bit = bit
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


class LinearInt(nn.Linear):
    def __init__(self, in_features, out_features, w_scale, q_a, int_dtype, device='cpu'):
        assert device == 'cpu', "LinearInt only supports cpu"
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
        q_out = super().forward(act_q.to(self.int_dtype))
        return q_out * (act_scale * self.w_scale)

    @classmethod
    def from_quantized(cls, quantized_fc: QALinear, int_dtype):
        in_features = quantized_fc.in_features
        out_features = quantized_fc.out_features
        weight_q, weight_scale = quantized_fc.quantizer_weight(quantized_fc.fc.weight.data)
        linear_int8 = cls(in_features, out_features, weight_scale, quantized_fc.quantizer_act, int_dtype)
        linear_int8.weight.data = weight_q.to(int_dtype)
        linear_int8.bias.data = quantized_fc.fc.bias.data.to(int_dtype)
        return linear_int8
