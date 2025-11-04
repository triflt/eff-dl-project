import torch
import torch.nn as nn


bits_to_dtype = {
    8: torch.int8,
    16: torch.int16,
    32: torch.int32,
}


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # One affine that produces all 4 gates at once
        self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier/Glorot for input part, orthogonal-ish behaviour for hidden part is optional.
        # A simple, solid default:
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
            # Encourage remembering at init (forget bias > 0)
            with torch.no_grad():
                # bias layout: [i | f | g | o]
                self.linear.bias[self.hidden_size:2*self.hidden_size].fill_(1.0)

    def forward(self, x_t, state):
        """
        x_t: (B, input_size)
        state: (h_{t-1}, c_{t-1}) where each is (B, hidden_size)
        returns (h_t, c_t)
        """
        h_prev, c_prev = state
        concat = torch.cat([x_t, h_prev], dim=-1)           # (B, input+hidden)
        gates = self.linear(concat)                         # (B, 4H)
        i, f, g, o = gates.chunk(4, dim=-1)                 # each (B, H)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t

    def to_qat(self, bits: int, qat_linear_class, **qat_kwargs) -> "LSTMCell":
        new_model = LSTMCell(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            bias=self.linear.bias is not None,
        )
        new_model.linear = qat_linear_class.from_linear(self.linear, bits, **qat_kwargs)
        return new_model

    def quantize(self, bits: int, linear_int_class) -> "LSTMCell":
        int_dtype = bits_to_dtype[bits]
        new_model = LSTMCell(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            bias=self.linear.fc.bias is not None,
        )
        new_model.linear = linear_int_class.from_qat(self.linear, int_dtype)
        return new_model


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, batch_first: bool = True):
        super().__init__()
        self.batch_first = batch_first
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size, bias=bias)

    def forward(self, x, hx=None):
        """
        x: (B, T, D) if batch_first else (T, B, D)
        hx: optional ((B, H), (B, H)) initial (h0, c0)
        returns:
          outputs: (B, T, H) if batch_first else (T, B, H)
          (h_T, c_T): each (B, H)
        """
        if not self.batch_first:
            # convert to batch_first for simpler looping
            x = x.transpose(0, 1)  # (T, B, D) -> (B, T, D)

        B, T, D = x.shape
        if hx is None:
            h_t = x.new_zeros(B, self.hidden_size)
            c_t = x.new_zeros(B, self.hidden_size)
        else:
            h_t, c_t = hx

        outputs = []
        for t in range(T):
            h_t, c_t = self.cell(x[:, t, :], (h_t, c_t))
            outputs.append(h_t.unsqueeze(1))  # (B, 1, H)

        outputs = torch.cat(outputs, dim=1)    # (B, T, H)

        if not self.batch_first:
            outputs = outputs.transpose(0, 1)  # back to (T, B, H)

        return outputs, (h_t, c_t)

    def to_qat(self, bits: int, qat_linear_class, **qat_kwargs) -> "LSTM":
        new_model = LSTM(
            input_size=self.cell.input_size,
            hidden_size=self.cell.hidden_size,
            bias=self.cell.linear.bias is not None,
            batch_first=self.batch_first,
        )
        new_model.cell = self.cell.to_qat(bits, qat_linear_class, **qat_kwargs)
        return new_model

    def quantize(self, bits: int, linear_int_class) -> "LSTM":
        new_model = LSTM(
            input_size=self.cell.input_size,
            hidden_size=self.cell.hidden_size,
            bias=self.cell.linear.fc.bias is not None,
            batch_first=self.batch_first,
        )
        new_model.cell = self.cell.quantize(bits, linear_int_class)
        return new_model
