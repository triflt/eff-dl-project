"""
LSQ (Learned Step Size Quantization) utilities for model quantization.
Based on the paper: "Learned Step Size Quantization" (https://arxiv.org/abs/1902.08153)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Quantizer(nn.Module):
    """
    LSQ Quantizer that learns the quantization scale parameter during training.
    
    Args:
        bit: Number of bits for quantization (e.g., 2, 4, 8, etc.)
    """
    def __init__(self, bit):
        super(Quantizer, self).__init__()
        self.bit = bit
        self.thd_neg = -(2 ** (bit - 1))
        self.thd_pos = 2 ** (bit - 1) - 1
        self.s = nn.Parameter(torch.ones(1))

    def init_from(self, x):
        """Initialize scale parameter from input tensor statistics."""
        s = (x.max() - x.min()) / (self.thd_pos - self.thd_neg)
        self.s = nn.Parameter(s)

    def skip_grad_scale(self, x, scale):
        """Apply gradient scaling without modifying forward pass."""
        y = x
        y_grad = x * scale
        return (y - y_grad).detach() + y_grad

    def round_pass(self, x):
        """Round operation with straight-through estimator."""
        y = x.round()
        y_grad = x
        return (y - y_grad).detach() + y_grad

    def forward(self, x):
        """
        Forward pass with quantization.
        If bit >= 32, no quantization is applied.
        """
        if self.bit >= 32:
            return x

        # Gradient scale factor for the scale parameter (empirical)
        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        device = x.device

        s_scale = self.skip_grad_scale(self.s, s_grad_scale).to(device)

        # Quantize
        x = x / s_scale
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = self.round_pass(x)

        # Dequantize
        x = x * s_scale
        return x


class QAConv1d(nn.Module):
    """
    Quantized 1D Convolution layer with activation and weight quantization.
    
    Args:
        bit: Number of bits for quantization
        ch_in: Input channels
        ch_out: Output channels
        kernel_size: Kernel size for convolution
        stride: Stride for convolution
        padding: Padding for convolution
        bias: Whether to include bias
    """
    def __init__(self, bit, ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=False):
        super(QAConv1d, self).__init__()
        self.bit = bit
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv1d(ch_in, ch_out, kernel_size, stride, padding, bias=bias)
        self.define_q_functions(self.bit)

    def define_q_functions(self, bit):
        """Initialize quantizers for activations and weights."""
        self.quantizer_act = Quantizer(bit)
        self.quantizer_weight = Quantizer(bit)
        self.quantizer_weight.init_from(self.conv.weight)

    def forward(self, input_x):
        """Forward pass with quantized weights and activations."""
        quantized_weight = self.quantizer_weight(self.conv.weight)
        quantized_act = self.quantizer_act(input_x)
        out = F.conv1d(quantized_act, quantized_weight, self.conv.bias, 
                       self.stride, self.padding)
        return out


class QALinear(nn.Module):
    """
    Quantized Linear layer with activation, weight, and bias quantization.
    
    Args:
        bit: Number of bits for quantization
        ch_in: Input features
        ch_out: Output features
        bias: Whether to include bias
    """
    def __init__(self, bit, ch_in, ch_out, bias=True):
        super(QALinear, self).__init__()
        self.bit = bit
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.fc = nn.Linear(ch_in, ch_out, bias=bias)
        self.define_q_functions(self.bit)

    def define_q_functions(self, bit):
        """Initialize quantizers for activations, weights, and bias."""
        self.quantizer_act = Quantizer(bit)
        self.quantizer_weight = Quantizer(bit)
        self.quantizer_weight.init_from(self.fc.weight)
        
        if self.fc.bias is not None:
            self.quantizer_bias = Quantizer(bit)

    def forward(self, input_x):
        """Forward pass with quantized weights, activations, and bias."""
        quantized_weight = self.quantizer_weight(self.fc.weight)
        quantized_act = self.quantizer_act(input_x)
        
        if self.fc.bias is not None:
            quantized_bias = self.quantizer_bias(self.fc.bias)
        else:
            quantized_bias = None
            
        out = F.linear(quantized_act, quantized_weight, bias=quantized_bias)
        return out


class QAEmbedding(nn.Module):
    """
    Quantized Embedding layer with weight quantization.
    
    Args:
        bit: Number of bits for quantization
        num_embeddings: Size of the dictionary of embeddings
        embedding_dim: The size of each embedding vector
        padding_idx: If given, pads the output with the embedding vector at padding_idx
    """
    def __init__(self, bit, num_embeddings, embedding_dim, padding_idx=None):
        super(QAEmbedding, self).__init__()
        self.bit = bit
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.define_q_functions(self.bit)

    def define_q_functions(self, bit):
        """Initialize quantizer for embedding weights."""
        self.quantizer_weight = Quantizer(bit)
        self.quantizer_weight.init_from(self.embedding.weight)

    def forward(self, input_ids):
        """Forward pass with quantized embedding weights."""
        quantized_weight = self.quantizer_weight(self.embedding.weight)
        out = F.embedding(input_ids, quantized_weight, self.padding_idx)
        return out


class QALayerNorm(nn.Module):
    """
    Quantized LayerNorm with activation quantization.
    
    Args:
        bit: Number of bits for quantization
        normalized_shape: Input shape from an expected input
        eps: A value added to the denominator for numerical stability
    """
    def __init__(self, bit, normalized_shape, eps=1e-8):
        super(QALayerNorm, self).__init__()
        self.bit = bit
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)
        self.define_q_functions(self.bit)

    def define_q_functions(self, bit):
        """Initialize quantizer for output activations."""
        self.quantizer_act = Quantizer(bit)

    def forward(self, input_x):
        """Forward pass with quantized output."""
        out = self.layer_norm(input_x)
        quantized_out = self.quantizer_act(out)
        return quantized_out


class QAMultiheadAttention(nn.Module):
    """
    Quantized MultiheadAttention with activation and weight quantization.
    
    Args:
        bit: Number of bits for quantization
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        dropout: Dropout probability
    """
    def __init__(self, bit, embed_dim, num_heads, dropout=0.0):
        super(QAMultiheadAttention, self).__init__()
        self.bit = bit
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.define_q_functions(self.bit)

    def define_q_functions(self, bit):
        """Initialize quantizers for input activations and output."""
        self.quantizer_input = Quantizer(bit)
        self.quantizer_output = Quantizer(bit)

    def forward(self, query, key, value, attn_mask=None):
        """Forward pass with quantized inputs and outputs."""
        # Quantize inputs
        q_query = self.quantizer_input(query)
        q_key = self.quantizer_input(key)
        q_value = self.quantizer_input(value)
        
        # Apply attention
        out, attn_weights = self.mha(q_query, q_key, q_value, attn_mask=attn_mask)
        
        # Quantize output
        quantized_out = self.quantizer_output(out)
        return quantized_out, attn_weights

