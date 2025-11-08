"""
Quantized SASRec model with LSQ (Learned Step Size Quantization).
"""

import numpy as np
import torch
from lsq_utils import (
    QAConv1d,
    QALinear,
    QAEmbedding,
    QALayerNorm,
    QAMultiheadAttention,
    Quantizer
)


class QAPointWiseFeedForward(torch.nn.Module):
    """
    Quantized Point-wise Feed-Forward Network with LSQ quantization.
    
    Args:
        bit: Number of bits for quantization
        hidden_units: Number of hidden units
        dropout_rate: Dropout rate
    """
    def __init__(self, bit, hidden_units, dropout_rate):
        super(QAPointWiseFeedForward, self).__init__()
        
        self.bit = bit
        self.conv1 = QAConv1d(bit, hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = QAConv1d(bit, hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(
                self.relu(
                    self.dropout1(
                        self.conv1(inputs.transpose(-1, -2))
                    )
                )
            )
        )
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        return outputs


class SASRecLSQ(torch.nn.Module):
    """
    Quantized SASRec model with LSQ quantization.
    
    Args:
        user_num: Number of users
        item_num: Number of items
        args: Arguments object containing model hyperparameters
        bit: Number of bits for quantization (default: 8)
    """
    def __init__(self, user_num, item_num, args, bit=8):
        super(SASRecLSQ, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.bit = bit

        # Quantized embeddings
        self.item_emb = QAEmbedding(bit, self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = QAEmbedding(bit, args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # Quantized layers
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = QALayerNorm(bit, args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = QALayerNorm(bit, args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = QAMultiheadAttention(
                bit,
                args.hidden_units,
                args.num_heads,
                args.dropout_rate
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = QALayerNorm(bit, args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = QAPointWiseFeedForward(bit, args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs):
        """
        Convert log sequences to feature representations.
        
        Args:
            log_seqs: Input sequences
            
        Returns:
            log_feats: Feature representations
        """
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        """
        Forward pass for training.
        
        Args:
            user_ids: User IDs
            log_seqs: Log sequences
            pos_seqs: Positive sequences
            neg_seqs: Negative sequences
            
        Returns:
            pos_logits: Positive logits
            neg_logits: Negative logits
        """
        log_feats = self.log2feats(log_seqs)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        """
        Prediction for inference.
        
        Args:
            user_ids: User IDs
            log_seqs: Log sequences
            item_indices: Item indices to predict
            
        Returns:
            logits: Prediction logits
        """
        log_feats = self.log2feats(log_seqs)

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits  # (U, I)

