"""
Spatio-Temporal Relational Encoder (STRE)

Graph attention answers which EEG entities interact within a window.
A temporal Transformer answers how relational EEG states evolve across windows.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Relation-aware GAT layer.

    s_ij = a^T [W h_i || W h_j] + b^T e_ij
    alpha_ij = softmax_{j in N(i)} s_ij
    h_i' = sum_j alpha_ij W h_j
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 1,
        dropout: float = 0.1,
        edge_dim: int = 2
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.edge_dim = edge_dim

        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * self.head_dim, 1)))
        self.b = nn.Linear(edge_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
        nn.init.xavier_uniform_(self.b.weight)

    def forward(
        self,
        h: torch.Tensor,
        edge_mask: torch.Tensor,
        edge_attr: torch.Tensor
    ):
        """
        Args:
            h: (batch, num_nodes, in_dim)
            edge_mask: (batch, num_nodes, num_nodes), 1 = neighbor
            edge_attr: (batch, num_nodes, num_nodes, edge_dim)
        """
        batch_size, num_nodes, _ = h.shape

        Wh = self.W(h).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        Wh = Wh.permute(0, 2, 1, 3)  # (B, H, N, D)

        Wh1 = torch.matmul(Wh, self.a[:self.head_dim, :])  # (B, H, N, 1)
        Wh2 = torch.matmul(Wh, self.a[self.head_dim:, :])  # (B, H, N, 1)
        scores = Wh1 + Wh2.transpose(-1, -2)  # a^T [Wh_i || Wh_j]

        relation_bias = self.b(edge_attr).squeeze(-1).unsqueeze(1)  # (B, 1, N, N)
        scores = scores + relation_bias

        neighbor = edge_mask.unsqueeze(1) > 0
        scores = scores.masked_fill(~neighbor, -1e9)
        attention = F.softmax(scores, dim=-1)
        attention = attention.masked_fill(~neighbor, 0.0)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)  # (B, H, N, D)
        h_prime = h_prime.permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes, self.out_dim)
        return h_prime


class MultiLayerGAT(nn.Module):
    """Stacked relation-aware GAT with ELU after each layer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        edge_dim: int = 2
    ):
        super().__init__()
        self.num_layers = num_layers
        layers = []
        if num_layers == 1:
            layers.append(GraphAttentionLayer(input_dim, output_dim, num_heads, dropout, edge_dim))
        else:
            layers.append(GraphAttentionLayer(input_dim, hidden_dim, num_heads, dropout, edge_dim))
            for _ in range(num_layers - 2):
                layers.append(GraphAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout, edge_dim))
            layers.append(GraphAttentionLayer(hidden_dim, output_dim, num_heads, dropout, edge_dim))
        self.layers = nn.ModuleList(layers)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_mask, edge_attr):
        for layer in self.layers:
            h = self.activation(layer(h, edge_mask, edge_attr))
            h = self.dropout(h)
        return h


class AttentionReadout(nn.Module):
    """Attention pooling over nodes to one relational state g_w per window."""

    def __init__(self, node_dim: int, output_dim: int):
        super().__init__()
        self.W = nn.Linear(node_dim, output_dim)
        self.query = nn.Parameter(torch.randn(output_dim))
        self.activation = nn.Tanh()

    def forward(self, h: torch.Tensor):
        Wh = self.activation(self.W(h))
        scores = torch.matmul(Wh, self.query)
        attention = F.softmax(scores, dim=1)
        return torch.sum(attention.unsqueeze(-1) * Wh, dim=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        return x + self.pe[:, :x.size(1), :]


class STRE(nn.Module):
    """
    Within-window GAT, then optional across-window Transformer.
    """

    def __init__(
        self,
        node_dim: int,
        graph_embed_dim: int,
        num_gat_layers: int = 2,
        num_gat_heads: int = 4,
        gat_dropout: float = 0.1,
        num_temporal_layers: int = 4,
        num_temporal_heads: int = 8,
        temporal_ff_dim: int = 512,
        temporal_dropout: float = 0.1,
        max_seq_len: int = 512,
        edge_dim: int = 2
    ):
        super().__init__()
        self.node_dim = node_dim
        self.graph_embed_dim = graph_embed_dim
        self.use_temporal_transformer = num_temporal_layers > 0

        self.node_proj = (
            nn.Linear(node_dim, graph_embed_dim)
            if node_dim != graph_embed_dim
            else nn.Identity()
        )

        hidden_gat_dim = graph_embed_dim // 2 if num_gat_layers > 1 else graph_embed_dim
        self.gat = MultiLayerGAT(
            input_dim=graph_embed_dim,
            hidden_dim=hidden_gat_dim,
            output_dim=graph_embed_dim,
            num_layers=num_gat_layers,
            num_heads=num_gat_heads,
            dropout=gat_dropout,
            edge_dim=edge_dim
        )
        self.readout = AttentionReadout(graph_embed_dim, graph_embed_dim)

        if self.use_temporal_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=graph_embed_dim,
                nhead=num_temporal_heads,
                dim_feedforward=temporal_ff_dim,
                dropout=temporal_dropout,
                batch_first=True
            )
            self.temporal_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_temporal_layers
            )
            self.pos_encoding = PositionalEncoding(graph_embed_dim, max_seq_len)
            self.dropout = nn.Dropout(temporal_dropout)
        else:
            self.temporal_encoder = None
            self.pos_encoding = None
            self.dropout = None

    def forward(
        self,
        edge_mask: torch.Tensor,
        node_features: torch.Tensor,
        edge_attr: torch.Tensor,
        window_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            edge_mask: (B, W, N, N)
            node_features: (B, W, N, node_dim)
            edge_attr: (B, W, N, N, 2)
            window_mask: (B, W), 1 = valid window

        Returns:
            Z: (B, W, graph_embed_dim)
        """
        batch_size, num_windows, num_nodes, node_dim = node_features.shape
        if node_dim != self.node_dim:
            raise ValueError(
                f"Input node_dim {node_dim} does not match expected {self.node_dim}"
            )

        # GAT one window at a time so peak memory is (B, N, N), not (B*W, N, N).
        h = self.node_proj(node_features)
        window_embeds = []
        for w in range(num_windows):
            h_w = self.gat(h[:, w], edge_mask[:, w], edge_attr[:, w])
            window_embeds.append(self.readout(h_w))
        graph_embeds = torch.stack(window_embeds, dim=1)

        if not self.use_temporal_transformer:
            return graph_embeds

        graph_embeds = self.dropout(self.pos_encoding(graph_embeds))
        src_key_padding_mask = (window_mask == 0) if window_mask is not None else None
        return self.temporal_encoder(graph_embeds, src_key_padding_mask=src_key_padding_mask)
