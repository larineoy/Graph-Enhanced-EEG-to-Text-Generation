"""
Spatio-Temporal Relational Embeddings (STRE)
Implements graph encoding with GAT and temporal modeling with Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) as described in Velickovic et al. 2018
    """
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 1, dropout: float = 0.1):
        super(GraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * self.head_dim, 1)))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, h: torch.Tensor, A: torch.Tensor):
        """
        Args:
            h: Node features of shape (batch_size, num_nodes, in_dim)
            A: Adjacency matrix of shape (batch_size, num_nodes, num_nodes)
            
        Returns:
            h_out: Updated node features of shape (batch_size, num_nodes, out_dim)
        """
        batch_size, num_nodes, _ = h.shape
        
        # Linear transformation
        Wh = self.W(h)  # (batch_size, num_nodes, out_dim)
        
        # Reshape for multi-head attention
        Wh = Wh.view(batch_size, num_nodes, self.num_heads, self.head_dim)
        Wh = Wh.permute(2, 0, 1, 3)  # (num_heads, batch_size, num_nodes, head_dim)
        
        # Compute attention scores
        # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        Wh1 = torch.matmul(Wh, self.a[:self.head_dim, :])  # (num_heads, batch_size, num_nodes, 1)
        Wh2 = torch.matmul(Wh, self.a[self.head_dim:, :])  # (num_heads, batch_size, num_nodes, 1)
        
        e = Wh1 + Wh2.transpose(2, 3)  # (num_heads, batch_size, num_nodes, num_nodes)
        e = self.leaky_relu(e)
        
        # Mask attention with adjacency matrix
        A_expanded = A.unsqueeze(0)  # (1, batch_size, num_nodes, num_nodes)
        attention_mask = -9e15 * torch.ones_like(e)
        e = torch.where(A_expanded > 0, e, attention_mask)
        
        # Compute attention weights
        attention = F.softmax(e, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to nodes
        h_prime = torch.matmul(attention, Wh)  # (num_heads, batch_size, num_nodes, head_dim)
        h_prime = h_prime.permute(1, 2, 0, 3)  # (batch_size, num_nodes, num_heads, head_dim)
        h_prime = h_prime.contiguous().view(batch_size, num_nodes, self.out_dim)
        
        return h_prime


class MultiLayerGAT(nn.Module):
    """Multi-layer Graph Attention Network"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super(MultiLayerGAT, self).__init__()
        self.num_layers = num_layers
        
        layers = []
        if num_layers == 1:
            layers.append(
                GraphAttentionLayer(input_dim, output_dim, num_heads, dropout)
            )
        else:
            # Input layer
            layers.append(
                GraphAttentionLayer(input_dim, hidden_dim, num_heads, dropout)
            )
            # Hidden layers
            for _ in range(num_layers - 2):
                layers.append(
                    GraphAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout)
                )
            # Output layer
            layers.append(
                GraphAttentionLayer(hidden_dim, output_dim, num_heads, dropout)
            )
        
        self.layers = nn.ModuleList(layers)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h: torch.Tensor, A: torch.Tensor):
        """
        Args:
            h: Node features (batch_size, num_nodes, input_dim)
            A: Adjacency matrix (batch_size, num_nodes, num_nodes)
            
        Returns:
            h_out: Graph embeddings (batch_size, num_nodes, output_dim)
        """
        for i, layer in enumerate(self.layers):
            h = layer(h, A)
            if i < len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


class AttentionReadout(nn.Module):
    """Attention-based graph-level readout"""
    
    def __init__(self, node_dim: int, output_dim: int):
        super(AttentionReadout, self).__init__()
        self.W = nn.Linear(node_dim, output_dim)
        self.query = nn.Parameter(torch.randn(output_dim))
        self.activation = nn.Tanh()
    
    def forward(self, h: torch.Tensor):
        """
        Args:
            h: Node embeddings (batch_size, num_nodes, node_dim)
            
        Returns:
            graph_embed: Graph-level embedding (batch_size, output_dim)
        """
        # Compute attention weights
        Wh = self.activation(self.W(h))  # (batch_size, num_nodes, output_dim)
        scores = torch.matmul(Wh, self.query)  # (batch_size, num_nodes)
        attention = F.softmax(scores, dim=1)  # (batch_size, num_nodes)
        
        # Weighted aggregation
        graph_embed = torch.sum(attention.unsqueeze(-1) * h, dim=1)  # (batch_size, node_dim)
        
        return graph_embed


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class STRE(nn.Module):
    """
    Spatio-Temporal Relational Embeddings (STRE)
    
    Generates graph-aware representations by:
    1. Encoding STRG with GAT layers
    2. Aggregating node embeddings to graph-level representations
    3. Modeling temporal dependencies with Transformer encoder
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
        max_seq_len: int = 512
    ):
        """
        Args:
            node_dim: Input node feature dimension
            graph_embed_dim: Output graph embedding dimension
            num_gat_layers: Number of GAT layers
            num_gat_heads: Number of attention heads in GAT
            gat_dropout: Dropout rate for GAT
            num_temporal_layers: Number of Transformer encoder layers
            num_temporal_heads: Number of attention heads in Transformer
            temporal_ff_dim: Feed-forward dimension in Transformer
            temporal_dropout: Dropout rate for Transformer
            max_seq_len: Maximum sequence length for positional encoding
        """
        super(STRE, self).__init__()
        
        # Project node features if needed
        if node_dim != graph_embed_dim:
            self.node_proj = nn.Linear(node_dim, graph_embed_dim)
        else:
            self.node_proj = nn.Identity()
        
        # GAT layers
        hidden_gat_dim = graph_embed_dim // 2 if num_gat_layers > 1 else graph_embed_dim
        self.gat = MultiLayerGAT(
            input_dim=node_dim,
            hidden_dim=hidden_gat_dim,
            output_dim=graph_embed_dim,
            num_layers=num_gat_layers,
            num_heads=num_gat_heads,
            dropout=gat_dropout
        )
        
        # Graph-level readout
        self.readout = AttentionReadout(graph_embed_dim, graph_embed_dim)
        
        # Temporal Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=graph_embed_dim,
            nhead=num_temporal_heads,
            dim_feedforward=temporal_ff_dim,
            dropout=temporal_dropout,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_temporal_layers)
        self.pos_encoding = PositionalEncoding(graph_embed_dim, max_seq_len)
        
        self.dropout = nn.Dropout(temporal_dropout)
    
    def forward(self, A: torch.Tensor, node_features: torch.Tensor):
        """
        Args:
            A: Adjacency matrices (batch_size, num_windows, num_nodes, num_nodes)
            node_features: Node features (batch_size, num_windows, num_nodes, node_dim)
            
        Returns:
            STRE: Spatio-temporal relational embeddings (batch_size, num_windows, graph_embed_dim)
        """
        batch_size, num_windows, num_nodes, node_dim = node_features.shape
        graph_embed_dim = self.readout.query.shape[0]
        
        # Reshape for batch processing
        A_flat = A.view(-1, num_nodes, num_nodes)  # (batch_size * num_windows, num_nodes, num_nodes)
        h_flat = node_features.view(-1, num_nodes, node_dim)  # (batch_size * num_windows, num_nodes, node_dim)
        
        # Graph encoding with GAT
        h_gat = self.gat(h_flat, A_flat)  # (batch_size * num_windows, num_nodes, graph_embed_dim)
        
        # Graph-level readout
        graph_embeds = self.readout(h_gat)  # (batch_size * num_windows, graph_embed_dim)
        
        # Reshape back to windows
        graph_embeds = graph_embeds.view(batch_size, num_windows, graph_embed_dim)
        
        # Temporal encoding with Transformer
        graph_embeds = self.pos_encoding(graph_embeds)
        graph_embeds = self.dropout(graph_embeds)
        
        # Transformer expects (seq_len, batch, dim) or (batch, seq_len, dim) with batch_first=True
        STRE_embeds = self.temporal_encoder(graph_embeds)  # (batch_size, num_windows, graph_embed_dim)
        
        return STRE_embeds

