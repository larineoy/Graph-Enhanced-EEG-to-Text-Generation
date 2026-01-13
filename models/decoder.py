"""
Transformer Decoder for EEG-to-Text Generation
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor):
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerDecoder(nn.Module):
    """
    Autoregressive Transformer Decoder for text generation
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        max_decoder_length: int = 128
    ):
        """
        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            num_layers: Number of decoder layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
            max_decoder_length: Maximum decoder sequence length
        """
        super(TransformerDecoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_decoder_length)
        
        # Decoder layers
        # Note: TransformerDecoder doesn't support batch_first, so we use default (seq_len, batch_size, embed_dim)
        # and manually transpose inputs/outputs
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=False  # TransformerDecoder expects (seq_len, batch_size, embed_dim)
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.max_decoder_length = max_decoder_length
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
        # Initialize output projection with small weights to prevent degenerate predictions
        # This helps prevent the model from always predicting the same token (e.g., comma)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None
    ):
        """
        Args:
            tgt: Target sequence (batch_size, tgt_len)
            memory: Encoder output / STRE embeddings (batch_size, src_len, embed_dim)
            tgt_mask: Attention mask for decoder self-attention
            tgt_key_padding_mask: Padding mask for target sequence
            
        Returns:
            output: Decoder output (batch_size, tgt_len, vocab_size)
        """
        # Embed tokens
        tgt_embeds = self.token_embedding(tgt)  # (batch_size, tgt_len, embed_dim)
        tgt_embeds = self.pos_encoding(tgt_embeds)
        tgt_embeds = self.dropout(tgt_embeds)
        
        # Decoder forward pass
        # TransformerDecoder expects (seq_len, batch_size, embed_dim) format
        memory_t = memory.transpose(0, 1)  # (src_len, batch_size, embed_dim)
        tgt_embeds_t = tgt_embeds.transpose(0, 1)  # (tgt_len, batch_size, embed_dim)
        
        # Ensure mask is correct shape for TransformerDecoder: (tgt_len, tgt_len)
        # After transposing, tgt_len is the first dimension
        expected_tgt_len = tgt_embeds_t.shape[0]
        
        if tgt_mask is not None:
            # Check if mask shape matches expected sequence length
            if len(tgt_mask.shape) == 2 and tgt_mask.shape[0] == expected_tgt_len and tgt_mask.shape[1] == expected_tgt_len:
                # Mask shape is correct, use it
                pass
            else:
                # Regenerate mask with correct sequence length
                tgt_mask = self.generate_mask(expected_tgt_len, tgt_embeds_t.device)
        else:
            # Generate causal mask if not provided
            tgt_mask = self.generate_mask(expected_tgt_len, tgt_embeds_t.device)
        
        decoder_output = self.decoder(
            tgt=tgt_embeds_t,
            memory=memory_t,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Transpose back to batch_first format
        decoder_output = decoder_output.transpose(0, 1)  # (batch_size, tgt_len, embed_dim)
        
        # Project to vocabulary
        output = self.output_proj(decoder_output)  # (batch_size, tgt_len, vocab_size)
        
        return output
    
    def generate_mask(self, sz: int, device: torch.device):
        """
        Generate causal mask for autoregressive decoding
        
        Args:
            sz: Sequence length
            device: Device to create mask on
            
        Returns:
            mask: Causal attention mask (sz, sz) where -inf means masked positions
        """
        # Create upper triangular mask (causal mask)
        # Positions with value 1 (upper triangle) should be masked (set to -inf)
        # Positions with value 0 (lower triangle including diagonal) can attend (set to 0)
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

