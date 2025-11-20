"""
Sequential baseline model for ablation study (no graph structure)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from .decoder import TransformerDecoder


class SequentialEEG2Text(nn.Module):
    """
    Sequential baseline model without graph structure
    Averages frequency bands and uses Transformer encoder-decoder
    """
    
    def __init__(
        self,
        num_channels: int,
        num_frequency_bands: int = 5,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        vocab_size: int = 10000,
        max_decoder_length: int = 128,
        device: str = 'cuda'
    ):
        super(SequentialEEG2Text, self).__init__()
        
        self.num_channels = num_channels
        self.num_frequency_bands = num_frequency_bands
        self.device = device
        
        # Project frequency bands to embedding dimension
        self.band_proj = nn.Linear(num_channels, embed_dim)
        
        # Transformer encoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Project to decoder dimension
        self.encoder_proj = nn.Linear(embed_dim, embed_dim)
        
        # Transformer decoder
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            max_decoder_length=max_decoder_length
        )
    
    def forward(
        self,
        eeg_bands: Dict[str, torch.Tensor],
        tgt_tokens: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            eeg_bands: Dictionary of frequency bands
            tgt_tokens: Target tokens for training
            tgt_mask: Causal mask
        """
        batch_size = list(eeg_bands.values())[0].shape[0]
        
        # Average across frequency bands: (batch, C, T) -> (batch, T, C)
        band_list = []
        for band_name in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            if band_name in eeg_bands:
                band_data = eeg_bands[band_name]  # (batch, C, T)
                band_list.append(band_data)
        
        # Stack and average: (batch, num_bands, C, T) -> (batch, C, T)
        if len(band_list) > 0:
            all_bands = torch.stack(band_list, dim=1)  # (batch, num_bands, C, T)
            avg_eeg = torch.mean(all_bands, dim=1)  # (batch, C, T)
        else:
            raise ValueError("No frequency bands found in eeg_bands")
        
        # Transpose to (batch, T, C) for sequence processing
        avg_eeg = avg_eeg.transpose(1, 2)  # (batch, T, C)
        
        # Project to embedding dimension
        eeg_embed = self.band_proj(avg_eeg)  # (batch, T, embed_dim)
        
        # Encode with Transformer
        memory = self.encoder(eeg_embed)  # (batch, T, embed_dim)
        memory = self.encoder_proj(memory)  # (batch, T, embed_dim)
        
        # Average over time to get single representation
        memory = memory.mean(dim=1, keepdim=True)  # (batch, 1, embed_dim)
        
        # Decode to text
        if tgt_tokens is not None:
            tgt_input = tgt_tokens[:, :-1]
            if tgt_mask is None:
                tgt_len = tgt_input.shape[1]
                tgt_mask = self.decoder.generate_mask(tgt_len, self.device)
            
            logits = self.decoder(
                tgt=tgt_input,
                memory=memory,
                tgt_mask=tgt_mask
            )
            
            return logits, {
                'memory': memory,
                'eeg_embed': eeg_embed
            }
        else:
            return memory, {
                'memory': memory,
                'eeg_embed': eeg_embed
            }
    
    def generate(
        self,
        eeg_bands: Dict[str, torch.Tensor],
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        max_length: int = 128,
        beam_size: int = 5
    ):
        """Generate text from EEG"""
        self.eval()
        device = list(eeg_bands.values())[0].device
        batch_size = list(eeg_bands.values())[0].shape[0]
        
        memory, _ = self.forward(eeg_bands)
        
        generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for _ in range(max_length - 1):
                tgt_mask = self.decoder.generate_mask(generated.shape[1], device)
                logits = self.decoder(generated, memory, tgt_mask)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                if (next_token == eos_token_id).all():
                    break
        
        return generated

