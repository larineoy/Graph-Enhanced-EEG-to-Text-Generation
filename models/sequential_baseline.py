"""
No-graph ablation: the same windowed frequency features without relational structure.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .decoder import BartEEGDecoder


class SequentialEEG2Text(nn.Module):
    """
    Sequence encoder over windowed bandpower vectors, then the same pretrained
    BART decoder used by the full model.
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
        decoder_pretrained_name: str = 'facebook/bart-base',
        max_decoder_length: int = 128,
        device: str = 'cuda',
        **deprecated
    ):
        super().__init__()
        del deprecated
        self.num_channels = num_channels
        self.num_frequency_bands = num_frequency_bands
        self.device = device
        self.max_decoder_length = max_decoder_length

        self.input_dim = num_channels * num_frequency_bands
        self.band_proj = nn.Linear(self.input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = BartEEGDecoder(pretrained_name=decoder_pretrained_name)
        self.memory_proj = nn.Linear(embed_dim, self.decoder.d_model)
        self.memory_norm = nn.LayerNorm(self.decoder.d_model)

    def _windowed_features(self, eeg_bands: Dict[str, torch.Tensor]) -> torch.Tensor:
        bands = []
        for name in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            if name not in eeg_bands:
                continue
            x = eeg_bands[name]
            if x.dim() == 3:
                x = x.unsqueeze(1)
            bands.append(torch.var(x, dim=-1))
        if not bands:
            raise ValueError("No frequency bands found in eeg_bands")
        stacked = torch.stack(bands, dim=-1)  # (B, W, C, F)
        return stacked.reshape(stacked.shape[0], stacked.shape[1], -1)

    def encode(
        self,
        eeg_bands: Dict[str, torch.Tensor],
        window_mask: Optional[torch.Tensor] = None
    ):
        features = self._windowed_features(eeg_bands)
        encoded = self.encoder(
            self.band_proj(features),
            src_key_padding_mask=(window_mask == 0) if window_mask is not None else None
        )
        memory = self.memory_norm(self.memory_proj(encoded))
        if window_mask is None:
            window_mask = torch.ones(memory.shape[:2], device=memory.device, dtype=memory.dtype)
        return memory, window_mask

    def forward(
        self,
        eeg_bands: Dict[str, torch.Tensor],
        tgt_tokens: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        window_mask: Optional[torch.Tensor] = None,
        **unused
    ):
        del unused
        memory, encoder_attention_mask = self.encode(eeg_bands, window_mask)
        extras = {'memory': memory, 'encoder_attention_mask': encoder_attention_mask}
        if tgt_tokens is None:
            return memory, extras

        logits = self.decoder(
            tgt=tgt_tokens,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            encoder_attention_mask=encoder_attention_mask,
            tgt_mask=tgt_mask
        )
        return logits, extras

    def generate(
        self,
        eeg_bands: Dict[str, torch.Tensor],
        bos_token_id: int = None,
        eos_token_id: int = None,
        pad_token_id: int = None,
        max_length: int = 128,
        beam_size: int = 5,
        window_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        del bos_token_id, eos_token_id, pad_token_id, kwargs
        self.eval()
        memory, encoder_attention_mask = self.encode(eeg_bands, window_mask)
        return self.decoder.generate(
            memory=memory,
            encoder_attention_mask=encoder_attention_mask,
            max_length=max_length,
            num_beams=beam_size
        )
