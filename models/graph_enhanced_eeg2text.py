"""
Graph-Enhanced EEG-to-Text model: STRG → STRE → pretrained BART decoder.
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .decoder import BartEEGDecoder
from .strg import STRG
from .stre import STRE


class GraphEnhancedEEG2Text(nn.Module):
    """
    STRG constructs per-window relational graphs.
    STRE produces EEG memory M.
    A pretrained BART decoder generates text from M.
    """

    def __init__(
        self,
        num_channels: int = 64,
        num_frequency_bands: int = 5,
        sampling_rate: float = 250.0,
        k_spatial: int = 6,
        k_functional: int = 6,
        use_spatial_topology: bool = True,
        use_functional_connectivity: bool = True,
        use_frequency_nodes: bool = True,
        dynamic_functional: bool = True,
        node_dim: int = 1,
        graph_embed_dim: int = 256,
        num_gat_layers: int = 2,
        num_gat_heads: int = 4,
        gat_dropout: float = 0.1,
        num_temporal_layers: int = 4,
        num_temporal_heads: int = 8,
        temporal_ff_dim: int = 512,
        temporal_dropout: float = 0.1,
        decoder_pretrained_name: str = 'facebook/bart-base',
        max_decoder_length: int = 128,
        device: str = 'cuda',
        **deprecated
    ):
        super().__init__()
        del deprecated
        self.num_channels = num_channels
        self.num_frequency_bands = num_frequency_bands
        self.sampling_rate = sampling_rate
        self.device = device
        self.max_decoder_length = max_decoder_length
        self.use_frequency_nodes = use_frequency_nodes

        if not use_frequency_nodes:
            node_dim = num_frequency_bands

        self.strg = STRG(
            num_channels=num_channels,
            num_frequency_bands=num_frequency_bands,
            k_spatial=k_spatial,
            k_functional=k_functional,
            use_spatial_topology=use_spatial_topology,
            use_functional_connectivity=use_functional_connectivity,
            use_frequency_nodes=use_frequency_nodes,
            dynamic_functional=dynamic_functional,
            device=device,
            electrode_positions=None
        )
        node_dim = self.strg.node_dim

        self.stre = STRE(
            node_dim=node_dim,
            graph_embed_dim=graph_embed_dim,
            num_gat_layers=num_gat_layers,
            num_gat_heads=num_gat_heads,
            gat_dropout=gat_dropout,
            num_temporal_layers=num_temporal_layers,
            num_temporal_heads=num_temporal_heads,
            temporal_ff_dim=temporal_ff_dim,
            temporal_dropout=temporal_dropout
        )

        self.decoder = BartEEGDecoder(pretrained_name=decoder_pretrained_name)
        self.stre_proj = nn.Linear(graph_embed_dim, self.decoder.d_model)
        self.memory_norm = nn.LayerNorm(self.decoder.d_model)

    def set_electrode_positions(self, electrode_positions: torch.Tensor):
        if electrode_positions is None:
            return
        if isinstance(electrode_positions, np.ndarray):
            electrode_positions = torch.from_numpy(electrode_positions).float()
        self.strg.set_electrode_positions(electrode_positions)

    def encode(
        self,
        eeg_bands: Dict[str, torch.Tensor],
        window_mask: Optional[torch.Tensor] = None,
        eeg_bands_full: Optional[Dict[str, torch.Tensor]] = None,
        eeg_windows: Optional[torch.Tensor] = None
    ):
        strg_out = self.strg(
            eeg_bands,
            eeg_bands_full=eeg_bands_full,
            eeg_windows=eeg_windows
        )
        stre_embeds = self.stre(
            strg_out['edge_mask'],
            strg_out['node_features'],
            strg_out['edge_attr'],
            window_mask=window_mask
        )
        memory = self.memory_norm(self.stre_proj(stre_embeds))
        encoder_attention_mask = window_mask
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                memory.shape[:2], device=memory.device, dtype=memory.dtype
            )
        return memory, encoder_attention_mask, strg_out, stre_embeds

    def forward(
        self,
        eeg_bands: Dict[str, torch.Tensor],
        tgt_tokens: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        window_mask: Optional[torch.Tensor] = None,
        eeg_bands_full: Optional[Dict[str, torch.Tensor]] = None,
        eeg_windows: Optional[torch.Tensor] = None
    ):
        memory, encoder_attention_mask, strg_out, stre_embeds = self.encode(
            eeg_bands,
            window_mask=window_mask,
            eeg_bands_full=eeg_bands_full,
            eeg_windows=eeg_windows
        )

        extras = {
            'stre_embeds': stre_embeds,
            'memory': memory,
            'encoder_attention_mask': encoder_attention_mask,
            'node_features': strg_out.get('node_features')
        }

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
        eeg_bands_full: Optional[Dict[str, torch.Tensor]] = None,
        eeg_windows: Optional[torch.Tensor] = None,
        **kwargs
    ):
        del bos_token_id, eos_token_id, pad_token_id
        self.eval()
        memory, encoder_attention_mask, _, _ = self.encode(
            eeg_bands,
            window_mask=window_mask,
            eeg_bands_full=eeg_bands_full,
            eeg_windows=eeg_windows
        )
        return self.decoder.generate(
            memory=memory,
            encoder_attention_mask=encoder_attention_mask,
            max_length=max_length,
            num_beams=beam_size,
            **kwargs
        )
