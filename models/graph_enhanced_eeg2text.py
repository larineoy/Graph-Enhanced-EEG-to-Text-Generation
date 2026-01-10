"""
Graph-Enhanced EEG-to-Text Decoding Model
Main model that integrates STRG, STRE, and Transformer Decoder

Uses preprocessed eeg_bands from preprocessing pipeline.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from .strg import STRG
from .stre import STRE
from .decoder import TransformerDecoder


class GraphEnhancedEEG2Text(nn.Module):
    """
    Graph-Enhanced EEG-to-Text Decoding Framework
    
    Integrates:
    1. STRG: Spectro-Topographic Relational Graph construction
    2. STRE: Spatio-Temporal Relational Embeddings
    3. Transformer Decoder: Text generation
    """
    
    def __init__(
        self,
        # EEG parameters
        num_channels: int = 64,
        num_frequency_bands: int = 5,
        sampling_rate: float = 250.0,  # Not used anymore but kept for compatibility
        
        # STRG parameters
        strg_alpha: float = 0.5,
        strg_beta: float = 0.5,
        use_spatial_topology: bool = True,
        use_functional_connectivity: bool = True,
        
        # STRE parameters
        node_dim: int = 1,  # Bandpower is scalar feature per node
        graph_embed_dim: int = 256,
        num_gat_layers: int = 2,
        num_gat_heads: int = 4,
        gat_dropout: float = 0.1,
        num_temporal_layers: int = 4,
        num_temporal_heads: int = 8,
        temporal_ff_dim: int = 512,
        temporal_dropout: float = 0.1,
        
        # Decoder parameters
        vocab_size: int = 10000,
        decoder_embed_dim: int = 256,
        num_decoder_layers: int = 4,
        num_decoder_heads: int = 8,
        decoder_ff_dim: int = 512,
        decoder_dropout: float = 0.1,
        max_decoder_length: int = 128,
        
        # Device
        device: str = 'cuda'
    ):
        super(GraphEnhancedEEG2Text, self).__init__()
        
        self.num_channels = num_channels
        self.num_frequency_bands = num_frequency_bands
        self.sampling_rate = sampling_rate
        self.device = device
        
        # STRG construction
        # electrode_positions will be set after model creation if available from ZuCo
        self.strg = STRG(
            num_channels=num_channels,
            num_frequency_bands=num_frequency_bands,
            alpha=strg_alpha,
            beta=strg_beta,
            use_spatial_topology=use_spatial_topology,
            use_functional_connectivity=use_functional_connectivity,
            device=device,
            electrode_positions=None  # Will be set from dataset if available
        )
        
        # STRE generation
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
        
        # Project STRE embeddings to decoder dimension if needed
        if graph_embed_dim != decoder_embed_dim:
            self.stre_proj = nn.Linear(graph_embed_dim, decoder_embed_dim)
        else:
            self.stre_proj = nn.Identity()
        
        # Transformer decoder
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=decoder_embed_dim,
            num_layers=num_decoder_layers,
            num_heads=num_decoder_heads,
            ff_dim=decoder_ff_dim,
            dropout=decoder_dropout,
            max_decoder_length=max_decoder_length
        )
        
        # Text encoder for contrastive loss
        # Uses shared token embeddings with decoder, then aggregates to sentence-level embedding
        self.text_encoder = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
            nn.ReLU(),
            nn.Dropout(decoder_dropout),
            nn.Linear(decoder_embed_dim, graph_embed_dim)  # Match graph_embed_dim for contrastive loss
        )
    
    def set_electrode_positions(self, electrode_positions: torch.Tensor):
        """
        Set electrode positions from ZuCo dataset chanlocs.
        Rebuilds spatial adjacency matrix with actual positions.
        
        Args:
            electrode_positions: Tensor of shape (num_channels, 3) with X, Y, Z coordinates
        """
        if electrode_positions is not None:
            # Convert to tensor if numpy array
            if isinstance(electrode_positions, np.ndarray):
                electrode_positions = torch.from_numpy(electrode_positions).float()
            
            # Update STRG with actual positions
            self.strg.electrode_positions = electrode_positions
            # Rebuild spatial adjacency with actual positions
            self.strg.register_buffer('A_spatial', self.strg._build_spatial_adjacency())
    
    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode text tokens to embeddings for contrastive loss.
        
        Args:
            text_tokens: Token IDs of shape (batch_size, seq_len)
            
        Returns:
            text_embeds: Sentence-level text embeddings of shape (batch_size, graph_embed_dim)
        """
        # Get token embeddings from decoder (shared embeddings)
        token_embeds = self.decoder.token_embedding(text_tokens)  # (batch_size, seq_len, decoder_embed_dim)
        
        # Average pool over sequence length (can also use attention pooling)
        # Mask out padding tokens (assume pad_token_id = 0)
        mask = (text_tokens != 0).float().unsqueeze(-1)  # (batch_size, seq_len, 1)
        masked_embeds = token_embeds * mask  # (batch_size, seq_len, decoder_embed_dim)
        seq_lengths = mask.sum(dim=1, keepdim=True)  # (batch_size, 1, 1)
        seq_lengths = torch.clamp(seq_lengths, min=1.0)  # Avoid division by zero
        avg_embeds = masked_embeds.sum(dim=1) / seq_lengths.squeeze(-1)  # (batch_size, decoder_embed_dim)
        
        # Project to graph embedding dimension
        text_embeds = self.text_encoder(avg_embeds)  # (batch_size, graph_embed_dim)
        
        return text_embeds
    
    def forward(
        self,
        eeg_bands: Dict[str, torch.Tensor],
        tgt_tokens: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ):
        """
        Forward pass
        
        Args:
            eeg_bands: Dictionary of frequency bands, each of shape (batch_size, num_channels, time_steps)
                Keys: 'delta', 'theta', 'alpha', 'beta', 'gamma'
            tgt_tokens: Target text tokens for training (batch_size, tgt_len)
            tgt_mask: Causal mask for decoder (optional)
            
        Returns:
            If training (tgt_tokens provided):
                logits: Decoder output logits (batch_size, tgt_len-1, vocab_size)
                strg_output: STRG outputs (A, node_features, bandpowers, stre_embeds, text_embeds) for loss computation
            If inference:
                memory: STRE embeddings (batch_size, 1, decoder_embed_dim)
                strg_output: STRG outputs
        """
        # Step 1: Construct STRG from preprocessed frequency bands
        A, node_features, bandpowers = self.strg(eeg_bands)
        # A: (batch_size, num_nodes, num_nodes)
        # node_features: (batch_size, num_nodes, node_dim) where node_dim=1
        # bandpowers: (batch_size, num_channels, num_frequency_bands)
        
        # Reshape for STRE: treat as single temporal window (sentence-level alignment)
        # Note: The paper mentions "word or sentence level windows aligned with text"
        # Current implementation uses sentence-level windows, where each sentence corresponds to one EEG segment
        # STRE architecture supports multiple temporal windows if word-level segmentation is desired in future
        A_windowed = A.unsqueeze(1)  # (batch_size, 1, num_nodes, num_nodes)
        node_features_windowed = node_features.unsqueeze(1)  # (batch_size, 1, num_nodes, node_dim)
        
        # Step 2: Generate STRE embeddings
        # STRE processes temporal sequence: Z_{1:T} = Transformer(h_{1:T})
        # For sentence-level: T=1 (single window per sentence)
        stre_embeds = self.stre(A_windowed, node_features_windowed)  # (batch_size, 1, graph_embed_dim)
        
        # Project to decoder dimension
        memory = self.stre_proj(stre_embeds)  # (batch_size, 1, decoder_embed_dim)
        
        # Step 3: Decode to text
        if tgt_tokens is not None:
            # Training mode
            # Shift tokens for teacher forcing
            tgt_input = tgt_tokens[:, :-1]  # Remove last token
            # tgt_output = tgt_tokens[:, 1:]  # For reference (targets in loss)
            
            # Generate causal mask
            if tgt_mask is None:
                tgt_len = tgt_input.shape[1]
                tgt_mask = self.decoder.generate_mask(tgt_len, self.device)
            
            logits = self.decoder(
                tgt=tgt_input,
                memory=memory,
                tgt_mask=tgt_mask
            )
            # logits: (batch_size, tgt_len-1, vocab_size)
            
            # Encode text for contrastive loss
            text_embeds = self.encode_text(tgt_tokens)  # (batch_size, graph_embed_dim)
            
            return logits, {
                'A': A,
                'node_features': node_features,
                'bandpowers': bandpowers,
                'stre_embeds': stre_embeds,
                'text_embeds': text_embeds
            }
        else:
            # Inference mode - return memory for autoregressive generation
            return memory, {
                'A': A,
                'node_features': node_features,
                'bandpowers': bandpowers,
                'stre_embeds': stre_embeds
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
        """
        Generate text from preprocessed EEG frequency bands
        
        Args:
            eeg_bands: Dictionary of frequency bands, each of shape (batch_size, num_channels, time_steps)
                Keys: 'delta', 'theta', 'alpha', 'beta', 'gamma'
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            max_length: Maximum generation length
            beam_size: Beam search size (not used in greedy generation)
            
        Returns:
            generated_tokens: Generated token sequences (batch_size, seq_len)
        """
        self.eval()
        
        # Get first band to determine batch size and device
        first_band = list(eeg_bands.values())[0]
        batch_size = first_band.shape[0]
        device = first_band.device
        
        # Get STRE embeddings (memory)
        memory, _ = self.forward(eeg_bands)
        
        # Simple greedy generation (can be extended to beam search)
        generated = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        with torch.no_grad():
            for _ in range(max_length - 1):
                tgt_mask = self.decoder.generate_mask(generated.shape[1], device)
                logits = self.decoder(
                    tgt=generated,
                    memory=memory,
                    tgt_mask=tgt_mask
                )
                
                # Get next token (greedy)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check if all sequences have reached EOS
                if (next_token == eos_token_id).all():
                    break
        
        return generated

