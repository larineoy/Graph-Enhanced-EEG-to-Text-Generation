"""
Spectro-Topographic Relational Graph (STRG) Construction
Implements graph construction that jointly encodes spatial topology 
and dynamic functional connectivity across electrodes and frequency bands.

Uses preprocessed eeg_bands from preprocessing pipeline (no internal filtering).
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from typing import Dict


class STRG(nn.Module):
    """
    Spectro-Topographic Relational Graph (STRG) Construction
    
    Constructs graphs that encode:
    - Static spatial adjacency based on electrode topology
    - Dynamic functional connectivity from EEG signals
    - Frequency-specific features across bands
    """
    
    def __init__(
        self,
        num_channels: int,
        num_frequency_bands: int = 5,
        alpha: float = 0.5,
        beta: float = 0.5,
        use_spatial_topology: bool = True,
        use_functional_connectivity: bool = True,
        device: str = 'cuda'
    ):
        """
        Args:
            num_channels: Number of EEG electrodes/channels
            num_frequency_bands: Number of frequency bands (delta, theta, alpha, beta, gamma)
            alpha: Weight for spatial adjacency in adjacency matrix
            beta: Weight for functional connectivity in adjacency matrix
            use_spatial_topology: Whether to include static spatial adjacency
            use_functional_connectivity: Whether to include dynamic functional connectivity
            device: Device to run computations on
        """
        super(STRG, self).__init__()
        self.num_channels = num_channels
        self.num_frequency_bands = num_frequency_bands
        self.alpha = alpha
        self.beta = beta
        self.use_spatial_topology = use_spatial_topology
        self.use_functional_connectivity = use_functional_connectivity
        self.device = device
        
        # Define frequency band names (must match preprocessing)
        self.frequency_band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        
        # Build spatial adjacency matrix (10-20 system topology)
        self.register_buffer('A_spatial', self._build_spatial_adjacency())
        
    def _build_spatial_adjacency(self):
        """
        Build static spatial adjacency matrix based on 10-20 electrode system.
        Adjacent electrodes are connected if they are neighbors on the scalp.
        """
        A_spatial = torch.zeros(
            self.num_channels * self.num_frequency_bands,
            self.num_channels * self.num_frequency_bands,
            device=self.device
        )
        
        # For each frequency band, connect channels that are spatially adjacent
        # This is a simplified version - in practice, you'd use actual electrode positions
        for f in range(self.num_frequency_bands):
            base_idx = f * self.num_channels
            # Connect adjacent channels (simplified grid topology)
            for i in range(self.num_channels):
                for j in range(self.num_channels):
                    if i != j:
                        # Simplified: connect if channels are close (you'd use actual distances)
                        # Here we use a simple heuristic based on channel indices
                        dist = abs(i - j)
                        if dist <= 3:  # Connect if channels are relatively close
                            A_spatial[base_idx + i, base_idx + j] = 1.0
                        
        # Symmetrize
        A_spatial = (A_spatial + A_spatial.T) / 2
        
        return A_spatial
    
    def _compute_bandpower(self, band_data: torch.Tensor) -> torch.Tensor:
        """
        Compute bandpower (variance) for each channel in a frequency band.
        
        Args:
            band_data: Band-filtered EEG of shape (batch_size, num_channels, time_steps)
            
        Returns:
            bandpowers: Band power of shape (batch_size, num_channels)
        """
        # Compute variance over time dimension
        bandpowers = torch.var(band_data, dim=2, keepdim=False)
        return bandpowers
    
    def _compute_functional_connectivity(self, band_data: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamic functional connectivity using Pearson correlation.
        
        Args:
            band_data: Band-filtered EEG of shape (num_channels, time_steps)
            
        Returns:
            A_functional: Functional connectivity matrix of shape (num_channels, num_channels)
        """
        num_channels, time_steps = band_data.shape
        device = band_data.device
        
        # Normalize data (zero mean, unit variance per channel)
        band_data_norm = (band_data - torch.mean(band_data, dim=1, keepdim=True)) / (
            torch.std(band_data, dim=1, keepdim=True) + 1e-8
        )
        
        # Compute correlation matrix: (num_channels, num_channels)
        # Correlation = (X @ X^T) / (time_steps - 1) for normalized data
        A_functional = torch.matmul(band_data_norm, band_data_norm.T) / (time_steps - 1)
        
        # Ensure diagonal is 1 and non-negative
        A_functional = torch.abs(A_functional)
        A_functional = A_functional - torch.diag(torch.diag(A_functional)) + torch.eye(num_channels, device=device)
        
        # Ensure symmetry
        A_functional = (A_functional + A_functional.T) / 2
        
        return A_functional
    
    def forward(self, eeg_bands: Dict[str, torch.Tensor]):
        """
        Construct STRG from preprocessed EEG frequency bands.
        
        Args:
            eeg_bands: Dictionary of frequency bands, each of shape (batch_size, num_channels, time_steps)
                Keys: 'delta', 'theta', 'alpha', 'beta', 'gamma'
            
        Returns:
            A: Combined adjacency matrix of shape (batch_size, num_nodes, num_nodes)
               where num_nodes = num_channels * num_frequency_bands
            node_features: Node features of shape (batch_size, num_nodes, node_dim)
               node_dim = 1 (bandpower feature)
            bandpowers: Band power features of shape (batch_size, num_channels, num_frequency_bands)
        """
        # Get batch size and check all bands have same shape
        first_band = list(eeg_bands.values())[0]
        batch_size, num_channels, time_steps = first_band.shape
        
        num_nodes = self.num_channels * self.num_frequency_bands
        device = first_band.device
        
        # Compute bandpowers for all bands and channels
        bandpowers_list = []
        for band_name in self.frequency_band_names:
            if band_name not in eeg_bands:
                raise ValueError(f"Missing frequency band: {band_name} in eeg_bands")
            band_data = eeg_bands[band_name]  # (batch_size, C, T)
            bandpowers_band = self._compute_bandpower(band_data)  # (batch_size, C)
            bandpowers_list.append(bandpowers_band)
        
        # Stack to get (batch_size, C, num_bands)
        bandpowers = torch.stack(bandpowers_list, dim=2)  # (batch_size, C, num_bands)
        
        # Build node features: each node = (channel, frequency_band) pair
        # Node order: (ch0, delta), (ch1, delta), ..., (chC, delta), (ch0, theta), ...
        node_features_list = []
        for b in range(batch_size):
            node_feat_batch = []
            for f_idx, band_name in enumerate(self.frequency_band_names):
                for ch in range(self.num_channels):
                    # Node feature = bandpower for this (channel, band) pair
                    bandpower_val = bandpowers[b, ch, f_idx]
                    node_feat_batch.append(bandpower_val)
            
            node_features_list.append(torch.stack(node_feat_batch))  # (num_nodes,)
        
        # Stack to (batch_size, num_nodes, 1)
        node_features = torch.stack(node_features_list).unsqueeze(-1)  # (batch_size, num_nodes, 1)
        
        # Build adjacency matrices
        batch_A = []
        
        for b in range(batch_size):
            # Initialize functional connectivity matrix
            A_functional_full = torch.zeros(num_nodes, num_nodes, device=device)
            
            if self.use_functional_connectivity:
                # Compute functional connectivity for each frequency band
                for f_idx, band_name in enumerate(self.frequency_band_names):
                    band_data = eeg_bands[band_name][b]  # (C, T) for this batch item
                    
                    # Compute functional connectivity for this band
                    A_func_band = self._compute_functional_connectivity(band_data)  # (C, C)
                    
                    # Place in full matrix at correct position
                    base_idx = f_idx * self.num_channels
                    A_functional_full[
                        base_idx:base_idx + self.num_channels,
                        base_idx:base_idx + self.num_channels
                    ] = A_func_band
            
            # Combine spatial and functional adjacency
            if self.use_spatial_topology and self.use_functional_connectivity:
                A = self.alpha * self.A_spatial + self.beta * A_functional_full
            elif self.use_spatial_topology:
                A = self.A_spatial
            elif self.use_functional_connectivity:
                A = A_functional_full
            else:
                A = torch.eye(num_nodes, device=device)
            
            # Normalize adjacency matrix (symmetric normalization)
            D = torch.sum(A, dim=1)  # (num_nodes,)
            D_inv_sqrt = torch.pow(D + 1e-8, -0.5)
            D_inv_sqrt = torch.where(torch.isinf(D_inv_sqrt), torch.zeros_like(D_inv_sqrt), D_inv_sqrt)
            D_inv_sqrt = torch.diag(D_inv_sqrt)
            A_normalized = D_inv_sqrt @ A @ D_inv_sqrt
            
            batch_A.append(A_normalized)
        
        # Stack adjacency matrices
        A = torch.stack(batch_A)  # (batch_size, num_nodes, num_nodes)
        
        return A, node_features, bandpowers

