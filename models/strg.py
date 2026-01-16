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
from typing import Dict, Optional
from utils.graph_utils import get_standard_10_20_positions, build_spatial_adjacency_from_positions


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
        device: str = 'cuda',
        electrode_positions: Optional[torch.Tensor] = None  # Actual positions from ZuCo chanlocs (X, Y, Z)
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
        # Enforce that num_frequency_bands matches the standard 5 bands
        assert num_frequency_bands == 5, f"num_frequency_bands must be 5 (got {num_frequency_bands}). Standard bands: delta, theta, alpha, beta, gamma"
        self.frequency_band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        
        # Store electrode positions if provided (from ZuCo chanlocs)
        self.electrode_positions = electrode_positions
        
        # Build spatial adjacency matrix using actual ZuCo positions if available, else standard 10-20
        # Build on CPU, register_buffer will handle device movement via model.to(device)
        self.register_buffer('A_spatial', self._build_spatial_adjacency())
        
    def _build_spatial_adjacency(self):
        """
        Build static spatial adjacency matrix using ACTUAL electrode positions from ZuCo chanlocs.
        
        If electrode_positions are provided (extracted from ZuCo files), uses those.
        Otherwise falls back to standard 10-20 positions based on ZuCo's known configuration.
        
        Uses electrode positions to determine spatial neighbors.
        Adjacent electrodes are connected based on physical proximity on the scalp.
        """
        # Use ACTUAL electrode positions from ZuCo chanlocs if available
        if self.electrode_positions is not None:
            # Convert to numpy if tensor
            if isinstance(self.electrode_positions, torch.Tensor):
                electrode_positions = self.electrode_positions.cpu().numpy()
            else:
                electrode_positions = self.electrode_positions
            
            # Ensure correct shape: (num_channels, 3) with X, Y, Z
            if electrode_positions.shape[0] != self.num_channels:
                raise ValueError(
                    f"Electrode positions shape {electrode_positions.shape} doesn't match "
                    f"num_channels {self.num_channels}"
                )
        else:
            # Fallback: Get standard 10-20 system electrode positions
            # This is used only if ZuCo files don't contain chanlocs metadata
            electrode_positions = get_standard_10_20_positions(self.num_channels)
        
        # Build spatial adjacency for channels using k-nearest neighbors
        # Use k=6 to connect each electrode to its 6 nearest neighbors (typical for 10-20 system)
        A_spatial_channels = build_spatial_adjacency_from_positions(
            electrode_positions,
            k_nearest=6  # Connect to 6 nearest neighbors (matches typical 10-20 topology)
        )
        
        # Convert to torch tensor (on CPU - device will be handled by register_buffer)
        A_spatial_channels = torch.from_numpy(A_spatial_channels).float()
        
        # Build full adjacency matrix: (num_channels * num_frequency_bands, num_channels * num_frequency_bands)
        # Same spatial structure is replicated for each frequency band
        # Nodes are ordered as: (ch0, delta), (ch1, delta), ..., (chC, delta), (ch0, theta), ...
        A_spatial = torch.zeros(
            self.num_channels * self.num_frequency_bands,
            self.num_channels * self.num_frequency_bands
        )
        
        # For each frequency band, use the same spatial adjacency structure
        for f in range(self.num_frequency_bands):
            base_idx = f * self.num_channels
            A_spatial[
                base_idx:base_idx + self.num_channels,
                base_idx:base_idx + self.num_channels
            ] = A_spatial_channels
        
        # Ensure symmetry (should already be symmetric, but enforce it)
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
        
        Note: Uses absolute correlation to ensure non-negative adjacency weights.
        This is intentional for graph construction, though it differs from signed Pearson correlation.
        
        Args:
            band_data: Band-filtered EEG of shape (num_channels, time_steps)
            
        Returns:
            A_functional: Functional connectivity matrix of shape (num_channels, num_channels)
        """
        num_channels, time_steps = band_data.shape
        device = band_data.device
        
        # Check minimum time steps for reliable correlation estimation
        if time_steps < 10:
            print(f"Warning: Short time window ({time_steps} samples) may lead to unreliable correlation estimates")
        
        # Normalize data (zero mean, unit variance per channel)
        band_data_norm = (band_data - torch.mean(band_data, dim=1, keepdim=True)) / (
            torch.std(band_data, dim=1, keepdim=True) + 1e-8
        )
        
        # Compute correlation matrix: (num_channels, num_channels)
        # Pearson correlation = (X_norm @ X_norm^T) / (time_steps - 1) for normalized data
        A_functional = torch.matmul(band_data_norm, band_data_norm.T) / (time_steps - 1)
        
        # Use absolute correlation for non-negative adjacency weights
        # Note: This differs from signed Pearson correlation but is standard for graph construction
        # where edge weights should be non-negative. The absolute value preserves magnitude of correlation.
        A_functional = (A_functional + 1.0) / 2.0  # Maps [-1, 1] → [0, 1]
        
        # Ensure diagonal is 1 (self-connection)
        A_functional = A_functional - torch.diag(torch.diag(A_functional)) + torch.eye(num_channels, device=device)
        
        # Ensure symmetry
        A_functional = (A_functional + A_functional.T) / 2
        
        # Optional: Threshold sparse edges to reduce noise
        # Uncomment to keep only top correlations (recommended for sparse graphs)
        # threshold_percentile = 0.8  # Keep top 20% of edges
        # threshold = torch.quantile(A_functional, threshold_percentile)
        # A_functional = torch.where(A_functional >= threshold, A_functional, torch.zeros_like(A_functional))
        
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
        # Get batch size and validate input shapes
        first_band = list(eeg_bands.values())[0]
        batch_size, num_channels, time_steps = first_band.shape
        
        # Shape consistency checks
        assert num_channels == self.num_channels, (
            f"Input has {num_channels} channels but model expects {self.num_channels} channels"
        )
        assert len(eeg_bands) == len(self.frequency_band_names), (
            f"Input has {len(eeg_bands)} bands but expected {len(self.frequency_band_names)} bands"
        )
        
        # Verify all bands have same shape
        for band_name, band_data in eeg_bands.items():
            assert band_data.shape == (batch_size, num_channels, time_steps), (
                f"Band {band_name} has shape {band_data.shape} but expected ({batch_size}, {num_channels}, {time_steps})"
            )
        
        num_nodes = self.num_channels * self.num_frequency_bands
        device = first_band.device
        
        # Ensure A_spatial is on the same device as input
        A_spatial = self.A_spatial.to(device)
        
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
        # Vectorized construction: reshape (B, C, F) -> (B, F, C) -> (B, F*C) -> (B, F*C, 1)
        # This matches node order: (ch0,delta), (ch1,delta), ..., (ch0,theta), ...
        node_features = bandpowers.permute(0, 2, 1).reshape(batch_size, -1, 1)  # (batch_size, num_nodes, 1)
        
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
                A = self.alpha * A_spatial + self.beta * A_functional_full
            elif self.use_spatial_topology:
                A = A_spatial
            elif self.use_functional_connectivity:
                A = A_functional_full
            else:
                A = torch.eye(num_nodes, device=device)
            
            # Note: GAT uses A > 0 only as a binary mask, so normalization values are ignored
            # We keep adjacency as-is (unnormalized) since GAT computes its own attention weights
            # If weighted adjacency is desired in future, normalization can be added here
            batch_A.append(A)
        
        # Stack adjacency matrices
        A = torch.stack(batch_A)  # (batch_size, num_nodes, num_nodes)
        
        return A, node_features, bandpowers
    
    def extract_strg_components(self, eeg_bands: Dict[str, torch.Tensor], return_separate: bool = False):
        """
        Extract STRG components separately for visualization purposes
        
        Args:
            eeg_bands: Dictionary of frequency bands
            return_separate: If True, return spatial, functional, and combined separately
            
        Returns:
            If return_separate=False: Same as forward() - (A, node_features, bandpowers)
            If return_separate=True: Dict with keys:
                - 'A_combined': Combined adjacency (batch_size, num_nodes, num_nodes)
                - 'A_spatial': Spatial adjacency (num_nodes, num_nodes)
                - 'A_functional_per_band': Dict of functional connectivity per band (num_channels, num_channels)
                - 'A_functional_full': Full functional matrix (num_nodes, num_nodes)
                - 'node_features': Node features (batch_size, num_nodes, node_dim)
                - 'bandpowers': Bandpowers (batch_size, num_channels, num_frequency_bands)
        """
        if not return_separate:
            return self.forward(eeg_bands)
        
        # Get batch size and device
        first_band = list(eeg_bands.values())[0]
        batch_size, num_channels, time_steps = first_band.shape
        num_nodes = self.num_channels * self.num_frequency_bands
        device = first_band.device
        
        # Ensure A_spatial is on correct device
        A_spatial = self.A_spatial.to(device)
        
        # Compute bandpowers and node features (same as forward)
        bandpowers_list = []
        for band_name in self.frequency_band_names:
            if band_name not in eeg_bands:
                raise ValueError(f"Missing frequency band: {band_name}")
            band_data = eeg_bands[band_name]
            bandpowers_band = self._compute_bandpower(band_data)
            bandpowers_list.append(bandpowers_band)
        
        bandpowers = torch.stack(bandpowers_list, dim=2)
        node_features = bandpowers.permute(0, 2, 1).reshape(batch_size, -1, 1)
        
        # Extract functional connectivity per band
        A_functional_per_band = {}
        A_functional_full = torch.zeros(num_nodes, num_nodes, device=device)
        
        if self.use_functional_connectivity:
            for f_idx, band_name in enumerate(self.frequency_band_names):
                # Use first sample for visualization
                band_data = eeg_bands[band_name][0]  # (C, T)
                A_func_band = self._compute_functional_connectivity(band_data)  # (C, C)
                A_functional_per_band[band_name] = A_func_band.cpu().numpy()
                
                # Place in full matrix
                base_idx = f_idx * self.num_channels
                A_functional_full[base_idx:base_idx+self.num_channels, 
                                base_idx:base_idx+self.num_channels] = A_func_band
        
        # Compute combined (for first sample)
        if self.use_spatial_topology and self.use_functional_connectivity:
            A_combined_sample = self.alpha * A_spatial + self.beta * A_functional_full
        elif self.use_spatial_topology:
            A_combined_sample = A_spatial
        elif self.use_functional_connectivity:
            A_combined_sample = A_functional_full
        else:
            A_combined_sample = torch.eye(num_nodes, device=device)
        
        # Build full batch (for consistency with forward)
        A_combined_batch = A_combined_sample.unsqueeze(0).repeat(batch_size, 1, 1)
        
        return {
            'A_combined': A_combined_batch.cpu().numpy(),
            'A_spatial': A_spatial.cpu().numpy(),
            'A_functional_per_band': A_functional_per_band,
            'A_functional_full': A_functional_full.cpu().numpy(),
            'node_features': node_features.cpu().numpy(),
            'bandpowers': bandpowers.cpu().numpy()
        }

