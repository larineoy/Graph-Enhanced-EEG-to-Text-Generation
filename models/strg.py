"""
Spectro-Topographic Relational Graph (STRG) Construction
Implements graph construction that jointly encodes spatial topology 
and dynamic functional connectivity across electrodes and frequency bands.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler


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
        
        # Define frequency bands (Hz)
        self.frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
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
    
    def _extract_frequency_features(self, eeg_data: np.ndarray, sampling_rate: float = 250.0):
        """
        Extract frequency-specific features for each channel and frequency band.
        
        Args:
            eeg_data: EEG signal of shape (num_channels, time_steps)
            sampling_rate: Sampling rate of EEG signal in Hz
            
        Returns:
            features: Dictionary of frequency band features, each of shape (num_channels, time_steps)
            bandpowers: Band power features of shape (num_channels, num_frequency_bands)
        """
        num_channels, time_steps = eeg_data.shape
        features = {}
        bandpowers = np.zeros((num_channels, self.num_frequency_bands))
        
        for band_idx, (band_name, (low_freq, high_freq)) in enumerate(self.frequency_bands.items()):
            band_features = np.zeros((num_channels, time_steps))
            
            for ch in range(num_channels):
                # Apply bandpass filter
                nyquist = sampling_rate / 2
                low = low_freq / nyquist
                high = high_freq / nyquist
                
                try:
                    b, a = signal.butter(4, [low, high], btype='band')
                    filtered = signal.filtfilt(b, a, eeg_data[ch, :])
                    band_features[ch, :] = filtered
                    
                    # Compute band power
                    bandpowers[ch, band_idx] = np.var(filtered)
                except:
                    band_features[ch, :] = eeg_data[ch, :]
                    bandpowers[ch, band_idx] = np.var(eeg_data[ch, :])
            
            features[band_name] = band_features
        
        return features, bandpowers
    
    def _compute_functional_connectivity(self, eeg_data: np.ndarray):
        """
        Compute dynamic functional connectivity using Pearson correlation.
        
        Args:
            eeg_data: EEG signal of shape (num_channels, time_steps)
            
        Returns:
            A_functional: Functional connectivity matrix of shape (num_channels, num_channels)
        """
        num_channels = eeg_data.shape[0]
        A_functional = np.zeros((num_channels, num_channels))
        
        for i in range(num_channels):
            for j in range(num_channels):
                if i == j:
                    A_functional[i, j] = 1.0
                else:
                    try:
                        corr, _ = pearsonr(eeg_data[i, :], eeg_data[j, :])
                        A_functional[i, j] = corr if not np.isnan(corr) else 0.0
                    except:
                        A_functional[i, j] = 0.0
        
        # Ensure symmetry and non-negative
        A_functional = (A_functional + A_functional.T) / 2
        A_functional = np.abs(A_functional)
        
        return A_functional
    
    def forward(self, eeg_data: torch.Tensor, sampling_rate: float = 250.0):
        """
        Construct STRG from EEG data.
        
        Args:
            eeg_data: EEG signal of shape (batch_size, num_channels, time_steps)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            A: Combined adjacency matrix of shape (batch_size, num_nodes, num_nodes)
               where num_nodes = num_channels * num_frequency_bands
            node_features: Node features of shape (batch_size, num_nodes, node_dim)
            bandpowers: Band power features of shape (batch_size, num_channels, num_frequency_bands)
        """
        batch_size = eeg_data.shape[0]
        num_nodes = self.num_channels * self.num_frequency_bands
        
        # Convert to numpy for signal processing
        eeg_np = eeg_data.cpu().numpy()
        
        batch_A = []
        batch_node_features = []
        batch_bandpowers = []
        
        for b in range(batch_size):
            # Extract frequency features
            freq_features, bandpowers = self._extract_frequency_features(
                eeg_np[b], sampling_rate
            )
            batch_bandpowers.append(bandpowers)
            
            # Build node features: each node is (channel, frequency_band) pair
            node_features = np.zeros((num_nodes, 1))
            for f in range(self.num_frequency_bands):
                for ch in range(self.num_channels):
                    node_idx = f * self.num_channels + ch
                    node_features[node_idx, 0] = bandpowers[ch, f]
            
            # Compute functional connectivity for each frequency band
            A_functional_full = np.zeros((num_nodes, num_nodes))
            
            if self.use_functional_connectivity:
                for f in range(self.num_frequency_bands):
                    band_name = list(self.frequency_bands.keys())[f]
                    band_data = freq_features[band_name]
                    
                    # Compute functional connectivity for this band
                    A_func_band = self._compute_functional_connectivity(band_data)
                    
                    # Place in full matrix
                    base_idx = f * self.num_channels
                    A_functional_full[
                        base_idx:base_idx + self.num_channels,
                        base_idx:base_idx + self.num_channels
                    ] = A_func_band
            
            # Combine spatial and functional adjacency
            A_spatial_np = self.A_spatial.cpu().numpy()
            
            if self.use_spatial_topology and self.use_functional_connectivity:
                A = self.alpha * A_spatial_np + self.beta * A_functional_full
            elif self.use_spatial_topology:
                A = A_spatial_np
            elif self.use_functional_connectivity:
                A = A_functional_full
            else:
                A = np.eye(num_nodes)
            
            # Normalize adjacency matrix
            D = np.sum(A, axis=1)
            D_inv_sqrt = np.power(D + 1e-8, -0.5)
            D_inv_sqrt = np.where(np.isinf(D_inv_sqrt), 0, D_inv_sqrt)
            A_normalized = np.diag(D_inv_sqrt) @ A @ np.diag(D_inv_sqrt)
            
            batch_A.append(A_normalized)
            batch_node_features.append(node_features)
        
        # Convert to tensors
        A = torch.FloatTensor(np.stack(batch_A)).to(self.device)
        node_features = torch.FloatTensor(np.stack(batch_node_features)).to(self.device)
        bandpowers = torch.FloatTensor(np.stack(batch_bandpowers)).to(self.device)
        
        return A, node_features, bandpowers

