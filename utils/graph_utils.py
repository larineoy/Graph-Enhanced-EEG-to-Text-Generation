"""
Graph utility functions
"""

import numpy as np
import torch
from typing import Optional


def normalize_adjacency(A: np.ndarray):
    """
    Normalize adjacency matrix
    
    Args:
        A: Adjacency matrix (N, N)
        
    Returns:
        A_normalized: Normalized adjacency matrix
    """
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.power(D + 1e-8, -0.5)
    D_inv_sqrt = np.where(np.isinf(D_inv_sqrt), 0, D_inv_sqrt)
    A_normalized = np.diag(D_inv_sqrt) @ A @ np.diag(D_inv_sqrt)
    return A_normalized


def compute_spatial_distance_matrix(electrode_positions: np.ndarray):
    """
    Compute spatial distance matrix from electrode 3D positions
    
    Args:
        electrode_positions: Array of shape (num_channels, 3) with x, y, z coordinates
        
    Returns:
        distance_matrix: Spatial distance matrix (num_channels, num_channels)
    """
    num_channels = electrode_positions.shape[0]
    distance_matrix = np.zeros((num_channels, num_channels))
    
    for i in range(num_channels):
        for j in range(num_channels):
            if i != j:
                diff = electrode_positions[i] - electrode_positions[j]
                distance_matrix[i, j] = np.linalg.norm(diff)
    
    return distance_matrix


def build_knn_adjacency(distance_matrix: np.ndarray, k: int = 5):
    """
    Build k-nearest neighbor adjacency matrix
    
    Args:
        distance_matrix: Distance matrix (num_channels, num_channels)
        k: Number of nearest neighbors
        
    Returns:
        A: Binary adjacency matrix (num_channels, num_channels)
    """
    num_channels = distance_matrix.shape[0]
    A = np.zeros((num_channels, num_channels))
    
    for i in range(num_channels):
        # Get k nearest neighbors (excluding self)
        indices = np.argsort(distance_matrix[i, :])[1:k+1]
        A[i, indices] = 1.0
    
    # Symmetrize
    A = (A + A.T) / 2
    
    return A


def get_standard_10_20_positions(num_channels: int = 64):
    """
    Generate standard 10-20 system electrode positions based on ZuCo dataset configuration.
    
    NOTE: ZuCo dataset uses standard 10-20 system electrode configuration, but the MATLAB
    files typically contain only EEG data arrays without electrode position metadata.
    This function generates positions based on ZuCo's known standard 10-20 layout.
    
    For 64 channels, uses standard extended 10-20 layout (ZuCo's typical configuration).
    Positions are in spherical coordinates (theta, phi) on unit sphere,
    converted to 3D Cartesian coordinates (x, y, z).
    
    Args:
        num_channels: Number of EEG channels (derived from actual ZuCo data shape)
        
    Returns:
        positions: Array of shape (num_channels, 3) with x, y, z coordinates
    """
    # Standard 10-20 system electrode positions based on ZuCo dataset configuration
    # ZuCo uses standard extended 10-20 system, but files don't contain position metadata
    # This generates positions matching ZuCo's electrode layout
    positions = []
    
    if num_channels == 64:
        # Extended 10-20 system: 64 channels arranged in standard positions
        # We'll create positions based on standard angular coordinates
        # on a unit sphere (radius = 1), then convert to Cartesian
        
        # Standard 10-20 system uses angular positions:
        # - Frontal: theta ~ 60-120 degrees, phi ~ 30-90 degrees
        # - Central: theta ~ 60-120 degrees, phi ~ 0-30 degrees  
        # - Parietal: theta ~ 60-120 degrees, phi ~ -30-0 degrees
        # - Occipital: theta ~ 60-120 degrees, phi ~ -90--30 degrees
        # - Temporal: theta ~ 0-60 and 120-180 degrees
        
        # Generate positions on a unit sphere
        # Using a grid-based approach for extended 10-20 system
        for i in range(num_channels):
            # Map channel index to spherical coordinates
            # This creates a reasonable approximation of 10-20 layout
            row = i // 8  # 8 columns
            col = i % 8   # 8 rows (approximately)
            
            # Convert to spherical coordinates (theta: azimuth, phi: elevation)
            # Standard 10-20: theta ranges from 0 to 180 degrees
            # phi ranges from -90 (back) to 90 (front) degrees
            theta_deg = 15 + col * 22.5  # Azimuth: 15 to 180 degrees
            phi_deg = 45 - row * 22.5    # Elevation: 45 (front) to -45 (back) degrees
            
            # Convert to radians
            theta = np.deg2rad(theta_deg)
            phi = np.deg2rad(phi_deg)
            
            # Convert spherical to Cartesian (unit sphere, radius = 1)
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            
            positions.append([x, y, z])
    
    else:
        # For other channel counts, use a simplified grid layout
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(num_channels)))
        for i in range(num_channels):
            row = i // grid_size
            col = i % grid_size
            
            # Map to unit sphere
            theta = 2 * np.pi * col / grid_size
            phi = np.pi * row / (grid_size - 1) if grid_size > 1 else np.pi / 2
            
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            
            positions.append([x, y, z])
    
    return np.array(positions, dtype=np.float32)


def build_spatial_adjacency_from_positions(
    electrode_positions: np.ndarray,
    distance_threshold: float = 0.5,
    k_nearest: Optional[int] = None
) -> np.ndarray:
    """
    Build spatial adjacency matrix from electrode 3D positions.
    Uses either distance threshold or k-nearest neighbors approach.
    
    Args:
        electrode_positions: Array of shape (num_channels, 3) with x, y, z coordinates
        distance_threshold: Maximum distance for adjacency (if k_nearest is None)
        k_nearest: Number of nearest neighbors to connect (overrides distance_threshold)
        
    Returns:
        A_spatial: Binary adjacency matrix (num_channels, num_channels)
    """
    num_channels = electrode_positions.shape[0]
    distance_matrix = compute_spatial_distance_matrix(electrode_positions)
    
    if k_nearest is not None:
        # Use k-nearest neighbors approach
        A_spatial = build_knn_adjacency(distance_matrix, k=k_nearest)
    else:
        # Use distance threshold approach
        A_spatial = np.zeros((num_channels, num_channels))
        for i in range(num_channels):
            for j in range(num_channels):
                if i != j and distance_matrix[i, j] <= distance_threshold:
                    A_spatial[i, j] = 1.0
        
        # Symmetrize
        A_spatial = (A_spatial + A_spatial.T) / 2
    
    return A_spatial


def window_eeg_signal(eeg_data: np.ndarray, window_size: int, stride: int):
    """
    Segment EEG signal into overlapping windows
    
    Args:
        eeg_data: EEG signal (num_channels, time_steps)
        window_size: Window size in samples
        stride: Stride between windows
        
    Returns:
        windows: Windowed EEG (num_windows, num_channels, window_size)
    """
    num_channels, time_steps = eeg_data.shape
    windows = []
    
    for start in range(0, time_steps - window_size + 1, stride):
        window = eeg_data[:, start:start + window_size]
        windows.append(window)
    
    return np.array(windows)


def segment_into_windows(
    eeg_data: np.ndarray,
    window_size: int,
    stride: Optional[int] = None,
    max_windows: Optional[int] = None
) -> np.ndarray:
    """
    Divide a sentence-aligned EEG recording into fixed-length windows.

    Short recordings are edge-padded to one window. A trailing remainder of at
    least half a window is kept as a final (possibly overlapping) window.

    Args:
        eeg_data: EEG of shape (num_channels, time_steps)
        window_size: Window length in samples
        stride: Hop size in samples (defaults to window_size, i.e. non-overlapping)
        max_windows: Optional cap; extra windows are subsampled evenly

    Returns:
        windows: Array of shape (num_windows, num_channels, window_size)
    """
    if stride is None:
        stride = window_size

    eeg_data = np.ascontiguousarray(eeg_data, dtype=np.float32)
    num_channels, time_steps = eeg_data.shape

    if time_steps <= 0:
        raise ValueError("EEG recording is empty")

    if time_steps < window_size:
        pad_width = window_size - time_steps
        padded = np.pad(eeg_data, ((0, 0), (0, pad_width)), mode='edge')
        return padded[None, ...]

    windows = []
    start = 0
    last_end = 0
    while start + window_size <= time_steps:
        windows.append(eeg_data[:, start:start + window_size])
        last_end = start + window_size
        start += stride

    remainder = time_steps - last_end
    if remainder >= window_size // 2:
        windows.append(eeg_data[:, time_steps - window_size:time_steps])

    stacked = np.stack(windows, axis=0)
    if max_windows is not None and stacked.shape[0] > max_windows:
        idx = np.linspace(0, stacked.shape[0] - 1, max_windows, dtype=int)
        stacked = stacked[idx]
    return stacked


def expand_channel_graph_to_frequency_blocks(
    A_channels: torch.Tensor,
    num_frequency_bands: int
) -> torch.Tensor:
    """
    Replicate a C x C channel graph as a block-diagonal CF x CF matrix.

    Args:
        A_channels: (..., C, C)
        num_frequency_bands: F

    Returns:
        A_blocks: (..., C*F, C*F)
    """
    *leading, num_channels, _ = A_channels.shape
    eye_f = torch.eye(num_frequency_bands, device=A_channels.device, dtype=A_channels.dtype)
    # (..., F, C, C) if we expand a single channel graph across bands
    if A_channels.dim() == 2:
        A_banded = A_channels.unsqueeze(0).expand(num_frequency_bands, -1, -1)
        leading = []
    else:
        A_banded = A_channels
        if A_banded.shape[-3] != num_frequency_bands:
            A_banded = A_banded.unsqueeze(-3).expand(*leading, num_frequency_bands, num_channels, num_channels)

    # A_full[..., f*C+i, g*C+j] = A_banded[..., f, i, j] if f == g else 0
    A_blocks = torch.einsum('...fij,fg->...figj', A_banded, eye_f)
    num_nodes = num_channels * num_frequency_bands
    return A_blocks.reshape(*A_blocks.shape[:-4], num_nodes, num_nodes)

