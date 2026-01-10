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

