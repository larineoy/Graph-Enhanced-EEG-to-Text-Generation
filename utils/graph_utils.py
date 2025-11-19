"""
Graph utility functions
"""

import numpy as np
import torch


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

