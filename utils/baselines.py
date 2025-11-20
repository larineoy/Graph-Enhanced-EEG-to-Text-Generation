"""
Random-input baseline utilities for sanity checks
Tests model on shuffled/random inputs to ensure it learns meaningful patterns
"""

import torch
import numpy as np
from typing import Dict, Optional
import random


def create_shuffled_channel_baseline(eeg_bands: Dict[str, torch.Tensor], seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """
    Create baseline by shuffling EEG channels (destroys spatial relationships)
    
    Args:
        eeg_bands: Original EEG bands dictionary
        seed: Random seed for reproducibility
        
    Returns:
        shuffled_bands: Bands with shuffled channels
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    shuffled_bands = {}
    for band_name, band_data in eeg_bands.items():
        batch_size, num_channels, time_steps = band_data.shape
        
        shuffled_data = band_data.clone()
        
        # Shuffle channels for each sample in batch
        for b in range(batch_size):
            perm = torch.randperm(num_channels)
            shuffled_data[b] = band_data[b, perm, :]
        
        shuffled_bands[band_name] = shuffled_data
    
    return shuffled_bands


def create_shuffled_time_baseline(eeg_bands: Dict[str, torch.Tensor], seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """
    Create baseline by shuffling time windows (destroys temporal relationships)
    
    Args:
        eeg_bands: Original EEG bands dictionary
        seed: Random seed for reproducibility
        
    Returns:
        shuffled_bands: Bands with shuffled time dimension
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    shuffled_bands = {}
    for band_name, band_data in eeg_bands.items():
        batch_size, num_channels, time_steps = band_data.shape
        
        shuffled_data = band_data.clone()
        
        # Shuffle time dimension for each sample in batch
        for b in range(batch_size):
            perm = torch.randperm(time_steps)
            shuffled_data[b] = band_data[b, :, perm]
        
        shuffled_bands[band_name] = shuffled_data
    
    return shuffled_bands


def create_random_gaussian_baseline(
    eeg_bands: Dict[str, torch.Tensor],
    mean: float = 0.0,
    std: float = 1.0,
    seed: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Create baseline with random Gaussian noise (no meaningful signal)
    
    Args:
        eeg_bands: Original EEG bands (used for shape reference)
        mean: Mean of Gaussian noise
        std: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility
        
    Returns:
        random_bands: Random Gaussian noise matching original shape
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    random_bands = {}
    for band_name, band_data in eeg_bands.items():
        shape = band_data.shape
        random_data = torch.randn(shape, device=band_data.device, dtype=band_data.dtype) * std + mean
        random_bands[band_name] = random_data
    
    return random_bands


def create_random_uniform_baseline(
    eeg_bands: Dict[str, torch.Tensor],
    low: float = -1.0,
    high: float = 1.0,
    seed: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Create baseline with random uniform noise
    
    Args:
        eeg_bands: Original EEG bands (used for shape reference)
        low: Lower bound of uniform distribution
        high: Upper bound of uniform distribution
        seed: Random seed for reproducibility
        
    Returns:
        random_bands: Random uniform noise matching original shape
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    random_bands = {}
    for band_name, band_data in eeg_bands.items():
        shape = band_data.shape
        random_data = torch.rand(shape, device=band_data.device, dtype=band_data.dtype) * (high - low) + low
        random_bands[band_name] = random_data
    
    return random_bands

