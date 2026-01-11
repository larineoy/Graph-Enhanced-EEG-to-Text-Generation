"""
Inference script for Graph-Enhanced EEG-to-Text model
"""

import argparse
import yaml
import torch
import numpy as np
from scipy import signal
from transformers import AutoTokenizer
from typing import Dict

from models import GraphEnhancedEEG2Text
from train import load_config, create_model, set_seed


def load_eeg_from_file(filepath: str):
    """
    Load EEG data from file (MATLAB, CSV, or numpy)
    """
    if filepath.endswith('.mat'):
        import scipy.io
        data = scipy.io.loadmat(filepath)
        # Extract EEG data (adjust based on file structure)
        keys = [k for k in data.keys() if not k.startswith('__')]
        if keys:
            eeg = data[keys[0]]
        else:
            raise ValueError(f"Could not find EEG data in {filepath}")
    elif filepath.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(filepath)
        # Assume first column is channel names, rest are time points
        eeg = df.iloc[:, 1:].values.T
    elif filepath.endswith('.npy'):
        eeg = np.load(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    return eeg


def extract_frequency_bands(eeg: np.ndarray, sampling_rate: float = 250.0) -> Dict[str, np.ndarray]:
    """
    Extract 5 frequency bands from EEG signal (same as preprocessing)
    
    Args:
        eeg: EEG signal of shape (num_channels, time_steps)
        sampling_rate: Sampling rate in Hz
        
    Returns:
        bands: Dictionary with keys ['delta', 'theta', 'alpha', 'beta', 'gamma']
    """
    frequency_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (13, 30),
        'gamma': (30, 100)
    }
    
    num_channels, time_steps = eeg.shape
    bands = {}
    nyquist = sampling_rate / 2
    
    for band_name, (low_freq, high_freq) in frequency_bands.items():
        band_eeg = np.zeros((num_channels, time_steps))
        
        # Normalize frequencies
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        high_norm = min(high_norm, 0.99)  # Cap at Nyquist
        
        for ch in range(num_channels):
            try:
                # Apply bandpass filter
                b, a = signal.butter(4, [low_norm, high_norm], btype='band')
                filtered = signal.filtfilt(b, a, eeg[ch, :])
                band_eeg[ch, :] = filtered
            except:
                # If filtering fails, use original signal
                band_eeg[ch, :] = eeg[ch, :]
        
        bands[band_name] = band_eeg
    
    return bands


def preprocess_eeg(
    eeg: np.ndarray, 
    normalize: bool = True,
    sampling_rate: float = 250.0,
    apply_notch_filter: bool = True,
    notch_freq: float = 50.0,
    apply_highpass_filter: bool = True,
    highpass_cutoff: float = 0.5
):
    """
    Preprocess EEG signal with artifact removal
    
    Args:
        eeg: EEG signal of shape (num_channels, time_steps)
        normalize: Whether to normalize
        sampling_rate: Sampling rate in Hz
        apply_notch_filter: Whether to apply notch filter for line noise
        notch_freq: Line noise frequency (50 Hz EU, 60 Hz US)
        apply_highpass_filter: Whether to apply high-pass filter for slow drifts
        highpass_cutoff: High-pass cutoff frequency in Hz
    """
    num_channels, time_steps = eeg.shape
    
    # Step 1: High-pass filter (remove slow drifts)
    if apply_highpass_filter:
        nyquist = sampling_rate / 2
        cutoff_norm = highpass_cutoff / nyquist
        for ch in range(num_channels):
            try:
                b, a = signal.butter(4, cutoff_norm, btype='high')
                eeg[ch, :] = signal.filtfilt(b, a, eeg[ch, :])
            except:
                pass
    
    # Step 2: Notch filter (remove power line noise)
    if apply_notch_filter:
        quality_factor = 30.0
        for ch in range(num_channels):
            try:
                b, a = signal.iirnotch(notch_freq, quality_factor, sampling_rate)
                eeg[ch, :] = signal.filtfilt(b, a, eeg[ch, :])
            except:
                pass
    
    # Step 3: Normalize per channel
    if normalize:
        eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
    
    return eeg


def main():
    parser = argparse.ArgumentParser(description='Run inference with Graph-Enhanced EEG-to-Text model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--eeg_file', type=str, required=True,
                       help='Path to EEG data file (MATLAB, CSV, or numpy)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output text (optional)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    set_seed(config['seed'])
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model = create_model(config, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f'Loaded model from {args.checkpoint}')
    
    # Load EEG data
    print(f'Loading EEG data from {args.eeg_file}')
    eeg = load_eeg_from_file(args.eeg_file)
    
    # Ensure correct shape: (num_channels, time_steps)
    if eeg.ndim == 2:
        if eeg.shape[0] > eeg.shape[1]:
            eeg = eeg.T  # Transpose if needed
    else:
        raise ValueError(f"Expected 2D EEG array, got shape {eeg.shape}")
    
    # Preprocess with artifact removal
    eeg = preprocess_eeg(
        eeg,
        normalize=True,
        sampling_rate=config.get('model', {}).get('sampling_rate', 250.0),
        apply_notch_filter=True,
        notch_freq=50.0,  # Adjust to 60.0 for US power line frequency
        apply_highpass_filter=True,
        highpass_cutoff=0.5
    )
    
    # Extract frequency bands (same as preprocessing pipeline)
    print('Extracting frequency bands...')
    eeg_bands_np = extract_frequency_bands(eeg, sampling_rate=config.get('model', {}).get('sampling_rate', 250.0))
    
    # Preprocess each band (note: bands are already filtered, just normalize)
    eeg_bands_processed = {}
    for band_name, band_eeg in eeg_bands_np.items():
        # Bands are already bandpass filtered, so only normalize (no additional filtering needed)
        eeg_bands_processed[band_name] = preprocess_eeg(
            band_eeg.copy(),
            normalize=True,
            apply_notch_filter=False,  # Already filtered
            apply_highpass_filter=False  # Already filtered
        )
    
    # Convert to tensors and add batch dimension
    eeg_bands = {
        band_name: torch.FloatTensor(band_eeg).unsqueeze(0).to(device)
        for band_name, band_eeg in eeg_bands_processed.items()
    }
    
    print(f'EEG bands shape: {[f"{k}: {v.shape}" for k, v in eeg_bands.items()]}')
    
    # Create tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except:
        print("Warning: Could not load tokenizer")
        tokenizer = None
    
    # Generate text
    print('Generating text...')
    with torch.no_grad():
        generated = model.generate(
            eeg_bands,
            bos_token_id=tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else 1,
            eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 2,
            pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0,
            max_length=config['model']['decoder']['max_decoder_length']
        )
    
    # Decode
    if tokenizer and hasattr(tokenizer, 'decode'):
        text = tokenizer.decode(generated[0].cpu().tolist(), skip_special_tokens=True)
    else:
        text = ' '.join([str(t.item()) for t in generated[0]])
    
    print('\nGenerated Text:')
    print('=' * 50)
    print(text)
    print('=' * 50)
    
    # Save output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(text)
        print(f'\nOutput saved to {args.output}')


if __name__ == '__main__':
    main()

