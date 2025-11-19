"""
Inference script for Graph-Enhanced EEG-to-Text model
"""

import argparse
import yaml
import torch
import numpy as np
from transformers import AutoTokenizer

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


def preprocess_eeg(eeg: np.ndarray, normalize: bool = True):
    """Preprocess EEG signal"""
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
    
    # Preprocess
    eeg = preprocess_eeg(eeg)
    
    # Convert to tensor and add batch dimension
    eeg_tensor = torch.FloatTensor(eeg).unsqueeze(0).to(device)
    
    print(f'EEG shape: {eeg_tensor.shape}')
    
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
            eeg_tensor,
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

