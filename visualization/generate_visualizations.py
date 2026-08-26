#!/usr/bin/env python3
"""
Generate visualizations from a trained model checkpoint
"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train import load_config, create_model, dataset_kwargs_from_config, get_decoder_tokenizer, sync_model_config_from_dataset
from preprocessing.preprocessing import ZuCoDataset, collate_fn
from utils.visualization import save_adjacency_heatmap, visualize_strg_comprehensive


def main():
    parser = argparse.ArgumentParser(description='Generate visualizations from a trained model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file (if not in checkpoint)')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='checkpoints/visualizations',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    print("="*70)
    print("GENERATING MODEL VISUALIZATIONS")
    print("="*70)
    
    # Load checkpoint
    print(f"\n[1/4] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Get config from checkpoint or load from file
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("  ✓ Config loaded from checkpoint")
    else:
        config = load_config(args.config)
        print(f"  ✓ Config loaded from {args.config}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  ✓ Using device: {device}")
    
    print("\n[2/4] Loading sample data...")
    tokenizer = get_decoder_tokenizer(config)
    print("  ✓ BART tokenizer loaded")
    val_dataset = ZuCoDataset(
        args.data_dir,
        split='val',
        **dataset_kwargs_from_config(config)
    )
    sync_model_config_from_dataset(config, val_dataset)

    print("\n[3/4] Creating model...")
    model = create_model(config, device)
    if getattr(val_dataset, 'electrode_positions', None) is not None:
        model.set_electrode_positions(val_dataset.electrode_positions)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    if 'epoch' in checkpoint:
        print(f"  ✓ Model from epoch {checkpoint['epoch'] + 1}")
    if 'best_val_loss' in checkpoint:
        print(f"  ✓ Best validation loss: {checkpoint['best_val_loss']:.4f}")

    data_config = config['data']
    
    # Get max_eeg_length from config
    max_eeg_length = config['model'].get('max_eeg_length', 20000)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Use batch_size=1 for visualization
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer, data_config['max_seq_length'], max_eeg_length=max_eeg_length),
        num_workers=0
    )
    
    print(f"  ✓ Loaded validation dataset: {len(val_dataset)} samples")
    
    # Get a sample batch
    sample_batch = next(iter(val_loader))
    sample_eeg_bands = {k: v.to(device) for k, v in sample_batch['eeg_bands'].items()}
    print("  ✓ Sample batch prepared")
    
    # Generate visualizations
    print("\n[4/4] Generating visualizations...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Simple adjacency heatmap
        with torch.no_grad():
            strg_out = model.strg(sample_eeg_bands)
            A_np = strg_out['edge_mask'][0, 0].cpu().numpy()
            
            save_adjacency_heatmap(
                A_np,
                os.path.join(args.output_dir, 'adjacency_heatmap.png'),
                title='Learned Adjacency Matrix (Best Model)',
                frequency_bands=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                num_channels=config['model']['num_channels']
            )
            print("  ✓ Saved: adjacency_heatmap.png")
        
        # Comprehensive STRG visualizations
        try:
            visualize_strg_comprehensive(
                model.strg,
                sample_eeg_bands,
                args.output_dir,
                epoch=None,
                electrode_positions=None,
                channel_names=None
            )
            print("  ✓ Saved: comprehensive STRG visualizations")
        except Exception as e:
            print(f"  ⚠ Warning: Could not generate comprehensive visualizations: {e}")
            print("  ✓ Basic adjacency heatmap was saved successfully")
        
    except Exception as e:
        print(f"  ✗ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"  ✓ Visualizations saved to: {args.output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
