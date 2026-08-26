"""
Visualize example predictions: EEG → Ground Truth → Model Prediction
Shows qualitative results with correct/incorrect parts highlighted
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import json
import os
import sys
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train import (
    load_config,
    create_model,
    set_seed,
    dataset_kwargs_from_config,
    get_decoder_tokenizer,
    sync_model_config_from_dataset,
)
from preprocessing.preprocessing import ZuCoDataset, collate_fn


def highlight_differences(text1: str, text2: str) -> tuple:
    """
    Highlight differences between two texts
    Returns tokens with matching/non-matching indicators
    """
    tokens1 = text1.split()
    tokens2 = text2.split()
    
    # Simple word-level comparison
    matched = []
    ref_only = []
    pred_only = []
    
    max_len = max(len(tokens1), len(tokens2))
    for i in range(max_len):
        ref_token = tokens1[i] if i < len(tokens1) else None
        pred_token = tokens2[i] if i < len(tokens2) else None
        
        if ref_token == pred_token and ref_token is not None:
            matched.append((ref_token, 'match'))
        elif ref_token is not None and pred_token is not None:
            matched.append((ref_token, 'ref_only'))
            matched.append((pred_token, 'pred_only'))
        elif ref_token is not None:
            matched.append((ref_token, 'ref_only'))
        elif pred_token is not None:
            matched.append((pred_token, 'pred_only'))
    
    return tokens1, tokens2


def visualize_prediction_examples(
    model,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    num_examples: int = 5,
    output_path: str = 'checkpoints/example_predictions.png'
):
    """
    Visualize example predictions
    
    Args:
        model: Trained model
        dataloader: DataLoader with test/val data
        tokenizer: Text tokenizer
        device: Device
        num_examples: Number of examples to show
        output_path: Path to save figure
    """
    model.eval()
    
    examples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if len(examples) >= num_examples:
                break
            
            eeg_bands = {k: v.to(device) for k, v in batch['eeg_bands'].items()}
            texts = batch['text']
            text_tokens = batch['text_tokens'].to(device)
            
            # Generate predictions
            try:
                # Get token IDs for generation
                if tokenizer is not None:
                    try:
                        cls_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token) if hasattr(tokenizer, 'cls_token') else 101
                        sep_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token) if hasattr(tokenizer, 'sep_token') else 102
                        pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token) if hasattr(tokenizer, 'pad_token') else 0
                        bos_token_id = cls_id[0] if isinstance(cls_id, list) else cls_id
                        eos_token_id = sep_id[0] if isinstance(sep_id, list) else sep_id
                        pad_token_id = pad_id[0] if isinstance(pad_id, list) else pad_id
                    except:
                        bos_token_id = 101
                        eos_token_id = 102
                        pad_token_id = 0
                else:
                    bos_token_id = 1
                    eos_token_id = 2
                    pad_token_id = 0
                
                generated = model.generate(
                    eeg_bands,
                    bos_token_id=bos_token_id,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    max_length=128
                )
                
                # Decode predictions
                for i in range(len(texts)):
                    if len(examples) >= num_examples:
                        break
                    
                    ref_text = texts[i]
                    if tokenizer is not None and hasattr(tokenizer, 'decode'):
                        pred_text = tokenizer.decode(generated[i].cpu().tolist(), skip_special_tokens=True)
                    else:
                        pred_text = ' '.join([str(t.item()) for t in generated[i]])
                    
                    examples.append({
                        'reference': ref_text,
                        'prediction': pred_text,
                        'index': batch_idx * dataloader.batch_size + i
                    })
            except Exception as e:
                print(f"Warning: Error generating prediction for batch {batch_idx}: {e}")
                continue
    
    # Create visualization
    n_examples = len(examples)
    if n_examples == 0:
        print("Error: No examples generated")
        return
    
    fig, axes = plt.subplots(n_examples, 1, figsize=(14, 2.5 * n_examples))
    if n_examples == 1:
        axes = [axes]
    
    for idx, example in enumerate(examples):
        ax = axes[idx]
        ax.axis('off')
        
        ref_text = example['reference']
        pred_text = example['prediction']
        
        # Simple word-level comparison
        ref_words = ref_text.split()
        pred_words = pred_text.split()
        
        # Calculate word overlap
        ref_set = set(ref_words)
        pred_set = set(pred_words)
        overlap = len(ref_set & pred_set)
        total_unique = len(ref_set | pred_set)
        overlap_pct = (overlap / total_unique * 100) if total_unique > 0 else 0
        
        # Display text
        y_pos = 0.9
        ax.text(0.05, y_pos, f'Example {idx + 1} (Overlap: {overlap_pct:.1f}%)', 
               fontsize=11, fontweight='bold', transform=ax.transAxes)
        
        y_pos -= 0.15
        ax.text(0.05, y_pos, 'Ground Truth:', fontsize=10, fontweight='bold', 
               color='#2E86AB', transform=ax.transAxes)
        ax.text(0.05, y_pos - 0.08, ref_text, fontsize=9, 
               color='#333333', wrap=True, transform=ax.transAxes)
        
        y_pos -= 0.25
        ax.text(0.05, y_pos, 'Prediction:', fontsize=10, fontweight='bold', 
               color='#A23B72', transform=ax.transAxes)
        ax.text(0.05, y_pos - 0.08, pred_text, fontsize=9, 
               color='#333333', wrap=True, transform=ax.transAxes)
        
        # Add similarity indicator
        if overlap_pct > 50:
            status_color = '#6A994E'  # Green
            status_text = '✓ Good match'
        elif overlap_pct > 20:
            status_color = '#F18F01'  # Orange
            status_text = '△ Partial match'
        else:
            status_color = '#C73E1D'  # Red
            status_text = '✗ Poor match'
        
        ax.text(0.95, 0.9, status_text, fontsize=10, fontweight='bold',
               color=status_color, ha='right', transform=ax.transAxes)
        
        # Add horizontal line
        ax.axhline(y=0.02, color='gray', linewidth=1, alpha=0.3, transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {n_examples} example predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize example predictions')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                       help='Dataset split')
    parser.add_argument('--num_examples', type=int, default=5,
                       help='Number of examples to visualize')
    parser.add_argument('--output', type=str, default='checkpoints/example_predictions.png',
                       help='Path to save figure')
    
    args = parser.parse_args()
    
    print("="*70)
    print("VISUALIZING EXAMPLE PREDICTIONS")
    print("="*70)
    
    # Load config
    config = load_config(args.config)
    set_seed(config['seed'])
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = get_decoder_tokenizer(config)
    print("✓ BART tokenizer loaded")

    print(f"\nLoading {args.split} dataset...")
    dataset = ZuCoDataset(
        args.data_dir,
        split=args.split,
        **dataset_kwargs_from_config(config)
    )
    sync_model_config_from_dataset(config, dataset)

    print(f"\nLoading model from {args.checkpoint}...")
    model = create_model(config, device)
    if getattr(dataset, 'electrode_positions', None) is not None:
        model.set_electrode_positions(dataset.electrode_positions)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✓ Model loaded")

    data_config = config['data']
    max_eeg_length = config['model'].get('max_eeg_length', 20000)
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Use batch_size=1 for visualization
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer, data_config['max_seq_length'], max_eeg_length=max_eeg_length),
        num_workers=0
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # Generate visualizations
    print(f"\nGenerating {args.num_examples} example predictions...")
    visualize_prediction_examples(
        model,
        dataloader,
        tokenizer,
        device,
        num_examples=args.num_examples,
        output_path=args.output
    )
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
