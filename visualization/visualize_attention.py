"""
Visualize attention patterns from decoder
Shows which EEG regions attend to which words
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import sys
from typing import Dict, List, Optional

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


def extract_attention_from_decoder(model, eeg_bands, text_tokens, device):
    """
    Extract attention weights from decoder
    
    Note: This requires modifying the decoder to return attention weights.
    For now, we'll use a simplified approach by accessing intermediate layers.
    
    Args:
        model: Trained model
        eeg_bands: EEG frequency bands
        text_tokens: Text tokens
        device: Device
    
    Returns:
        attention_weights: Attention weights if available, None otherwise
    """
    model.eval()
    
    # For now, we'll visualize decoder self-attention patterns
    # This would require modifying the decoder to return attention weights
    # Since PyTorch's TransformerDecoder doesn't easily expose attention,
    # we'll use a workaround: visualize the logits as a proxy for attention
    
    with torch.no_grad():
        # Get logits (as proxy for attention patterns)
        logits, _ = model(eeg_bands, text_tokens)
        
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
        
        # Get top-k tokens per position (as proxy for attention)
        top_k = 5
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
        
        return {
            'logits': logits.cpu().numpy(),
            'probs': probs.cpu().numpy(),
            'top_probs': top_probs.cpu().numpy(),
            'top_indices': top_indices.cpu().numpy()
        }


def visualize_attention_heatmap(
    attention_data: Dict,
    tokenizer,
    text_tokens: torch.Tensor,
    output_path: str,
    num_examples: int = 3
):
    """
    Visualize attention heatmap
    
    Args:
        attention_data: Dictionary with attention/logit data
        tokenizer: Text tokenizer
        text_tokens: Original text tokens
        output_path: Path to save figure
        num_examples: Number of examples to show
    """
    # For now, we'll visualize the probability distribution over vocabulary
    # This is a simplified visualization - real attention would show source-target alignment
    
    probs = attention_data['probs']  # (batch_size, seq_len, vocab_size)
    top_probs = attention_data['top_probs']  # (batch_size, seq_len, top_k)
    top_indices = attention_data['top_indices']  # (batch_size, seq_len, top_k)
    
    batch_size = min(num_examples, probs.shape[0])
    
    # Create figure
    fig, axes = plt.subplots(batch_size, 1, figsize=(14, 3 * batch_size))
    if batch_size == 1:
        axes = [axes]
    
    for batch_idx in range(batch_size):
        ax = axes[batch_idx]
        
        # Get tokens for this example
        tokens = text_tokens[batch_idx].cpu().tolist()
        if tokenizer is not None and hasattr(tokenizer, 'decode'):
            try:
                token_strings = [tokenizer.decode([t]) for t in tokens[:20]]  # Limit to 20 tokens
            except:
                token_strings = [str(t) for t in tokens[:20]]
        else:
            token_strings = [str(t) for t in tokens[:20]]
        
        # Get top-k probabilities for each position
        seq_len = min(len(token_strings), probs.shape[1])
        top_probs_example = top_probs[batch_idx, :seq_len, :]  # (seq_len, top_k)
        
        # Create heatmap: position vs top-k tokens
        im = ax.imshow(top_probs_example.T, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        
        # Customize axes
        ax.set_xlabel('Token Position', fontsize=10, fontweight='bold')
        ax.set_ylabel('Top-K Tokens', fontsize=10, fontweight='bold')
        ax.set_title(f'Example {batch_idx + 1}: Token Probability Distribution (Top-5)', 
                    fontsize=11, fontweight='bold', pad=10)
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels([f"{i}" for i in range(seq_len)], rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(top_probs_example.shape[1]))
        ax.set_yticklabels([f"Top-{i+1}" for i in range(top_probs_example.shape[1])], fontsize=8)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Probability')
        
        # Add token labels below
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(range(seq_len))
        ax2.set_xticklabels([s[:5] for s in token_strings[:seq_len]], 
                           rotation=45, ha='left', fontsize=7, alpha=0.7)
        ax2.set_xlabel('Tokens', fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved attention heatmap to {output_path}")


def visualize_attention_examples(
    model,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    num_examples: int = 5,
    output_path: str = 'checkpoints/attention_heatmaps.png'
):
    """
    Visualize attention patterns for multiple examples
    
    Args:
        model: Trained model
        dataloader: DataLoader with test/val data
        tokenizer: Text tokenizer
        device: Device
        num_examples: Number of examples to show
        output_path: Path to save figure
    """
    examples_collected = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if examples_collected >= num_examples:
            break
        
        eeg_bands = {k: v.to(device) for k, v in batch['eeg_bands'].items()}
        text_tokens = batch['text_tokens'].to(device)
        
        # Extract attention (simplified - using logits as proxy)
        try:
            attention_data = extract_attention_from_decoder(model, eeg_bands, text_tokens, device)
            
            # Visualize
            visualize_attention_heatmap(
                attention_data,
                tokenizer,
                text_tokens,
                output_path,
                num_examples=min(num_examples - examples_collected, text_tokens.shape[0])
            )
            
            examples_collected += text_tokens.shape[0]
        except Exception as e:
            print(f"Warning: Error extracting attention for batch {batch_idx}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description='Visualize decoder attention patterns')
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
    parser.add_argument('--output', type=str, default='checkpoints/attention_heatmaps.png',
                       help='Path to save figure')
    
    args = parser.parse_args()
    
    print("="*70)
    print("VISUALIZING DECODER ATTENTION PATTERNS")
    print("="*70)
    print("\nNote: This visualization shows token probability distributions")
    print("as a proxy for attention. For true attention weights, the decoder")
    print("would need to be modified to return attention matrices.")
    print("="*70)
    
    # Load config
    config = load_config(args.config)
    set_seed(config['seed'])
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
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
        batch_size=3,  # Use small batch for visualization
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer, data_config['max_seq_length'], max_eeg_length=max_eeg_length),
        num_workers=0
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # Generate visualizations
    print(f"\nGenerating attention visualizations for {args.num_examples} examples...")
    visualize_attention_examples(
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
