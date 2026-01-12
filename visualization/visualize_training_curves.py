"""
Plot training curves from training log
Shows loss and metrics over epochs
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from typing import Dict, List, Optional


def plot_training_curves(
    log_path: str,
    output_path: str,
    show_metrics: bool = True,
    figsize: tuple = (14, 5)
):
    """
    Plot training curves from training log JSON file
    
    Args:
        log_path: Path to training_log.json
        output_path: Path to save figure
        show_metrics: Whether to show validation metrics
        figsize: Figure size (width, height)
    """
    # Load training log
    with open(log_path, 'r') as f:
        log = json.load(f)
    
    epochs = [entry['epoch'] for entry in log]
    train_loss = [entry['train_loss'] for entry in log]
    val_loss = [entry['val_loss'] for entry in log]
    
    # Extract metrics if available
    val_metrics = {}
    if 'val_metrics' in log[0]:
        all_metrics = set()
        for entry in log:
            if 'val_metrics' in entry:
                all_metrics.update(entry['val_metrics'].keys())
        
        for metric in all_metrics:
            val_metrics[metric] = [
                entry.get('val_metrics', {}).get(metric, 0.0) 
                for entry in log
            ]
    
    # Create figure
    if show_metrics and len(val_metrics) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 5))
        axes = [axes]
    
    # Plot 1: Loss curves
    ax = axes[0]
    ax.plot(epochs, train_loss, label='Train Loss', marker='o', linewidth=2, markersize=6, color='#2E86AB')
    ax.plot(epochs, val_loss, label='Val Loss', marker='s', linewidth=2, markersize=6, color='#A23B72')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training and Validation Loss', fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=min(epochs), right=max(epochs))
    
    # Plot 2: BLEU scores (if available)
    if show_metrics and len(val_metrics) > 0 and len(axes) > 1:
        ax = axes[1]
        bleu_metrics = {k: v for k, v in val_metrics.items() if 'bleu' in k.lower()}
        if len(bleu_metrics) > 0:
            for metric_name, values in sorted(bleu_metrics.items()):
                # Convert to percentage if needed (BLEU is typically 0-1, but sometimes stored as 0-100)
                display_values = [v * 100 if v < 1 else v for v in values]
                label = metric_name.upper().replace('_', '-')
                ax.plot(epochs, display_values, label=label, marker='o', linewidth=1.5, markersize=4)
            ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax.set_ylabel('BLEU Score (%)', fontsize=12, fontweight='bold')
            ax.set_title('BLEU Scores', fontsize=13, fontweight='bold', pad=10)
            ax.legend(fontsize=9, frameon=True, fancybox=True, shadow=True, ncol=2)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(left=min(epochs), right=max(epochs))
    
    # Plot 3: ROUGE/BERTScore (if available)
    if show_metrics and len(val_metrics) > 0 and len(axes) > 2:
        ax = axes[2]
        rouge_metrics = {k: v for k, v in val_metrics.items() if 'rouge' in k.lower() or 'bert' in k.lower()}
        if len(rouge_metrics) > 0:
            # Show key metrics: ROUGE-L-F and BERTScore-F1
            key_metrics = ['rougeL_F', 'bertscore_f1', 'rouge1_F']
            colors = ['#F18F01', '#C73E1D', '#6A994E']
            markers = ['o', 's', '^']
            
            for idx, metric_name in enumerate(key_metrics):
                if metric_name in rouge_metrics:
                    values = rouge_metrics[metric_name]
                    # Convert to percentage if needed
                    display_values = [v * 100 if v < 1 and 'bert' not in metric_name else v for v in values]
                    label = metric_name.upper().replace('_', '-')
                    ax.plot(epochs, display_values, label=label, marker=markers[idx % len(markers)], 
                           linewidth=1.5, markersize=4, color=colors[idx % len(colors)])
            
            ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax.set_title('ROUGE & BERTScore', fontsize=13, fontweight='bold', pad=10)
            ax.legend(fontsize=9, frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(left=min(epochs), right=max(epochs))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training curves to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot training curves from training log')
    parser.add_argument('--log', type=str, default='checkpoints/training_log.json',
                       help='Path to training_log.json')
    parser.add_argument('--output', type=str, default='checkpoints/training_curves.png',
                       help='Path to save figure')
    parser.add_argument('--no_metrics', action='store_true',
                       help='Only plot loss, skip metrics')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log):
        print(f"Error: Training log not found: {args.log}")
        return
    
    plot_training_curves(
        args.log,
        args.output,
        show_metrics=not args.no_metrics
    )


if __name__ == '__main__':
    main()
