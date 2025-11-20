"""
Neuroscientific visualization utilities for learned graphs
Includes adjacency heatmaps and graph structure visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import Dict, Optional, List
import os


def save_adjacency_heatmap(
    adjacency_matrix: np.ndarray,
    output_path: str,
    title: str = "Adjacency Matrix Heatmap",
    figsize: tuple = (12, 10),
    cmap: str = 'viridis',
    show_values: bool = False,
    node_labels: Optional[List[str]] = None,
    frequency_bands: Optional[List[str]] = None,
    num_channels: Optional[int] = None
):
    """
    Save adjacency matrix as heatmap with neuroscientific annotations
    
    Args:
        adjacency_matrix: Adjacency matrix (num_nodes, num_nodes) or (batch, num_nodes, num_nodes)
        output_path: Path to save the figure
        title: Title of the figure
        figsize: Figure size (width, height)
        cmap: Colormap for heatmap
        show_values: Whether to display values in cells
        node_labels: Optional labels for nodes
        frequency_bands: List of frequency band names
        num_channels: Number of EEG channels
    """
    # Handle batch dimension
    if adjacency_matrix.ndim == 3:
        # Average over batch for visualization
        adjacency_matrix = np.mean(adjacency_matrix, axis=0)
    
    num_nodes = adjacency_matrix.shape[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(adjacency_matrix, cmap=cmap, aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Edge Weight', rotation=270, labelpad=20)
    
    # Add grid lines to separate frequency bands if provided
    if frequency_bands and num_channels:
        num_bands = len(frequency_bands)
        for i in range(1, num_bands):
            pos = i * num_channels
            ax.axhline(y=pos, color='white', linewidth=2, alpha=0.5)
            ax.axvline(x=pos, color='white', linewidth=2, alpha=0.5)
    
    # Add labels if provided
    if node_labels:
        ax.set_xticks(range(len(node_labels)))
        ax.set_yticks(range(len(node_labels)))
        ax.set_xticklabels(node_labels, rotation=90, fontsize=8)
        ax.set_yticklabels(node_labels, fontsize=8)
    elif frequency_bands and num_channels:
        # Create labels: ch0-delta, ch1-delta, ..., ch0-theta, ...
        labels = []
        for band in frequency_bands:
            for ch in range(num_channels):
                labels.append(f"ch{ch}-{band}")
        # Sample labels to avoid overcrowding
        step = max(1, len(labels) // 20)
        tick_positions = list(range(0, len(labels), step))
        tick_labels = [labels[i] for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
        ax.set_yticklabels(tick_labels, fontsize=6)
    
    ax.set_xlabel('Node Index')
    ax.set_ylabel('Node Index')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Show values if requested (only for small matrices)
    if show_values and num_nodes <= 50:
        for i in range(num_nodes):
            for j in range(num_nodes):
                text = ax.text(j, i, f'{adjacency_matrix[i, j]:.2f}',
                             ha="center", va="center", color="white" if adjacency_matrix[i, j] < 0.5 else "black",
                             fontsize=6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved adjacency heatmap to {output_path}")


def save_spatial_functional_comparison(
    A_spatial: np.ndarray,
    A_functional: np.ndarray,
    A_combined: np.ndarray,
    output_path: str,
    frequency_bands: Optional[List[str]] = None,
    num_channels: Optional[int] = None
):
    """
    Create side-by-side comparison of spatial, functional, and combined adjacency matrices
    
    Args:
        A_spatial: Spatial adjacency matrix
        A_functional: Functional connectivity matrix
        A_combined: Combined adjacency matrix
        output_path: Path to save figure
        frequency_bands: List of frequency band names
        num_channels: Number of EEG channels
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    matrices = [
        (A_spatial, 'Spatial Adjacency', 'Blues'),
        (A_functional, 'Functional Connectivity', 'Reds'),
        (A_combined, 'Combined (α·Spatial + β·Functional)', 'Purples')
    ]
    
    for idx, (matrix, title, cmap) in enumerate(matrices):
        if matrix.ndim == 3:
            matrix = np.mean(matrix, axis=0)
        
        im = axes[idx].imshow(matrix, cmap=cmap, aspect='auto', interpolation='nearest')
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Node Index')
        axes[idx].set_ylabel('Node Index')
        
        plt.colorbar(im, ax=axes[idx])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved spatial/functional comparison to {output_path}")


def save_graph_evolution(
    adjacency_matrices: List[np.ndarray],
    output_path: str,
    titles: Optional[List[str]] = None,
    ncols: int = 3
):
    """
    Visualize graph evolution across epochs or layers
    
    Args:
        adjacency_matrices: List of adjacency matrices to visualize
        output_path: Path to save figure
        titles: Optional titles for each subplot
        ncols: Number of columns in subplot grid
    """
    n_plots = len(adjacency_matrices)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, A in enumerate(adjacency_matrices):
        if A.ndim == 3:
            A = np.mean(A, axis=0)
        
        ax = axes[idx]
        im = ax.imshow(A, cmap='viridis', aspect='auto', interpolation='nearest')
        ax.set_title(titles[idx] if titles else f'Graph {idx+1}', fontsize=10)
        ax.set_xlabel('Node Index')
        ax.set_ylabel('Node Index')
        plt.colorbar(im, ax=ax)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved graph evolution to {output_path}")


def visualize_learned_graphs_from_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    model_config: Dict,
    sample_eeg_bands: Optional[Dict[str, torch.Tensor]] = None
):
    """
    Extract and visualize learned graph structures from a trained model
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save visualizations
        model_config: Model configuration dictionary
        sample_eeg_bands: Sample EEG bands to generate graphs (optional)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from models import GraphEnhancedEEG2Text
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphEnhancedEEG2Text(**model_config, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if sample_eeg_bands is not None:
        # Generate graphs from sample data
        with torch.no_grad():
            A, node_features, bandpowers = model.strg(sample_eeg_bands)
            
            # Save adjacency heatmap
            A_np = A.cpu().numpy()
            save_adjacency_heatmap(
                A_np,
                os.path.join(output_dir, 'adjacency_heatmap.png'),
                title='Learned Adjacency Matrix',
                frequency_bands=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                num_channels=model_config.get('num_channels', 64)
            )
            
            # If available, extract spatial and functional separately
            # Note: This requires accessing internal STRG state
            if hasattr(model.strg, 'A_spatial'):
                A_spatial = model.strg.A_spatial.cpu().numpy()
                # Functional would need to be computed
                # For now, save what we have
                save_adjacency_heatmap(
                    A_spatial,
                    os.path.join(output_dir, 'spatial_adjacency.png'),
                    title='Spatial Adjacency Matrix',
                    frequency_bands=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                    num_channels=model_config.get('num_channels', 64)
                )

