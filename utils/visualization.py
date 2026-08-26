"""
Neuroscientific visualization utilities for learned graphs
Includes adjacency heatmaps, graph structure visualization, and STRG illustrations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import Dict, Optional, List, Tuple
import os
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Network graph visualizations will be limited.")


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
            strg_out = model.strg(sample_eeg_bands)
            A_np = strg_out['edge_mask'][0, 0].cpu().numpy() if strg_out['edge_mask'].dim() == 4 else strg_out['edge_mask'][0].cpu().numpy()
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


def visualize_strg_comprehensive(
    strg_module,
    eeg_bands: Dict[str, torch.Tensor],
    output_dir: str,
    epoch: Optional[int] = None,
    electrode_positions: Optional[np.ndarray] = None,
    channel_names: Optional[List[str]] = None
):
    """
    Generate comprehensive STRG visualizations for paper:
    - Spatial vs Functional vs Combined adjacency
    - Frequency-specific connectivity
    - Topographic plots
    - Graph network structure
    
    Args:
        strg_module: STRG module instance
        eeg_bands: Dictionary of frequency bands (batch_size, num_channels, time_steps)
        output_dir: Directory to save visualizations
        epoch: Current epoch (for filename)
        electrode_positions: Electrode 3D positions (num_channels, 3) for topographic plots
        channel_names: Channel names (optional) for labeling
    """
    os.makedirs(output_dir, exist_ok=True)
    device = next(strg_module.parameters()).device
    
    # Process single sample
    if isinstance(eeg_bands, dict):
        sample_eeg_bands = {k: v[:1].to(device) for k, v in eeg_bands.items()}
    else:
        sample_eeg_bands = eeg_bands[:1].to(device) if torch.is_tensor(eeg_bands) else eeg_bands
    
    strg_module.eval()
    with torch.no_grad():
        # Extract components separately for comprehensive visualization
        components = strg_module.extract_strg_components(sample_eeg_bands, return_separate=True)
        
        A_combined = components['A_combined'][0]  # (num_nodes, num_nodes)
        A_spatial = components['A_spatial']  # (num_nodes, num_nodes)
        A_functional_full = components['A_functional_full']  # (num_nodes, num_nodes)
        A_functional_per_band = components['A_functional_per_band']  # Dict
        bandpowers = components['bandpowers'][0]  # (num_channels, num_frequency_bands)
        node_features_np = components['node_features'][0]  # (num_nodes, 1)
    
    num_channels = strg_module.num_channels
    num_frequency_bands = strg_module.num_frequency_bands
    frequency_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    epoch_suffix = f"_epoch_{epoch}" if epoch is not None else ""
    
    # 1. STRG Component Comparison (Spatial, Functional, Combined)
    if A_spatial is not None and A_functional_full is not None:
        save_spatial_functional_comparison(
            A_spatial, A_functional_full, A_combined,
            os.path.join(output_dir, f'strg_components{epoch_suffix}.png'),
            frequency_bands=frequency_bands,
            num_channels=num_channels
        )
    
    # 2. Frequency-Specific Connectivity Visualization
    if len(A_functional_per_band) > 0:
        save_frequency_specific_connectivity(
            A_functional_per_band,
            os.path.join(output_dir, f'frequency_specific_connectivity{epoch_suffix}.png'),
            num_channels=num_channels
        )
    
    # 3. Bandpower Topographic Maps
    save_bandpower_topographic_maps(
        bandpowers,
        os.path.join(output_dir, f'bandpower_topography{epoch_suffix}.png'),
        electrode_positions=electrode_positions,
        channel_names=channel_names,
        frequency_bands=frequency_bands
    )
    
    # 4. Graph Structure Network Visualization
    if HAS_NETWORKX:
        save_graph_network_structure(
            A_combined,
            os.path.join(output_dir, f'graph_network_structure{epoch_suffix}.png'),
            node_features=node_features_np,
            frequency_bands=frequency_bands,
            num_channels=num_channels,
            max_nodes=100  # Limit for visualization clarity
        )
    
    # 5. Combined Adjacency with Frequency Band Blocks
    save_strg_adjacency_with_bands(
        A_combined,
        os.path.join(output_dir, f'strg_adjacency_bands{epoch_suffix}.png'),
        frequency_bands=frequency_bands,
        num_channels=num_channels
    )
    
    print(f"Saved comprehensive STRG visualizations to {output_dir}")


def save_frequency_specific_connectivity(
    A_functional_per_band: Dict[str, np.ndarray],
    output_path: str,
    num_channels: int
):
    """
    Visualize functional connectivity for each frequency band separately
    """
    num_bands = len(A_functional_per_band)
    fig, axes = plt.subplots(1, num_bands, figsize=(5*num_bands, 5))
    if num_bands == 1:
        axes = [axes]
    
    frequency_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    band_colors = {'delta': 'Purples', 'theta': 'Blues', 'alpha': 'Greens', 
                   'beta': 'Oranges', 'gamma': 'Reds'}
    
    for idx, band_name in enumerate(frequency_bands):
        if band_name in A_functional_per_band:
            A_band = A_functional_per_band[band_name]
            im = axes[idx].imshow(A_band, cmap=band_colors.get(band_name, 'viridis'), 
                                 aspect='auto', interpolation='nearest')
            axes[idx].set_title(f'{band_name.capitalize()} Band\nFunctional Connectivity', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Channel Index')
            axes[idx].set_ylabel('Channel Index')
            plt.colorbar(im, ax=axes[idx], label='Correlation')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved frequency-specific connectivity to {output_path}")


def save_bandpower_topographic_maps(
    bandpowers: np.ndarray,
    output_path: str,
    electrode_positions: Optional[np.ndarray] = None,
    channel_names: Optional[List[str]] = None,
    frequency_bands: Optional[List[str]] = None
):
    """
    Create topographic maps showing bandpower across electrodes for each frequency band
    """
    num_channels, num_frequency_bands = bandpowers.shape
    if frequency_bands is None:
        frequency_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    fig, axes = plt.subplots(1, num_frequency_bands, figsize=(5*num_frequency_bands, 4))
    if num_frequency_bands == 1:
        axes = [axes]
    
    for idx, band_name in enumerate(frequency_bands):
        bandpower = bandpowers[:, idx]
        
        # Simple 2D scatter plot (if 3D positions available, could use interpolation)
        if electrode_positions is not None and electrode_positions.shape[1] >= 2:
            # Use X, Y coordinates for 2D projection
            scatter = axes[idx].scatter(electrode_positions[:, 0], electrode_positions[:, 1],
                                       c=bandpower, s=100, cmap='viridis', edgecolors='black')
            axes[idx].set_xlabel('X Position')
            axes[idx].set_ylabel('Y Position')
        else:
            # Fallback: bar plot
            axes[idx].bar(range(num_channels), bandpower, color='steelblue', alpha=0.7)
            axes[idx].set_xlabel('Channel Index')
            axes[idx].set_ylabel('Bandpower')
            if channel_names:
                axes[idx].set_xticks(range(min(len(channel_names), num_channels)))
                axes[idx].set_xticklabels(channel_names[:num_channels], rotation=45, ha='right', fontsize=6)
        
        axes[idx].set_title(f'{band_name.capitalize()} Band\nBandpower Distribution', 
                           fontsize=11, fontweight='bold')
        
        if electrode_positions is not None and electrode_positions.shape[1] >= 2:
            plt.colorbar(scatter, ax=axes[idx], label='Bandpower')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved bandpower topographic maps to {output_path}")


def save_graph_network_structure(
    adjacency_matrix: np.ndarray,
    output_path: str,
    node_features: Optional[np.ndarray] = None,
    frequency_bands: Optional[List[str]] = None,
    num_channels: Optional[int] = None,
    max_nodes: int = 100
):
    """
    Visualize graph as network structure (using networkx)
    Shows actual graph topology, not just adjacency matrix
    """
    if not HAS_NETWORKX:
        print(f"Skipping network structure visualization (networkx not available)")
        return
    
    # Downsample if too many nodes
    num_nodes = adjacency_matrix.shape[0]
    if num_nodes > max_nodes:
        step = num_nodes // max_nodes
        idxs = np.arange(0, num_nodes, step)
        adjacency_matrix = adjacency_matrix[np.ix_(idxs, idxs)]
        if node_features is not None:
            node_features = node_features[idxs]
        num_nodes = len(idxs)
    
    # Create networkx graph
    G = nx.from_numpy_array(adjacency_matrix)
    
    # Threshold edges for clarity (keep top 20% by weight)
    if G.number_of_edges() > num_nodes * 5:  # If too many edges
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        threshold = np.percentile(edge_weights, 80)
        edges_to_remove = [(u, v) for u, v in G.edges() if G[u][v]['weight'] < threshold]
        G.remove_edges_from(edges_to_remove)
    
    # Create layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Node colors based on features or frequency band
    if node_features is not None:
        node_colors = node_features.flatten()
    elif frequency_bands and num_channels:
        # Color by frequency band
        node_colors = []
        for i in range(num_nodes):
            band_idx = i // num_channels if num_channels > 0 else 0
            node_colors.append(band_idx)
    else:
        node_colors = 'lightblue'
    
    # Edge colors based on weight
    edge_weights_list = [G[u][v]['weight'] for u, v in G.edges()]
    edge_colors = edge_weights_list
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100,
                          cmap=plt.cm.viridis, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.3, edge_color=edge_colors,
                          edge_cmap=plt.cm.Greys, ax=ax)
    
    ax.set_title('STRG Network Structure', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved graph network structure to {output_path}")


def save_strg_adjacency_with_bands(
    adjacency_matrix: np.ndarray,
    output_path: str,
    frequency_bands: List[str],
    num_channels: int
):
    """
    Visualize STRG adjacency matrix with clear frequency band block structure
    Annotated version highlighting the spectro-topographic structure
    """
    num_bands = len(frequency_bands)
    fig, ax = plt.subplots(figsize=(14, 12))
    
    im = ax.imshow(adjacency_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
    
    # Add grid lines to separate frequency bands
    for i in range(1, num_bands):
        pos = i * num_channels
        ax.axhline(y=pos, color='white', linewidth=3, alpha=0.8)
        ax.axvline(x=pos, color='white', linewidth=3, alpha=0.8)
    
    # Add band labels
    band_centers = [(i + 0.5) * num_channels for i in range(num_bands)]
    for idx, (center, band_name) in enumerate(zip(band_centers, frequency_bands)):
        ax.text(center, -num_channels*0.05, band_name.upper(), 
               ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')
        ax.text(-num_channels*0.05, center, band_name.upper(), 
               ha='right', va='center', fontsize=11, fontweight='bold', color='black',
               rotation=90)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Edge Weight (α·Spatial + β·Functional)', rotation=270, labelpad=25, fontsize=12)
    
    ax.set_xlabel('Node Index (Channel-Frequency Pairs)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Node Index (Channel-Frequency Pairs)', fontsize=12, fontweight='bold')
    ax.set_title('Spectro-Topographic Relational Graph (STRG) Adjacency Matrix', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved STRG adjacency with bands to {output_path}")

