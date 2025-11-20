"""
Hyperparameter sensitivity analysis utilities
Tests different combinations of hyperparameters
"""

from typing import Dict, List, Tuple
import itertools
import numpy as np


def generate_hyperparameter_grid(base_config: Dict) -> List[Dict]:
    """
    Generate grid of hyperparameter combinations for sensitivity analysis
    
    Args:
        base_config: Base configuration
        
    Returns:
        config_grid: List of configuration dictionaries
    """
    # Define parameter ranges
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    beta_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Filter: only valid combinations (alpha + beta can vary, but typically sum to 1)
    # Or test independently
    configs = []
    
    # Test alpha values (with beta = 1 - alpha)
    for alpha in alpha_values:
        config = base_config.copy()
        config['strg_alpha'] = alpha
        config['strg_beta'] = 1.0 - alpha
        config['experiment_name'] = f'alpha_{alpha}_beta_{1.0-alpha}'
        configs.append(config)
    
    # Test beta values independently (with alpha fixed at 0.5)
    for beta in beta_values:
        if beta != 0.5:  # Already covered above
            config = base_config.copy()
            config['strg_alpha'] = 0.5
            config['strg_beta'] = beta
            config['experiment_name'] = f'alpha_0.5_beta_{beta}'
            configs.append(config)
    
    return configs


def generate_loss_weight_grid(base_config: Dict) -> List[Dict]:
    """
    Generate grid of loss weight combinations
    
    Args:
        base_config: Base configuration
        
    Returns:
        config_grid: List of configurations with different loss weights
    """
    lambda_smooth_values = [0.0, 0.05, 0.1, 0.2, 0.5]
    lambda_contrastive_values = [0.0, 0.1, 0.2, 0.3, 0.5]
    
    configs = []
    for lambda_smooth in lambda_smooth_values:
        for lambda_contrastive in lambda_contrastive_values:
            config = base_config.copy()
            config['lambda_smooth'] = lambda_smooth
            config['lambda_contrastive'] = lambda_contrastive
            config['experiment_name'] = f'lambda_smooth_{lambda_smooth}_contrastive_{lambda_contrastive}'
            configs.append(config)
    
    return configs


def generate_architecture_grid(base_config: Dict) -> List[Dict]:
    """
    Generate grid of architecture hyperparameters
    
    Args:
        base_config: Base configuration
        
    Returns:
        config_grid: List of configurations with different architectures
    """
    num_gat_layers_values = [1, 2, 3, 4]
    num_gat_heads_values = [2, 4, 8]
    graph_embed_dim_values = [128, 256, 512]
    
    configs = []
    for num_gat_layers in num_gat_layers_values:
        for num_gat_heads in num_gat_heads_values:
            for graph_embed_dim in graph_embed_dim_values:
                config = base_config.copy()
                config['num_gat_layers'] = num_gat_layers
                config['num_gat_heads'] = num_gat_heads
                config['graph_embed_dim'] = graph_embed_dim
                config['experiment_name'] = f'gat_layers_{num_gat_layers}_heads_{num_gat_heads}_embed_{graph_embed_dim}'
                configs.append(config)
    
    return configs

