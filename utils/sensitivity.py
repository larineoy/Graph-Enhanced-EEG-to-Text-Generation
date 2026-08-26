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
    k_spatial_values = [3, 6, 8, 12]
    k_functional_values = [3, 6, 8, 12]
    configs = []
    for k_spatial in k_spatial_values:
        for k_functional in k_functional_values:
            config = base_config.copy()
            config['k_spatial'] = k_spatial
            config['k_functional'] = k_functional
            config['experiment_name'] = f'ks_{k_spatial}_kf_{k_functional}'
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

