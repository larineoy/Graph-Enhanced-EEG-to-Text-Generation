"""
Ablation study utilities
Creates model variants for ablation experiments
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from models import GraphEnhancedEEG2Text, STRG, STRE


class AblationModelFactory:
    """Factory for creating model variants for ablation studies"""
    
    @staticmethod
    def create_model(
        ablation_type: str,
        base_config: Dict,
        device: str = 'cuda'
    ) -> nn.Module:
        """
        Create model variant based on ablation type
        
        Args:
            ablation_type: One of:
                - 'full': Full model (baseline)
                - 'no_graph': Sequential model without graph structure
                - 'static_only': Only spatial adjacency, no functional connectivity
                - 'dynamic_only': Only functional connectivity, no spatial topology
                - 'no_contrastive': Without contrastive alignment loss
                - 'no_smoothness': Without graph smoothness regularization
                - 'graph_only': Graph encoding without temporal modeling
            base_config: Base model configuration
            device: Device to run on
            
        Returns:
            model: Model instance for ablation study
        """
        if ablation_type == 'full':
            return GraphEnhancedEEG2Text(
                num_channels=base_config['num_channels'],
                num_frequency_bands=base_config['num_frequency_bands'],
                strg_alpha=base_config.get('strg_alpha', 0.5),
                strg_beta=base_config.get('strg_beta', 0.5),
                use_spatial_topology=True,
                use_functional_connectivity=True,
                node_dim=base_config.get('node_dim', 1),
                graph_embed_dim=base_config['graph_embed_dim'],
                num_gat_layers=base_config.get('num_gat_layers', 2),
                num_gat_heads=base_config.get('num_gat_heads', 4),
                gat_dropout=base_config.get('gat_dropout', 0.1),
                num_temporal_layers=base_config.get('num_temporal_layers', 4),
                num_temporal_heads=base_config.get('num_temporal_heads', 8),
                temporal_ff_dim=base_config.get('temporal_ff_dim', 512),
                temporal_dropout=base_config.get('temporal_dropout', 0.1),
                vocab_size=base_config['vocab_size'],
                decoder_embed_dim=base_config['decoder_embed_dim'],
                num_decoder_layers=base_config.get('num_decoder_layers', 4),
                num_decoder_heads=base_config.get('num_decoder_heads', 8),
                decoder_ff_dim=base_config.get('decoder_ff_dim', 512),
                decoder_dropout=base_config.get('decoder_dropout', 0.1),
                max_decoder_length=base_config.get('max_decoder_length', 128),
                device=device
            )
        
        elif ablation_type == 'no_graph':
            # Sequential model: average frequency bands and use LSTM/Transformer
            from models.sequential_baseline import SequentialEEG2Text
            return SequentialEEG2Text(
                num_channels=base_config['num_channels'],
                num_frequency_bands=base_config['num_frequency_bands'],
                embed_dim=base_config['decoder_embed_dim'],
                num_layers=base_config.get('num_decoder_layers', 4),
                vocab_size=base_config['vocab_size'],
                device=device
            )
        
        elif ablation_type == 'static_only':
            # Only spatial topology
            return GraphEnhancedEEG2Text(
                **{k: v for k, v in base_config.items() if k != 'use_functional_connectivity'},
                use_spatial_topology=True,
                use_functional_connectivity=False,
                strg_beta=0.0,
                device=device
            )
        
        elif ablation_type == 'dynamic_only':
            # Only functional connectivity
            return GraphEnhancedEEG2Text(
                **{k: v for k, v in base_config.items() if k != 'use_spatial_topology'},
                use_spatial_topology=False,
                use_functional_connectivity=True,
                strg_alpha=0.0,
                device=device
            )
        
        elif ablation_type == 'graph_only':
            # No temporal modeling - just graph readout
            return GraphEnhancedEEG2Text(
                **base_config,
                num_temporal_layers=0,  # Disable temporal Transformer
                device=device
            )
        
        else:
            raise ValueError(f"Unknown ablation type: {ablation_type}")


def get_ablation_loss_weights(ablation_type: str, base_config: Dict) -> Dict[str, float]:
    """
    Get loss weights for ablation study
    
    Args:
        ablation_type: Ablation type
        base_config: Base configuration
        
    Returns:
        loss_weights: Dictionary with lambda_smooth and lambda_contrastive
    """
    base_lambda_smooth = base_config.get('lambda_smooth', 0.1)
    base_lambda_contrastive = base_config.get('lambda_contrastive', 0.2)
    
    if ablation_type == 'no_contrastive':
        return {
            'lambda_smooth': base_lambda_smooth,
            'lambda_contrastive': 0.0
        }
    elif ablation_type == 'no_smoothness':
        return {
            'lambda_smooth': 0.0,
            'lambda_contrastive': base_lambda_contrastive
        }
    elif ablation_type == 'full':
        return {
            'lambda_smooth': base_lambda_smooth,
            'lambda_contrastive': base_lambda_contrastive
        }
    else:
        return {
            'lambda_smooth': base_lambda_smooth,
            'lambda_contrastive': base_lambda_contrastive
        }

