"""
Ablation study factory.

Each variant maps onto one row of the paper's hypothesis-driven ablation table.
"""

from typing import Dict

import torch.nn as nn


class AblationModelFactory:
    """Factory for creating model variants for ablation experiments."""

    @staticmethod
    def _full_kwargs(base_config: Dict, device: str) -> Dict:
        decoder_name = base_config.get(
            'decoder_pretrained_name',
            base_config.get('pretrained_name', 'facebook/bart-base')
        )
        return dict(
            num_channels=base_config['num_channels'],
            num_frequency_bands=base_config['num_frequency_bands'],
            k_spatial=base_config.get('k_spatial', 6),
            k_functional=base_config.get('k_functional', 6),
            use_spatial_topology=base_config.get('use_spatial_topology', True),
            use_functional_connectivity=base_config.get('use_functional_connectivity', True),
            use_frequency_nodes=base_config.get('use_frequency_nodes', True),
            dynamic_functional=base_config.get('dynamic_functional', True),
            node_dim=base_config.get('node_dim', 1),
            graph_embed_dim=base_config['graph_embed_dim'],
            num_gat_layers=base_config.get('num_gat_layers', 2),
            num_gat_heads=base_config.get('num_gat_heads', 4),
            gat_dropout=base_config.get('gat_dropout', 0.1),
            num_temporal_layers=base_config.get('num_temporal_layers', 4),
            num_temporal_heads=base_config.get('num_temporal_heads', 8),
            temporal_ff_dim=base_config.get('temporal_ff_dim', 512),
            temporal_dropout=base_config.get('temporal_dropout', 0.1),
            decoder_pretrained_name=decoder_name,
            max_decoder_length=base_config.get('max_decoder_length', 128),
            device=device
        )

    @staticmethod
    def create_model(
        ablation_type: str,
        base_config: Dict,
        device: str = 'cuda'
    ) -> nn.Module:
        from models import GraphEnhancedEEG2Text, SequentialEEG2Text

        aliases = {
            'static_only': 'spatial_only',
            'dynamic_only': 'functional_only',
            'graph_only': 'gat_only',
            'spatial_functional': 'full',
            'electrode_frequency_nodes': 'full',
            'dynamic_functional': 'full',
            'gat_temporal': 'full',
        }
        ablation_type = aliases.get(ablation_type, ablation_type)
        kwargs = AblationModelFactory._full_kwargs(base_config, device)

        if ablation_type == 'full':
            return GraphEnhancedEEG2Text(**kwargs)

        if ablation_type == 'no_graph':
            return SequentialEEG2Text(
                num_channels=base_config['num_channels'],
                num_frequency_bands=base_config['num_frequency_bands'],
                embed_dim=base_config['graph_embed_dim'],
                num_layers=base_config.get('num_temporal_layers', 4),
                num_heads=base_config.get('num_temporal_heads', 8),
                ff_dim=base_config.get('temporal_ff_dim', 512),
                dropout=base_config.get('temporal_dropout', 0.1),
                decoder_pretrained_name=kwargs['decoder_pretrained_name'],
                max_decoder_length=kwargs['max_decoder_length'],
                device=device
            )

        if ablation_type == 'spatial_only':
            kwargs['use_spatial_topology'] = True
            kwargs['use_functional_connectivity'] = False
            return GraphEnhancedEEG2Text(**kwargs)

        if ablation_type == 'functional_only':
            kwargs['use_spatial_topology'] = False
            kwargs['use_functional_connectivity'] = True
            return GraphEnhancedEEG2Text(**kwargs)

        if ablation_type == 'electrode_nodes':
            kwargs['use_frequency_nodes'] = False
            kwargs['node_dim'] = base_config['num_frequency_bands']
            return GraphEnhancedEEG2Text(**kwargs)

        if ablation_type == 'static_functional':
            kwargs['dynamic_functional'] = False
            kwargs['use_functional_connectivity'] = True
            return GraphEnhancedEEG2Text(**kwargs)

        if ablation_type == 'gat_only':
            kwargs['num_temporal_layers'] = 0
            return GraphEnhancedEEG2Text(**kwargs)

        raise ValueError(
            f"Unknown ablation type: {ablation_type}. "
            "Expected one of: full, no_graph, spatial_only, functional_only, "
            "electrode_nodes, static_functional, gat_only"
        )


def get_ablation_loss_weights(ablation_type: str, base_config: Dict) -> Dict[str, float]:
    """Paper uses generation loss only for every variant."""
    del ablation_type
    return {
        'lambda_smooth': float(base_config.get('lambda_smooth', 0.0)),
        'lambda_contrastive': float(base_config.get('lambda_contrastive', 0.0))
    }
