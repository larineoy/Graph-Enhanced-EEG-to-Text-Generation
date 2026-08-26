"""
Hyperparameter sensitivity analysis
Tests different combinations of alpha, beta, loss weights, etc.
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import json
from tqdm import tqdm
import numpy as np
from typing import Dict, List

from models import GraphEnhancedEEG2Text
from preprocessing.preprocessing import ZuCoDataset, collate_fn
from utils.losses import CompositeLoss
from utils.metrics import evaluate_predictions
from utils.sensitivity import generate_hyperparameter_grid, generate_loss_weight_grid
from utils.statistics import compute_statistics
from train import (
    load_config,
    set_seed,
    dataset_kwargs_from_config,
    get_decoder_tokenizer,
    move_eeg_batch,
    sync_model_config_from_dataset,
)


def evaluate_hyperparameter_config(
    config: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    tokenizer,
    num_epochs: int = 20,
    electrode_positions=None
):
    """Quick evaluation of a hyperparameter configuration"""
    model = GraphEnhancedEEG2Text(
        num_channels=config['num_channels'],
        num_frequency_bands=config['num_frequency_bands'],
        k_spatial=config.get('k_spatial', 6),
        k_functional=config.get('k_functional', 6),
        use_spatial_topology=config.get('use_spatial_topology', True),
        use_functional_connectivity=config.get('use_functional_connectivity', True),
        use_frequency_nodes=config.get('use_frequency_nodes', True),
        dynamic_functional=config.get('dynamic_functional', True),
        node_dim=config.get('node_dim', 1),
        graph_embed_dim=config['graph_embed_dim'],
        num_gat_layers=config.get('num_gat_layers', 2),
        num_gat_heads=config.get('num_gat_heads', 4),
        gat_dropout=config.get('gat_dropout', 0.1),
        num_temporal_layers=config.get('num_temporal_layers', 4),
        num_temporal_heads=config.get('num_temporal_heads', 8),
        temporal_ff_dim=config.get('temporal_ff_dim', 512),
        temporal_dropout=config.get('temporal_dropout', 0.1),
        decoder_pretrained_name=config.get('decoder_pretrained_name', 'facebook/bart-base'),
        max_decoder_length=config.get('max_decoder_length', 128),
        device=device
    )
    if electrode_positions is not None:
        model.set_electrode_positions(electrode_positions)
    model = model.to(device)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    criterion = CompositeLoss(
        lambda_smooth=config.get('lambda_smooth', 0.0),
        lambda_contrastive=config.get('lambda_contrastive', 0.0),
        vocab_size=getattr(tokenizer, 'vocab_size', len(tokenizer)),
        ignore_index=pad_token_id
    )

    best_val_metrics = None
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            eeg_bands, window_mask, eeg_bands_full, eeg_windows = move_eeg_batch(batch, device)
            text_tokens = batch['text_tokens'].to(device)
            logits, _ = model(
                eeg_bands,
                text_tokens,
                window_mask=window_mask,
                eeg_bands_full=eeg_bands_full,
                eeg_windows=eeg_windows
            )
            targets = text_tokens
            if logits.shape[1] != targets.shape[1]:
                min_len = min(logits.shape[1], targets.shape[1])
                logits = logits[:, :min_len, :]
                targets = targets[:, :min_len]
            loss, _ = criterion(logits=logits, targets=targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            model.eval()
            all_references = []
            all_candidates = []
            with torch.no_grad():
                for batch in val_loader:
                    eeg_bands, window_mask, eeg_bands_full, eeg_windows = move_eeg_batch(batch, device)
                    generated = model.generate(
                        eeg_bands,
                        max_length=config.get('max_decoder_length', 128),
                        window_mask=window_mask,
                        eeg_bands_full=eeg_bands_full,
                        eeg_windows=eeg_windows
                    )
                    for i, text in enumerate(batch['text']):
                        all_references.append(text.split())
                        all_candidates.append(
                            tokenizer.decode(generated[i].cpu().tolist(), skip_special_tokens=True).split()
                        )
            metrics = evaluate_predictions(all_references, all_candidates, compute_bert=False)
            if best_val_metrics is None or metrics.get('bleu_4', 0) > best_val_metrics.get('bleu_4', 0):
                best_val_metrics = metrics

    return best_val_metrics or {}


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter sensitivity analysis')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='sensitivity_results')
    parser.add_argument('--analysis_type', type=str, default='k_neighbors',
                       choices=['k_neighbors', 'alpha_beta', 'loss_weights', 'architecture'],
                       help='Type of sensitivity analysis')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of epochs per configuration (reduced for speed)')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    set_seed(config['seed'])
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer = get_decoder_tokenizer(config)
    train_dataset = ZuCoDataset(args.data_dir, split='train', **dataset_kwargs_from_config(config))
    val_dataset = ZuCoDataset(args.data_dir, split='val', **dataset_kwargs_from_config(config))
    sync_model_config_from_dataset(config, train_dataset)

    max_eeg_length = config['model'].get('max_eeg_length', 20000)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=lambda x: collate_fn(
            x, tokenizer, config['data']['max_seq_length'], max_eeg_length=max_eeg_length
        ),
        num_workers=config.get('num_workers', 0)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=lambda x: collate_fn(
            x, tokenizer, config['data']['max_seq_length'], max_eeg_length=max_eeg_length
        ),
        num_workers=config.get('num_workers', 0)
    )

    base_config = {
        'num_channels': config['model']['num_channels'],
        'num_frequency_bands': config['model']['num_frequency_bands'],
        'k_spatial': config['model']['strg'].get('k_spatial', 6),
        'k_functional': config['model']['strg'].get('k_functional', 6),
        'use_spatial_topology': config['model']['strg']['use_spatial_topology'],
        'use_functional_connectivity': config['model']['strg']['use_functional_connectivity'],
        'use_frequency_nodes': config['model']['strg'].get('use_frequency_nodes', True),
        'dynamic_functional': config['model']['strg'].get('dynamic_functional', True),
        'node_dim': config['model']['stre']['node_dim'],
        'graph_embed_dim': config['model']['stre']['graph_embed_dim'],
        'num_gat_layers': config['model']['stre']['num_gat_layers'],
        'num_gat_heads': config['model']['stre']['num_gat_heads'],
        'gat_dropout': config['model']['stre']['gat_dropout'],
        'num_temporal_layers': config['model']['stre']['num_temporal_layers'],
        'num_temporal_heads': config['model']['stre']['num_temporal_heads'],
        'temporal_ff_dim': config['model']['stre']['temporal_ff_dim'],
        'temporal_dropout': config['model']['stre']['temporal_dropout'],
        'decoder_pretrained_name': config['model']['decoder'].get('pretrained_name', 'facebook/bart-base'),
        'max_decoder_length': config['model']['decoder']['max_decoder_length'],
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'lambda_smooth': config['training']['lambda_smooth'],
        'lambda_contrastive': config['training']['lambda_contrastive']
    }

    if args.analysis_type in ('k_neighbors', 'alpha_beta'):
        from utils.sensitivity import generate_hyperparameter_grid
        config_grid = generate_hyperparameter_grid(base_config)
    elif args.analysis_type == 'loss_weights':
        from utils.sensitivity import generate_loss_weight_grid
        config_grid = generate_loss_weight_grid(base_config)
    else:
        from utils.sensitivity import generate_architecture_grid
        config_grid = generate_architecture_grid(base_config)
    
    print(f'Testing {len(config_grid)} hyperparameter configurations...')
    
    all_results = []
    for idx, hp_config in enumerate(config_grid):
        exp_name = hp_config.get('experiment_name', f'config_{idx}')
        print(f'\n[{idx+1}/{len(config_grid)}] {exp_name}')
        
        metrics = evaluate_hyperparameter_config(
            hp_config,
            train_loader,
            val_loader,
            device,
            tokenizer,
            num_epochs=args.num_epochs,
            electrode_positions=getattr(train_dataset, 'electrode_positions', None)
        )
        
        result = {
            'config': hp_config,
            'metrics': metrics
        }
        all_results.append(result)
        
        # Save individual result
        with open(os.path.join(args.output_dir, f'{exp_name}.json'), 'w') as f:
            json.dump(result, f, indent=2)
    
    # Save all results
    with open(os.path.join(args.output_dir, f'all_{args.analysis_type}_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create summary table
    print('\nSensitivity Analysis Results:')
    print('=' * 80)
    for result in all_results:
        exp_name = result['config'].get('experiment_name', 'unknown')
        metrics = result['metrics']
        bleu4 = metrics.get('bleu_4', 0)
        print(f'{exp_name:<50} BLEU-4: {bleu4:.2f}')
    
    print('\nSensitivity analysis completed!')


if __name__ == '__main__':
    main()

