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
from train import load_config, set_seed


def evaluate_hyperparameter_config(
    config: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    tokenizer,
    num_epochs: int = 20
):
    """Quick evaluation of a hyperparameter configuration"""
    
    # Create model with this config
    model = GraphEnhancedEEG2Text(
        num_channels=config['num_channels'],
        num_frequency_bands=config['num_frequency_bands'],
        strg_alpha=config.get('strg_alpha', 0.5),
        strg_beta=config.get('strg_beta', 0.5),
        use_spatial_topology=config.get('use_spatial_topology', True),
        use_functional_connectivity=config.get('use_functional_connectivity', True),
        node_dim=config.get('node_dim', 1),
        graph_embed_dim=config['graph_embed_dim'],
        num_gat_layers=config.get('num_gat_layers', 2),
        num_gat_heads=config.get('num_gat_heads', 4),
        gat_dropout=config.get('gat_dropout', 0.1),
        num_temporal_layers=config.get('num_temporal_layers', 4),
        num_temporal_heads=config.get('num_temporal_heads', 8),
        temporal_ff_dim=config.get('temporal_ff_dim', 512),
        temporal_dropout=config.get('temporal_dropout', 0.1),
        vocab_size=config['vocab_size'],
        decoder_embed_dim=config['decoder_embed_dim'],
        num_decoder_layers=config.get('num_decoder_layers', 4),
        num_decoder_heads=config.get('num_decoder_heads', 8),
        decoder_ff_dim=config.get('decoder_ff_dim', 512),
        decoder_dropout=config.get('decoder_dropout', 0.1),
        max_decoder_length=config.get('max_decoder_length', 128),
        device=device
    )
    model = model.to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    criterion = CompositeLoss(
        lambda_smooth=config.get('lambda_smooth', 0.1),
        lambda_contrastive=config.get('lambda_contrastive', 0.2),
        vocab_size=config['vocab_size']
    )
    
    # Quick training
    best_val_metrics = None
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            eeg_bands = {k: v.to(device) for k, v in batch['eeg_bands'].items()}
            text_tokens = batch['text_tokens'].to(device)
            
            logits, strg_output = model(eeg_bands, text_tokens)
            targets = text_tokens[:, 1:]
            
            loss, _ = criterion(
                logits=logits,
                targets=targets,
                node_embeddings=strg_output.get('node_features'),
                adjacency_matrix=strg_output.get('A')
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Quick validation
        if epoch % 5 == 0:
            model.eval()
            all_references = []
            all_candidates = []
            
            with torch.no_grad():
                for batch in val_loader:
                    eeg_bands = {k: v.to(device) for k, v in batch['eeg_bands'].items()}
                    texts = batch['text']
                    
                    generated = model.generate(
                        eeg_bands,
                        bos_token_id=tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else 1,
                        eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 2,
                        max_length=config.get('max_decoder_length', 128)
                    )
                    
                    for i, text in enumerate(texts):
                        ref = text.split()
                        if hasattr(tokenizer, 'decode'):
                            cand = tokenizer.decode(generated[i].cpu().tolist()).split()
                        else:
                            cand = [str(t.item()) for t in generated[i]]
                        all_references.append(ref)
                        all_candidates.append(cand)
            
            metrics = evaluate_predictions(all_references, all_candidates, compute_bert=False)
            if best_val_metrics is None or metrics.get('bleu_4', 0) > best_val_metrics.get('bleu_4', 0):
                best_val_metrics = metrics
    
    return best_val_metrics or {}


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter sensitivity analysis')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='sensitivity_results')
    parser.add_argument('--analysis_type', type=str, default='alpha_beta',
                       choices=['alpha_beta', 'loss_weights', 'architecture'],
                       help='Type of sensitivity analysis')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of epochs per configuration (reduced for speed)')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    set_seed(config['seed'])
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create datasets
    train_dataset = ZuCoDataset(
        args.data_dir,
        split='train',
        max_seq_length=config['data']['max_seq_length']
    )
    val_dataset = ZuCoDataset(
        args.data_dir,
        split='val',
        max_seq_length=config['data']['max_seq_length']
    )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except:
        tokenizer = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer, config['data']['max_seq_length']),
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer, config['data']['max_seq_length']),
        num_workers=config['num_workers']
    )
    
    # Generate hyperparameter grid
    base_config = {
        'num_channels': config['model']['num_channels'],
        'num_frequency_bands': config['model']['num_frequency_bands'],
        'strg_alpha': config['model']['strg']['alpha'],
        'strg_beta': config['model']['strg']['beta'],
        'use_spatial_topology': config['model']['strg']['use_spatial_topology'],
        'use_functional_connectivity': config['model']['strg']['use_functional_connectivity'],
        'node_dim': config['model']['stre']['node_dim'],
        'graph_embed_dim': config['model']['stre']['graph_embed_dim'],
        'num_gat_layers': config['model']['stre']['num_gat_layers'],
        'num_gat_heads': config['model']['stre']['num_gat_heads'],
        'gat_dropout': config['model']['stre']['gat_dropout'],
        'num_temporal_layers': config['model']['stre']['num_temporal_layers'],
        'num_temporal_heads': config['model']['stre']['num_temporal_heads'],
        'temporal_ff_dim': config['model']['stre']['temporal_ff_dim'],
        'temporal_dropout': config['model']['stre']['temporal_dropout'],
        'vocab_size': config['model']['decoder']['vocab_size'],
        'decoder_embed_dim': config['model']['decoder']['embed_dim'],
        'num_decoder_layers': config['model']['decoder']['num_layers'],
        'num_decoder_heads': config['model']['decoder']['num_heads'],
        'decoder_ff_dim': config['model']['decoder']['ff_dim'],
        'decoder_dropout': config['model']['decoder']['dropout'],
        'max_decoder_length': config['model']['decoder']['max_decoder_length'],
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'lambda_smooth': config['training']['lambda_smooth'],
        'lambda_contrastive': config['training']['lambda_contrastive']
    }
    
    if args.analysis_type == 'alpha_beta':
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
            num_epochs=args.num_epochs
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

