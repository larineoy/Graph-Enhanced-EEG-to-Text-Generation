"""
Training script with ablation study support
Runs multiple model variants and collects results
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
from utils.ablation import AblationModelFactory, get_ablation_loss_weights
from utils.statistics import compute_statistics, format_metric_with_std
from utils.visualization import save_adjacency_heatmap
from train import load_config, set_seed


def train_ablation_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    ablation_type: str,
    device: torch.device,
    checkpoint_dir: str
):
    """Train a single ablation model variant"""
    
    # Get loss weights for this ablation
    loss_weights = get_ablation_loss_weights(ablation_type, config['training'])
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    criterion = CompositeLoss(
        lambda_smooth=loss_weights['lambda_smooth'],
        lambda_contrastive=loss_weights['lambda_contrastive'],
        vocab_size=config['model']['decoder']['vocab_size']
    )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except:
        tokenizer = None
    
    best_val_loss = float('inf')
    all_val_metrics = []
    
    for epoch in range(config['training']['num_epochs']):
        # Train
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Training {ablation_type} epoch {epoch+1}'):
            eeg_bands = {k: v.to(device) for k, v in batch['eeg_bands'].items()}
            text_tokens = batch['text_tokens'].to(device)
            
            logits, strg_output = model(eeg_bands, text_tokens)
            targets = text_tokens[:, 1:]
            
            loss, _ = criterion(
                logits=logits,
                targets=targets,
                node_embeddings=strg_output.get('node_features'),
                adjacency_matrix=strg_output.get('A'),
                eeg_embeddings=strg_output.get('stre_embeds', torch.zeros(1, device=device)),
                text_embeddings=None
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0.0
        all_references = []
        all_candidates = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Validating {ablation_type}'):
                eeg_bands = {k: v.to(device) for k, v in batch['eeg_bands'].items()}
                text_tokens = batch['text_tokens'].to(device)
                texts = batch['text']
                
                logits, _ = model(eeg_bands, text_tokens)
                targets = text_tokens[:, 1:]
                loss, _ = criterion(logits=logits, targets=targets)
                val_loss += loss.item()
                
                generated = model.generate(
                    eeg_bands,
                    bos_token_id=tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else 1,
                    eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 2,
                    max_length=config['model']['decoder']['max_decoder_length']
                )
                
                for i, text in enumerate(texts):
                    ref = text.split()
                    if hasattr(tokenizer, 'decode'):
                        cand = tokenizer.decode(generated[i].cpu().tolist()).split()
                    else:
                        cand = [str(t.item()) for t in generated[i]]
                    all_references.append(ref)
                    all_candidates.append(cand)
        
        val_metrics = evaluate_predictions(all_references, all_candidates, compute_bert=True)
        all_val_metrics.append(val_metrics)
        
        # Save best model
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config
            }, os.path.join(checkpoint_dir, f'{ablation_type}_best_model.pt'))
        
        # Visualize learned graphs
        if hasattr(model, 'strg') and epoch % 10 == 0:
            sample_batch = next(iter(val_loader))
            sample_eeg_bands = {k: v[:1].to(device) for k, v in sample_batch['eeg_bands'].items()}
            with torch.no_grad():
                A, _, _ = model.strg(sample_eeg_bands)
                A_np = A[0].cpu().numpy()
                os.makedirs(os.path.join(checkpoint_dir, 'visualizations'), exist_ok=True)
                save_adjacency_heatmap(
                    A_np,
                    os.path.join(checkpoint_dir, 'visualizations', f'{ablation_type}_epoch_{epoch}.png'),
                    title=f'{ablation_type} - Epoch {epoch}',
                    num_channels=config['model']['num_channels']
                )
    
    # Return final metrics
    return all_val_metrics[-1] if all_val_metrics else {}


def main():
    parser = argparse.ArgumentParser(description='Train ablation study models')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='ablation_results')
    parser.add_argument('--ablation_types', nargs='+', 
                       default=['full', 'no_graph', 'static_only', 'dynamic_only', 
                               'no_contrastive', 'no_smoothness', 'graph_only'],
                       help='List of ablation types to run')
    parser.add_argument('--num_seeds', type=int, default=3,
                       help='Number of random seeds to run')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = {}
    
    # Run each ablation type with multiple seeds
    for ablation_type in args.ablation_types:
        print(f'\n{"="*60}')
        print(f'Running ablation: {ablation_type}')
        print(f'{"="*60}\n')
        
        ablation_results = []
        
        for seed in range(args.num_seeds):
            print(f'\nSeed {seed+1}/{args.num_seeds}')
            set_seed(config['seed'] + seed)
            
            # Create model
            model_config = {
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
                'max_decoder_length': config['model']['decoder']['max_decoder_length']
            }
            
            model = AblationModelFactory.create_model(ablation_type, model_config, device)
            model = model.to(device)
            
            # Create datasets
            train_dataset = ZuCoDataset(
                args.data_dir,
                split='train',
                max_seq_length=config['data']['max_seq_length'],
                apply_notch_filter=config['data'].get('apply_notch_filter', True),
                notch_freq=config['data'].get('notch_freq', 50.0),
                apply_highpass_filter=config['data'].get('apply_highpass_filter', True),
                highpass_cutoff=config['data'].get('highpass_cutoff', 0.5)
            )
            val_dataset = ZuCoDataset(
                args.data_dir,
                split='val',
                max_seq_length=config['data']['max_seq_length'],
                apply_notch_filter=config['data'].get('apply_notch_filter', True),
                notch_freq=config['data'].get('notch_freq', 50.0),
                apply_highpass_filter=config['data'].get('apply_highpass_filter', True),
                highpass_cutoff=config['data'].get('highpass_cutoff', 0.5)
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
            
            # Train
            checkpoint_dir = os.path.join(args.output_dir, ablation_type, f'seed_{seed+1}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            metrics = train_ablation_model(
                model, train_loader, val_loader, config, ablation_type, device, checkpoint_dir
            )
            
            ablation_results.append(metrics)
            
            # Save per-seed results
            with open(os.path.join(checkpoint_dir, 'results.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
        
        all_results[ablation_type] = ablation_results
        
        # Compute statistics across seeds
        stats = compute_statistics(ablation_results)
        with open(os.path.join(args.output_dir, f'{ablation_type}_statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f'\n{ablation_type} Results (mean ± std across {args.num_seeds} seeds):')
        for metric, values in stats.items():
            print(f'  {metric}: {format_metric_with_std(values["mean"], values["std"])}')
    
    # Save all results
    with open(os.path.join(args.output_dir, 'all_ablation_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print('\nAblation study completed!')


if __name__ == '__main__':
    main()

