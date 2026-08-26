"""
Train paper ablations and report test-set metrics from the best validation checkpoint.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocessing.preprocessing import ZuCoDataset, collate_fn
from train import (
    build_optimizer,
    configure_pretrained_decoder,
    dataset_kwargs_from_config,
    get_decoder_tokenizer,
    load_config,
    move_eeg_batch,
    resolve_batch_size,
    resolve_num_workers,
    set_seed,
    sync_model_config_from_dataset,
)
from utils.ablation import AblationModelFactory, get_ablation_loss_weights
from utils.losses import CompositeLoss
from utils.metrics import evaluate_predictions
from utils.statistics import compute_statistics, format_metric_with_std


def _make_loader(dataset, tokenizer, config, shuffle: bool) -> DataLoader:
    max_eeg_length = config['model'].get('max_eeg_length', 4000)
    return DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=shuffle,
        collate_fn=lambda x: collate_fn(
            x, tokenizer, config['data']['max_seq_length'], max_eeg_length=max_eeg_length
        ),
        num_workers=resolve_num_workers(config),
        pin_memory=False
    )


def _decode_batch(model, batch, device, tokenizer, config):
    eeg_bands, window_mask, eeg_bands_full, eeg_windows = move_eeg_batch(batch, device)
    generated = model.generate(
        eeg_bands,
        max_length=config['model']['decoder']['max_decoder_length'],
        beam_size=config.get('evaluation', {}).get('beam_size', 5),
        window_mask=window_mask,
        eeg_bands_full=eeg_bands_full,
        eeg_windows=eeg_windows
    )
    refs, cands = [], []
    for i, text in enumerate(batch['text']):
        refs.append(text.split())
        cands.append(tokenizer.decode(generated[i].cpu().tolist(), skip_special_tokens=True).split())
    return refs, cands


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tokenizer,
    config: Dict,
    criterion: Optional[CompositeLoss] = None
):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_refs, all_cands = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            eeg_bands, window_mask, eeg_bands_full, eeg_windows = move_eeg_batch(batch, device)
            text_tokens = batch['text_tokens'].to(device)
            if criterion is not None:
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
                total_loss += loss.item()
                n_batches += 1
            refs, cands = _decode_batch(model, batch, device, tokenizer, config)
            all_refs.extend(refs)
            all_cands.extend(cands)
    metrics = evaluate_predictions(all_refs, all_cands, compute_bert=True)
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, metrics


def train_ablation_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: Dict,
    ablation_type: str,
    device: torch.device,
    checkpoint_dir: str,
    tokenizer
) -> Dict:
    loss_weights = get_ablation_loss_weights(ablation_type, config['training'])
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
    optimizer = build_optimizer(model, config)
    criterion = CompositeLoss(
        lambda_smooth=loss_weights['lambda_smooth'],
        lambda_contrastive=loss_weights['lambda_contrastive'],
        vocab_size=getattr(tokenizer, 'vocab_size', len(tokenizer)),
        ignore_index=pad_token_id
    )

    best_val_loss = float('inf')
    best_path = os.path.join(checkpoint_dir, f'{ablation_type}_best_model.pt')
    patience = config['training'].get('early_stopping_patience', 5)
    min_delta = config['training'].get('early_stopping_min_delta', 0.001)
    use_early_stop = config['training'].get('early_stopping', False)
    patience_counter = 0

    for epoch in range(config['training']['num_epochs']):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Training {ablation_type} epoch {epoch + 1}'):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            optimizer.step()
            train_loss += loss.item()

        val_loss, val_metrics = evaluate_loader(
            model, val_loader, device, tokenizer, config, criterion
        )
        print(
            f'  Epoch {epoch + 1}: train {train_loss / max(len(train_loader), 1):.4f} | '
            f'val {val_loss:.4f} | BLEU-4 {val_metrics.get("bleu_4", 0):.2f}'
        )

        improved = val_loss < best_val_loss - min_delta
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'config': config
            }, best_path)
        if use_early_stop:
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'  Early stopping after {epoch + 1} epochs')
                    break

    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'  Loaded best checkpoint (val loss {checkpoint["best_val_loss"]:.4f})')

    _, test_metrics = evaluate_loader(model, test_loader, device, tokenizer, config, criterion)
    test_metrics['best_val_loss'] = best_val_loss
    return test_metrics


def _ablation_model_config(config: Dict) -> Dict:
    return {
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
        'max_decoder_length': config['model']['decoder']['max_decoder_length']
    }


def main():
    parser = argparse.ArgumentParser(description='Train paper ablation variants')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='ablation_results')
    parser.add_argument(
        '--ablation_types',
        nargs='+',
        default=['full', 'no_graph', 'spatial_only', 'functional_only',
                 'electrode_nodes', 'static_functional', 'gat_only'],
        help='Ablation variants from the paper table'
    )
    parser.add_argument('--num_seeds', type=int, default=3)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    resolve_batch_size(config, device)
    tokenizer = get_decoder_tokenizer(config)
    config['model']['decoder']['vocab_size'] = getattr(tokenizer, 'vocab_size', len(tokenizer))

    print('Loading sentence-identity splits (once)...')
    ds_kwargs = dataset_kwargs_from_config(config)
    splits = ZuCoDataset.load_splits(
        args.data_dir, splits=('train', 'val', 'test'), **ds_kwargs
    )
    train_dataset = splits['train']
    val_dataset = splits['val']
    test_dataset = splits['test']
    sync_model_config_from_dataset(config, train_dataset)
    print(f'  Channels: {config["model"]["num_channels"]}')
    print(f'  Train/val/test: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}')

    train_loader = _make_loader(train_dataset, tokenizer, config, shuffle=True)
    val_loader = _make_loader(val_dataset, tokenizer, config, shuffle=False)
    test_loader = _make_loader(test_dataset, tokenizer, config, shuffle=False)

    all_results = {}
    for ablation_type in args.ablation_types:
        print(f'\n{"=" * 60}\nRunning ablation: {ablation_type}\n{"=" * 60}')
        ablation_results = []
        for seed in range(args.num_seeds):
            print(f'\nSeed {seed + 1}/{args.num_seeds}')
            set_seed(config['seed'] + seed)
            model = AblationModelFactory.create_model(
                ablation_type, _ablation_model_config(config), device
            )
            configure_pretrained_decoder(model, config)
            if hasattr(model, 'set_electrode_positions') and getattr(train_dataset, 'electrode_positions', None) is not None:
                model.set_electrode_positions(train_dataset.electrode_positions)
            model = model.to(device)

            checkpoint_dir = os.path.join(args.output_dir, ablation_type, f'seed_{seed + 1}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            metrics = train_ablation_model(
                model, train_loader, val_loader, test_loader,
                config, ablation_type, device, checkpoint_dir, tokenizer
            )
            ablation_results.append(metrics)
            with open(os.path.join(checkpoint_dir, 'test_results.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

        all_results[ablation_type] = ablation_results
        stats = compute_statistics(ablation_results)
        with open(os.path.join(args.output_dir, f'{ablation_type}_statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        print(f'\n{ablation_type} test results (mean ± std across {len(ablation_results)} seeds):')
        for metric, values in stats.items():
            if 'mean' in values:
                print(f'  {metric}: {format_metric_with_std(values["mean"], values["std"])}')

    with open(os.path.join(args.output_dir, 'all_ablation_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print('\nAblation study completed. Metrics are from the test split of the best val checkpoint.')


if __name__ == '__main__':
    main()
