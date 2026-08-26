"""
Training script for cross-subject evaluation
Implements leave-one-subject-out (LOSO) and custom train/test subject splits
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
import os
import json
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple

from models import GraphEnhancedEEG2Text
from preprocessing.preprocessing import ZuCoDataset, collate_fn
from utils.losses import CompositeLoss
from utils.metrics import evaluate_predictions
from utils.cross_subject import CrossSubjectEvaluator
from utils.statistics import compute_statistics
from train import (
    load_config,
    create_model,
    set_seed,
    dataset_kwargs_from_config,
    get_decoder_tokenizer,
    move_eeg_batch,
    sync_model_config_from_dataset,
)


def train_and_evaluate_loso(
    config: Dict,
    data_dir: str,
    output_dir: str,
    device: torch.device,
    num_seeds: int = 3
):
    """
    Perform leave-one-subject-out (LOSO) cross-validation
    
    Args:
        config: Configuration dictionary
        data_dir: Data directory
        output_dir: Output directory for results
        device: Device to run on
        num_seeds: Number of random seeds per split
    """
    # Load full dataset to get subject information
    full_dataset = ZuCoDataset(
        data_dir,
        split='all',
        **dataset_kwargs_from_config(config)
    )
    sync_model_config_from_dataset(config, full_dataset)
    tokenizer = get_decoder_tokenizer(config)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
    max_eeg_length = config['model'].get('max_eeg_length', 20000)

    evaluator = CrossSubjectEvaluator(full_dataset)
    loso_splits = evaluator.leave_one_subject_out_splits()
    
    all_results = {}
    
    for train_indices, test_indices, test_subject in loso_splits:
        print(f'\n{"="*60}')
        print(f'LOSO: Testing on subject {test_subject}')
        print(f'Training subjects: {len(set([full_dataset.samples[i]["subject"] for i in train_indices]))}')
        print(f'Test samples: {len(test_indices)}')
        print(f'{"="*60}\n')
        
        subject_results = []
        
        for seed in range(num_seeds):
            print(f'Seed {seed+1}/{num_seeds}')
            set_seed(config['seed'] + seed)
            
            # Create train/test datasets
            train_dataset = Subset(full_dataset, train_indices)
            test_dataset = Subset(full_dataset, test_indices)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=True,
                collate_fn=lambda x: collate_fn(
                    x, tokenizer, config['data']['max_seq_length'], max_eeg_length=max_eeg_length
                ),
                num_workers=config.get('num_workers', 0)
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=False,
                collate_fn=lambda x: collate_fn(
                    x, tokenizer, config['data']['max_seq_length'], max_eeg_length=max_eeg_length
                ),
                num_workers=config.get('num_workers', 0)
            )

            model = create_model(config, device)
            if getattr(full_dataset, 'electrode_positions', None) is not None:
                model.set_electrode_positions(full_dataset.electrode_positions)
            model = model.to(device)

            optimizer = optim.AdamW(
                model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
            criterion = CompositeLoss(
                lambda_smooth=config['training']['lambda_smooth'],
                lambda_contrastive=config['training']['lambda_contrastive'],
                vocab_size=getattr(tokenizer, 'vocab_size', len(tokenizer)),
                ignore_index=pad_token_id
            )

            for epoch in range(config['training']['num_epochs']):
                model.train()
                for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
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

            model.eval()
            all_references = []
            all_candidates = []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc='Evaluating'):
                    eeg_bands, window_mask, eeg_bands_full, eeg_windows = move_eeg_batch(batch, device)
                    generated = model.generate(
                        eeg_bands,
                        max_length=config['model']['decoder']['max_decoder_length'],
                        window_mask=window_mask,
                        eeg_bands_full=eeg_bands_full,
                        eeg_windows=eeg_windows
                    )
                    for i, text in enumerate(batch['text']):
                        all_references.append(text.split())
                        all_candidates.append(
                            tokenizer.decode(generated[i].cpu().tolist(), skip_special_tokens=True).split()
                        )
            
            metrics = evaluate_predictions(all_references, all_candidates, compute_bert=True)
            subject_results.append(metrics)
            
            # Save per-seed result
            seed_dir = os.path.join(output_dir, f'test_subject_{test_subject}', f'seed_{seed+1}')
            os.makedirs(seed_dir, exist_ok=True)
            with open(os.path.join(seed_dir, 'results.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
        
        # Compute statistics across seeds
        stats = compute_statistics(subject_results)
        all_results[test_subject] = stats
        
        with open(os.path.join(output_dir, f'test_subject_{test_subject}_statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f'\nSubject {test_subject} Results (mean ± std):')
        for metric, values in stats.items():
            if 'mean' in values:
                print(f'  {metric}: {values["mean"]:.2f} ± {values["std"]:.2f}')
    
    # Save aggregated results
    with open(os.path.join(output_dir, 'loso_all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print('\nLOSO cross-validation completed!')


def main():
    parser = argparse.ArgumentParser(description='Cross-subject evaluation')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='cross_subject_results')
    parser.add_argument('--mode', type=str, default='loso', choices=['loso', 'custom'],
                       help='Evaluation mode: loso (leave-one-subject-out) or custom')
    parser.add_argument('--train_subjects', nargs='+', default=None,
                       help='Subject IDs for training (custom mode only)')
    parser.add_argument('--test_subjects', nargs='+', default=None,
                       help='Subject IDs for testing (custom mode only)')
    parser.add_argument('--num_seeds', type=int, default=3)
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'loso':
        train_and_evaluate_loso(config, args.data_dir, args.output_dir, device, args.num_seeds)
    else:
        # Custom train/test split
        full_dataset = ZuCoDataset(
            args.data_dir,
            split='all',
            **dataset_kwargs_from_config(config)
        )
        sync_model_config_from_dataset(config, full_dataset)
        evaluator = CrossSubjectEvaluator(full_dataset)
        train_indices, test_indices = evaluator.train_test_subject_split(
            args.train_subjects or [],
            args.test_subjects or []
        )
        
        print(f'Custom split: {len(train_indices)} train, {len(test_indices)} test')
        # Similar training/evaluation code as LOSO...


if __name__ == '__main__':
    main()

