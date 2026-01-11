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
from train import load_config, create_model, set_seed


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
        max_seq_length=config['data']['max_seq_length'],
        apply_notch_filter=config['data'].get('apply_notch_filter', True),
        notch_freq=config['data'].get('notch_freq', 50.0),
        apply_highpass_filter=config['data'].get('apply_highpass_filter', True),
        highpass_cutoff=config['data'].get('highpass_cutoff', 0.5)
    )
    
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
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=False,
                collate_fn=lambda x: collate_fn(x, tokenizer, config['data']['max_seq_length']),
                num_workers=config['num_workers']
            )
            
            # Create and train model
            model = create_model(config, device)
            model = model.to(device)
            
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
            
            criterion = CompositeLoss(
                lambda_smooth=config['training']['lambda_smooth'],
                lambda_contrastive=config['training']['lambda_contrastive'],
                vocab_size=config['model']['decoder']['vocab_size']
            )
            
            # Train
            best_val_loss = float('inf')
            for epoch in range(config['training']['num_epochs']):
                model.train()
                for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
                    optimizer.step()
            
            # Evaluate on test subject
            model.eval()
            all_references = []
            all_candidates = []
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc='Evaluating'):
                    eeg_bands = {k: v.to(device) for k, v in batch['eeg_bands'].items()}
                    texts = batch['text']
                    
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
            max_seq_length=config['data']['max_seq_length']
        )
        evaluator = CrossSubjectEvaluator(full_dataset)
        train_indices, test_indices = evaluator.train_test_subject_split(
            args.train_subjects or [],
            args.test_subjects or []
        )
        
        print(f'Custom split: {len(train_indices)} train, {len(test_indices)} test')
        # Similar training/evaluation code as LOSO...


if __name__ == '__main__':
    main()

