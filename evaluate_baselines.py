"""
Evaluate model on random-input baselines
Tests shuffled channels, shuffled time, and random noise
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import json
from tqdm import tqdm

from models import GraphEnhancedEEG2Text
from preprocessing.preprocessing import ZuCoDataset, collate_fn
from utils.metrics import evaluate_predictions
from utils.baselines import (
    create_shuffled_channel_baseline,
    create_shuffled_time_baseline,
    create_random_gaussian_baseline,
    create_random_uniform_baseline
)
from train import load_config, create_model, set_seed


def evaluate_baseline(
    model: GraphEnhancedEEG2Text,
    dataloader: DataLoader,
    baseline_type: str,
    device: torch.device,
    tokenizer,
    config: dict
):
    """
    Evaluate model on a specific baseline
    
    Args:
        model: Trained model
        dataloader: DataLoader with real data (will be modified for baseline)
        baseline_type: Type of baseline ('shuffled_channels', 'shuffled_time', 'random_gaussian', 'random_uniform')
        device: Device
        tokenizer: Tokenizer
        config: Configuration
        
    Returns:
        metrics: Evaluation metrics
    """
    model.eval()
    all_references = []
    all_candidates = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Evaluating {baseline_type}'):
            # Create baseline version of EEG bands
            original_eeg_bands = {k: v.to(device) for k, v in batch['eeg_bands'].items()}
            
            if baseline_type == 'shuffled_channels':
                eeg_bands = create_shuffled_channel_baseline(original_eeg_bands)
            elif baseline_type == 'shuffled_time':
                eeg_bands = create_shuffled_time_baseline(original_eeg_bands)
            elif baseline_type == 'random_gaussian':
                eeg_bands = create_random_gaussian_baseline(original_eeg_bands, mean=0.0, std=1.0)
            elif baseline_type == 'random_uniform':
                eeg_bands = create_random_uniform_baseline(original_eeg_bands, low=-1.0, high=1.0)
            else:
                eeg_bands = original_eeg_bands
            
            texts = batch['text']
            
            generated = model.generate(
                eeg_bands,
                bos_token_id=tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else 1,
                eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 2,
                pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0,
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
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate on random-input baselines')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--output', type=str, default='baseline_results.json')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config['seed'])
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = create_model(config, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create dataset
    dataset = ZuCoDataset(
        args.data_dir,
        split=args.split,
        max_seq_length=config['data']['max_seq_length']
    )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except:
        tokenizer = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer, config['data']['max_seq_length']),
        num_workers=config['num_workers']
    )
    
    # Evaluate on different baselines
    baseline_types = ['shuffled_channels', 'shuffled_time', 'random_gaussian', 'random_uniform']
    
    # Also evaluate on real data for comparison
    print('Evaluating on real data...')
    real_metrics = evaluate_baseline(model, dataloader, 'real', device, tokenizer, config)
    
    all_results = {
        'real_data': real_metrics
    }
    
    for baseline_type in baseline_types:
        print(f'\nEvaluating on {baseline_type}...')
        baseline_metrics = evaluate_baseline(model, dataloader, baseline_type, device, tokenizer, config)
        all_results[baseline_type] = baseline_metrics
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print('\nBaseline Evaluation Results:')
    print('=' * 60)
    print(f"{'Baseline':<25} {'BLEU-4':<15} {'ROUGE-1-F':<15}")
    print('=' * 60)
    for baseline_name, metrics in all_results.items():
        bleu4 = metrics.get('bleu_4', 0)
        rouge1f = metrics.get('rouge1_fmeasure', 0)
        print(f"{baseline_name:<25} {bleu4:<15.2f} {rouge1f:<15.2f}")
    print('=' * 60)
    
    print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()

