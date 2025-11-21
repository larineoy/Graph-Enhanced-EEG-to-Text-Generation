"""
Evaluation script for Graph-Enhanced EEG-to-Text model
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import json
import os
from tqdm import tqdm

from models import GraphEnhancedEEG2Text
from preprocessing.preprocessing import ZuCoDataset, collate_fn
from utils.metrics import evaluate_predictions
from utils.statistics import compute_statistics, format_metric_with_std
from train import load_config, create_model, set_seed


def evaluate(model, dataloader, device, tokenizer, config):
    """Evaluate model on dataset"""
    model.eval()
    all_references = []
    all_candidates = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            # Use eeg_bands dict from preprocessing
            eeg_bands = {band_name: band_tensor.to(device) for band_name, band_tensor in batch['eeg_bands'].items()}
            texts = batch['text']
            
            # Generate predictions
            generated = model.generate(
                eeg_bands,
                bos_token_id=tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else 1,
                eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 2,
                pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0,
                max_length=config['model']['decoder']['max_decoder_length'],
                beam_size=config['evaluation']['beam_size']
            )
            
            # Decode predictions
            for i in range(len(texts)):
                ref = texts[i].split()
                if hasattr(tokenizer, 'decode'):
                    cand = tokenizer.decode(generated[i].cpu().tolist()).split()
                else:
                    cand = [str(t.item()) for t in generated[i]]
                
                all_references.append(ref)
                all_candidates.append(cand)
    
    # Compute metrics
    metrics = evaluate_predictions(all_references, all_candidates, compute_bert=True)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Graph-Enhanced EEG-to-Text model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint or directory with multiple checkpoints')
    parser.add_argument('--data_dir', type=str, default='ZuCo Data',
                       help='Path to data directory')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Path to save evaluation results')
    parser.add_argument('--multi_seed', action='store_true',
                       help='Evaluate multiple checkpoints (seed_1, seed_2, etc.) and compute statistics')
    parser.add_argument('--num_seeds', type=int, default=5,
                       help='Number of seeds to evaluate (if multi_seed)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    set_seed(config['seed'])
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except:
        print("Warning: Could not load tokenizer, using simple tokenizer")
        tokenizer = None
    
    # Create dataset with artifact removal settings from config
    data_config = config['data']
    dataset = ZuCoDataset(
        args.data_dir,
        split=args.split,
        max_seq_length=data_config['max_seq_length'],
        apply_notch_filter=data_config.get('apply_notch_filter', True),
        notch_freq=data_config.get('notch_freq', 50.0),
        apply_highpass_filter=data_config.get('apply_highpass_filter', True),
        highpass_cutoff=data_config.get('highpass_cutoff', 0.5),
        detect_bad_channels=data_config.get('detect_bad_channels', False),
        bad_channel_threshold=data_config.get('bad_channel_threshold', 3.0)
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer, config['data']['max_seq_length']),
        num_workers=config['num_workers']
    )
    
    if args.multi_seed:
        # Multi-seed evaluation mode
        checkpoint_dir = os.path.dirname(args.checkpoint) if os.path.isfile(args.checkpoint) else args.checkpoint
        all_results = []
        
        print(f'\n{"="*60}')
        print(f'Multi-Seed Evaluation: {args.num_seeds} seeds')
        print(f'{"="*60}\n')
        
        for seed in range(1, args.num_seeds + 1):
            # Step 1: Find checkpoint for this seed (Line ~XXX)
            checkpoint_path = os.path.join(checkpoint_dir, f'seed_{seed}', 'best_model.pt')
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join(checkpoint_dir, f'seed_{seed}', 'checkpoint.pt')
            if not os.path.exists(checkpoint_path):
                print(f"Warning: Checkpoint not found for seed {seed}: {checkpoint_path}")
                continue
            
            print(f'Evaluating seed {seed}/{args.num_seeds}...')
            
            # Step 2: Load model for this seed (Line ~XXX)
            model = create_model(config, device)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            
            # Step 3: Evaluate model on test set (Line ~XXX)
            metrics = evaluate(model, dataloader, device, tokenizer, config)
            all_results.append(metrics)
            
            # Step 4: Save per-seed results (Line ~XXX)
            seed_output = os.path.join(checkpoint_dir, f'eval_{args.split}_seed_{seed}.json')
            with open(seed_output, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f'  Saved results to {seed_output}')
        
        # Step 5: Compute statistics across seeds (Line ~XXX)
        if len(all_results) > 0:
            statistics = compute_statistics(all_results)
            
            print(f'\n{"="*60}')
            print(f'Multi-Seed Evaluation Results ({len(all_results)} runs):')
            print(f'{"="*60}')
            for metric, stats in statistics.items():
                if 'mean' in stats:
                    print(f'{metric:<25} {format_metric_with_std(stats["mean"], stats["std"])}')
            print(f'{"="*60}')
            
            # Step 6: Save aggregated results (Line ~XXX)
            if args.output:
                output_path = args.output
            else:
                output_path = os.path.join(checkpoint_dir, f'eval_{args.split}_multi_seed.json')
            
            with open(output_path, 'w') as f:
                json.dump({
                    'statistics': statistics,
                    'all_results': all_results,
                    'num_seeds': len(all_results)
                }, f, indent=2)
            
            print(f'\nAggregated results saved to {output_path}')
        else:
            print('No valid checkpoints found for multi-seed evaluation')
    
    else:
        # Single checkpoint evaluation mode (original code)
        # Load model
        model = create_model(config, device)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        print(f'Loaded model from {args.checkpoint}')
        print(f'Evaluating on {args.split} split ({len(dataset)} samples)')
        
        # Evaluate
        metrics = evaluate(model, dataloader, device, tokenizer, config)
        
        # Print results
        print('\nEvaluation Results:')
        print('=' * 50)
        print(f"{'Metric':<20} {'Score':<10}")
        print('=' * 50)
        for metric, score in sorted(metrics.items()):
            print(f"{metric:<20} {score:>8.2f}")
        print('=' * 50)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()

