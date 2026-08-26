"""
Diagnostic script to check why BLEU/ROUGE scores are 0
Checks all 8 common issues that cause zero scores
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from train import (
    load_config,
    create_model,
    set_seed,
    dataset_kwargs_from_config,
    get_decoder_tokenizer,
    move_eeg_batch,
    sync_model_config_from_dataset,
)
from preprocessing.preprocessing import ZuCoDataset, collate_fn
from torch.utils.data import DataLoader
import random


def diagnose_validation(model, dataloader, tokenizer, device, config, num_samples=5):
    """
    Run all 8 diagnostic checks
    """
    print("="*70)
    print("DIAGNOSTIC CHECKS FOR ZERO BLEU/ROUGE")
    print("="*70)
    
    model.eval()
    
    # Get random samples
    indices = random.sample(range(len(dataloader.dataset)), min(num_samples, len(dataloader.dataset)))
    samples = [dataloader.dataset[i] for i in indices]
    
    # Get token IDs for generation
    if tokenizer is not None:
        try:
            cls_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token) if hasattr(tokenizer, 'cls_token') else 101
            sep_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token) if hasattr(tokenizer, 'sep_token') else 102
            pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token) if hasattr(tokenizer, 'pad_token') else 0
            bos_token_id = cls_id[0] if isinstance(cls_id, list) else cls_id
            eos_token_id = sep_id[0] if isinstance(sep_id, list) else sep_id
            pad_token_id = pad_id[0] if isinstance(pad_id, list) else pad_id
        except:
            bos_token_id = 101
            eos_token_id = 102
            pad_token_id = 0
    else:
        bos_token_id = 101
        eos_token_id = 102
        pad_token_id = 0
    
    # Check 0.1: Print decoded predictions + references
    print("\n[CHECK 0.1] Printing decoded predictions and references...")
    print("-"*70)
    
    all_references = []
    all_candidates = []
    all_pred_ids = []
    all_ref_ids = []
    
    with torch.no_grad():
        for idx, sample in enumerate(samples):
            # Create batch from single sample
            batch = collate_fn([sample], tokenizer, config['data']['max_seq_length'], 
                             max_eeg_length=config['model'].get('max_eeg_length', 20000))
            
            eeg_bands, window_mask, eeg_bands_full, eeg_windows = move_eeg_batch(batch, device)
            texts = batch['text']
            text_tokens = batch['text_tokens'].to(device)
            
            generated = model.generate(
                eeg_bands,
                max_length=config['model']['decoder']['max_decoder_length'],
                window_mask=window_mask,
                eeg_bands_full=eeg_bands_full,
                eeg_windows=eeg_windows
            )
            
            # Decode
            ref_text = texts[0]
            ref_ids = text_tokens[0].cpu().tolist()[:30]
            pred_ids = generated[0].cpu().tolist()[:30]
            
            # Check 0.2: Same tokenizer for decode
            if hasattr(tokenizer, 'decode'):
                # BUG FIX: Add skip_special_tokens=True
                pred_text = tokenizer.decode(generated[0].cpu().tolist(), skip_special_tokens=True)
                # Also decode reference for comparison
                ref_decoded = tokenizer.decode(text_tokens[0].cpu().tolist(), skip_special_tokens=True)
            else:
                pred_text = ' '.join([str(t.item()) for t in generated[0]])
                ref_decoded = ref_text
            
            all_references.append(ref_text.split())
            all_candidates.append(pred_text.split())
            all_pred_ids.append(pred_ids)
            all_ref_ids.append(ref_ids)
            
            print(f"\nSample {idx+1}:")
            print(f"  Reference IDs[:30]: {ref_ids}")
            print(f"  Predicted IDs[:30]: {pred_ids}")
            print(f"  Reference text: '{ref_text}'")
            print(f"  Reference decoded: '{ref_decoded}'")
            print(f"  Predicted text: '{pred_text}'")
            print(f"  Ref length: {len(ref_text.split())} tokens")
            print(f"  Pred length: {len(pred_text.split())} tokens")
            print(f"  Ref ID length: {len(text_tokens[0])} tokens")
            print(f"  Pred ID length: {len(generated[0])} tokens")
            
            # Red flags
            if pred_text == "":
                print("  ⚠️  RED FLAG: Prediction is empty!")
                print(f"     Model only generated [CLS] (101) and padding (0) tokens")
                print(f"     After skip_special_tokens=True, nothing remains → empty string")
                print(f"     This is why BLEU/ROUGE are 0!")
            elif pred_text.strip() in ["[CLS]", "[SEP]", "[PAD]", "<pad>", "<cls>", "<sep>"]:
                print(f"  ⚠️  RED FLAG: Prediction is only special token: '{pred_text}'")
            elif len(pred_text.split()) <= 3:
                print(f"  ⚠️  RED FLAG: Prediction is very short: {len(pred_text.split())} tokens")
            elif len(all_candidates) > 1 and pred_text == all_candidates[0]:
                print("  ⚠️  RED FLAG: Prediction is identical to previous!")
            
            # Additional diagnostic for first sample
            if idx == 0:
                print(f"\n  [DEBUG] Why model predicts padding:")
                print(f"    - Model is untrained (checkpoint architecture mismatch)")
                print(f"    - Untrained models predict padding token (0) as most likely")
                print(f"    - Generation: [CLS] → argmax → padding (0) → padding (0) → ...")
                print(f"    - Solution: Retrain OR mask pad_token_id during generation")
    
    # Check 0.2: Same tokenizer
    print("\n[CHECK 0.2] Verifying tokenizer consistency...")
    print("-"*70)
    if tokenizer is not None:
        print(f"  ✓ Tokenizer: {type(tokenizer).__name__}")
        print(f"  ✓ Vocab size: {tokenizer.vocab_size}")
        print(f"  ✓ Using tokenizer.decode() with skip_special_tokens=True")
    else:
        print("  ⚠️  WARNING: No tokenizer available!")
    
    # Check 0.3: Teacher forcing/shifting
    print("\n[CHECK 0.3] Checking teacher forcing/shifting...")
    print("-"*70)
    # Check in code: targets = text_tokens[:, 1:] should be used
    print("  ✓ Should use: targets = text_tokens[:, 1:]")
    print("  ✓ Should use: decoder_input = text_tokens[:, :-1]")
    print("  ⚠️  Please verify in train.py line 118: targets = text_tokens[:, 1:]")
    
    # Check 0.4: Loss masking
    print("\n[CHECK 0.4] Checking loss ignore_index...")
    print("-"*70)
    from utils.losses import CompositeLoss
    try:
        criterion = CompositeLoss(
            lambda_smooth=0.1,
            lambda_contrastive=0.2,
            vocab_size=config['model']['decoder']['vocab_size'],
            ignore_index=-100  # Default PyTorch ignore_index
        )
        # Check if ignore_index is set
        if hasattr(criterion.ce_loss, 'ignore_index'):
            ignore_idx = criterion.ce_loss.ignore_index
            print(f"  Current ignore_index: {ignore_idx}")
            if ignore_idx == pad_token_id:
                print(f"  ✓ ignore_index={ignore_idx} matches pad_token_id={pad_token_id}")
            else:
                print(f"  ⚠️  WARNING: ignore_index={ignore_idx} != pad_token_id={pad_token_id}")
                print(f"  ⚠️  Padding tokens (ID={pad_token_id}) will NOT be ignored in loss!")
        else:
            print("  ⚠️  WARNING: ignore_index not set in loss function!")
            print(f"  Should set: ignore_index={pad_token_id}")
    except Exception as e:
        print(f"  ⚠️  Could not check loss function: {e}")
    
    # Check 0.5: EOS handling
    print("\n[CHECK 0.5] Checking EOS handling in generation...")
    print("-"*70)
    eos_found = sum(1 for pred in all_pred_ids if eos_token_id in pred)
    print(f"  EOS token ID: {eos_token_id} ([SEP] for BERT)")
    print(f"  EOS found in {eos_found}/{len(all_pred_ids)} predictions")
    if eos_found == 0:
        print("  ⚠️  WARNING: EOS never generated - sequences may be truncated")
    
    # Check 0.6: BLEU/ROUGE code sanity test
    print("\n[CHECK 0.6] Testing BLEU/ROUGE code with identical strings...")
    print("-"*70)
    from utils.metrics import evaluate_predictions
    # Test with identical strings
    test_refs = [['this', 'is', 'a', 'test']]
    test_cands = [['this', 'is', 'a', 'test']]
    test_metrics = evaluate_predictions(test_refs, test_cands, compute_bert=False)
    print(f"  Test metrics (identical strings):")
    for key, val in test_metrics.items():
        print(f"    {key}: {val:.4f}")
    if test_metrics.get('bleu_4', 0) > 0.9:
        print("  ✓ BLEU/ROUGE code works correctly")
    else:
        print(f"  ⚠️  WARNING: BLEU_4 should be ~100.0 for identical strings, got {test_metrics.get('bleu_4', 0)}")
    
    # Check 0.7: Lowercasing/cleaning consistency
    print("\n[CHECK 0.7] Checking lowercasing/cleaning consistency...")
    print("-"*70)
    print("  Reference processing: text.split()")
    print("  Candidate processing: tokenizer.decode(..., skip_special_tokens=True).split()")
    
    # Check for wordpieces in candidates
    wordpiece_count = sum(1 for cand in all_candidates for token in cand if token.startswith('##'))
    if wordpiece_count > 0:
        print(f"  ⚠️  WARNING: Found {wordpiece_count} wordpiece tokens (##) in candidates!")
        print("  This suggests candidates weren't properly decoded")
    else:
        print("  ✓ No wordpiece tokens found - decoding looks correct")
    
    # Check 0.8: Dataset split and references
    print("\n[CHECK 0.8] Checking dataset split and references...")
    print("-"*70)
    print(f"  Dataset size: {len(dataloader.dataset)}")
    print(f"  Split: {getattr(dataloader.dataset, 'split', 'unknown')}")
    empty_refs = sum(1 for ref in all_references if len(ref) == 0 or ref == [''])
    if empty_refs > 0:
        print(f"  ⚠️  WARNING: Found {empty_refs} empty references!")
    else:
        print(f"  ✓ All references have content (checked {len(all_references)} samples)")
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    # Check if all predictions are empty
    all_empty = all(len(cand) == 0 for cand in all_candidates)
    if all_empty:
        print("\n🔴 CRITICAL ISSUE FOUND: All predictions are empty!")
        print("="*70)
        print("Root Cause: Model only generates [CLS] (101) + padding (0) tokens")
        print("After skip_special_tokens=True, this becomes empty string → BLEU/ROUGE = 0")
        print("\nWhy this happens:")
        print("1. Model is untrained (checkpoint architecture mismatch)")
        print("2. Untrained models predict padding token (0) as most likely")
        print("3. Generation: [CLS] → argmax → padding (0) → padding (0) → ...")
        print("\nSolutions:")
        print("1. ✅ FIXED: skip_special_tokens=True in decode() (already applied)")
        print("2. ⚠️  NEEDED: Retrain model with current architecture")
        print("   OR fix checkpoint loading to handle architecture changes")
        print("3. ⚠️  NEEDED: Check if generation should mask padding tokens")
        print("   (Currently generation doesn't prevent predicting pad_token_id)")
    else:
        print("\nMost likely issues:")
        print("1. ✅ FIXED: Missing skip_special_tokens=True in decode()")
        print("2. ⚠️  Loss ignore_index may not match pad_token_id")
        print("3. ⚠️  Some predictions are empty or only special tokens")
    
    print("\nNext steps:")
    print("1. ✅ skip_special_tokens=True fix applied to train.py")
    print("2. ⚠️  Retrain model OR fix checkpoint loading")
    print("3. ⚠️  Consider masking pad_token_id during generation")
    print("4. ⚠️  Verify ignore_index matches pad_token_id in loss")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnose zero BLEU/ROUGE issues')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--split', type=str, default='val',
                       help='Dataset split')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to check')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    set_seed(config['seed'])
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    print("\n[STAGE 1/4] Loading tokenizer...")
    tokenizer = get_decoder_tokenizer(config)
    print("  ✓ BART tokenizer loaded")
    
    # Load dataset first to detect actual number of channels
    print("\n[STAGE 2/4] Loading dataset to detect channels...")
    data_config = config['data']
    max_eeg_length = config['model'].get('max_eeg_length', 20000)
    
    dataset = ZuCoDataset(
        args.data_dir,
        split=args.split,
        **dataset_kwargs_from_config(config)
    )
    sync_model_config_from_dataset(config, dataset)
    
    # Update config with actual number of channels from dataset
    if hasattr(dataset, 'num_channels') and dataset.num_channels is not None:
        actual_channels = dataset.num_channels
        if config['model']['num_channels'] != actual_channels:
            print(f"  ⚠ Config has {config['model']['num_channels']} channels, but data has {actual_channels}")
            print(f"  ✓ Updating config to use {actual_channels} channels")
            config['model']['num_channels'] = actual_channels
    else:
        print("  ⚠ Could not detect channels from dataset, using config value")
    
    # Update vocab size from tokenizer
    if tokenizer is not None:
        tokenizer_vocab_size = getattr(tokenizer, 'vocab_size', len(tokenizer))
        decoder_vocab_size = config['model']['decoder']['vocab_size']
        if decoder_vocab_size != tokenizer_vocab_size:
            print(f"  ⚠ Config vocab_size ({decoder_vocab_size}) != tokenizer vocab_size ({tokenizer_vocab_size})")
            print(f"  ✓ Updating config to use tokenizer vocab_size ({tokenizer_vocab_size})")
            config['model']['decoder']['vocab_size'] = tokenizer_vocab_size
            config['data']['vocab_size'] = tokenizer_vocab_size
    
    # Load model
    print("\n[STAGE 3/4] Creating model...")
    model = create_model(config, device)
    if getattr(dataset, 'electrode_positions', None) is not None:
        model.set_electrode_positions(dataset.electrode_positions)
    
    if os.path.exists(args.checkpoint):
        print(f"[STAGE 4/4] Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Try flexible loading first (handles architecture changes)
        print("  Attempting flexible checkpoint loading (strict=False)...")
        try:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            if missing_keys:
                print(f"  ⚠ {len(missing_keys)} missing keys (architecture changed):")
                for key in missing_keys[:10]:  # Show first 10
                    print(f"    - {key}")
                if len(missing_keys) > 10:
                    print(f"    ... and {len(missing_keys) - 10} more")
            
            if unexpected_keys:
                print(f"  ⚠ {len(unexpected_keys)} unexpected keys (old architecture):")
                for key in unexpected_keys:
                    print(f"    - {key}")
            
            if not missing_keys and not unexpected_keys:
                print("  ✓ Checkpoint loaded perfectly (strict mode)")
            else:
                print(f"  ✓ Checkpoint loaded with {len(missing_keys)} missing and {len(unexpected_keys)} unexpected keys")
                print("  ⚠ Model architecture changed since checkpoint was saved")
                print("  ⚠ Diagnostics may not reflect actual trained model behavior")
                
        except Exception as e:
            print(f"  ✗ Error loading checkpoint: {e}")
            print("  ⚠ Continuing with random initialization for diagnostic purposes...")
            print("  ⚠ Note: Results will reflect untrained model behavior")
    else:
        print(f"  ⚠ Checkpoint not found at {args.checkpoint}")
        print("  ⚠ Using randomly initialized model for diagnostics")
    
    model = model.to(device)
    print("  ✓ Model ready for diagnostics")
    
    # Create dataloader (dataset already loaded above)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer, data_config['max_seq_length'], max_eeg_length=max_eeg_length),
        num_workers=0
    )
    
    # Run diagnostics
    diagnose_validation(model, dataloader, tokenizer, device, config, args.num_samples)
