"""
Debug script to diagnose why model always predicts commas
"""

import torch
import argparse
import os
from preprocessing.preprocessing import ZuCoDataset, collate_fn
from torch.utils.data import DataLoader
from train import (
    create_model,
    dataset_kwargs_from_config,
    get_decoder_tokenizer,
    move_eeg_batch,
    sync_model_config_from_dataset,
)
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--data_dir', type=str, default='data')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading tokenizer...")
    tokenizer = get_decoder_tokenizer(config)
    actual_vocab_size = getattr(tokenizer, 'vocab_size', len(tokenizer))
    config['model']['decoder']['vocab_size'] = actual_vocab_size
    print(f"  ✓ BART tokenizer loaded (vocab_size={actual_vocab_size})")

    print("\nLoading dataset...")
    train_dataset = ZuCoDataset(
        args.data_dir,
        split='train',
        **dataset_kwargs_from_config(config)
    )
    sync_model_config_from_dataset(config, train_dataset)
    print(f"  ✓ Detected num_channels: {config['model']['num_channels']}")
    print(f"  ✓ Using vocab_size: {actual_vocab_size}")

    print("Creating model...")
    model = create_model(config, device)
    if getattr(train_dataset, 'electrode_positions', None) is not None:
        model.set_electrode_positions(train_dataset.electrode_positions)
    model = model.to(device)
    
    # Load checkpoint if exists
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Checkpoint loaded (strict=False)")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            print("Using randomly initialized model")
    else:
        print("No checkpoint found, using randomly initialized model")
    
    model.eval()
    
    # Get a sample from dataset
    print("\nGetting sample from dataset...")
    max_eeg_length = config['model'].get('max_eeg_length', 20000)
    dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer, config['model']['decoder']['max_decoder_length'], max_eeg_length=max_eeg_length),
        num_workers=0
    )
    
    batch = next(iter(dataloader))
    eeg_bands, window_mask, eeg_bands_full, eeg_windows = move_eeg_batch(batch, device)
    text_tokens = batch['text_tokens'].to(device)
    text = batch['text'][0]
    
    print(f"Ground truth text: {text}")
    print(f"Text tokens (first 20): {text_tokens[0][:20].tolist()}")
    
    # Check tokenizer
    print(f"\nTokenizer info:")
    print(f"  pad_token_id: {tokenizer.pad_token_id}")
    print(f"  cls_token_id: {tokenizer.cls_token_id}")
    print(f"  sep_token_id: {tokenizer.sep_token_id}")
    print(f"  Comma token ID: {tokenizer.convert_tokens_to_ids(',')}")
    
    # Forward pass to check logits during training
    print("\n" + "="*60)
    print("CHECKING TRAINING FORWARD PASS")
    print("="*60)
    with torch.no_grad():
        logits, strg_output = model(
            eeg_bands,
            text_tokens,
            window_mask=window_mask,
            eeg_bands_full=eeg_bands_full,
            eeg_windows=eeg_windows
        )
        
        print(f"Logits shape: {logits.shape}")
        print(f"Targets shape: {text_tokens[:, 1:].shape}")
        
        # Check first position logits
        first_logits = logits[0, 0, :].cpu()
        top10_logits, top10_indices = torch.topk(first_logits, 10)
        
        print(f"\nFirst position logits:")
        print(f"  Range: [{first_logits.min().item():.2f}, {first_logits.max().item():.2f}]")
        print(f"  Std: {first_logits.std().item():.2f}")
        print(f"  Top 10 token IDs: {top10_indices.tolist()}")
        print(f"  Top 10 logits: {top10_logits.tolist()}")
        
        # Decode top tokens
        print(f"\nTop 10 tokens:")
        for idx, (logit_val, token_id) in enumerate(zip(top10_logits, top10_indices)):
            token_str = tokenizer.decode([token_id.item()])
            print(f"  {idx+1}. ID={token_id.item():4d}, logit={logit_val.item():6.2f}, token='{token_str}'")
        
        # Check if logits are degenerate
        if first_logits.std().item() < 0.1:
            print(f"\n⚠️  WARNING: Logits are too uniform (std={first_logits.std().item():.2f})!")
            print("  This will cause the model to always predict the same token.")
        
        # Check STRE embeddings
        stre_embeds = strg_output['stre_embeds'][0].cpu()
        print(f"\nSTRE embeddings:")
        print(f"  Norm: {torch.norm(stre_embeds).item():.4f}")
        print(f"  Std: {stre_embeds.std().item():.4f}")
        if torch.norm(stre_embeds).item() < 1e-6:
            print(f"  ⚠️  WARNING: STRE embeddings are near zero!")
    
    # Generate predictions
    print("\n" + "="*60)
    print("CHECKING GENERATION")
    print("="*60)
    
    print(
        f"Using bos={tokenizer.bos_token_id}, eos={tokenizer.eos_token_id}, "
        f"pad={tokenizer.pad_token_id}"
    )
    generated = model.generate(
        eeg_bands,
        max_length=50,
        window_mask=window_mask,
        eeg_bands_full=eeg_bands_full,
        eeg_windows=eeg_windows
    )
    
    print(f"\nGenerated token IDs (first 30): {generated[0][:30].tolist()}")
    
    # Decode
    pred_text = tokenizer.decode(generated[0].cpu().tolist(), skip_special_tokens=True)
    print(f"Generated text: {pred_text}")
    
    # Count comma frequency
    comma_count = pred_text.count(',')
    print(f"Comma count: {comma_count} out of {len(pred_text)} characters")
    
    # Check for repetitive predictions (any token repeated many times)
    pred_tokens = pred_text.split()
    repetition_ratio = 0.0
    most_common_token = ""
    if len(pred_tokens) > 0:
        most_common_token = max(set(pred_tokens), key=pred_tokens.count)
        most_common_count = pred_tokens.count(most_common_token)
        repetition_ratio = most_common_count / len(pred_tokens)
        print(f"Most repeated token: '{most_common_token}' ({most_common_count}/{len(pred_tokens)} = {repetition_ratio:.1%})")
        
        if repetition_ratio > 0.5:
            print(f"\n⚠️  WARNING: Prediction is mostly repetitive ('{most_common_token}')!")
            print("  This indicates the model is untrained or logits are degenerate.")
    
    if comma_count > len(pred_text) * 0.5:
        print("\n⚠️  WARNING: Prediction is mostly commas!")
        print("  This indicates the model is untrained or logits are degenerate.")
    
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    if first_logits.std().item() < 0.1:
        print("❌ ISSUE: Logits are too uniform")
        print("   → Model is untrained or initialization is poor")
        print("   → Solution: Train the model")
    elif torch.norm(stre_embeds).item() < 1e-6:
        print("❌ ISSUE: STRE embeddings are near zero")
        print("   → STRG/STRE modules are not working")
        print("   → Solution: Check STRG/STRE initialization and forward pass")
    elif len(pred_tokens) > 0 and repetition_ratio > 0.5:
        print("❌ ISSUE: Model predicts repetitive tokens")
        print(f"   → Model keeps predicting '{most_common_token}' repeatedly")
        print("   → This is normal for an untrained model")
        print("   → Solution: Train the model - it will learn to predict diverse tokens")
    elif comma_count > len(pred_text) * 0.5:
        print("❌ ISSUE: Model predicts mostly commas")
        print("   → Model is untrained (most likely)")
        print("   → Solution: Train the model with proper loss function")
    else:
        print("✅ No obvious issues detected")
        print("   → Model architecture is working correctly")
        print("   → Model needs training to learn meaningful predictions")
        print("   → Run: python train.py --config config/config.yaml --data_dir data")

if __name__ == '__main__':
    import os
    main()
