"""
Debug script to diagnose why model always predicts commas
"""

import torch
import argparse
import os
from transformers import BertTokenizer
from models.graph_enhanced_eeg2text import GraphEnhancedEEG2Text
from preprocessing.preprocessing import ZuCoDataset, collate_fn
from torch.utils.data import DataLoader
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
    
    # Load tokenizer
    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        actual_vocab_size = getattr(tokenizer, 'vocab_size', len(tokenizer))
        print(f"  ✓ BERT tokenizer loaded (vocab_size={actual_vocab_size})")
    except Exception as e:
        print(f"  ⚠ Warning: Could not load tokenizer: {e}")
        print("  Using BertTokenizer as fallback")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        actual_vocab_size = len(tokenizer)
    
    # Load dataset to get actual num_channels and vocab_size
    print("\nLoading dataset...")
    data_config = config.get('data', {})
    train_dataset = ZuCoDataset(
        args.data_dir,
        split='train',
        max_seq_length=config['model']['decoder']['max_decoder_length'],
        apply_notch_filter=data_config.get('apply_notch_filter', True),
        notch_freq=data_config.get('notch_freq', 50.0),
        apply_highpass_filter=data_config.get('apply_highpass_filter', True),
        highpass_cutoff=data_config.get('highpass_cutoff', 0.5),
        detect_bad_channels=data_config.get('detect_bad_channels', False),
        bad_channel_threshold=data_config.get('bad_channel_threshold', 3.0)
    )
    
    # Update config with actual values
    actual_num_channels = train_dataset.num_channels
    if actual_num_channels is None:
        # Fallback: get from first sample
        sample = train_dataset[0]
        actual_num_channels = sample['eeg_bands']['delta'].shape[0]
    
    print(f"  ✓ Detected num_channels: {actual_num_channels}")
    print(f"  ✓ Using vocab_size: {actual_vocab_size}")
    
    config['model']['num_channels'] = actual_num_channels
    config['model']['decoder']['vocab_size'] = actual_vocab_size
    
    # Create model
    print("Creating model...")
    model_config = config['model']
    strg_config = model_config.get('strg', {})
    stre_config = model_config.get('stre', {})
    decoder_config = model_config.get('decoder', {})
    
    model = GraphEnhancedEEG2Text(
        num_channels=model_config['num_channels'],
        num_frequency_bands=model_config['num_frequency_bands'],
        sampling_rate=250.0,
        
        # STRG
        strg_alpha=strg_config.get('alpha', 0.5),
        strg_beta=strg_config.get('beta', 0.5),
        use_spatial_topology=strg_config.get('use_spatial_topology', True),
        use_functional_connectivity=strg_config.get('use_functional_connectivity', True),
        
        # STRE
        node_dim=stre_config.get('node_dim', 1),
        graph_embed_dim=stre_config.get('graph_embed_dim', 256),
        num_gat_layers=stre_config.get('num_gat_layers', 2),
        num_gat_heads=stre_config.get('num_gat_heads', 4),
        gat_dropout=stre_config.get('gat_dropout', 0.1),
        num_temporal_layers=stre_config.get('num_temporal_layers', 4),
        num_temporal_heads=stre_config.get('num_temporal_heads', 8),
        temporal_ff_dim=stre_config.get('temporal_ff_dim', 512),
        temporal_dropout=stre_config.get('temporal_dropout', 0.1),
        
        # Decoder
        vocab_size=decoder_config.get('vocab_size', 10000),
        decoder_embed_dim=decoder_config.get('embed_dim', 256),
        num_decoder_layers=decoder_config.get('num_layers', 4),
        num_decoder_heads=decoder_config.get('num_heads', 8),
        decoder_ff_dim=decoder_config.get('ff_dim', 512),
        decoder_dropout=decoder_config.get('dropout', 0.1),
        max_decoder_length=decoder_config.get('max_decoder_length', 128),
        
        device=device
    ).to(device)
    
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
    eeg_bands = {k: v.to(device) for k, v in batch['eeg_bands'].items()}
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
        logits, strg_output = model(eeg_bands, text_tokens)
        
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
        stre_embeds = strg_output['stre_embeds'].squeeze(1)[0].cpu()
        print(f"\nSTRE embeddings:")
        print(f"  Norm: {torch.norm(stre_embeds).item():.4f}")
        print(f"  Std: {stre_embeds.std().item():.4f}")
        if torch.norm(stre_embeds).item() < 1e-6:
            print(f"  ⚠️  WARNING: STRE embeddings are near zero!")
    
    # Generate predictions
    print("\n" + "="*60)
    print("CHECKING GENERATION")
    print("="*60)
    
    bos_token_id = tokenizer.cls_token_id
    eos_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    
    print(f"Using bos_token_id={bos_token_id}, eos_token_id={eos_token_id}, pad_token_id={pad_token_id}")
    
    generated = model.generate(
        eeg_bands,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_length=50  # Shorter for debugging
    )
    
    print(f"\nGenerated token IDs (first 30): {generated[0][:30].tolist()}")
    
    # Decode
    pred_text = tokenizer.decode(generated[0].cpu().tolist(), skip_special_tokens=True)
    print(f"Generated text: {pred_text}")
    
    # Count comma frequency
    comma_count = pred_text.count(',')
    print(f"Comma count: {comma_count} out of {len(pred_text)} characters")
    
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
    elif comma_count > len(pred_text) * 0.5:
        print("❌ ISSUE: Model predicts mostly commas")
        print("   → Model is untrained (most likely)")
        print("   → Solution: Train the model with proper loss function")
    else:
        print("✅ No obvious issues detected")
        print("   → Model might be working, but needs training")

if __name__ == '__main__':
    import os
    main()
