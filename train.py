"""
Training script for Graph-Enhanced EEG-to-Text model
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

from models import GraphEnhancedEEG2Text
from preprocessing.preprocessing import ZuCoDataset, collate_fn
from utils.losses import CompositeLoss
from utils.metrics import evaluate_predictions


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def load_config(config_path: str):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure numeric types are properly converted (YAML might load 1e-4 as string)
    if 'training' in config:
        if 'learning_rate' in config['training']:
            lr = config['training']['learning_rate']
            if isinstance(lr, str):
                config['training']['learning_rate'] = float(lr)
            elif not isinstance(lr, (int, float)):
                config['training']['learning_rate'] = float(lr)
        
        if 'weight_decay' in config['training']:
            wd = config['training']['weight_decay']
            if isinstance(wd, str):
                config['training']['weight_decay'] = float(wd)
            elif not isinstance(wd, (int, float)):
                config['training']['weight_decay'] = float(wd)
    
    return config


def create_model(config, device):
    """Create model from config"""
    model_config = config['model']
    strg_config = model_config['strg']
    stre_config = model_config['stre']
    decoder_config = model_config['decoder']
    
    model = GraphEnhancedEEG2Text(
        num_channels=model_config['num_channels'],
        num_frequency_bands=model_config['num_frequency_bands'],
        sampling_rate=250.0,
        
        # STRG
        strg_alpha=strg_config['alpha'],
        strg_beta=strg_config['beta'],
        use_spatial_topology=strg_config['use_spatial_topology'],
        use_functional_connectivity=strg_config['use_functional_connectivity'],
        
        # STRE
        node_dim=stre_config['node_dim'],
        graph_embed_dim=stre_config['graph_embed_dim'],
        num_gat_layers=stre_config['num_gat_layers'],
        num_gat_heads=stre_config['num_gat_heads'],
        gat_dropout=stre_config['gat_dropout'],
        num_temporal_layers=stre_config['num_temporal_layers'],
        num_temporal_heads=stre_config['num_temporal_heads'],
        temporal_ff_dim=stre_config['temporal_ff_dim'],
        temporal_dropout=stre_config['temporal_dropout'],
        
        # Decoder
        vocab_size=decoder_config['vocab_size'],
        decoder_embed_dim=decoder_config['embed_dim'],
        num_decoder_layers=decoder_config['num_layers'],
        num_decoder_heads=decoder_config['num_heads'],
        decoder_ff_dim=decoder_config['ff_dim'],
        decoder_dropout=decoder_config['dropout'],
        max_decoder_length=decoder_config['max_decoder_length'],
        
        device=device
    )
    
    return model


def train_epoch(model, dataloader, optimizer, criterion, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    loss_history = []
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        try:
            # Use eeg_bands dict from preprocessing
            eeg_bands = {band_name: band_tensor.to(device) for band_name, band_tensor in batch['eeg_bands'].items()}
            text_tokens = batch['text_tokens'].to(device)
            
            # Forward pass
            logits, strg_output = model(eeg_bands, text_tokens)
            
            # Compute loss
            # Shift for teacher forcing
            targets = text_tokens[:, 1:]
            
            # Get text embeddings for contrastive loss (using frozen BERT)
            text_embeds = strg_output.get('text_embeds')  # (batch_size, graph_embed_dim) from frozen BERT
            eeg_embeds = strg_output['stre_embeds'].squeeze(1).mean(dim=1)  # (batch_size, embed_dim)
            
            loss, loss_dict = criterion(
                logits=logits,
                targets=targets,
                node_embeddings=strg_output['node_features'],
                adjacency_matrix=strg_output['A'],
                eeg_embeddings=eeg_embeds,
                text_embeddings=text_embeds
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            
            optimizer.step()
            
            total_loss += loss.item()
            loss_history.append(loss_dict)
            
            # Update progress bar
            if batch_idx % config['training']['log_every'] == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'ce_loss': f'{loss_dict.get("ce_loss", 0):.4f}'
                })
        except Exception as e:
            import traceback
            import sys
            print(f"\n[ERROR] Failed during training batch {batch_idx}", file=sys.stderr, flush=True)
            print(f"[ERROR] Error: {str(e)}", file=sys.stderr, flush=True)
            print(f"[ERROR] Traceback:", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            raise
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, loss_history


def validate(model, dataloader, criterion, device, tokenizer, config):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    all_references = []
    all_candidates = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            # Use eeg_bands dict from preprocessing
            eeg_bands = {band_name: band_tensor.to(device) for band_name, band_tensor in batch['eeg_bands'].items()}
            text_tokens = batch['text_tokens'].to(device)
            texts = batch['text']
            
            # Forward pass
            logits, strg_output = model(eeg_bands, text_tokens)
            
            # Compute loss
            targets = text_tokens[:, 1:]
            loss, _ = criterion(logits=logits, targets=targets)
            total_loss += loss.item()
            
            # Generate predictions
            # BERT uses [CLS] (101) and [SEP] (102) as special tokens
            # Get token IDs - BERT doesn't have bos_token_id/eos_token_id attributes
            if tokenizer is not None:
                try:
                    # Use convert_tokens_to_ids for reliable token ID conversion
                    if hasattr(tokenizer, 'cls_token') and tokenizer.cls_token:
                        cls_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
                        bos_token_id = cls_id[0] if isinstance(cls_id, list) else cls_id
                    else:
                        bos_token_id = 101  # Default [CLS] token ID for BERT
                    
                    if hasattr(tokenizer, 'sep_token') and tokenizer.sep_token:
                        sep_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
                        eos_token_id = sep_id[0] if isinstance(sep_id, list) else sep_id
                    else:
                        eos_token_id = 102  # Default [SEP] token ID for BERT
                    
                    if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token:
                        pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
                        pad_token_id = pad_id[0] if isinstance(pad_id, list) else pad_id
                    else:
                        pad_token_id = 0  # Default [PAD] token ID for BERT
                except (AttributeError, KeyError):
                    # Fallback to default BERT token IDs
                    bos_token_id = 101  # [CLS]
                    eos_token_id = 102  # [SEP]
                    pad_token_id = 0    # [PAD]
            else:
                # Fallback values if tokenizer not available
                bos_token_id = 101  # [CLS] token ID for BERT
                eos_token_id = 102  # [SEP] token ID for BERT
                pad_token_id = 0    # [PAD] token ID for BERT
            
            generated = model.generate(
                eeg_bands,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                max_length=config['model']['decoder']['max_decoder_length']
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
    
    avg_loss = total_loss / len(dataloader)
    
    # Compute metrics
    metrics = evaluate_predictions(all_references, all_candidates, compute_bert=True)
    
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description='Train Graph-Enhanced EEG-to-Text model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='ZuCo Data',
                       help='Path to data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    print("="*70)
    print("GRAPH-ENHANCED EEG-TO-TEXT TRAINING")
    print("="*70)
    print("\n[STAGE 1/7] Loading configuration...")
    
    # Load config
    config = load_config(args.config)
    set_seed(config['seed'])
    print(f"  ✓ Config loaded from {args.config}")
    print(f"  ✓ Random seed set to {config['seed']}")
    
    print("\n[STAGE 2/7] Setting up device...")
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"  ✓ Using device: {device}")
    if torch.cuda.is_available():
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ CUDA version: {torch.version.cuda}")
    
    print("\n[STAGE 3/7] Initializing tokenizer...")
    # Create tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Get actual vocabulary size from tokenizer (BERT-base-uncased has 30522 tokens)
        actual_vocab_size = getattr(tokenizer, 'vocab_size', len(tokenizer))
        # Update config to use tokenizer's vocab size for decoder
        config['model']['decoder']['vocab_size'] = actual_vocab_size
        config['data']['vocab_size'] = actual_vocab_size
        print(f"  ✓ BERT tokenizer loaded successfully (vocab_size={actual_vocab_size})")
    except Exception as e:
        print(f"  ⚠ Warning: Could not load tokenizer: {e}")
        print("  ⚠ Using simple tokenizer with config vocab_size")
        tokenizer = None
    
    print("\n[STAGE 4/7] Loading and preprocessing datasets...")
    print("  This may take a few minutes (loading EEG files, applying filters, extracting frequency bands)...")
    
    # Create datasets with artifact removal settings from config
    data_config = config['data']
    train_dataset = ZuCoDataset(
        args.data_dir,
        split='train',
        max_seq_length=data_config['max_seq_length'],
        apply_notch_filter=data_config.get('apply_notch_filter', True),
        notch_freq=data_config.get('notch_freq', 50.0),
        apply_highpass_filter=data_config.get('apply_highpass_filter', True),
        highpass_cutoff=data_config.get('highpass_cutoff', 0.5),
        detect_bad_channels=data_config.get('detect_bad_channels', False),
        bad_channel_threshold=data_config.get('bad_channel_threshold', 3.0)
    )
    val_dataset = ZuCoDataset(
        args.data_dir,
        split='val',
        max_seq_length=data_config['max_seq_length'],
        apply_notch_filter=data_config.get('apply_notch_filter', True),
        notch_freq=data_config.get('notch_freq', 50.0),
        apply_highpass_filter=data_config.get('apply_highpass_filter', True),
        highpass_cutoff=data_config.get('highpass_cutoff', 0.5),
        detect_bad_channels=data_config.get('detect_bad_channels', False),
        bad_channel_threshold=data_config.get('bad_channel_threshold', 3.0)
    )
    
    print(f"  ✓ Train dataset: {len(train_dataset)} samples")
    print(f"  ✓ Validation dataset: {len(val_dataset)} samples")
    
    # Extract actual number of channels from ZuCo data (not from config)
    # This ensures we use the actual channel count from the dataset, not a hardcoded value
    actual_num_channels = train_dataset.num_channels
    if actual_num_channels is None:
        raise ValueError("Could not determine number of channels from ZuCo dataset. Check data loading.")
    
    # Verify validation dataset has same channel count
    val_num_channels = val_dataset.num_channels
    if val_num_channels is not None and val_num_channels != actual_num_channels:
        raise ValueError(
            f"Train and validation datasets have different channel counts: "
            f"train={actual_num_channels}, val={val_num_channels}"
        )
    
    # Update config to use actual channel count
    config['model']['num_channels'] = actual_num_channels
    print(f"  ✓ Using {actual_num_channels} channels (detected from ZuCo data)")
    
    print("\n[STAGE 5/7] Creating data loaders...")
    # Use num_workers=0 on Windows to avoid multiprocessing memory issues
    # On Linux/Mac, you can use num_workers > 0 if you have enough RAM
    import sys
    if sys.platform == 'win32':
        num_workers = 0  # Windows multiprocessing can cause MemoryError
        print("  ⚠ Using num_workers=0 (Windows compatibility mode to avoid memory issues)")
    else:
        num_workers = config.get('num_workers', 0)  # Default to 0 if not specified
    
    # Get max EEG length from config to limit memory usage
    # Default: 20000 time steps (~80 seconds at 250Hz) - prevents memory issues
    # Original sequences can be 100k+ time steps, which would cause OOM errors when batching
    max_eeg_length = config['model'].get('max_eeg_length', 20000)
    # Cap at 50000 to prevent extreme memory usage
    if max_eeg_length > 50000:
        max_eeg_length = 50000
        print(f"  ⚠ Limiting max_eeg_length to 50000 to avoid memory issues")
    print(f"  ✓ Max EEG sequence length: {max_eeg_length} time steps (longer sequences will be truncated)")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer, config['data']['max_seq_length'], max_eeg_length=max_eeg_length),
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer, config['data']['max_seq_length'], max_eeg_length=max_eeg_length),
        num_workers=num_workers
    )
    print(f"  ✓ Train batches: {len(train_loader)}")
    print(f"  ✓ Validation batches: {len(val_loader)}")
    
    print("\n[STAGE 6/7] Initializing model...")
    # Verify vocab size is set correctly
    decoder_vocab_size = config['model']['decoder']['vocab_size']
    if tokenizer is not None:
        tokenizer_vocab_size = getattr(tokenizer, 'vocab_size', len(tokenizer))
        if decoder_vocab_size != tokenizer_vocab_size:
            print(f"  ⚠ Warning: Decoder vocab_size ({decoder_vocab_size}) != tokenizer vocab_size ({tokenizer_vocab_size})")
            print(f"  ⚠ Updating decoder vocab_size to match tokenizer")
            config['model']['decoder']['vocab_size'] = tokenizer_vocab_size
            decoder_vocab_size = tokenizer_vocab_size
    print(f"  ✓ Decoder vocab_size: {decoder_vocab_size}")
    
    # Create model
    model = create_model(config, device)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Model created and moved to {device}")
    print(f"  ✓ Total parameters: {total_params:,}")
    print(f"  ✓ Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    print(f"  ✓ Optimizer: AdamW (lr={config['training']['learning_rate']})")
    
    # Create loss function
    criterion = CompositeLoss(
        lambda_smooth=config['training']['lambda_smooth'],
        lambda_contrastive=config['training']['lambda_contrastive'],
        vocab_size=config['model']['decoder']['vocab_size']
    )
    print(f"  ✓ Loss function: Composite (λ_smooth={config['training']['lambda_smooth']}, λ_contrastive={config['training']['lambda_contrastive']})")
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"  ✓ Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"  ✓ Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    else:
        print("  ✓ Starting training from scratch")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"  ✓ Checkpoint directory: {args.checkpoint_dir}")
    
    # Create visualization directory
    viz_dir = os.path.join(args.checkpoint_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    print("\n[STAGE 7/7] Starting training loop...")
    print("="*70)
    
    # Training loop
    training_log = []
    patience_counter = 0
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f'\nEpoch {epoch+1}/{config["training"]["num_epochs"]}')
        
        # Train
        train_loss, train_loss_history = train_epoch(model, train_loader, optimizer, criterion, device, config)
        print(f'Train Loss: {train_loss:.4f}')
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device, tokenizer, config)
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val Metrics: {json.dumps({k: f"{v:.2f}" for k, v in val_metrics.items()}, indent=2)}')
        
        # Log training metrics
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metrics': val_metrics
        }
        training_log.append(log_entry)
        
        # Visualize learned graphs periodically
        if (epoch + 1) % 10 == 0:
            try:
                from utils.visualization import save_adjacency_heatmap
                model.eval()
                sample_batch = next(iter(val_loader))
                sample_eeg_bands = {k: v[:1].to(device) for k, v in sample_batch['eeg_bands'].items()}
                with torch.no_grad():
                    A, _, _ = model.strg(sample_eeg_bands)
                    A_np = A[0].cpu().numpy()
                    save_adjacency_heatmap(
                        A_np,
                        os.path.join(viz_dir, f'adjacency_epoch_{epoch+1}.png'),
                        title=f'Learned Adjacency Matrix - Epoch {epoch+1}',
                        frequency_bands=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                        num_channels=config['model']['num_channels']
                    )
            except Exception as e:
                print(f"Warning: Could not save visualization: {e}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best_model.pt'))
            print(f'Saved best model with val loss: {best_val_loss:.4f}')
        
        # Periodic checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
        # Early stopping check
        if config['training'].get('early_stopping', False):
            patience = config['training'].get('early_stopping_patience', 5)
            min_delta = config['training'].get('early_stopping_min_delta', 0.001)
            
            # Track best validation loss
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'\nEarly stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)')
                    break
    
    # Save training log
    log_path = os.path.join(args.checkpoint_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    print(f"  ✓ Training log saved to: {log_path}")
    print(f"  ✓ Best model saved to: {os.path.join(args.checkpoint_dir, 'best_model.pt')}")
    print(f"  ✓ Total epochs trained: {len(training_log)}")
    print(f"  ✓ Best validation loss: {best_val_loss:.4f}")
    if len(training_log) > 0:
        final_metrics = training_log[-1].get('val_metrics', {})
        if final_metrics:
            print(f"  ✓ Final validation metrics:")
            for metric, value in final_metrics.items():
                print(f"      - {metric}: {value:.4f}")
    print("="*70)


if __name__ == '__main__':
    main()

