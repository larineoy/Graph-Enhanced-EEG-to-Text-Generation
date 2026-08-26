"""
Training script for Graph-Enhanced EEG-to-Text model
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
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
        k_spatial=strg_config.get('k_spatial', 6),
        k_functional=strg_config.get('k_functional', 6),
        use_spatial_topology=strg_config['use_spatial_topology'],
        use_functional_connectivity=strg_config['use_functional_connectivity'],
        use_frequency_nodes=strg_config.get('use_frequency_nodes', True),
        dynamic_functional=strg_config.get('dynamic_functional', True),
        
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
        decoder_pretrained_name=decoder_config.get('pretrained_name', 'facebook/bart-base'),
        max_decoder_length=decoder_config.get('max_decoder_length', 128),
        
        device=device
    )
    
    return model


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    loss_history = []
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        # Use eeg_bands dict from preprocessing
        eeg_bands = {band_name: band_tensor.to(device) for band_name, band_tensor in batch['eeg_bands'].items()}
        window_mask = batch['window_mask'].to(device) if 'window_mask' in batch else None
        eeg_bands_full = None
        if batch.get('eeg_bands_full') is not None:
            eeg_bands_full = {k: v.to(device) for k, v in batch['eeg_bands_full'].items()}
        eeg_windows = batch['eeg_windows'].to(device) if batch.get('eeg_windows') is not None else None
        text_tokens = batch['text_tokens'].to(device)

        logits, strg_output = model(
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
        loss, loss_dict = criterion(logits=logits, targets=targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        
        optimizer.step()
        scheduler.step()  # Update learning rate
        
        total_loss += loss.item()
        loss_history.append(loss_dict)
        
        # Update progress bar
        if batch_idx % config['training']['log_every'] == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ce_loss': f'{loss_dict.get("ce_loss", 0):.4f}'
            })
    
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
            targets = text_tokens
            loss, _ = criterion(logits=logits, targets=targets)
            total_loss += loss.item()
            
            # Generate predictions
            generated = model.generate(
                eeg_bands,
                bos_token_id=tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else 1,
                eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 2,
                pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0,
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
        decoder_name = config['model']['decoder'].get('pretrained_name', 'facebook/bart-base')
        tokenizer = AutoTokenizer.from_pretrained(decoder_name, use_fast=True)
        print(f"  ✓ {decoder_name} tokenizer loaded successfully")
    except:
        print("  ⚠ Warning: Could not load tokenizer, using simple tokenizer")
        tokenizer = None
    
    print("\n[STAGE 4/7] Loading and preprocessing datasets...")
    print("  This may take a few minutes (loading EEG files, applying filters, extracting frequency bands)...")
    
    # Create datasets with artifact removal settings from config
    data_config = config['data']
    ds_kwargs = dict(
        max_seq_length=data_config['max_seq_length'],
        apply_notch_filter=data_config.get('apply_notch_filter', True),
        notch_freq=data_config.get('notch_freq', 50.0),
        apply_highpass_filter=data_config.get('apply_highpass_filter', True),
        highpass_cutoff=data_config.get('highpass_cutoff', 0.5),
        detect_bad_channels=data_config.get('detect_bad_channels', False),
        bad_channel_threshold=data_config.get('bad_channel_threshold', 3.0),
        split_seed=data_config.get('split_seed', 42),
        window_size_sec=data_config.get('window_size_sec', 1.0),
        window_stride_sec=data_config.get('window_stride_sec', 1.0),
        max_windows=data_config.get('max_windows', 16),
    )
    train_dataset = ZuCoDataset(args.data_dir, split='train', **ds_kwargs)
    val_dataset = ZuCoDataset(args.data_dir, split='val', **ds_kwargs)
    
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
            f"Channel count mismatch: train has {actual_num_channels} channels, "
            f"validation has {val_num_channels} channels. All samples must have the same channel count."
        )
    
    print(f"  ✓ Using {actual_num_channels} channels from ZuCo data (not hardcoded)")
    
    # Override config with actual channel count from ZuCo data
    config['model']['num_channels'] = actual_num_channels
    
    print("\n[STAGE 5/7] Creating data loaders...")
    # Use num_workers=0 on Windows to avoid multiprocessing memory issues
    # On Linux/Mac, you can use num_workers > 0 if you have enough RAM
    import sys
    if sys.platform == 'win32':
        num_workers = 0  # Windows multiprocessing can cause MemoryError
        print("  ⚠ Using num_workers=0 (Windows compatibility mode to avoid memory issues)")
    else:
        num_workers = config.get('num_workers', 0)  # Default to 0 if not specified
    
    max_eeg_length = config['model'].get('max_eeg_length', 20000)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=lambda x: collate_fn(
            x, tokenizer, config['data']['max_seq_length'], max_eeg_length=max_eeg_length
        ),
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=lambda x: collate_fn(
            x, tokenizer, config['data']['max_seq_length'], max_eeg_length=max_eeg_length
        ),
        num_workers=num_workers
    )
    print(f"  ✓ Train batches: {len(train_loader)}")
    print(f"  ✓ Validation batches: {len(val_loader)}")
    
    print("\n[STAGE 6/7] Initializing model...")
    # Create model with actual channel count from ZuCo data
    model = create_model(config, device)
    
    # Set electrode positions from ZuCo dataset if available (from chanlocs)
    if train_dataset.electrode_positions is not None:
        import torch
        electrode_positions_tensor = torch.from_numpy(train_dataset.electrode_positions).float()
        model.set_electrode_positions(electrode_positions_tensor)
        print(f"  ✓ Using {len(train_dataset.electrode_positions)} electrode positions from ZuCo chanlocs (X, Y, Z)")
    else:
        print(f"  ⚠ No electrode positions found in ZuCo files, using standard 10-20 positions")
    
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
    
    # Create learning rate scheduler with warmup
    warmup_steps = config['training'].get('warmup_steps', 1000)
    total_training_steps = len(train_loader) * config['training']['num_epochs']
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )
    print(f"  ✓ Learning rate scheduler: Linear warmup ({warmup_steps} steps) + linear decay")
    print(f"    Total training steps: {total_training_steps}")
    
    # Create loss function
    pad_token_id = tokenizer.pad_token_id if tokenizer is not None and tokenizer.pad_token_id is not None else 1
    criterion = CompositeLoss(
        lambda_smooth=config['training']['lambda_smooth'],
        lambda_contrastive=config['training']['lambda_contrastive'],
        vocab_size=getattr(tokenizer, 'vocab_size', config['model']['decoder'].get('vocab_size', 50265)),
        ignore_index=pad_token_id
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
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
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
    print(f"  ⚠ IMPORTANT: With num_workers=0 (Windows compatibility), the first batch")
    print(f"    processes samples sequentially. This is NORMAL and expected behavior.")
    print(f"  ")
    print(f"  First batch processing involves:")
    print(f"    - Preprocessing each sample (filtering, frequency extraction)")
    print(f"    - Text tokenization")
    print(f"    - Model initialization on first forward pass")
    print(f"  ")
    print(f"  Progress messages will show: [DataLoader] Processing sample X/Y...")
    print(f"  Be patient - subsequent batches will be faster after caching/warmup.")
    print("="*70)
    
    # Training loop
    training_log = []
    patience_counter = 0
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f'\nEpoch {epoch+1}/{config["training"]["num_epochs"]}')
        
        # Train
        train_loss, train_loss_history = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, config)
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
        
        # Visualize learned graphs periodically (comprehensive STRG visualizations for paper)
        if (epoch + 1) % config['training'].get('visualize_every', 10) == 0:
            try:
                from utils.visualization import visualize_strg_comprehensive, save_graph_evolution
                import json
                
                model.eval()
                sample_batch = next(iter(val_loader))
                sample_eeg_bands = {k: v[:1].to(device) for k, v in sample_batch['eeg_bands'].items()}
                
                # Get electrode positions and channel names if available
                electrode_positions = None
                channel_names = None
                if hasattr(train_dataset, 'electrode_positions') and train_dataset.electrode_positions is not None:
                    electrode_positions = train_dataset.electrode_positions
                if hasattr(train_dataset, 'channel_names') and train_dataset.channel_names is not None:
                    channel_names = train_dataset.channel_names
                
                # Generate comprehensive STRG visualizations
                epoch_viz_dir = os.path.join(viz_dir, f'epoch_{epoch+1}')
                os.makedirs(epoch_viz_dir, exist_ok=True)
                
                visualize_strg_comprehensive(
                    model.strg,
                    sample_eeg_bands,
                    epoch_viz_dir,
                    epoch=epoch+1,
                    electrode_positions=electrode_positions,
                    channel_names=channel_names
                )
                
                # Also save simple adjacency heatmap for quick reference
                with torch.no_grad():
                    A, _, _ = model.strg(sample_eeg_bands)
                    A_np = A[0].cpu().numpy()
                    from utils.visualization import save_adjacency_heatmap
                    save_adjacency_heatmap(
                        A_np,
                        os.path.join(epoch_viz_dir, 'adjacency_heatmap.png'),
                        title=f'STRG Adjacency Matrix - Epoch {epoch+1}',
                        frequency_bands=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                        num_channels=config['model']['num_channels']
                    )
                
                print(f"  ✓ Saved comprehensive STRG visualizations to {epoch_viz_dir}")
                
                # Store graph evolution (save adjacency matrices for later comparison)
                if epoch == 0 or (epoch + 1) % 20 == 0:  # Every 20 epochs
                    graph_evolution_file = os.path.join(viz_dir, 'graph_evolution.npz')
                    evolution_data = {}
                    if os.path.exists(graph_evolution_file):
                        evolution_data = dict(np.load(graph_evolution_file))
                    evolution_data[f'epoch_{epoch+1}'] = A_np
                    np.savez(graph_evolution_file, **evolution_data)
                    
                    # Visualize evolution if we have enough epochs
                    if len(evolution_data) >= 3:
                        epochs_to_plot = sorted([k for k in evolution_data.keys()])[-6:]  # Last 6 epochs
                        evolution_matrices = [evolution_data[e] for e in epochs_to_plot]
                        evolution_titles = [f"Epoch {e.replace('epoch_', '')}" for e in epochs_to_plot]
                        save_graph_evolution(
                            evolution_matrices,
                            os.path.join(viz_dir, 'graph_evolution_timeline.png'),
                            titles=evolution_titles,
                            ncols=3
                        )
                
            except Exception as e:
                import traceback
                print(f"Warning: Could not save comprehensive visualization: {e}")
                traceback.print_exc()
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
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
                'scheduler_state_dict': scheduler.state_dict(),
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
