"""
Training script for Graph-Enhanced EEG-to-Text model
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


def dataset_kwargs_from_config(config):
    """Shared ZuCoDataset arguments that encode the paper's split and windowing."""
    data_config = config['data']
    return dict(
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


def move_eeg_batch(batch, device):
    eeg_bands = {k: v.to(device) for k, v in batch['eeg_bands'].items()}
    window_mask = batch['window_mask'].to(device) if 'window_mask' in batch else None
    eeg_bands_full = None
    if batch.get('eeg_bands_full') is not None:
        eeg_bands_full = {k: v.to(device) for k, v in batch['eeg_bands_full'].items()}
    eeg_windows = batch['eeg_windows'].to(device) if batch.get('eeg_windows') is not None else None
    return eeg_bands, window_mask, eeg_bands_full, eeg_windows


def get_decoder_tokenizer(config):
    name = config.get('model', {}).get('decoder', {}).get('pretrained_name', 'facebook/bart-base')
    return AutoTokenizer.from_pretrained(name, use_fast=True)


def sync_model_config_from_dataset(config, dataset):
    if getattr(dataset, 'num_channels', None) is not None:
        config['model']['num_channels'] = dataset.num_channels
    return config


def resolve_num_workers(config) -> int:
    """ZuCo is fully in RAM; forking workers duplicates it and OOMs laptops."""
    import sys
    requested = int(config.get('num_workers', 0) or 0)
    if sys.platform in ('darwin', 'win32') and requested > 0:
        print(f"  ⚠ {sys.platform}: forcing num_workers=0 (DataLoader fork copies the dataset)")
        return 0
    return requested


def resolve_batch_size(config, device) -> int:
    batch_size = int(config['training']['batch_size'])
    if getattr(device, 'type', str(device)) == 'cpu' and batch_size > 2:
        print(f"  ⚠ CPU: reducing batch_size {batch_size} → 2 (525-node graphs + BART)")
        config['training']['batch_size'] = 2
        return 2
    return batch_size


def configure_pretrained_decoder(model, config):
    """Freeze BART's unused text encoder. Train the decoder so it can read EEG memory M."""
    bart = getattr(getattr(model, 'decoder', None), 'bart', None)
    if bart is None:
        return model
    decoder_cfg = config.get('model', {}).get('decoder', {})
    if not decoder_cfg.get('freeze_pretrained', True):
        return model

    freeze_encoder_only = decoder_cfg.get('freeze_encoder_only', True)
    for param in bart.parameters():
        param.requires_grad = False

    if freeze_encoder_only:
        unfrozen = 0
        for name, param in bart.named_parameters():
            if 'model.encoder' not in name:
                param.requires_grad = True
                unfrozen += 1
        print(
            f"  ✓ Frozen BART text encoder; training {unfrozen} decoder/lm_head tensors + EEG encoder"
        )
        return model

    unfrozen = 0
    unfreeze_decoder = decoder_cfg.get('unfreeze_decoder_layers', True)
    unfreeze_xattn = decoder_cfg.get('unfreeze_cross_attention', True)
    for name, param in bart.named_parameters():
        train_layer = unfreeze_decoder and 'decoder.layers' in name
        train_xattn = (not unfreeze_decoder) and unfreeze_xattn and 'encoder_attn' in name
        if train_layer or train_xattn:
            param.requires_grad = True
            unfrozen += 1
    if unfreeze_decoder:
        print(
            f"  ✓ Frozen BART encoder, embeddings, and lm_head; "
            f"training {unfrozen} decoder-layer tensors + EEG encoder"
        )
    elif unfreeze_xattn:
        print(f"  ✓ Frozen BART language model; training {unfrozen} cross-attention tensors + EEG encoder")
    else:
        print("  ✓ Frozen entire BART (training EEG encoder and projection only)")
    return model


def build_optimizer(model, config):
    """Higher LR on the EEG encoder; lower LR on BART decoder layers."""
    eeg_params = []
    bart_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'decoder.bart' in name or '.bart.' in name:
            bart_params.append(param)
        else:
            eeg_params.append(param)
    if not eeg_params and not bart_params:
        raise RuntimeError('No trainable parameters after decoder freeze settings.')
    base_lr = float(config['training']['learning_rate'])
    eeg_lr = base_lr * float(config['training'].get('eeg_lr_multiplier', 5.0))
    param_groups = []
    if eeg_params:
        param_groups.append({'params': eeg_params, 'lr': eeg_lr})
    if bart_params:
        param_groups.append({'params': bart_params, 'lr': base_lr})
    optimizer = optim.AdamW(param_groups, weight_decay=config['training']['weight_decay'])
    print(
        f"  ✓ Optimizer: AdamW (EEG lr={eeg_lr:g} on {len(eeg_params)} tensors, "
        f"BART decoder lr={base_lr:g} on {len(bart_params)} tensors)"
    )
    return optimizer


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
        k_spatial=strg_config.get('k_spatial', 6),
        k_functional=strg_config.get('k_functional', 6),
        use_spatial_topology=strg_config.get('use_spatial_topology', True),
        use_functional_connectivity=strg_config.get('use_functional_connectivity', True),
        use_frequency_nodes=strg_config.get('use_frequency_nodes', True),
        dynamic_functional=strg_config.get('dynamic_functional', True),
        node_dim=stre_config.get('node_dim', 1),
        graph_embed_dim=stre_config['graph_embed_dim'],
        num_gat_layers=stre_config['num_gat_layers'],
        num_gat_heads=stre_config['num_gat_heads'],
        gat_dropout=stre_config['gat_dropout'],
        num_temporal_layers=stre_config['num_temporal_layers'],
        num_temporal_heads=stre_config['num_temporal_heads'],
        temporal_ff_dim=stre_config['temporal_ff_dim'],
        temporal_dropout=stre_config['temporal_dropout'],
        decoder_pretrained_name=decoder_config.get('pretrained_name', 'facebook/bart-base'),
        max_decoder_length=decoder_config.get('max_decoder_length', 128),
        device=device
    )
    configure_pretrained_decoder(model, config)
    return model


def train_epoch(model, dataloader, optimizer, criterion, device, config, tokenizer=None, pad_token_id=0, epoch=0):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    loss_history = []
    
    # Track first batch check per epoch
    first_batch_key = f'_first_batch_epoch_{epoch}'
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        try:
            eeg_bands, window_mask, eeg_bands_full, eeg_windows = move_eeg_batch(batch, device)
            text_tokens = batch['text_tokens'].to(device)
            pad_token_id = pad_token_id if pad_token_id is not None else 1

            logits, strg_output = model(
                eeg_bands,
                text_tokens,
                window_mask=window_mask,
                eeg_bands_full=eeg_bands_full,
                eeg_windows=eeg_windows
            )
            # BART logits are aligned with the full label sequence (not GPT-style shift)
            targets = text_tokens
            if logits.shape[1] != targets.shape[1]:
                min_len = min(logits.shape[1], targets.shape[1])
                logits = logits[:, :min_len, :]
                targets = targets[:, :min_len]
            
            # DIAGNOSTIC: Check if logits are reasonable (not all zeros or NaNs)
            if batch_idx == 0 and not hasattr(train_epoch, first_batch_key):  # Only check first batch of each epoch
                setattr(train_epoch, first_batch_key, True)
                with torch.no_grad():
                    logits_sample = logits[0, 0, :].cpu()  # First sample, first position
                    logits_max = logits_sample.max().item()
                    logits_min = logits_sample.min().item()
                    logits_std = logits_sample.std().item()
                    logits_top5 = torch.topk(logits_sample, 5).indices.tolist()
                    print(f"\n[DEBUG Epoch {epoch}] First batch logits check:")
                    print(f"  Logits range: [{logits_min:.2f}, {logits_max:.2f}], std: {logits_std:.2f}")
                    print(f"  Top 5 token IDs: {logits_top5}")
                    
                    # Decode top tokens
                    if tokenizer is not None:
                        print(f"  Top 5 tokens:")
                        for idx, token_id in enumerate(logits_top5):
                            token_str = tokenizer.decode([token_id])
                            print(f"    {idx+1}. ID={token_id}, token='{token_str}'")
                    
                    # Check EEG embeddings (pre-LayerNorm STRE states distinguish sentences)
                    eeg_embeds_check = strg_output['stre_embeds'][0, 0].cpu()
                    eeg_norm = torch.norm(eeg_embeds_check).item()
                    print(f"  EEG embedding norm: {eeg_norm:.4f}")
                    if eeg_norm < 1e-6:
                        print(f"  ⚠️  WARNING: EEG embeddings are near zero! This will cause poor predictions.")
                    nf = strg_output.get('node_features')
                    if nf is not None:
                        print(
                            f"  Node features: mean={nf.mean().item():.4f}, "
                            f"std={nf.std().item():.4f}"
                        )
                    embeds = strg_output.get('stre_embeds')
                    if embeds is not None and embeds.size(0) > 1:
                        pooled = embeds.mean(dim=1)
                        pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-8)
                        sim = pooled @ pooled.T
                        offdiag = sim[~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)]
                        print(f"  EEG memory pairwise cosine: {offdiag.mean().item():.3f}")
                        if offdiag.mean().item() > 0.99:
                            print("  ⚠️  WARNING: EEG memories are nearly identical across the batch.")
            
            # Compute loss
            # Targets already computed above (full BART label sequence)
            
            # DIAGNOSTIC: Check targets (first batch only, first epoch only)
            if batch_idx == 0 and epoch == 0 and tokenizer is not None:
                with torch.no_grad():
                    # Check target token distribution
                    targets_flat = targets.reshape(-1).cpu()
                    unique_tokens, counts = torch.unique(targets_flat, return_counts=True)
                    top10_indices = counts.argsort(descending=True)[:10]
                    top10_tokens = unique_tokens[top10_indices]
                    top10_counts = counts[top10_indices]
                    
                    print(f"\n[DEBUG] Target tokens analysis (first batch, epoch {epoch}):")
                    print(f"  Total target tokens: {len(targets_flat)}")
                    print(f"  Unique tokens: {len(unique_tokens)}")
                    
                    # Print raw text and tokens for verification
                    if len(batch['text']) > 0:
                        print(f"  RAW TEXT (first sample): {batch['text'][0][:100]}...")
                        print(f"  TOKENS (first 50): {tokenizer.convert_ids_to_tokens(text_tokens[0][:50].tolist())}")
                    
                    print(f"  Top 10 most frequent target tokens:")
                    for i, (token_id, count) in enumerate(zip(top10_tokens, top10_counts)):
                        token_str = tokenizer.decode([token_id.item()])
                        pct = (count.item() / len(targets_flat)) * 100
                        print(f"    {i+1}. ID={token_id.item():5d}, count={count.item():4d} ({pct:5.1f}%), token='{token_str}'")
                    
                    # Check if comma is dominating
                    comma_id = tokenizer.convert_tokens_to_ids(',')
                    if comma_id in unique_tokens:
                        comma_idx = (unique_tokens == comma_id).nonzero(as_tuple=True)[0]
                        if len(comma_idx) > 0:
                            comma_count = counts[comma_idx[0]].item()
                            comma_pct = (comma_count / len(targets_flat)) * 100
                            print(f"  Comma token (ID={comma_id}): {comma_count}/{len(targets_flat)} ({comma_pct:.1f}%)")
                            if comma_pct > 30:
                                print(f"  ⚠️  WARNING: Comma is >30% of targets! This might cause model to predict only commas.")
                    
                    # Check if targets are mostly padding
                    pad_count = (targets_flat == pad_token_id).sum().item()
                    pad_pct = (pad_count / len(targets_flat)) * 100
                    print(f"  Padding tokens: {pad_count}/{len(targets_flat)} ({pad_pct:.1f}%)")
                    if pad_pct > 50:
                        print(f"  ⚠️  WARNING: More than 50% of targets are padding! This might cause issues.")
            
            # DIAGNOSTIC: Check targets (first batch only, first epoch only)
            if batch_idx == 0 and hasattr(train_epoch, '_first_batch_checked'):
                pass  # Skip if already checked
            elif batch_idx == 0:
                train_epoch._first_batch_checked = True
                with torch.no_grad():
                    # Check target token distribution
                    targets_flat = targets.reshape(-1).cpu()
                    unique_tokens, counts = torch.unique(targets_flat, return_counts=True)
                    top10_tokens, top10_counts = unique_tokens[counts.argsort(descending=True)[:10]], counts[counts.argsort(descending=True)[:10]]
                    
                    print(f"\n[DEBUG] Target tokens analysis (first batch):")
                    print(f"  Total target tokens: {len(targets_flat)}")
                    print(f"  Unique tokens: {len(unique_tokens)}")
                    print(f"  Top 10 most frequent target tokens:")
                    for i, (token_id, count) in enumerate(zip(top10_tokens, top10_counts)):
                        token_str = tokenizer.decode([token_id.item()]) if hasattr(tokenizer, 'decode') else str(token_id.item())
                        pct = (count.item() / len(targets_flat)) * 100
                        print(f"    {i+1}. ID={token_id.item():5d}, count={count.item():4d} ({pct:5.1f}%), token='{token_str}'")
                    
                    # Check if targets are mostly padding
                    pad_count = (targets_flat == pad_token_id).sum().item()
                    pad_pct = (pad_count / len(targets_flat)) * 100
                    print(f"  Padding tokens: {pad_count}/{len(targets_flat)} ({pad_pct:.1f}%)")
                    if pad_pct > 50:
                        print(f"  ⚠️  WARNING: More than 50% of targets are padding! This might cause issues.")
            
            # Generation loss only (paper: no contrastive or graph-regularization terms)
            loss, loss_dict = criterion(
                logits=logits,
                targets=targets
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # DIAGNOSTIC: Check gradients (first batch only)
            if batch_idx == 0:
                total_grad_norm = 0.0
                param_count = 0
                zero_grad_count = 0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param_grad_norm = param.grad.norm().item()
                        total_grad_norm += param_grad_norm
                        param_count += 1
                        if param_grad_norm < 1e-8:
                            zero_grad_count += 1
                    else:
                        zero_grad_count += 1
                if param_count > 0:
                    avg_grad_norm = total_grad_norm / param_count
                    print(f"  Gradient check: avg_norm={avg_grad_norm:.6f}, zero_grad_params={zero_grad_count}")
                    if avg_grad_norm < 1e-6:
                        print(f"  ⚠️  WARNING: Gradients are very small! Model may not be learning.")
                else:
                    print(f"  ⚠️  WARNING: No gradients found!")
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config['training']['gradient_clip']
            )
            
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


def _decode_generated(generated, texts, text_tokens, tokenizer, debug_predictions, all_references, all_candidates):
    for i in range(len(texts)):
        ref = texts[i].split()
        if hasattr(tokenizer, 'decode'):
            cand = tokenizer.decode(generated[i].cpu().tolist(), skip_special_tokens=True).split()
            pred_text = ' '.join(cand)
        else:
            cand = [str(t.item()) for t in generated[i]]
            pred_text = ' '.join(cand)
        all_references.append(ref)
        all_candidates.append(cand)
        if len(debug_predictions) < 5:
            ref_ids = text_tokens[i].cpu().tolist()[:30]
            pred_ids = generated[i].cpu().tolist()[:30]
            debug_predictions.append((ref_ids, pred_ids, texts[i], pred_text))


def _print_debug_predictions(debug_predictions, all_candidates):
    if not debug_predictions:
        return
    print(f"\n[VALIDATION DEBUG] First {len(debug_predictions)} predictions:")
    for idx, (ref_ids, pred_ids, ref_text, pred_text) in enumerate(debug_predictions):
        print(f"\n  Example {idx+1}:")
        print(f"    Reference: {ref_text[:100]}...")
        print(f"    Prediction: {pred_text[:100] if pred_text else '<empty>'}...")
        print(f"    Ref tokens (first 20): {ref_ids[:20]}")
        print(f"    Pred tokens (first 20): {pred_ids[:20]}")
        if pred_text:
            comma_count = pred_text.count(',')
            if comma_count > len(pred_text) * 0.5:
                print(f"    ⚠️  WARNING: Prediction is mostly commas ({comma_count}/{len(pred_text)} chars)")
            pred_tokens = pred_text.split()
            if pred_tokens:
                most_common = max(set(pred_tokens), key=pred_tokens.count)
                repetition_ratio = pred_tokens.count(most_common) / len(pred_tokens)
                if repetition_ratio > 0.5:
                    print(f"    ⚠️  WARNING: Prediction is repetitive ('{most_common}' appears {repetition_ratio:.1%} of the time)")
    if all_candidates:
        n_empty = sum(1 for c in all_candidates if len(c) == 0)
        if n_empty:
            print(
                f"\n  ⚠️  WARNING: {n_empty}/{len(all_candidates)} predictions "
                "are empty after removing special tokens."
            )
        unique_preds = {' '.join(c) for c in all_candidates}
        if len(all_candidates) >= 2 and len(unique_preds) == 1:
            print("\n  ⚠️  WARNING: every validation prediction is identical.")
            print("  The decoder is ignoring EEG and emitting one high-likelihood sentence.")


def validate(model, dataloader, criterion, device, tokenizer, config, compute_generate=False):
    """Teacher-forced val CE always. Full-set generation metrics only when requested."""
    model.eval()
    total_loss = 0.0
    all_references = []
    all_candidates = []
    debug_predictions = []
    max_len = config['model']['decoder']['max_decoder_length']

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Validating')):
            eeg_bands, window_mask, eeg_bands_full, eeg_windows = move_eeg_batch(batch, device)
            text_tokens = batch['text_tokens'].to(device)
            texts = batch['text']
            targets = text_tokens

            logits, _ = model(
                eeg_bands,
                text_tokens,
                window_mask=window_mask,
                eeg_bands_full=eeg_bands_full,
                eeg_windows=eeg_windows
            )
            if logits.shape[1] != targets.shape[1]:
                min_len = min(logits.shape[1], targets.shape[1])
                logits = logits[:, :min_len, :]
                targets = targets[:, :min_len]
            loss, _ = criterion(logits=logits, targets=targets)
            total_loss += loss.item()

            want_debug = len(debug_predictions) < 5
            if compute_generate or want_debug:
                generated = model.generate(
                    eeg_bands,
                    max_length=max_len,
                    beam_size=config.get('evaluation', {}).get('beam_size', 5) if compute_generate else 1,
                    window_mask=window_mask,
                    eeg_bands_full=eeg_bands_full,
                    eeg_windows=eeg_windows
                )
                if compute_generate:
                    _decode_generated(
                        generated, texts, text_tokens, tokenizer,
                        debug_predictions, all_references, all_candidates
                    )
                elif want_debug:
                    _decode_generated(
                        generated, texts, text_tokens, tokenizer,
                        debug_predictions, [], []
                    )

    _print_debug_predictions(debug_predictions, all_candidates if compute_generate else [p[3].split() for p in debug_predictions])
    avg_loss = total_loss / max(len(dataloader), 1)
    if compute_generate and all_references:
        metrics = evaluate_predictions(all_references, all_candidates, compute_bert=True)
    else:
        metrics = {}
        print("  (full BLEU/ROUGE skipped this epoch; watching val CE and the 5 debug predictions)")
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
        actual_vocab_size = getattr(tokenizer, 'vocab_size', len(tokenizer))
        config['model']['decoder']['vocab_size'] = actual_vocab_size
        config['data']['vocab_size'] = actual_vocab_size
        print(f"  ✓ {decoder_name} tokenizer loaded (vocab_size={actual_vocab_size})")
        print(f"  ✓ pad_token_id={tokenizer.pad_token_id}, bos={tokenizer.bos_token_id}, eos={tokenizer.eos_token_id}")
    except Exception as e:
        print(f"  ⚠ Warning: Could not load tokenizer: {e}")
        print("  ⚠ Using simple tokenizer with config vocab_size")
        tokenizer = None
    
    print("\n[STAGE 4/7] Loading and preprocessing datasets...")
    print("  This may take a few minutes (loading EEG files, applying filters, extracting frequency bands)...")
    
    # SANITY TEST MODE: Train on 32 samples only to verify pipeline works
    # Set sanity_test: true in config to enable this (catches 80% of hidden bugs)
    sanity_test_mode = config.get('sanity_test', False)
    if sanity_test_mode:
        print("  ⚠️  SANITY TEST MODE: Training on 32 samples only to verify pipeline")
    
    ds_kwargs = dataset_kwargs_from_config(config)
    print("  Loading ZuCo once, then splitting in memory (avoids a second full MATLAB pass)...")
    splits = ZuCoDataset.load_splits(args.data_dir, splits=('train', 'val'), **ds_kwargs)
    train_dataset = splits['train']
    val_dataset = splits['val']
    
    # Apply sanity test subset if enabled
    if sanity_test_mode:
        train_dataset = Subset(train_dataset, range(min(32, len(train_dataset))))
        print(f"  ⚠️  SANITY TEST: Using only {len(train_dataset)} samples for training")
    
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
    num_workers = resolve_num_workers(config)
    resolve_batch_size(config, device)
    print(f"  ✓ DataLoader workers: {num_workers}")

    max_eeg_length = config['model'].get('max_eeg_length', 4000)
    print(f"  ✓ Max EEG sequence length: {max_eeg_length} time steps (longer sequences will be truncated)")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer, config['data']['max_seq_length'], max_eeg_length=max_eeg_length),
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer, config['data']['max_seq_length'], max_eeg_length=max_eeg_length),
        num_workers=num_workers,
        pin_memory=False
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
    if getattr(train_dataset, 'electrode_positions', None) is not None:
        model.set_electrode_positions(train_dataset.electrode_positions)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Model created and moved to {device}")
    print(f"  ✓ Total parameters: {total_params:,}")
    print(f"  ✓ Trainable parameters: {trainable_params:,}")
    
    optimizer = build_optimizer(model, config)
    
    # Get pad_token_id for loss function (must ignore padding tokens in loss)
    pad_token_id = tokenizer.pad_token_id if tokenizer is not None and tokenizer.pad_token_id is not None else 1
    
    # Create loss function
    criterion = CompositeLoss(
        lambda_smooth=config['training']['lambda_smooth'],
        lambda_contrastive=config['training']['lambda_contrastive'],
        vocab_size=config['model']['decoder']['vocab_size'],
        ignore_index=pad_token_id  # CRITICAL: Ignore padding tokens (0) in loss calculation
    )
    # Verify ignore_index is correctly set (CRITICAL DEBUG CHECK)
    print(f"  ✓ Loss function: Composite (λ_smooth={config['training']['lambda_smooth']}, λ_contrastive={config['training']['lambda_contrastive']})")
    print(f"  ✓ CE ignore_index = {criterion.ce_loss.ignore_index}")
    print(f"  ✓ pad_token_id = {pad_token_id}")
    if criterion.ce_loss.ignore_index != pad_token_id:
        print(f"  ⚠️  WARNING: ignore_index mismatch! This will cause training issues!")
    else:
        print(f"  ✓ ignore_index matches pad_token_id - padding tokens will be ignored in loss")
    
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
    best_val_metrics = {}
    best_epoch = 0
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f'\nEpoch {epoch+1}/{config["training"]["num_epochs"]}')
        
        # Train
        train_loss, train_loss_history = train_epoch(model, train_loader, optimizer, criterion, device, config, tokenizer=tokenizer, pad_token_id=pad_token_id, epoch=epoch)
        print(f'Train Loss: {train_loss:.4f}')
        
        # Validate
        eval_every = int(config['evaluation'].get('eval_generate_every', 5))
        compute_generate = ((epoch + 1) % eval_every == 0) or epoch == 0
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, tokenizer, config,
            compute_generate=compute_generate
        )
        print(f'Val Loss: {val_loss:.4f}')
        if val_metrics:
            print(f'Val Metrics: {json.dumps({k: f"{v:.2f}" for k, v in val_metrics.items()}, indent=2)}')
        else:
            print('Val Metrics: skipped (see debug predictions; full BLEU every eval_generate_every epochs)')
        
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
                    strg_out = model.strg(sample_eeg_bands)
                    A_np = strg_out['edge_mask'][0, 0].cpu().numpy()
                    save_adjacency_heatmap(
                        A_np,
                        os.path.join(viz_dir, f'adjacency_epoch_{epoch+1}.png'),
                        title=f'Learned Adjacency Matrix - Epoch {epoch+1}',
                        frequency_bands=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                        num_channels=config['model']['num_channels']
                    )
            except Exception as e:
                print(f"Warning: Could not save visualization: {e}")
        
        # Save checkpoint. Compare against the previous best *before* updating it,
        # otherwise patience increments on the same epoch that just improved.
        previous_best = best_val_loss
        min_delta = config['training'].get('early_stopping_min_delta', 0.001)
        if val_loss < previous_best:
            best_val_loss = val_loss
            best_val_metrics = dict(val_metrics)
            best_epoch = epoch + 1
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_metrics': best_val_metrics,
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
            if val_loss < previous_best - min_delta:
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
    print(f"  ✓ Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
    if best_val_metrics:
        print("  ✓ Best-checkpoint validation metrics:")
        for metric, value in best_val_metrics.items():
            print(f"      - {metric}: {value:.4f}")
    elif training_log:
        final_metrics = training_log[-1].get('val_metrics', {})
        if final_metrics:
            print("  ✓ Last-epoch validation metrics:")
            for metric, value in final_metrics.items():
                print(f"      - {metric}: {value:.4f}")
    print("="*70)


if __name__ == '__main__':
    main()

