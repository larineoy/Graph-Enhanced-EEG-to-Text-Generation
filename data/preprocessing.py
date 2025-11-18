"""
Data preprocessing utilities for ZuCo dataset
"""

import numpy as np
import pandas as pd
import scipy.io
from scipy import signal
from typing import Tuple, List, Dict
import torch
from torch.utils.data import Dataset
import os


class ZuCoDataset(Dataset):
    """
    Dataset class for ZuCo EEG-to-Text data
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        max_seq_length: int = 128,
        normalize: bool = True,
        train_split: float = 0.8,
        val_split: float = 0.1
    ):
        """
        Args:
            data_dir: Directory containing ZuCo data files
            split: Dataset split ('train', 'val', 'test', or 'all')
            max_seq_length: Maximum sequence length for text
            normalize: Whether to normalize EEG signals
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
        """
        self.data_dir = data_dir
        self.split = split
        self.max_seq_length = max_seq_length
        self.normalize = normalize
        self.train_split = train_split
        self.val_split = val_split
        
        # Load all data first
        all_eeg_data, all_text_data = self._load_data()
        
        # Split data if not 'all'
        if split == 'all':
            self.eeg_data, self.text_data = all_eeg_data, all_text_data
        else:
            self.eeg_data, self.text_data = self._split_data(all_eeg_data, all_text_data)
        
    def _load_data(self):
        """
        Load EEG and text data from ZuCo files
        """
        eeg_list = []
        text_list = []
        
        # Load MATLAB files
        mat_files = [f for f in os.listdir(self.data_dir) if f.endswith('.mat')]
        
        for mat_file in mat_files:
            filepath = os.path.join(self.data_dir, mat_file)
            
            try:
                data = scipy.io.loadmat(filepath)
                
                # Extract EEG data (structure may vary)
                # Adjust keys based on actual ZuCo file structure
                if 'YAC_NR1_EEG' in mat_file:
                    # Example: adjust based on actual structure
                    if 'eeg_data' in data:
                        eeg = data['eeg_data']
                    elif 'data' in data:
                        eeg = data['data']
                    else:
                        # Try to find EEG-like arrays
                        keys = [k for k in data.keys() if not k.startswith('__')]
                        if keys:
                            eeg = data[keys[0]]
                        else:
                            continue
                    
                    # Load corresponding text from CSV if available
                    text = self._load_text_for_eeg(mat_file)
                    
                    if eeg is not None and text is not None:
                        # Segment into samples
                        samples = self._segment_eeg_text(eeg, text)
                        eeg_list.extend([s[0] for s in samples])
                        text_list.extend([s[1] for s in samples])
            
            except Exception as e:
                print(f"Error loading {mat_file}: {e}")
                continue
        
        return eeg_list, text_list
    
    def _load_text_for_eeg(self, eeg_file: str) -> str:
        """
        Load corresponding text for EEG file
        """
        # Try to find corresponding CSV or text file
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        # Simple heuristic: match by subject/task identifier
        # Adjust based on actual ZuCo file naming convention
        for csv_file in csv_files:
            # Check if files correspond (adjust logic as needed)
            if 'nr_1' in csv_file.lower() and 'nr1' in eeg_file.lower():
                try:
                    df = pd.read_csv(os.path.join(self.data_dir, csv_file))
                    # Extract text column (adjust column name)
                    if 'text' in df.columns:
                        return ' '.join(df['text'].astype(str).tolist())
                    elif 'word' in df.columns:
                        return ' '.join(df['word'].astype(str).tolist())
                except:
                    pass
        
        # Fallback: return empty string
        return ""
    
    def _segment_eeg_text(self, eeg: np.ndarray, text: str):
        """
        Segment EEG and text into aligned samples
        """
        samples = []
        
        # Simple segmentation: split into fixed-size windows
        # In practice, you'd use word boundaries from wordbounds files
        window_size = 250 * 2  # 2 seconds at 250 Hz
        
        if eeg.ndim == 2:
            num_channels, time_steps = eeg.shape
            
            num_windows = time_steps // window_size
            text_words = text.split()
            words_per_window = max(1, len(text_words) // num_windows) if num_windows > 0 else len(text_words)
            
            for i in range(num_windows):
                start_idx = i * window_size
                end_idx = start_idx + window_size
                
                if end_idx <= time_steps:
                    eeg_window = eeg[:, start_idx:end_idx]
                    text_start = i * words_per_window
                    text_end = text_start + words_per_window
                    text_window = ' '.join(text_words[text_start:text_end])
                    
                    samples.append((eeg_window, text_window))
        
        return samples
    
    def _split_data(self, all_eeg_data, all_text_data):
        """Split data into train/val/test sets"""
        total_samples = len(all_eeg_data)
        
        train_end = int(total_samples * self.train_split)
        val_end = int(total_samples * (self.train_split + self.val_split))
        
        if self.split == 'train':
            return all_eeg_data[:train_end], all_text_data[:train_end]
        elif self.split == 'val':
            return all_eeg_data[train_end:val_end], all_text_data[train_end:val_end]
        elif self.split == 'test':
            return all_eeg_data[val_end:], all_text_data[val_end:]
        else:
            return all_eeg_data, all_text_data
    
    def _preprocess_eeg(self, eeg: np.ndarray):
        """
        Preprocess EEG signal
        """
        # Filter artifacts (simplified)
        # In practice, apply proper artifact removal
        
        if self.normalize:
            # Normalize per channel
            eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
        
        return eeg
    
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        text = self.text_data[idx]
        
        # Preprocess EEG
        eeg = self._preprocess_eeg(eeg)
        
        # Convert to tensor
        eeg_tensor = torch.FloatTensor(eeg)
        
        return {
            'eeg': eeg_tensor,
            'text': text,
            'text_tokens': text  # Will be tokenized in collate_fn
        }


def collate_fn(batch, tokenizer=None, max_seq_length=128):
    """
    Collate function for DataLoader
    
    Args:
        batch: List of samples
        tokenizer: Text tokenizer
        max_seq_length: Maximum sequence length
        
    Returns:
        batched_data: Dictionary of batched tensors
    """
    eegs = [item['eeg'] for item in batch]
    texts = [item['text'] for item in batch]
    
    # Pad EEG to same length
    max_eeg_len = max(e.shape[1] for e in eegs)
    num_channels = eegs[0].shape[0]
    
    eeg_padded = []
    for eeg in eegs:
        if eeg.shape[1] < max_eeg_len:
            padding = torch.zeros(num_channels, max_eeg_len - eeg.shape[1])
            eeg = torch.cat([eeg, padding], dim=1)
        eeg_padded.append(eeg)
    
    eeg_batch = torch.stack(eeg_padded)
    
    # Tokenize texts
    if tokenizer is not None:
        tokenized = tokenizer(
            texts,
            padding='max_length',
            max_length=max_seq_length,
            truncation=True,
            return_tensors='pt'
        )
        text_tokens = tokenized['input_ids']
    else:
        # Simple word-level tokenization (fallback)
        text_tokens = torch.zeros(len(texts), max_seq_length, dtype=torch.long)
        for i, text in enumerate(texts):
            words = text.split()[:max_seq_length]
            # Simple mapping (replace with proper tokenizer)
            for j, word in enumerate(words):
                text_tokens[i, j] = hash(word) % 10000  # Placeholder
    
    return {
        'eeg': eeg_batch,
        'text': texts,
        'text_tokens': text_tokens
    }


def load_zuco_data(data_dir: str):
    """
    Utility function to load ZuCo data
    
    Args:
        data_dir: Directory containing ZuCo files
        
    Returns:
        eeg_data: List of EEG arrays
        text_data: List of text strings
    """
    dataset = ZuCoDataset(data_dir, split='all')
    return dataset.eeg_data, dataset.text_data

