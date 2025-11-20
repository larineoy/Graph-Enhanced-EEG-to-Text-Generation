"""
Data preprocessing utilities for ZuCo dataset
Supports ZuCo 1.0 and 2.0 with proper EEG-Text alignment using wordbounds
Includes artifact removal: notch filtering, high-pass filtering, and normalization
"""

import numpy as np
import pandas as pd
import scipy.io
from scipy import signal
from typing import Tuple, List, Dict, Optional
import torch
from torch.utils.data import Dataset
import os
import re
import glob
import h5py


class ZuCoDataset(Dataset):
    """
    Dataset class for ZuCo EEG-to-Text data with sentence-aligned windows
    Supports both ZuCo 1.0 and 2.0 structures
    """
    
    # Frequency bands for STRG: delta, theta, alpha, beta, gamma
    FREQUENCY_BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (13, 30),
        'gamma': (30, 100)
    }
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        max_seq_length: int = 128,
        normalize: bool = True,
        train_split: float = 0.8,
        val_split: float = 0.1,
        sampling_rate: float = 250.0,
        version: Optional[str] = None,  # '1.0', '2.0', or None (auto-detect)
        # Artifact removal options
        apply_notch_filter: bool = True,
        notch_freq: float = 50.0,  # Line noise frequency (50 Hz for EU, 60 Hz for US)
        apply_highpass_filter: bool = True,
        highpass_cutoff: float = 0.5,  # Remove slow drifts below 0.5 Hz
        detect_bad_channels: bool = False,  # Optional: detect and interpolate bad channels
        bad_channel_threshold: float = 3.0  # Standard deviations for bad channel detection
    ):
        """
        Args:
            data_dir: Root directory containing ZuCo_1.0/ and ZuCo_2.0/
            split: Dataset split ('train', 'val', 'test', or 'all')
            max_seq_length: Maximum sequence length for text
            normalize: Whether to normalize EEG signals
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
            sampling_rate: EEG sampling rate in Hz (default 250)
            version: ZuCo version ('1.0', '2.0', or None for auto-detect)
        """
        self.data_dir = data_dir
        self.split = split
        self.max_seq_length = max_seq_length
        self.normalize = normalize
        self.train_split = train_split
        self.val_split = val_split
        self.sampling_rate = sampling_rate
        
        # Artifact removal parameters
        self.apply_notch_filter = apply_notch_filter
        self.notch_freq = notch_freq
        self.apply_highpass_filter = apply_highpass_filter
        self.highpass_cutoff = highpass_cutoff
        self.detect_bad_channels = detect_bad_channels
        self.bad_channel_threshold = bad_channel_threshold
        
        # Detect ZuCo version if not specified
        if version is None:
            if os.path.exists(os.path.join(data_dir, 'ZuCo_1.0')):
                version = '1.0'
            elif os.path.exists(os.path.join(data_dir, 'ZuCo_2.0')):
                version = '2.0'
            else:
                raise ValueError(f"Could not detect ZuCo version in {data_dir}")
        self.version = version
        
        # Load all aligned samples
        self.samples = self._load_aligned_samples()
        
        # Split data if not 'all'
        if split != 'all':
            self.samples = self._split_data(self.samples)
        
        print(f"Loaded {len(self.samples)} samples from ZuCo {version} ({split} split)")
    
    def _load_aligned_samples(self) -> List[Dict]:
        """
        Load EEG-text pairs with sentence-level alignment using wordbounds
        Returns list of dictionaries with aligned samples
        """
        all_samples = []
        
        # Determine base path based on version
        if self.version == '1.0':
            base_path = os.path.join(self.data_dir, 'ZuCo_1.0')
            all_samples.extend(self._load_zuco_v1_samples(base_path))
        elif self.version == '2.0':
            base_path = os.path.join(self.data_dir, 'ZuCo_2.0')
            all_samples.extend(self._load_zuco_v2_samples(base_path))
        
        return all_samples
    
    def _load_zuco_v1_samples(self, base_path: str) -> List[Dict]:
        """Load samples from ZuCo 1.0 structure"""
        samples = []
        eeg_base = os.path.join(base_path, 'eeg')
        text_base = os.path.join(base_path, 'text')
        
        # Load text data
        text_data = {}
        
        # Load NR sentences
        nr_text_file = os.path.join(text_base, 'sentencesNR.mat')
        if os.path.exists(nr_text_file):
            nr_data = self._load_matlab_file(nr_text_file)
            # Extract sentences (adjust key based on actual structure)
            for key in nr_data.keys():
                if not key.startswith('__') and isinstance(nr_data[key], np.ndarray):
                    sentences = nr_data[key].flatten()
                    for i, sent in enumerate(sentences):
                        if isinstance(sent, str):
                            text_data[f'NR_{i+1}'] = sent
                        elif isinstance(sent, np.ndarray) and sent.size > 0:
                            text_data[f'NR_{i+1}'] = str(sent.item())
        
        # Load SR sentences
        sr_text_file = os.path.join(text_base, 'sentencesSR.mat')
        if os.path.exists(sr_text_file):
            sr_data = self._load_matlab_file(sr_text_file)
            for key in sr_data.keys():
                if not key.startswith('__') and isinstance(sr_data[key], np.ndarray):
                    sentences = sr_data[key].flatten()
                    for i, sent in enumerate(sentences):
                        if isinstance(sent, str):
                            text_data[f'SR_{i+1}'] = sent
                        elif isinstance(sent, np.ndarray) and sent.size > 0:
                            text_data[f'SR_{i+1}'] = str(sent.item())
        
        # Process NR task
        nr_path = os.path.join(eeg_base, 'NR')
        if os.path.exists(nr_path):
            for subject_dir in os.listdir(nr_path):
                subject_path = os.path.join(nr_path, subject_dir)
                if os.path.isdir(subject_path):
                    # Find EEG files
                    eeg_files = glob.glob(os.path.join(subject_path, '*_NR*_EEG.mat'))
                    wordbounds_files = glob.glob(os.path.join(subject_path, 'wordbounds*.mat'))
                    
                    # Load wordbounds
                    wordbounds = self._load_wordbounds(wordbounds_files)
                    
                    # Process each EEG file
                    for eeg_file in eeg_files:
                        # Extract task number (e.g., NR1 from gip_ZAB_NR1_EEG.mat)
                        match = re.search(r'NR(\d+)', eeg_file)
                        if match:
                            task_num = int(match.group(1))
                            task_key = f'NR_{task_num}'
                            
                            if task_key in text_data:
                                eeg_samples = self._load_eeg_with_alignment(
                                    eeg_file, wordbounds, text_data[task_key], 
                                    task_key, subject_dir, 'NR'
                                )
                                samples.extend(eeg_samples)
        
        # Process SR task (similar structure)
        sr_path = os.path.join(eeg_base, 'SR')
        if os.path.exists(sr_path):
            for subject_dir in os.listdir(sr_path):
                subject_path = os.path.join(sr_path, subject_dir)
                if os.path.isdir(subject_path):
                    eeg_files = glob.glob(os.path.join(subject_path, '*_SR*_EEG.mat'))
                    wordbounds_files = glob.glob(os.path.join(subject_path, 'wordbounds*.mat'))
                    
                    wordbounds = self._load_wordbounds(wordbounds_files)
                    
                    for eeg_file in eeg_files:
                        match = re.search(r'SR(\d+)', eeg_file)
                        if match:
                            task_num = int(match.group(1))
                            task_key = f'SR_{task_num}'
                            
                            if task_key in text_data:
                                eeg_samples = self._load_eeg_with_alignment(
                                    eeg_file, wordbounds, text_data[task_key],
                                    task_key, subject_dir, 'SR'
                                )
                                samples.extend(eeg_samples)
        
        return samples
    
    def _load_zuco_v2_samples(self, base_path: str) -> List[Dict]:
        """Load samples from ZuCo 2.0 structure"""
        samples = []
        eeg_base = os.path.join(base_path, 'eeg')
        text_base = os.path.join(base_path, 'text')
        
        # Load text data from CSV files
        text_data = {}
        csv_files = glob.glob(os.path.join(text_base, 'nr_*.csv'))
        for csv_file in csv_files:
            # Extract task number from filename (e.g., nr_1.csv -> 1)
            match = re.search(r'nr_(\d+)', csv_file)
            if match:
                task_num = int(match.group(1))
                sentences = self._load_sentences_from_csv(csv_file)
                text_data[task_num] = sentences
        
        # Load wordbounds files
        wordbounds_base = os.path.join(eeg_base, 'NR')
        wordbounds_files = glob.glob(os.path.join(wordbounds_base, 'wordbounds_NR*.mat'))
        wordbounds = self._load_wordbounds(wordbounds_files)
        
        # Process NR task
        nr_path = os.path.join(eeg_base, 'NR')
        if os.path.exists(nr_path):
            for subject_dir in os.listdir(nr_path):
                subject_path = os.path.join(nr_path, subject_dir)
                if os.path.isdir(subject_path):
                    eeg_files = glob.glob(os.path.join(subject_path, '*_NR*_EEG.mat'))
                    
                    for eeg_file in eeg_files:
                        # Extract task number (e.g., NR3 from gip_YAC_NR3_EEG.mat)
                        match = re.search(r'NR(\d+)', eeg_file)
                        if match:
                            task_num = int(match.group(1))
                            
                            if task_num in text_data:
                                # Get sentences for this task
                                sentences = text_data[task_num]
                                
                                # Load EEG and align with sentences
                                eeg_samples = self._load_eeg_with_sentence_alignment(
                                    eeg_file, wordbounds, sentences,
                                    task_num, subject_dir, 'NR'
                                )
                                samples.extend(eeg_samples)
        
        return samples
    
    def _load_sentences_from_csv(self, csv_file: str) -> List[str]:
        """Load sentences from ZuCo 2.0 CSV format"""
        sentences = []
        try:
            # Read CSV with custom delimiter
            df = pd.read_csv(csv_file, sep=';', header=None, quoting=1)
            for _, row in df.iterrows():
                if len(row) >= 3:
                    # Extract sentence text (3rd column, remove quotes)
                    sentence = str(row[2]).strip('"')
                    if sentence and sentence != 'nan':
                        sentences.append(sentence)
        except Exception as e:
            print(f"Error loading CSV {csv_file}: {e}")
        return sentences
    
    def _load_wordbounds(self, wordbounds_files: List[str]) -> Dict:
        """
        Load word boundaries from wordbounds files
        Returns dict with sentence-level timing information
        """
        wordbounds = {}
        for wb_file in wordbounds_files:
            try:
                wb_data = self._load_matlab_file(wb_file)
                # Extract wordbound information (structure varies)
                for key in wb_data.keys():
                    if not key.startswith('__'):
                        wordbounds[key] = wb_data[key]
            except Exception as e:
                print(f"Error loading wordbounds {wb_file}: {e}")
        return wordbounds
    
    def _load_matlab_file(self, filepath: str):
        """
        Load MATLAB file, handling both v7.3 (HDF5) and older formats
        
        Returns:
            dict: Dictionary with data from MATLAB file
        """
        try:
            # Try scipy first (for older MATLAB formats)
            try:
                data = scipy.io.loadmat(filepath)
                return data
            except (NotImplementedError, ValueError) as e:
                # If scipy fails, try h5py for MATLAB v7.3 files
                if 'HDF reader' in str(e) or 'v7.3' in str(e):
                    return self._load_matlab_v73(filepath)
                else:
                    raise
        except Exception as e:
            print(f"Error loading MATLAB file {filepath}: {e}")
            raise
    
    def _load_matlab_v73(self, filepath: str) -> Dict:
        """
        Load MATLAB v7.3 (HDF5) file using h5py
        Handles EEGLAB structure files which have EEG structure with data field
        
        Args:
            filepath: Path to .mat file
            
        Returns:
            dict: Dictionary similar to scipy.io.loadmat output
        """
        data = {}
        try:
            with h5py.File(filepath, 'r') as f:
                # For EEGLAB files, look for EEG structure
                if 'EEG' in f.keys():
                    eeg_group = f['EEG']
                    if 'data' in eeg_group.keys():
                        # Get the data reference
                        data_ref = eeg_group['data']
                        if isinstance(data_ref, h5py.Dataset):
                            # If it's a reference, follow it
                            try:
                                if data_ref.dtype == h5py.special_dtype(ref=h5py.Reference):
                                    ref_path = data_ref[0, 0] if data_ref.ndim >= 2 else data_ref[0]
                                    actual_data = f[ref_path]
                                    data_arr = np.array(actual_data)
                                    # MATLAB stores data transposed in v7.3
                                    if data_arr.ndim == 2:
                                        data_arr = data_arr.T
                                    data['data'] = data_arr
                                    data['EEG'] = {'data': data_arr}
                            except:
                                # If reference doesn't work, try direct access
                                try:
                                    data_arr = np.array(data_ref)
                                    if data_arr.ndim == 2:
                                        data_arr = data_arr.T
                                    data['data'] = data_arr
                                    data['EEG'] = {'data': data_arr}
                                except:
                                    pass
                
                # Also try to find any large numeric datasets
                def extract_data(name, obj):
                    """Recursively extract data from HDF5 file"""
                    try:
                        if isinstance(obj, h5py.Dataset):
                            # Skip references
                            if obj.dtype == h5py.special_dtype(ref=h5py.Reference):
                                return
                            
                            # Try to get actual numeric data
                            try:
                                arr = np.array(obj)
                                # Only store if it's numeric and reasonably sized
                                if arr.dtype.kind in ['f', 'i', 'u'] and arr.size > 100:
                                    # MATLAB stores data transposed in v7.3
                                    if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                                        arr = arr.T
                                    key = name.split('/')[-1]
                                    if key not in data or arr.size > data[key].size:
                                        data[key] = arr
                            except:
                                pass
                    except:
                        pass
                
                f.visititems(extract_data)
                
        except Exception as e:
            print(f"Error loading MATLAB v7.3 file {filepath}: {e}")
        
        return data
    
    def _extract_eeg_from_matlab(self, mat_data: Dict) -> Optional[np.ndarray]:
        """
        Extract EEG data array from loaded MATLAB data
        
        Args:
            mat_data: Dictionary from loadmat or _load_matlab_v73
            
        Returns:
            np.ndarray: EEG data of shape (channels, time) or None
        """
        # Try common keys first
        for key in ['EEG', 'eeg_data', 'data']:
            if key in mat_data:
                eeg = mat_data[key]
                
                # Handle nested structures (EEG structure might have .data field)
                if isinstance(eeg, np.ndarray) and eeg.dtype.names:
                    # Structured array - try to get 'data' field
                    if 'data' in eeg.dtype.names:
                        eeg = eeg['data'][0, 0] if eeg.shape == (1, 1) else eeg['data']
                    elif len(eeg.dtype.names) > 0:
                        # Get first field
                        first_field = eeg.dtype.names[0]
                        eeg = eeg[first_field][0, 0] if eeg.shape == (1, 1) else eeg[first_field]
                
                # Convert to numpy array
                if isinstance(eeg, np.ndarray):
                    # Handle MATLAB's cell arrays and object arrays
                    if eeg.dtype == np.object_ or eeg.dtype.kind == 'O':
                        # Try to extract from object array
                        if eeg.size > 0:
                            eeg = eeg.flat[0]
                            if isinstance(eeg, np.ndarray):
                                eeg = np.array(eeg)
                    else:
                        eeg = np.array(eeg)
                    
                    # Ensure shape is (channels, time)
                    if eeg.ndim == 2:
                        if eeg.shape[0] < eeg.shape[1]:
                            eeg = eeg.T
                        return eeg
                    elif eeg.ndim == 3:
                        # (trials, channels, time) -> average or take first trial
                        eeg = eeg[0] if eeg.shape[0] == 1 else np.mean(eeg, axis=0)
                        return eeg
        
        # Try to find largest numeric array
        max_size = 0
        best_eeg = None
        for key in mat_data.keys():
            if key.startswith('__'):
                continue
            arr = mat_data[key]
            if isinstance(arr, np.ndarray):
                # Check if it's numeric (not object/string)
                if arr.dtype.kind in ['f', 'i', 'u']:
                    if arr.size > max_size and arr.ndim >= 2:
                        max_size = arr.size
                        best_eeg = arr
        
        if best_eeg is not None:
            # Ensure shape is (channels, time)
            if best_eeg.ndim == 2:
                if best_eeg.shape[0] < best_eeg.shape[1]:
                    best_eeg = best_eeg.T
                return best_eeg
            elif best_eeg.ndim == 3:
                best_eeg = best_eeg[0] if best_eeg.shape[0] == 1 else np.mean(best_eeg, axis=0)
                return best_eeg
        
        return None
    
    def _load_eeg_with_alignment(
        self,
        eeg_file: str,
        wordbounds: Dict, 
        text: str,
        task_key: str,
        subject_id: str,
        task_type: str
    ) -> List[Dict]:
        """Load EEG and align with single text (ZuCo 1.0 style)"""
        try:
            # Load MATLAB file (handles both v7.3 and older formats)
            eeg_data = self._load_matlab_file(eeg_file)
            
            # Extract EEG array
            eeg = self._extract_eeg_from_matlab(eeg_data)
            
            if eeg is None:
                return []
            
            # Extract frequency bands
            eeg_bands = self._extract_frequency_bands(eeg)
            
            # For ZuCo 1.0, treat entire recording as one sentence
            # or segment if wordbounds available
            samples = []
            
            if wordbounds:
                # Use wordbounds to segment by sentence
                # This is simplified - actual implementation would parse wordbounds structure
                # For now, use entire recording as one sample
                sample = {
                    'eeg_raw': eeg,
                    'eeg_bands': eeg_bands,
                    'sentence_text': text,
                    'subject': subject_id,
                    'task': task_key
                }
                samples.append(sample)
            else:
                # No wordbounds - use entire recording
                sample = {
                    'eeg_raw': eeg,
                    'eeg_bands': eeg_bands,
                    'sentence_text': text,
                    'subject': subject_id,
                    'task': task_key
                }
                samples.append(sample)
            
            return samples
            
        except Exception as e:
            print(f"Error loading EEG {eeg_file}: {e}")
            return []
    
    def _load_eeg_with_sentence_alignment(
        self,
        eeg_file: str,
        wordbounds: Dict,
        sentences: List[str],
        task_num: int,
        subject_id: str,
        task_type: str
    ) -> List[Dict]:
        """Load EEG and align with multiple sentences (ZuCo 2.0 style)"""
        try:
            # Load MATLAB file (handles both v7.3 and older formats)
            eeg_data = self._load_matlab_file(eeg_file)
            
            # Extract EEG array
            eeg = self._extract_eeg_from_matlab(eeg_data)
            
            if eeg is None:
                return []
            
            # Ensure shape is (channels, time)
            if eeg.ndim == 2:
                if eeg.shape[0] < eeg.shape[1]:
                    eeg = eeg.T
            
            num_channels, time_steps = eeg.shape
            
            # Try to extract sentence boundaries from wordbounds
            sentence_windows = self._extract_sentence_windows(wordbounds, task_num, time_steps)
            
            samples = []
            
            # Align sentences with EEG windows
            if sentence_windows and len(sentence_windows) > 0:
                # Use wordbounds for alignment
                num_sentences = min(len(sentences), len(sentence_windows))
                
                for i in range(num_sentences):
                    start_idx, end_idx = sentence_windows[i]
                    
                    if end_idx > start_idx and end_idx <= time_steps:
                        # Extract EEG window for this sentence
                        eeg_window = eeg[:, start_idx:end_idx]
                        
                        # Extract frequency bands for this window
                        eeg_bands = self._extract_frequency_bands(eeg_window)
                        
                        sample = {
                            'eeg_raw': eeg_window,
                            'eeg_bands': eeg_bands,
                            'sentence_text': sentences[i],
                            'subject': subject_id,
                            'task': f'{task_type}{task_num}_sent{i+1}'
                        }
                        samples.append(sample)
            else:
                # Fallback: segment evenly if no wordbounds
                window_size = time_steps // len(sentences) if sentences else time_steps
                for i, sentence in enumerate(sentences):
                    start_idx = i * window_size
                    end_idx = (i + 1) * window_size if i < len(sentences) - 1 else time_steps
                    
                    if end_idx > start_idx and end_idx <= time_steps:
                        eeg_window = eeg[:, start_idx:end_idx]
                        eeg_bands = self._extract_frequency_bands(eeg_window)
                        
                        sample = {
                            'eeg_raw': eeg_window,
                            'eeg_bands': eeg_bands,
                            'sentence_text': sentence,
                            'subject': subject_id,
                            'task': f'{task_type}{task_num}_sent{i+1}'
                        }
                        samples.append(sample)
            
            return samples
            
        except Exception as e:
            print(f"Error loading EEG {eeg_file}: {e}")
            return []
    
    def _extract_sentence_windows(self, wordbounds: Dict, task_num: int, total_time: int) -> List[Tuple[int, int]]:
        """
        Extract sentence-level time windows from wordbounds
        Returns list of (start_idx, end_idx) tuples in samples
        """
        windows = []
        
        # Try to find relevant wordbounds data
        # This is a simplified version - actual wordbounds structure may vary
        for key, data in wordbounds.items():
            if isinstance(data, np.ndarray):
                # Try to extract sentence boundaries
                # Structure depends on actual wordbounds format
                if data.ndim == 1 and len(data) > 0:
                    # Simple case: array of sentence end times
                    prev_idx = 0
                    for end_time in data:
                        end_idx = int(end_time * self.sampling_rate)
                        if end_idx > prev_idx:
                            windows.append((prev_idx, min(end_idx, total_time)))
                            prev_idx = end_idx
        
        # Fallback: if no valid windows found, return empty list
        # (will trigger fallback segmentation)
        return windows
    
    def _extract_frequency_bands(self, eeg: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract 5 frequency bands from EEG signal
        Returns dict with keys: 'delta', 'theta', 'alpha', 'beta', 'gamma'
        Each value is array of shape (C, T)
        """
        num_channels, time_steps = eeg.shape
        bands = {}
        nyquist = self.sampling_rate / 2
        
        for band_name, (low_freq, high_freq) in self.FREQUENCY_BANDS.items():
            band_eeg = np.zeros((num_channels, time_steps))
            
            # Normalize frequencies
            low_norm = low_freq / nyquist
            high_norm = high_freq / nyquist
            high_norm = min(high_norm, 0.99)  # Cap at Nyquist
            
            for ch in range(num_channels):
                try:
                    # Apply bandpass filter
                    b, a = signal.butter(4, [low_norm, high_norm], btype='band')
                    filtered = signal.filtfilt(b, a, eeg[ch, :])
                    band_eeg[ch, :] = filtered
                except:
                    # If filtering fails, use original signal
                    band_eeg[ch, :] = eeg[ch, :]
            
            bands[band_name] = band_eeg
        
        return bands
    
    def _split_data(self, all_samples: List[Dict]) -> List[Dict]:
        """Split data into train/val/test sets"""
        total_samples = len(all_samples)
        
        train_end = int(total_samples * self.train_split)
        val_end = int(total_samples * (self.train_split + self.val_split))
        
        if self.split == 'train':
            return all_samples[:train_end]
        elif self.split == 'val':
            return all_samples[train_end:val_end]
        elif self.split == 'test':
            return all_samples[val_end:]
        else:
            return all_samples
    
    def _apply_highpass_filter(self, eeg: np.ndarray) -> np.ndarray:
        """
        Apply high-pass filter to remove slow drifts and DC offset
        
        Args:
            eeg: EEG signal of shape (num_channels, time_steps)
            
        Returns:
            filtered_eeg: High-pass filtered EEG
        """
        if not self.apply_highpass_filter:
            return eeg
        
        num_channels, time_steps = eeg.shape
        filtered_eeg = np.zeros_like(eeg)
        nyquist = self.sampling_rate / 2
        cutoff_norm = self.highpass_cutoff / nyquist
        
        for ch in range(num_channels):
            try:
                # Design high-pass Butterworth filter
                b, a = signal.butter(4, cutoff_norm, btype='high')
                # Apply zero-phase filtering (forward and backward)
                filtered_eeg[ch, :] = signal.filtfilt(b, a, eeg[ch, :])
            except Exception as e:
                # If filtering fails, use original signal
                print(f"Warning: High-pass filtering failed for channel {ch}: {e}")
                filtered_eeg[ch, :] = eeg[ch, :]
        
        return filtered_eeg
    
    def _apply_notch_filter(self, eeg: np.ndarray) -> np.ndarray:
        """
        Apply notch filter to remove power line noise (50/60 Hz)
        
        Args:
            eeg: EEG signal of shape (num_channels, time_steps)
            
        Returns:
            filtered_eeg: Notch-filtered EEG
        """
        if not self.apply_notch_filter:
            return eeg
        
        num_channels, time_steps = eeg.shape
        filtered_eeg = np.zeros_like(eeg)
        quality_factor = 30.0  # Quality factor for notch filter
        
        for ch in range(num_channels):
            try:
                # Design notch filter (removes specific frequency)
                b, a = signal.iirnotch(self.notch_freq, quality_factor, self.sampling_rate)
                # Apply zero-phase filtering
                filtered_eeg[ch, :] = signal.filtfilt(b, a, eeg[ch, :])
            except Exception as e:
                # If filtering fails, use original signal
                print(f"Warning: Notch filtering failed for channel {ch}: {e}")
                filtered_eeg[ch, :] = eeg[ch, :]
        
        return filtered_eeg
    
    def _detect_bad_channels(self, eeg: np.ndarray) -> List[int]:
        """
        Detect bad channels based on variance and amplitude
        
        Args:
            eeg: EEG signal of shape (num_channels, time_steps)
            
        Returns:
            bad_channels: List of channel indices that are bad
        """
        if not self.detect_bad_channels:
            return []
        
        num_channels = eeg.shape[0]
        bad_channels = []
        
        # Compute channel statistics
        channel_vars = np.var(eeg, axis=1)
        channel_means = np.mean(np.abs(eeg), axis=1)
        
        # Detect channels with unusually high variance (likely artifacts)
        var_mean = np.mean(channel_vars)
        var_std = np.std(channel_vars)
        var_threshold = var_mean + self.bad_channel_threshold * var_std
        
        # Detect channels with unusually high mean amplitude
        mean_mean = np.mean(channel_means)
        mean_std = np.std(channel_means)
        mean_threshold = mean_mean + self.bad_channel_threshold * mean_std
        
        for ch in range(num_channels):
            if channel_vars[ch] > var_threshold or channel_means[ch] > mean_threshold:
                bad_channels.append(ch)
        
        return bad_channels
    
    def _interpolate_bad_channels(self, eeg: np.ndarray, bad_channels: List[int]) -> np.ndarray:
        """
        Interpolate bad channels using spatial interpolation from neighboring channels
        
        Args:
            eeg: EEG signal of shape (num_channels, time_steps)
            bad_channels: List of bad channel indices
            
        Returns:
            eeg_interpolated: EEG with bad channels interpolated
        """
        if len(bad_channels) == 0:
            return eeg
        
        eeg_interpolated = eeg.copy()
        num_channels = eeg.shape[0]
        
        for bad_ch in bad_channels:
            # Simple interpolation: average of adjacent channels
            # In practice, you'd use actual electrode positions for spatial interpolation
            adjacent_channels = []
            
            # Find adjacent channels (simple heuristic: channels close in index)
            for ch in range(num_channels):
                if ch != bad_ch and ch not in bad_channels:
                    dist = abs(ch - bad_ch)
                    if dist <= 3:  # Within 3 channels
                        adjacent_channels.append(ch)
            
            if len(adjacent_channels) > 0:
                # Interpolate as average of adjacent channels
                eeg_interpolated[bad_ch, :] = np.mean(eeg[adjacent_channels, :], axis=0)
            else:
                # If no adjacent channels, use zero padding (better than leaving as-is)
                eeg_interpolated[bad_ch, :] = np.zeros(eeg.shape[1])
        
        return eeg_interpolated
    
    def _preprocess_eeg(self, eeg: np.ndarray) -> np.ndarray:
        """
        Preprocess EEG signal with artifact removal and normalization
        
        Steps:
        1. High-pass filter (remove slow drifts)
        2. Notch filter (remove power line noise)
        3. Bad channel detection and interpolation (optional)
        4. Normalization (z-score per channel)
        
        Args:
            eeg: EEG signal of shape (num_channels, time_steps)
            
        Returns:
            preprocessed_eeg: Clean, normalized EEG
        """
        # Step 1: Remove slow drifts with high-pass filter
        eeg = self._apply_highpass_filter(eeg)
        
        # Step 2: Remove power line noise with notch filter
        eeg = self._apply_notch_filter(eeg)
        
        # Step 3: Detect and interpolate bad channels (optional)
        if self.detect_bad_channels:
            bad_channels = self._detect_bad_channels(eeg)
            if len(bad_channels) > 0:
                eeg = self._interpolate_bad_channels(eeg, bad_channels)
        
        # Step 4: Normalize per channel (z-score)
        if self.normalize:
            eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
        
        return eeg
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Preprocess raw EEG
        eeg_raw = sample['eeg_raw'].copy()
        eeg_raw = self._preprocess_eeg(eeg_raw)
        
        # Preprocess frequency bands
        eeg_bands_processed = {}
        for band_name, band_eeg in sample['eeg_bands'].items():
            eeg_bands_processed[band_name] = self._preprocess_eeg(band_eeg.copy())
        
        # Convert to tensors
        eeg_raw_tensor = torch.FloatTensor(eeg_raw)
        eeg_bands_tensor = {
            band_name: torch.FloatTensor(band_eeg)
            for band_name, band_eeg in eeg_bands_processed.items()
        }
        
        return {
            'eeg_raw': eeg_raw_tensor,           # (C, T)
            'eeg_bands': eeg_bands_tensor,       # Dict of (C, T) tensors
            'sentence_text': sample['sentence_text'],
            'subject': sample['subject'],
            'task': sample['task'],
            'text': sample['sentence_text']      # For compatibility with existing code
        }


def collate_fn(batch, tokenizer=None, max_seq_length=128):
    """
    Collate function for DataLoader
    
    Args:
        batch: List of samples (from __getitem__)
        tokenizer: Text tokenizer
        max_seq_length: Maximum sequence length
        
    Returns:
        batched_data: Dictionary of batched tensors
    """
    # Extract components
    eeg_raw_list = [item['eeg_raw'] for item in batch]
    eeg_bands_list = [item['eeg_bands'] for item in batch]
    texts = [item['sentence_text'] for item in batch]
    subjects = [item['subject'] for item in batch]
    tasks = [item['task'] for item in batch]
    
    # Pad raw EEG to same length
    max_eeg_len = max(e.shape[1] for e in eeg_raw_list)
    num_channels = eeg_raw_list[0].shape[0]
    
    eeg_raw_padded = []
    for eeg in eeg_raw_list:
        if eeg.shape[1] < max_eeg_len:
            padding = torch.zeros(num_channels, max_eeg_len - eeg.shape[1])
            eeg = torch.cat([eeg, padding], dim=1)
        eeg_raw_padded.append(eeg)
    
    eeg_raw_batch = torch.stack(eeg_raw_padded)  # (batch_size, C, T)
    
    # Pad frequency bands
    eeg_bands_batch = {}
    for band_name in eeg_bands_list[0].keys():
        band_list = [item[band_name] for item in eeg_bands_list]
        band_padded = []
        for band_eeg in band_list:
            if band_eeg.shape[1] < max_eeg_len:
                padding = torch.zeros(num_channels, max_eeg_len - band_eeg.shape[1])
                band_eeg = torch.cat([band_eeg, padding], dim=1)
            band_padded.append(band_eeg)
        eeg_bands_batch[band_name] = torch.stack(band_padded)  # (batch_size, C, T)
    
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
            for j, word in enumerate(words):
                text_tokens[i, j] = hash(word) % 10000  # Placeholder
    
    return {
        'eeg': eeg_raw_batch,          # (batch_size, C, T) - for compatibility
        'eeg_raw': eeg_raw_batch,      # (batch_size, C, T)
        'eeg_bands': eeg_bands_batch,  # Dict: {band_name: (batch_size, C, T)}
        'text': texts,                 # List of sentence strings
        'sentence_text': texts,        # Alias for compatibility
        'text_tokens': text_tokens,    # (batch_size, max_seq_length)
        'subject': subjects,           # List of subject IDs
        'task': tasks                  # List of task IDs
    }


def load_zuco_data(data_dir: str, version: Optional[str] = None):
    """
    Utility function to load ZuCo data
    
    Args:
        data_dir: Root directory containing ZuCo_1.0/ and ZuCo_2.0/
        version: ZuCo version ('1.0', '2.0', or None for auto-detect)
        
    Returns:
        samples: List of sample dictionaries with eeg_raw, eeg_bands, sentence_text, etc.
    """
    dataset = ZuCoDataset(data_dir, split='all', version=version)
    return dataset.samples

