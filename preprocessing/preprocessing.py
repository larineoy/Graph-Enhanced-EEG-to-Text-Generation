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
from tqdm import tqdm
import time
import gc
import json


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
        bad_channel_threshold: float = 3.0,  # Standard deviations for bad channel detection
        split_seed: int = 42,
        window_size_sec: float = 1.0,
        window_stride_sec: Optional[float] = None,
        max_windows: int = 16
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
        self.split_seed = split_seed
        self.window_size_sec = window_size_sec
        self.window_stride_sec = window_stride_sec if window_stride_sec is not None else window_size_sec
        self.max_windows = max_windows
        self.window_size = max(1, int(round(self.window_size_sec * self.sampling_rate))) if window_size_sec else 0
        self.window_stride = max(1, int(round(self.window_stride_sec * self.sampling_rate))) if self.window_size else 0
        
        # Detect ZuCo versions available
        # Support loading from both ZuCo 1.0 and ZuCo 2.0
        self.num_channels = None  # Will be determined from data
        self.versions_to_load = []
        
        if version is None:
            # Auto-detect: load both versions if available
            if os.path.exists(os.path.join(data_dir, 'ZuCo_1.0')):
                self.versions_to_load.append('1.0')
            if os.path.exists(os.path.join(data_dir, 'ZuCo_2.0')):
                self.versions_to_load.append('2.0')
            
            if len(self.versions_to_load) == 0:
                raise ValueError(f"Could not detect ZuCo version in {data_dir}. Expected ZuCo_1.0/ or ZuCo_2.0/")
            
            # For backward compatibility, set version to first available
            self.version = self.versions_to_load[0]
        else:
            # Specific version requested
            if version not in ['1.0', '2.0']:
                raise ValueError(f"Invalid version: {version}. Must be '1.0' or '2.0'")
            if not os.path.exists(os.path.join(data_dir, f'ZuCo_{version}')):
                raise ValueError(f"ZuCo {version} not found in {data_dir}")
            self.versions_to_load = [version]
            self.version = version

        keep = 0
        if self.window_size and self.max_windows:
            keep = self.window_size + (self.max_windows - 1) * self.window_stride
        self._cache_dir = os.path.join(
            self.data_dir, '.cache', f'zuco_win{self.max_windows}_keep{keep}'
        )
        self._cache_index_path = os.path.join(self._cache_dir, 'index.json')
        self._next_cache_id = 0
        
        # Load all aligned samples with progress tracking
        print(f"  Loading EEG files and aligning with text (this may take 2-10 minutes)...")
        start_time = time.time()
        self.samples = self._load_aligned_samples()
        load_time = time.time() - start_time
        self._canonicalize_loaded_samples()
        self._all_samples = self.samples

        # Split data if not 'all'
        if split != 'all':
            self.samples = self._split_data(self._all_samples)
        
        # Format version string for display
        if len(self.versions_to_load) > 1:
            version_str = f"{'+'.join(self.versions_to_load)}"
        else:
            version_str = self.version
        
        print(f"  ✓ Loaded {len(self.samples)} samples from ZuCo {version_str} ({split} split) in {load_time:.1f} seconds")
    
    def _load_aligned_samples(self) -> List[Dict]:
        """
        Load EEG-text pairs with sentence-level alignment using wordbounds
        Returns list of dictionaries with aligned samples
        Supports loading from both ZuCo 1.0 and ZuCo 2.0
        """
        cached = self._load_sample_cache()
        if cached is not None:
            return cached

        os.makedirs(self._cache_dir, exist_ok=True)
        all_samples = []
        
        # Load samples from all specified versions
        for version in self.versions_to_load:
            if version == '1.0':
                base_path = os.path.join(self.data_dir, 'ZuCo_1.0')
                samples_v1 = self._load_zuco_v1_samples(base_path)
                all_samples.extend(samples_v1)
                print(f"    ✓ Loaded {len(samples_v1)} samples from ZuCo 1.0")
            elif version == '2.0':
                base_path = os.path.join(self.data_dir, 'ZuCo_2.0')
                samples_v2 = self._load_zuco_v2_samples(base_path)
                all_samples.extend(samples_v2)
                print(f"    ✓ Loaded {len(samples_v2)} samples from ZuCo 2.0")
        
        print(f"    ✓ Total samples loaded: {len(all_samples)} (from {len(self.versions_to_load)} version(s))")
        self._write_sample_cache(all_samples)
        return all_samples

    def _cache_params(self) -> Dict:
        keep = 0
        if self.window_size and self.max_windows:
            keep = self.window_size + (self.max_windows - 1) * self.window_stride
        return {
            'versions': list(self.versions_to_load),
            'max_windows': self.max_windows,
            'window_size': self.window_size,
            'keep': keep,
        }

    def _load_sample_cache(self) -> Optional[List[Dict]]:
        if not os.path.exists(self._cache_index_path):
            return None
        try:
            with open(self._cache_index_path, 'r') as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get('params') != self._cache_params():
            return None
        samples = payload.get('samples') or []
        if not samples:
            return None
        missing = [s['eeg_path'] for s in samples if not os.path.exists(s.get('eeg_path', ''))]
        if missing:
            print(f"  ⚠ EEG cache incomplete ({len(missing)} missing files); rebuilding")
            return None
        print(f"  ✓ Loaded {len(samples)} clipped EEG sentences from {self._cache_dir}")
        return samples

    def _write_sample_cache(self, samples: List[Dict]):
        os.makedirs(self._cache_dir, exist_ok=True)
        slim = []
        for sample in samples:
            rec = {
                'eeg_path': sample.get('eeg_path'),
                'sentence_text': sample.get('sentence_text'),
                'subject': sample.get('subject'),
                'task': sample.get('task'),
                'num_channels': sample.get('num_channels'),
                'time_steps': sample.get('time_steps'),
            }
            info = sample.get('channel_info')
            if info and 'electrode_positions' in info:
                rec['channel_info'] = {
                    'electrode_positions': np.asarray(info['electrode_positions']).tolist()
                }
            slim.append(rec)
        with open(self._cache_index_path, 'w') as f:
            json.dump({'params': self._cache_params(), 'samples': slim}, f)
        print(f"  ✓ Wrote EEG cache to {self._cache_dir}")

    def _max_keep_samples(self) -> Optional[int]:
        if self.window_size and self.max_windows:
            return self.window_size + (self.max_windows - 1) * self.window_stride
        return None

    def _materialize_sentence_eeg(self, eeg: np.ndarray) -> Optional[np.ndarray]:
        """Copy a sentence slice so the parent recording can be freed."""
        eeg = self._ensure_channels_first(eeg)
        if eeg is None or eeg.ndim != 2:
            return None
        keep = self._max_keep_samples()
        if keep and eeg.shape[1] > keep:
            eeg = eeg[:, :keep]
        return np.ascontiguousarray(eeg, dtype=np.float32)

    def _commit_sample(
        self,
        eeg: np.ndarray,
        sentence_text: str,
        subject: str,
        task: str,
        channel_info: Optional[Dict] = None
    ) -> Optional[Dict]:
        eeg = self._materialize_sentence_eeg(eeg)
        if eeg is None:
            return None
        os.makedirs(self._cache_dir, exist_ok=True)
        path = os.path.join(self._cache_dir, f'{self._next_cache_id:06d}.npy')
        np.save(path, eeg)
        self._next_cache_id += 1
        sample = {
            'eeg_path': path,
            'sentence_text': sentence_text,
            'subject': subject,
            'task': task,
            'num_channels': int(eeg.shape[0]),
            'time_steps': int(eeg.shape[1]),
        }
        if channel_info:
            sample['channel_info'] = channel_info
        return sample

    def _load_sample_eeg(self, sample: Dict) -> np.ndarray:
        if sample.get('eeg_raw') is not None:
            return np.array(sample['eeg_raw'], copy=True)
        return np.load(sample['eeg_path'])

    @staticmethod
    def _ensure_channels_first(eeg: Optional[np.ndarray], max_channels: int = 256) -> Optional[np.ndarray]:
        """Force (channels, time). Never treat a long time axis as channels."""
        if eeg is None or not isinstance(eeg, np.ndarray):
            return eeg
        if eeg.ndim != 2:
            return eeg
        dim0, dim1 = eeg.shape
        dim0_looks_like_channels = 8 <= dim0 <= max_channels
        dim1_looks_like_channels = 8 <= dim1 <= max_channels
        if dim0_looks_like_channels and not dim1_looks_like_channels:
            return eeg
        if dim1_looks_like_channels and not dim0_looks_like_channels:
            return eeg.T
        if dim0_looks_like_channels and dim1_looks_like_channels:
            return eeg if dim0 <= dim1 else eeg.T
        return eeg

    def _canonicalize_loaded_samples(self):
        """Orient every array as (C, T) and keep one consistent channel count."""
        if len(self.samples) == 0:
            raise ValueError("No samples loaded from ZuCo dataset")

        kept = []
        for sample in self.samples:
            n_ch = sample.get('num_channels')
            if n_ch is None:
                if sample.get('eeg_raw') is not None:
                    eeg = self._ensure_channels_first(sample['eeg_raw'])
                    if eeg is None or eeg.ndim != 2:
                        continue
                    sample['eeg_raw'] = np.ascontiguousarray(eeg, dtype=np.float32)
                    sample['num_channels'] = int(eeg.shape[0])
                    n_ch = sample['num_channels']
                elif sample.get('eeg_path'):
                    n_ch = int(np.load(sample['eeg_path'], mmap_mode='r').shape[0])
                    sample['num_channels'] = n_ch
                else:
                    continue
            kept.append(sample)
        self.samples = kept

        plausible = {}
        for sample in self.samples:
            n_ch = sample['num_channels']
            if 8 <= n_ch <= 256:
                plausible[n_ch] = plausible.get(n_ch, 0) + 1
        if not plausible:
            raise ValueError(
                "Could not find any EEG arrays with a plausible channel count "
                "(expected 8–256 channels after orientation)."
            )

        self.num_channels = max(plausible, key=plausible.get)
        before = len(self.samples)
        self.samples = [
            sample for sample in self.samples
            if sample.get('num_channels') == self.num_channels
        ]
        dropped = before - len(self.samples)
        print(f"  ✓ Detected {self.num_channels} channels from ZuCo data "
              f"({len(self.samples)} usable samples)")
        if dropped:
            print(f"  ⚠ Dropped {dropped} samples that were not {self.num_channels}-channel")

        self.electrode_positions = None
        self.channel_names = None
        for sample in self.samples:
            channel_info = sample.get('channel_info') or {}
            positions = channel_info.get('electrode_positions')
            if positions is not None and len(positions) == self.num_channels:
                self.electrode_positions = np.asarray(positions, dtype=np.float32)
                print(f"  ✓ Extracted {len(self.electrode_positions)} electrode positions from ZuCo chanlocs (X, Y, Z)")
                break
            names = channel_info.get('channel_names')
            if names is not None and len(names) == self.num_channels:
                self.channel_names = names
        if self.electrode_positions is None and self.channel_names:
            print(f"  ✓ Extracted {len(self.channel_names)} channel names from ZuCo")

    def make_split(self, split: str) -> 'ZuCoDataset':
        """Reuse already-loaded EEG arrays for another sentence-identity split."""
        other = object.__new__(ZuCoDataset)
        other.__dict__.update(self.__dict__)
        other.split = split
        other.samples = (
            list(other._all_samples) if split == 'all'
            else other._split_data(other._all_samples)
        )
        return other

    @classmethod
    def load_splits(cls, data_dir: str, splits=('train', 'val'), **kwargs) -> Dict[str, 'ZuCoDataset']:
        """Load ZuCo once, then view train/val/test without rereading MATLAB files."""
        base = cls(data_dir, split='all', **kwargs)
        return {split: base.make_split(split) for split in splits}
    
    def _load_zuco_v1_samples(self, base_path: str) -> List[Dict]:
        """Load samples from ZuCo 1.0 structure"""
        samples = []
        eeg_base = os.path.join(base_path, 'eeg')
        text_base = os.path.join(base_path, 'text')
        
        print(f"    Loading text sentences...")
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
        
        print(f"    ✓ Loaded {len(text_data)} text sentences")
        
        # Collect all subjects and files for progress tracking
        all_tasks = []
        nr_path = os.path.join(eeg_base, 'NR')
        if os.path.exists(nr_path):
            for subject_dir in os.listdir(nr_path):
                subject_path = os.path.join(nr_path, subject_dir)
                if os.path.isdir(subject_path):
                    eeg_files = glob.glob(os.path.join(subject_path, '*_NR*_EEG.mat'))
                    wordbounds_files = glob.glob(os.path.join(subject_path, 'wordbounds*.mat'))
                    for eeg_file in eeg_files:
                        all_tasks.append(('NR', subject_dir, eeg_file, wordbounds_files))
        
        sr_path = os.path.join(eeg_base, 'SR')
        if os.path.exists(sr_path):
            for subject_dir in os.listdir(sr_path):
                subject_path = os.path.join(sr_path, subject_dir)
                if os.path.isdir(subject_path):
                    eeg_files = glob.glob(os.path.join(subject_path, '*_SR*_EEG.mat'))
                    wordbounds_files = glob.glob(os.path.join(subject_path, 'wordbounds*.mat'))
                    for eeg_file in eeg_files:
                        all_tasks.append(('SR', subject_dir, eeg_file, wordbounds_files))
        
        print(f"    Processing {len(all_tasks)} EEG files across {len(set(t[1] for t in all_tasks))} subjects...")
        
        # Load wordbounds per subject (cache to avoid reloading)
        wordbounds_cache = {}
        
        # Process all tasks with progress bar
        for task_type, subject_dir, eeg_file, wordbounds_files in tqdm(all_tasks, desc="      Loading EEG files", unit="file", leave=False):
            # Load wordbounds (cache by subject)
            if subject_dir not in wordbounds_cache:
                wordbounds_cache[subject_dir] = self._load_wordbounds(wordbounds_files)
            wordbounds = wordbounds_cache[subject_dir]
            
            # Process EEG file
            if task_type == 'NR':
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
                        gc.collect()
            
            elif task_type == 'SR':
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
                        gc.collect()
        
        print(f"    ✓ Processed {len(all_tasks)} files, extracted {len(samples)} aligned samples")
        return samples
    
    def _load_zuco_v2_samples(self, base_path: str) -> List[Dict]:
        """Load samples from ZuCo 2.0 structure"""
        samples = []
        eeg_base = os.path.join(base_path, 'eeg')
        text_base = os.path.join(base_path, 'text')
        
        print(f"    Loading text sentences from CSV files...")
        # Load text data from CSV files
        text_data = {}
        csv_files = glob.glob(os.path.join(text_base, 'nr_*.csv'))
        for csv_file in tqdm(csv_files, desc="      Loading CSV files", unit="file", leave=False):
            # Extract task number from filename (e.g., nr_1.csv -> 1)
            match = re.search(r'nr_(\d+)', csv_file)
            if match:
                task_num = int(match.group(1))
                sentences = self._load_sentences_from_csv(csv_file)
                text_data[task_num] = sentences
        
        print(f"    ✓ Loaded text data for {len(text_data)} tasks")
        
        # Load wordbounds files
        print(f"    Loading wordbounds...")
        wordbounds_base = os.path.join(eeg_base, 'NR')
        wordbounds_files = glob.glob(os.path.join(wordbounds_base, 'wordbounds_NR*.mat'))
        wordbounds = self._load_wordbounds(wordbounds_files)
        print(f"    ✓ Loaded wordbounds")
        
        # Collect all EEG files for progress tracking
        all_tasks = []
        nr_path = os.path.join(eeg_base, 'NR')
        if os.path.exists(nr_path):
            for subject_dir in os.listdir(nr_path):
                subject_path = os.path.join(nr_path, subject_dir)
                if os.path.isdir(subject_path):
                    eeg_files = glob.glob(os.path.join(subject_path, '*_NR*_EEG.mat'))
                    for eeg_file in eeg_files:
                        all_tasks.append((subject_dir, eeg_file))
        
        print(f"    Processing {len(all_tasks)} EEG files across {len(set(t[0] for t in all_tasks))} subjects...")
        
        # Process NR task with progress bar
        for subject_dir, eeg_file in tqdm(all_tasks, desc="      Loading EEG files", unit="file", leave=False):
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
                    del eeg_samples
                    gc.collect()
        
        print(f"    ✓ Processed {len(all_tasks)} files, extracted {len(samples)} aligned samples")
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
                # Convert float64 arrays to float32 to save memory
                for key, value in data.items():
                    if isinstance(value, np.ndarray) and value.dtype == np.float64:
                        # Only convert large arrays to save memory
                        if value.size > 10000:  # Threshold: ~80KB for 10000 floats
                            data[key] = value.astype(np.float32)
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
        Extracts channel metadata if available in ZuCo files
        
        Args:
            filepath: Path to .mat file
            
        Returns:
            dict: Dictionary with 'data' (EEG array) and optionally 'channel_info' (metadata)
        """
        data = {}
        channel_info = {}
        
        try:
            with h5py.File(filepath, 'r') as f:
                # For EEGLAB files, look for EEG structure
                if 'EEG' in f.keys():
                    eeg_group = f['EEG']
                    
                    # Extract channel metadata from ZuCo files (chanlocs contains X, Y, Z positions!)
                    if 'chanlocs' in eeg_group.keys():
                        try:
                            chanlocs = eeg_group['chanlocs']
                            if isinstance(chanlocs, h5py.Group):
                                # Extract X, Y, Z coordinates and labels from chanlocs
                                # ZuCo stores these as arrays in the chanlocs group
                                electrode_positions = []
                                channel_names = []
                                
                                # Check if chanlocs has direct X, Y, Z arrays (EEGLAB format)
                                if 'X' in chanlocs.keys() and 'Y' in chanlocs.keys() and 'Z' in chanlocs.keys():
                                    try:
                                        X_ref = chanlocs['X']
                                        Y_ref = chanlocs['Y']
                                        Z_ref = chanlocs['Z']
                                        
                                        # Follow references if needed
                                        # ZuCo stores X, Y, Z as HDF5 datasets or references
                                        def get_array(ref):
                                            """Extract array from HDF5 dataset or reference"""
                                            if isinstance(ref, h5py.Dataset):
                                                if ref.dtype == h5py.special_dtype(ref=h5py.Reference):
                                                    # Reference type - follow the reference
                                                    if ref.ndim >= 2:
                                                        ref_path = f[ref[0, 0]]
                                                    else:
                                                        ref_path = f[ref[0]]
                                                    return np.array(ref_path)
                                                else:
                                                    # Direct dataset - read values
                                                    return np.array(ref)
                                            else:
                                                # Already an array or group
                                                return np.array(ref)
                                        
                                        X_arr = get_array(X_ref)
                                        Y_arr = get_array(Y_ref)
                                        Z_arr = get_array(Z_ref)
                                        
                                        # Ensure they're 1D arrays
                                        if X_arr.ndim > 1:
                                            X_arr = X_arr.flatten()
                                        if Y_arr.ndim > 1:
                                            Y_arr = Y_arr.flatten()
                                        if Z_arr.ndim > 1:
                                            Z_arr = Z_arr.flatten()
                                        
                                        # Stack into (num_channels, 3) array
                                        num_chans = len(X_arr)
                                        if len(Y_arr) == num_chans and len(Z_arr) == num_chans:
                                            positions = np.stack([X_arr, Y_arr, Z_arr], axis=1)
                                            channel_info['electrode_positions'] = positions.astype(np.float32)
                                        
                                        # Extract channel labels if available
                                        if 'labels' in chanlocs.keys():
                                            labels_ref = chanlocs['labels']
                                            labels_arr = get_array(labels_ref)
                                            if labels_arr.ndim > 1:
                                                labels_arr = labels_arr.flatten()
                                            # Convert to strings (may be stored as character codes)
                                            if labels_arr.dtype.kind in ['U', 'S', 'O']:
                                                channel_names = [str(l) for l in labels_arr[:num_chans]]
                                            elif labels_arr.dtype == np.object_:
                                                # Try to decode each label
                                                channel_names = []
                                                for i in range(min(num_chans, len(labels_arr))):
                                                    try:
                                                        label_ref = labels_arr[i]
                                                        if isinstance(label_ref, h5py.Reference):
                                                            label_data = f[label_ref]
                                                            if isinstance(label_data, h5py.Dataset):
                                                                label_str = ''.join([chr(c) for c in label_data[:] if c > 0])
                                                                channel_names.append(label_str)
                                                    except:
                                                        channel_names.append(f'CH{i+1}')
                                            if channel_names:
                                                channel_info['channel_names'] = channel_names
                                    except Exception as e:
                                        pass  # Extraction failed, continue without positions
                        except Exception as e:
                            pass  # Channel metadata extraction failed, continue without it
                    
                    # Extract number of channels if available
                    if 'nbchan' in eeg_group.keys():
                        try:
                            nbchan_ref = eeg_group['nbchan']
                            if isinstance(nbchan_ref, h5py.Dataset):
                                nbchan = int(nbchan_ref[0, 0] if nbchan_ref.ndim >= 2 else nbchan_ref[0])
                                channel_info['num_channels'] = nbchan
                        except Exception as e:
                            pass
                    
                    if 'data' in eeg_group.keys():
                        # Get the data reference
                        data_ref = eeg_group['data']
                        if isinstance(data_ref, h5py.Dataset):
                            # If it's a reference, follow it
                            try:
                                if data_ref.dtype == h5py.special_dtype(ref=h5py.Reference):
                                    ref_path = data_ref[0, 0] if data_ref.ndim >= 2 else data_ref[0]
                                    actual_data = f[ref_path]
                                    # FORCE float32 for all floating point arrays to save memory
                                    if actual_data.dtype.kind == 'f':
                                        data_arr = np.array(actual_data, dtype=np.float32)
                                    else:
                                        data_arr = np.array(actual_data)
                                    # MATLAB stores data transposed in v7.3
                                    if data_arr.ndim == 2:
                                        data_arr = data_arr.T
                                    # Final safety check: convert to float32 if still float64
                                    if data_arr.dtype == np.float64:
                                        data_arr = data_arr.astype(np.float32)
                                    data['data'] = data_arr
                                    data['EEG'] = {'data': data_arr}
                                    if channel_info:
                                        data['channel_info'] = channel_info
                            except Exception as e:
                                # If reference doesn't work, try direct access
                                try:
                                    # FORCE float32 for all floating point arrays to save memory
                                    if data_ref.dtype.kind == 'f':
                                        data_arr = np.array(data_ref, dtype=np.float32)
                                    else:
                                        data_arr = np.array(data_ref)
                                    if data_arr.ndim == 2:
                                        data_arr = data_arr.T
                                    # Final safety check: convert to float32 if still float64
                                    if data_arr.dtype == np.float64:
                                        data_arr = data_arr.astype(np.float32)
                                    data['data'] = data_arr
                                    data['EEG'] = {'data': data_arr}
                                    if channel_info:
                                        data['channel_info'] = channel_info
                                except Exception as e2:
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
                                # Check shape first without loading full array
                                if obj.shape and obj.size > 100:
                                    # FORCE float32 conversion for ALL floating point arrays to save memory
                                    if obj.dtype.kind == 'f':
                                        arr = np.array(obj, dtype=np.float32)
                                    elif obj.dtype.kind in ['i', 'u']:
                                        # For integers, keep as-is but convert large int64 to int32
                                        if obj.dtype.itemsize >= 8:  # int64 or uint64
                                            arr = np.array(obj, dtype=np.int32 if obj.dtype.kind == 'i' else np.uint32)
                                        else:
                                            arr = np.array(obj)
                                    else:
                                        arr = np.array(obj)
                                    
                                    # Only store if it's numeric and reasonably sized
                                    if arr.dtype.kind in ['f', 'i', 'u']:
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
                
                # Check if it's a structured EEG object with metadata
                # ZuCo dataset may contain EEG structure with channel info
                if isinstance(eeg, dict) and 'data' in eeg:
                    # EEG structure from EEGLAB format
                    eeg = eeg['data']
                
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
                    
                    # Convert to float32 to save memory if still float64
                    if eeg.dtype == np.float64:
                        eeg = eeg.astype(np.float32)
                    
                    if eeg.ndim == 2:
                        return self._ensure_channels_first(eeg)
                    elif eeg.ndim == 3:
                        # (trials, channels, time) -> average or take first trial
                        eeg = eeg[0] if eeg.shape[0] == 1 else np.mean(eeg, axis=0)
                        # Convert to float32 if needed
                        if eeg.dtype == np.float64:
                            eeg = eeg.astype(np.float32)
                        return self._ensure_channels_first(eeg)
        
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
            # Convert to float32 to save memory if still float64
            if best_eeg.dtype == np.float64:
                best_eeg = best_eeg.astype(np.float32)
            
            if best_eeg.ndim == 2:
                return self._ensure_channels_first(best_eeg)
            elif best_eeg.ndim == 3:
                best_eeg = best_eeg[0] if best_eeg.shape[0] == 1 else np.mean(best_eeg, axis=0)
                # Convert to float32 if needed
                if best_eeg.dtype == np.float64:
                    best_eeg = best_eeg.astype(np.float32)
                return self._ensure_channels_first(best_eeg)
        
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
            
            # Convert to float32 to save memory if still float64
            if eeg.dtype == np.float64:
                eeg = eeg.astype(np.float32)
            eeg = self._ensure_channels_first(eeg)
            if eeg is None or eeg.ndim != 2:
                return []
            
            # Store raw EEG only - frequency bands will be extracted after preprocessing in __getitem__
            # For ZuCo 1.0, typically one sentence per EEG file
            # Paper: "For each sentence s_i, we extract its corresponding EEG interval by 
            # converting word-level timestamps into sample indices: [t_start, t_end] = floor(wordbound_i * f_s), f_s = 250 Hz"
            # For ZuCo 1.0, the text corresponds to the entire EEG recording
            samples = []
            
            # ZuCo 1.0: one sentence typically corresponds to one EEG file
            # If wordbounds are available, we would segment here, but ZuCo 1.0 structure
            # typically has one sentence per file, so use entire recording
            channel_info = eeg_data.get('channel_info')
            del eeg_data
            sample = self._commit_sample(eeg, text, subject_id, task_key, channel_info)
            del eeg
            if sample is not None:
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
            
            # Convert to float32 to save memory if still float64
            if eeg.dtype == np.float64:
                eeg = eeg.astype(np.float32)
            
            eeg = self._ensure_channels_first(eeg)
            if eeg is None or eeg.ndim != 2:
                return []

            num_channels, time_steps = eeg.shape
            
            # Try to extract sentence boundaries from wordbounds
            sentence_windows = self._extract_sentence_windows(wordbounds, task_num, time_steps)
            
            samples = []
            del eeg_data

            def _add_window(start_idx, end_idx, sentence, sent_i):
                if end_idx <= start_idx or end_idx > time_steps:
                    return
                sample = self._commit_sample(
                    eeg[:, start_idx:end_idx],
                    sentence,
                    subject_id,
                    f'{task_type}{task_num}_sent{sent_i}'
                )
                if sample is not None:
                    samples.append(sample)

            if sentence_windows:
                num_sentences = min(len(sentences), len(sentence_windows))
                for i in range(num_sentences):
                    start_idx, end_idx = sentence_windows[i]
                    _add_window(start_idx, end_idx, sentences[i], i + 1)
            elif sentences:
                window_size = time_steps // len(sentences)
                for i, sentence in enumerate(sentences):
                    start_idx = i * window_size
                    end_idx = (i + 1) * window_size if i < len(sentences) - 1 else time_steps
                    _add_window(start_idx, end_idx, sentence, i + 1)

            del eeg
            return samples
            
        except Exception as e:
            print(f"Error loading EEG {eeg_file}: {e}")
            return []
    
    def _extract_sentence_windows(self, wordbounds: Dict, task_num: int, total_time: int) -> List[Tuple[int, int]]:
        """
        Extract sentence-level time windows from wordbounds per paper specification.
        
        Paper: "For each sentence s_i, we extract its corresponding EEG interval by 
        converting word-level timestamps into sample indices:
        [t_start^(i), t_end^(i)] = floor(wordbound_i * f_s), f_s = 250 Hz"
        
        Args:
            wordbounds: Dictionary containing wordbound timing information
            task_num: Task number for matching wordbounds
            total_time: Total time steps in the EEG recording
            
        Returns:
            List of (start_idx, end_idx) tuples in sample indices
        """
        windows = []
        
        # Try to find relevant wordbounds data for this task
        # Wordbounds structure in ZuCo varies, but typically contains timing arrays
        for key, data in wordbounds.items():
            if isinstance(data, np.ndarray):
                # Handle different wordbound formats
                if data.ndim == 1 and len(data) > 0:
                    # Case 1: Array of sentence end times (cumulative)
                    # Convert to sample indices: floor(time * f_s)
                    prev_idx = 0
                    for end_time in data:
                        end_idx = int(np.floor(end_time * self.sampling_rate))  # Paper formula: floor(wordbound_i * f_s)
                        if end_idx > prev_idx and end_idx <= total_time:
                            windows.append((prev_idx, end_idx))
                            prev_idx = end_idx
                elif data.ndim == 2 and data.shape[0] >= 2:
                    # Case 2: (2, N) array with start and end times
                    # First row: start times, second row: end times
                    start_times = data[0, :] if data.shape[0] >= 1 else data[:, 0]
                    end_times = data[1, :] if data.shape[0] >= 2 else data[:, 1]
                    for start_time, end_time in zip(start_times, end_times):
                        start_idx = int(np.floor(start_time * self.sampling_rate))
                        end_idx = int(np.floor(end_time * self.sampling_rate))
                        if end_idx > start_idx and end_idx <= total_time:
                            windows.append((start_idx, end_idx))
                elif isinstance(data, dict):
                    # Case 3: Nested structure with timing information
                    # Try to extract timing arrays from nested structure
                    for subkey, subdata in data.items():
                        if isinstance(subdata, np.ndarray) and subdata.ndim == 1:
                            prev_idx = 0
                            for end_time in subdata:
                                end_idx = int(np.floor(end_time * self.sampling_rate))
                                if end_idx > prev_idx and end_idx <= total_time:
                                    windows.append((prev_idx, end_idx))
                                    prev_idx = end_idx
        
        # Paper: "If timestamp annotations are partially unavailable, EEG recordings 
        # are segmented proportionally across sentences following ZuCo's experimental protocol"
        # If no valid windows found, return empty list to trigger proportional segmentation
        return windows
    
    def _extract_frequency_bands(self, eeg: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract 5 frequency bands from EEG signal
        Returns dict with keys: 'delta', 'theta', 'alpha', 'beta', 'gamma'
        Each value is array of shape (C, T)
        """
        import sys
        
        num_channels, time_steps = eeg.shape
        
        # Check for invalid input
        if np.any(np.isnan(eeg)) or np.any(np.isinf(eeg)):
            print(f"  [WARNING] Input EEG contains NaN/Inf, filling with zeros", file=sys.stderr, flush=True)
            eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)
        
        bands = {}
        nyquist = self.sampling_rate / 2
        
        # Check if signal is too short for filtering (once, not per channel)
        if time_steps < 10:
            print(f"  [WARNING] Signal too short ({time_steps} samples) for filtering, using original", 
                  file=sys.stderr, flush=True)
            # Return original signal for all bands if too short (make contiguous)
            for band_name in self.FREQUENCY_BANDS.keys():
                bands[band_name] = np.ascontiguousarray(eeg.copy(), dtype=np.float32)
            return bands
        
        for band_name, (low_freq, high_freq) in self.FREQUENCY_BANDS.items():
            # Normalize frequencies
            low_norm = low_freq / nyquist
            high_norm = high_freq / nyquist
            high_norm = min(high_norm, 0.99)  # Cap at Nyquist
            
            try:
                # Design bandpass filter once for all channels
                b, a = signal.butter(4, [low_norm, high_norm], btype='band')
                
                # Vectorized filtering: apply to all channels at once
                band_eeg = signal.filtfilt(b, a, eeg, axis=1)
                
                # Check for NaN/Inf in filtered result
                if np.any(np.isnan(band_eeg)) or np.any(np.isinf(band_eeg)):
                    # Fallback: use original signal where filtering failed
                    invalid_mask = np.isnan(band_eeg) | np.isinf(band_eeg)
                    band_eeg[invalid_mask] = eeg[invalid_mask]
                    print(f"  [WARNING] Filtering produced NaN/Inf in {band_name} band, using original signal where needed", 
                          file=sys.stderr, flush=True)
                
                # Ensure contiguous memory layout to avoid negative stride issues when converting to tensor
                bands[band_name] = np.ascontiguousarray(band_eeg, dtype=np.float32)
            except Exception as e:
                # If filtering fails, use original signal (make contiguous)
                print(f"  [WARNING] Filtering failed for {band_name} band: {e}, using original signal", 
                      file=sys.stderr, flush=True)
                bands[band_name] = np.ascontiguousarray(eeg.copy(), dtype=np.float32)
        
        return bands
    
    def _normalize_sentence_id(self, text: str) -> str:
        """Canonical sentence key so the same stimulus cannot leak across splits."""
        return ' '.join(str(text).strip().lower().split())

    def _split_data(self, all_samples: List[Dict]) -> List[Dict]:
        """
        Split by unique sentence identity, not by EEG instance.

        All subject recordings of the same sentence are assigned to one partition.
        The assignment is deterministic given split_seed.
        """
        keys = [self._normalize_sentence_id(s.get('sentence_text', '')) for s in all_samples]
        unique = sorted(set(keys))
        rng = np.random.RandomState(self.split_seed)
        rng.shuffle(unique)

        n = len(unique)
        train_end = int(n * self.train_split)
        val_end = int(n * (self.train_split + self.val_split))
        if n >= 3:
            train_end = min(max(train_end, 1), n - 2)
            val_end = min(max(val_end, train_end + 1), n - 1)

        split_of = {}
        for i, sent in enumerate(unique):
            if i < train_end:
                split_of[sent] = 'train'
            elif i < val_end:
                split_of[sent] = 'val'
            else:
                split_of[sent] = 'test'

        selected = [sample for sample, key in zip(all_samples, keys) if split_of[key] == self.split]
        n_sents = sum(1 for sent in unique if split_of[sent] == self.split)
        print(
            f"  Sentence-identity split (seed={self.split_seed}): "
            f"{n} unique sentences → {n_sents} in '{self.split}' "
            f"({len(selected)} EEG instances)"
        )
        return selected
    
    def _apply_highpass_filter(self, eeg: np.ndarray) -> np.ndarray:
        """
        Apply high-pass filter to remove slow drifts and DC offset
        
        Args:
            eeg: EEG signal of shape (num_channels, time_steps)
            
        Returns:
            filtered_eeg: High-pass filtered EEG
        """
        if not self.apply_highpass_filter:
            return np.ascontiguousarray(eeg, dtype=np.float32)
        
        num_channels, time_steps = eeg.shape
        
        # Check for invalid input
        if np.any(np.isnan(eeg)) or np.any(np.isinf(eeg)):
            eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Skip filtering if signal is too short
        if time_steps < 10:
            return np.ascontiguousarray(eeg, dtype=np.float32)
        
        nyquist = self.sampling_rate / 2
        cutoff_norm = self.highpass_cutoff / nyquist
        cutoff_norm = min(cutoff_norm, 0.99)  # Cap at Nyquist
        
        try:
            # Design high-pass Butterworth filter (once for all channels)
            b, a = signal.butter(4, cutoff_norm, btype='high')
            
            # Vectorized filtering: apply to all channels at once using axis parameter
            # filtfilt can handle 2D arrays with axis parameter for vectorized operation
            filtered_eeg = signal.filtfilt(b, a, eeg, axis=1)
            
            # Check for NaN/Inf in results
            if np.any(np.isnan(filtered_eeg)) or np.any(np.isinf(filtered_eeg)):
                # Fallback: use original signal where filtering failed
                invalid_mask = np.isnan(filtered_eeg) | np.isinf(filtered_eeg)
                filtered_eeg[invalid_mask] = eeg[invalid_mask]
            
            # Ensure contiguous memory layout to avoid negative stride issues when converting to tensor
            return np.ascontiguousarray(filtered_eeg, dtype=np.float32)
        except Exception as e:
            # If filtering fails, return original signal (make contiguous)
            import sys
            print(f"Warning: High-pass filtering failed: {e}, using original signal", 
                  file=sys.stderr, flush=True)
            return np.ascontiguousarray(eeg, dtype=np.float32)
    
    def _apply_notch_filter(self, eeg: np.ndarray) -> np.ndarray:
        """
        Apply notch filter to remove power line noise (50/60 Hz)
        
        Args:
            eeg: EEG signal of shape (num_channels, time_steps)
            
        Returns:
            filtered_eeg: Notch-filtered EEG
        """
        if not self.apply_notch_filter:
            return np.ascontiguousarray(eeg, dtype=np.float32)
        
        num_channels, time_steps = eeg.shape
        
        # Check for invalid input
        if np.any(np.isnan(eeg)) or np.any(np.isinf(eeg)):
            eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Skip filtering if signal is too short
        if time_steps < 10:
            return np.ascontiguousarray(eeg, dtype=np.float32)
        
        quality_factor = 30.0  # Quality factor for notch filter
        
        try:
            # Design notch filter (once for all channels)
            b, a = signal.iirnotch(self.notch_freq, quality_factor, self.sampling_rate)
            
            # Vectorized filtering: apply to all channels at once
            filtered_eeg = signal.filtfilt(b, a, eeg, axis=1)
            
            # Check for NaN/Inf in results
            if np.any(np.isnan(filtered_eeg)) or np.any(np.isinf(filtered_eeg)):
                # Fallback: use original signal where filtering failed
                invalid_mask = np.isnan(filtered_eeg) | np.isinf(filtered_eeg)
                filtered_eeg[invalid_mask] = eeg[invalid_mask]
            
            # Ensure contiguous memory layout to avoid negative stride issues when converting to tensor
            return np.ascontiguousarray(filtered_eeg, dtype=np.float32)
        except Exception as e:
            # If filtering fails, return original signal (make contiguous)
            import sys
            print(f"Warning: Notch filtering failed: {e}, using original signal", 
                  file=sys.stderr, flush=True)
            return np.ascontiguousarray(eeg, dtype=np.float32)
    
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
            return np.ascontiguousarray(eeg, dtype=np.float32)
        
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
        
        # Ensure contiguous memory layout
        return np.ascontiguousarray(eeg_interpolated, dtype=np.float32)
    
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
        
        # Ensure contiguous memory layout to avoid negative stride issues when converting to tensor
        return np.ascontiguousarray(eeg, dtype=np.float32)

    def _segment_into_windows(self, eeg: np.ndarray) -> np.ndarray:
        """Divide a sentence EEG array (C, T) into fixed-length windows."""
        from utils.graph_utils import segment_into_windows
        return segment_into_windows(
            eeg,
            window_size=self.window_size,
            stride=self.window_stride,
            max_windows=self.max_windows
        )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        try:
            sample = self.samples[idx]
            
            # Step 1: Preprocess raw EEG first (following paper order):
            # 1. High-pass filtering (0.5 Hz)
            # 2. Notch filtering (50/60 Hz)
            # 3. Z-score normalization
            eeg_raw = self._load_sample_eeg(sample)
            
            # CRITICAL: Ensure correct format (C, T) before any processing
            # ZuCo has ~105 channels, so proper shape is (105, time_steps)
            shape = eeg_raw.shape
            
            # Transpose detection: handle all cases including edge cases like (T, 1)
            if len(shape) == 2:
                dim0, dim1 = shape
                
                # Case 1: dim0 is in channel range (50-200), dim1 is large -> already correct (C, T)
                if 50 <= dim0 <= 200 and dim1 > 1000:
                    pass  # Already correct format
                # Case 2: dim1 is in channel range (50-200), dim0 is large -> transpose needed (T, C) -> (C, T)
                elif 50 <= dim1 <= 200 and dim0 > 1000:
                    eeg_raw = eeg_raw.T
                # Case 3: dim0 is huge (>1000) and dim1 is small (<50) -> likely (T, 1) or (T, C) with few channels -> transpose
                elif dim0 > 1000 and dim1 < 50:
                    eeg_raw = eeg_raw.T
                # Case 4: dim1 is huge and dim0 is small -> likely (C, T) with few channels, already correct
                elif dim1 > 1000 and dim0 < 50:
                    pass  # Already correct
                # Case 5: Both large or both small -> check ratio: larger/smaller > 100 suggests time/channels
                elif max(dim0, dim1) / min(dim0, dim1) > 100:
                    # The larger dimension is likely time, smaller is likely channels
                    if dim0 > dim1:
                        eeg_raw = eeg_raw.T
                # Default: if neither dimension is obviously channels, don't transpose (will be handled by padding later)
            
            # Show progress for first batch processing (important for num_workers=0 on Windows)
            # With sequential processing, the first batch won't be ready until all samples are processed
            if not hasattr(self, '_last_progress_idx'):
                self._last_progress_idx = -1
            
            # Show progress every 5 samples or for first 10 samples
            should_show = getattr(self, '_getitem_prints', 0) < 3
            
            if should_show and idx != self._last_progress_idx:
                import sys
                import time
                if not hasattr(self, '_first_sample_time'):
                    self._first_sample_time = time.time()
                    elapsed = 0
                else:
                    elapsed = time.time() - self._first_sample_time
                
                # Estimate time remaining (rough estimate based on first few samples)
                if idx > 0 and elapsed > 0:
                    samples_per_sec = idx / elapsed
                    remaining_samples = len(self.samples) - idx - 1
                    eta_seconds = remaining_samples / samples_per_sec if samples_per_sec > 0 else 0
                    eta_str = f" (ETA: {eta_seconds:.0f}s)" if eta_seconds > 0 else ""
                else:
                    eta_str = ""
                
                print(f"  [DataLoader] Processing sample {idx+1}/{len(self.samples)} (preprocessing + frequency extraction){eta_str}...", 
                      file=sys.stderr, flush=True)
                self._last_progress_idx = idx
                self._getitem_prints = getattr(self, '_getitem_prints', 0) + 1
            
            # Preprocess EEG
            eeg_preprocessed = self._preprocess_eeg(eeg_raw)
            
            # Step 2: Extract frequency bands from preprocessed EEG
            # Paper: "Following artifact removal, each EEG window is decomposed into five canonical oscillatory bands"
            eeg_bands = self._extract_frequency_bands(eeg_preprocessed)

            eeg_preprocessed_contiguous = np.ascontiguousarray(eeg_preprocessed.copy(), dtype=np.float32)
            eeg_raw_tensor = torch.from_numpy(eeg_preprocessed_contiguous)
            eeg_bands_full = None
            if getattr(self, 'keep_full_sentence_bands', False):
                eeg_bands_full = {
                    band_name: torch.from_numpy(np.ascontiguousarray(band_eeg.copy(), dtype=np.float32))
                    for band_name, band_eeg in eeg_bands.items()
                }

            if self.window_size:
                eeg_windows_np = self._segment_into_windows(eeg_preprocessed_contiguous)
                eeg_bands_tensor = {}
                for band_name, band_eeg in eeg_bands.items():
                    band_windows = self._segment_into_windows(
                        np.ascontiguousarray(band_eeg.copy(), dtype=np.float32)
                    )
                    eeg_bands_tensor[band_name] = torch.from_numpy(
                        np.ascontiguousarray(band_windows, dtype=np.float32)
                    )
                eeg_windows_tensor = torch.from_numpy(np.ascontiguousarray(eeg_windows_np, dtype=np.float32))
                num_windows = eeg_windows_tensor.shape[0]
                window_mask = torch.ones(num_windows, dtype=torch.float32)
            else:
                eeg_bands_tensor = {
                    k: torch.from_numpy(np.ascontiguousarray(band_eeg.copy(), dtype=np.float32)).unsqueeze(0)
                    for k, band_eeg in eeg_bands.items()
                }
                eeg_windows_tensor = eeg_raw_tensor.unsqueeze(0)
                window_mask = torch.ones(1, dtype=torch.float32)

            return {
                'eeg_raw': eeg_raw_tensor,
                'eeg_windows': eeg_windows_tensor,
                'eeg_bands': eeg_bands_tensor,
                'eeg_bands_full': eeg_bands_full,
                'window_mask': window_mask,
                'sentence_text': sample['sentence_text'],
                'subject': sample['subject'],
                'task': sample['task'],
                'text': sample['sentence_text']
            }
        except Exception as e:
            import sys
            import traceback
            print(f"\n  [ERROR] Failed to process sample {idx+1}/{len(self.samples)}", file=sys.stderr, flush=True)
            print(f"  [ERROR] Subject: {sample.get('subject', 'unknown')}, Task: {sample.get('task', 'unknown')}", 
                  file=sys.stderr, flush=True)
            print(f"  [ERROR] Error: {str(e)}", file=sys.stderr, flush=True)
            print(f"  [ERROR] Traceback:", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            # Re-raise to see full error
            raise


def collate_fn(batch, tokenizer=None, max_seq_length=128, max_eeg_length=20000):
    """
    Collate function for DataLoader.

    Windowed band tensors are padded along the window axis.
    Full-sentence bands are padded along time for the static-functional ablation.
    """
    eeg_raw_list = [item['eeg_raw'] for item in batch]
    eeg_bands_list = [item['eeg_bands'] for item in batch]
    texts = [item['sentence_text'] for item in batch]
    subjects = [item['subject'] for item in batch]
    tasks = [item['task'] for item in batch]

    first_shape = eeg_raw_list[0].shape
    needs_transpose = first_shape[0] > first_shape[1] and 50 < first_shape[1] < 200
    if needs_transpose:
        eeg_raw_list = [eeg.T for eeg in eeg_raw_list]

    max_eeg_len = max(e.shape[1] for e in eeg_raw_list)
    if max_eeg_length is not None and max_eeg_len > max_eeg_length:
        eeg_raw_list = [
            eeg[:, :max_eeg_length] if eeg.shape[1] > max_eeg_length else eeg
            for eeg in eeg_raw_list
        ]
        max_eeg_len = max_eeg_length

    num_channels = eeg_raw_list[0].shape[0]
    eeg_raw_padded = []
    for eeg in eeg_raw_list:
        if eeg.shape[1] < max_eeg_len:
            eeg = torch.cat([eeg, torch.zeros(num_channels, max_eeg_len - eeg.shape[1])], dim=1)
        eeg_raw_padded.append(eeg)
    eeg_raw_batch = torch.stack(eeg_raw_padded)

    def _as_windowed(band: torch.Tensor) -> torch.Tensor:
        if band.dim() == 2:
            if needs_transpose:
                band = band.T
            return band.unsqueeze(0)
        if band.dim() == 3:
            return band
        raise ValueError(f"Unexpected band rank {band.dim()}")

    windowed_lists = {name: [] for name in eeg_bands_list[0].keys()}
    for sample_bands in eeg_bands_list:
        for name, band in sample_bands.items():
            windowed_lists[name].append(_as_windowed(band))

    max_windows = max(b.shape[0] for b in next(iter(windowed_lists.values())))
    window_len = next(iter(windowed_lists.values()))[0].shape[-1]
    eeg_bands_batch = {}
    for name, band_list in windowed_lists.items():
        padded = []
        for band in band_list:
            W, C, T = band.shape
            if T != window_len:
                if T < window_len:
                    band = torch.cat([band, torch.zeros(W, C, window_len - T)], dim=-1)
                else:
                    band = band[:, :, :window_len]
            if W < max_windows:
                band = torch.cat([band, torch.zeros(max_windows - W, C, window_len)], dim=0)
            padded.append(band)
        eeg_bands_batch[name] = torch.stack(padded)

    window_masks = []
    for item in batch:
        mask = item.get('window_mask')
        if mask is None:
            mask = torch.ones(_as_windowed(item['eeg_bands'][next(iter(item['eeg_bands']))]).shape[0])
        if mask.numel() < max_windows:
            mask = torch.cat([mask.float(), torch.zeros(max_windows - mask.numel())])
        else:
            mask = mask.float()[:max_windows]
        window_masks.append(mask)
    window_mask_batch = torch.stack(window_masks)

    eeg_windows_batch = None
    if all('eeg_windows' in item for item in batch):
        windows = [_as_windowed(item['eeg_windows']) for item in batch]
        padded_w = []
        for win in windows:
            W, C, T = win.shape
            if T != window_len:
                win = torch.cat([win, torch.zeros(W, C, window_len - T)], dim=-1) if T < window_len else win[:, :, :window_len]
            if W < max_windows:
                win = torch.cat([win, torch.zeros(max_windows - W, C, window_len)], dim=0)
            padded_w.append(win)
        eeg_windows_batch = torch.stack(padded_w)

    eeg_bands_full_batch = None
    if all(item.get('eeg_bands_full') is not None for item in batch):
        eeg_bands_full_batch = {}
        for name in eeg_bands_list[0].keys():
            full_list = [item['eeg_bands_full'][name] for item in batch]
            if needs_transpose:
                full_list = [
                    band.T if band.dim() == 2 and band.shape[0] > band.shape[1] else band
                    for band in full_list
                ]
            padded_full = []
            for band in full_list:
                if band.shape[1] > max_eeg_len:
                    band = band[:, :max_eeg_len]
                if band.shape[1] < max_eeg_len:
                    band = torch.cat([band, torch.zeros(num_channels, max_eeg_len - band.shape[1])], dim=1)
                padded_full.append(band)
            eeg_bands_full_batch[name] = torch.stack(padded_full)

    if tokenizer is None:
        raise ValueError(
            "Tokenizer is required for text tokenization. "
            "Please provide a tokenizer (e.g., from transformers library) in collate_fn."
        )
    tokenized = tokenizer(
        texts,
        padding='max_length',
        max_length=max_seq_length,
        truncation=True,
        return_tensors='pt'
    )

    batch_out = {
        'eeg': eeg_raw_batch,
        'eeg_raw': eeg_raw_batch,
        'eeg_bands': eeg_bands_batch,
        'window_mask': window_mask_batch,
        'text': texts,
        'sentence_text': texts,
        'text_tokens': tokenized['input_ids'],
        'subject': subjects,
        'task': tasks
    }
    if eeg_windows_batch is not None:
        batch_out['eeg_windows'] = eeg_windows_batch
    if eeg_bands_full_batch is not None:
        batch_out['eeg_bands_full'] = eeg_bands_full_batch
    return batch_out


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
