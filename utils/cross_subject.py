"""
Cross-subject evaluation utilities
Includes leave-one-subject-out (LOSO) and train/test on different subjects
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from preprocessing.preprocessing import ZuCoDataset


class CrossSubjectEvaluator:
    """Utilities for cross-subject evaluation"""
    
    def __init__(self, dataset: ZuCoDataset):
        """
        Args:
            dataset: Dataset containing all samples with subject information
        """
        self.dataset = dataset
        self.samples = dataset.samples
        
        # Group samples by subject
        self.subject_samples = self._group_by_subject()
    
    def _group_by_subject(self) -> Dict[str, List[int]]:
        """Group sample indices by subject ID"""
        subject_samples = {}
        for idx, sample in enumerate(self.samples):
            subject = sample['subject']
            if subject not in subject_samples:
                subject_samples[subject] = []
            subject_samples[subject].append(idx)
        return subject_samples
    
    def get_subjects(self) -> List[str]:
        """Get list of all subject IDs"""
        return list(self.subject_samples.keys())
    
    def leave_one_subject_out_splits(self) -> List[Tuple[List[int], List[int], str]]:
        """
        Generate leave-one-subject-out (LOSO) splits
        
        Returns:
            splits: List of (train_indices, test_indices, test_subject) tuples
        """
        subjects = self.get_subjects()
        splits = []
        
        for test_subject in subjects:
            train_indices = []
            test_indices = []
            
            for subject, indices in self.subject_samples.items():
                if subject == test_subject:
                    test_indices.extend(indices)
                else:
                    train_indices.extend(indices)
            
            splits.append((train_indices, test_indices, test_subject))
        
        return splits
    
    def train_test_subject_split(
        self,
        train_subjects: List[str],
        test_subjects: List[str]
    ) -> Tuple[List[int], List[int]]:
        """
        Split data by subject lists
        
        Args:
            train_subjects: List of subject IDs for training
            test_subjects: List of subject IDs for testing
            
        Returns:
            train_indices, test_indices: Sample indices for train and test
        """
        train_indices = []
        test_indices = []
        
        for subject, indices in self.subject_samples.items():
            if subject in train_subjects:
                train_indices.extend(indices)
            elif subject in test_subjects:
                test_indices.extend(indices)
        
        return train_indices, test_indices
    
    def create_subject_specific_dataset(
        self,
        subject_indices: List[int],
        split: str = 'all'
    ) -> ZuCoDataset:
        """
        Create dataset for specific subjects
        
        Args:
            subject_indices: List of sample indices to include
            split: Split type (not used, kept for compatibility)
            
        Returns:
            dataset: Dataset with only specified samples
        """
        # Create new dataset with filtered samples
        filtered_samples = [self.samples[idx] for idx in subject_indices]
        
        # Create new dataset instance
        filtered_dataset = ZuCoDataset.__new__(ZuCoDataset)
        filtered_dataset.samples = filtered_samples
        filtered_dataset.split = split
        filtered_dataset.max_seq_length = self.dataset.max_seq_length
        filtered_dataset.normalize = self.dataset.normalize
        filtered_dataset.train_split = self.dataset.train_split
        filtered_dataset.val_split = self.dataset.val_split
        
        return filtered_dataset

