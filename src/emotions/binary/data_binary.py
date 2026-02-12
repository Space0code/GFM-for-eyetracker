"""
Binary classification data wrappers.

Wraps existing datasets and converts emotion labels to binary classification
based on a threshold. Supports single-emotion selection.
"""

import torch
from typing import List, Optional
from torch.utils.data import Dataset


class BinarySpacioTemporalDataset(Dataset):
    """
    Wrapper around SpacioTemporalDataset for binary emotion classification.
    
    Converts emotion labels to binary (0 or 1) based on threshold and selects
    a single target emotion for training.
    
    Args:
        base_dataset: The original SpacioTemporalDataset
        target_emotion: Name of emotion to predict (e.g., 'emotion-anger')
        threshold: Intensity threshold for binary classification (label <= threshold -> 0, else -> 1)
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        target_emotion: str,
        threshold: float = 0.0
    ):
        self.base_dataset = base_dataset
        self.target_emotion = target_emotion
        self.threshold = threshold
        
        # Validate emotion exists
        if hasattr(base_dataset, 'emotion_names'):
            if target_emotion not in base_dataset.emotion_names:
                available = ', '.join(base_dataset.emotion_names)
                raise ValueError(
                    f"Target emotion '{target_emotion}' not found. "
                    f"Available emotions: {available}"
                )
            self.emotion_idx = base_dataset.emotion_names.index(target_emotion)
        else:
            # Fallback: try to get from first sample
            sample = base_dataset[0]
            if hasattr(sample, 'emotion_names'):
                if target_emotion not in sample.emotion_names:
                    raise ValueError(f"Target emotion '{target_emotion}' not found")
                self.emotion_idx = sample.emotion_names.index(target_emotion)
            else:
                raise ValueError("Cannot determine emotion names from dataset")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get original graph data
        data = self.base_dataset[idx]
        
        # Extract target emotion value
        if hasattr(data, 'y') and data.y is not None:
            emotion_value = data.y[self.emotion_idx].item()
            
            # Convert to binary label
            binary_label = 1.0 if emotion_value > self.threshold else 0.0
            
            # Replace multi-emotion target with binary target
            data.y = torch.tensor([binary_label], dtype=torch.float32)
        
        return data
    
    @property
    def emotion_names(self):
        """Return list containing only the target emotion."""
        return [self.target_emotion]


class BinaryTabularSample:
    """
    Wrapper for tabular samples with binary emotion labels.
    
    Converts TabularWindowSample to binary classification format.
    
    Args:
        base_sample: Original TabularWindowSample
        target_emotion: Name of emotion to predict
        threshold: Intensity threshold for binary classification
    """
    
    def __init__(self, base_sample, target_emotion: str, threshold: float = 0.0):
        self.base_sample = base_sample
        self.target_emotion = target_emotion
        self.threshold = threshold
        
        # Copy metadata
        self.subject = base_sample.subject
        self.recording = base_sample.recording
        self.features = base_sample.features
        
        # Convert target to binary
        if target_emotion not in base_sample.targets:
            available = ', '.join(base_sample.targets.keys())
            raise ValueError(
                f"Target emotion '{target_emotion}' not found. "
                f"Available: {available}"
            )
        
        emotion_value = base_sample.targets[target_emotion]
        binary_label = 1.0 if emotion_value > threshold else 0.0
        
        # Store only binary target for selected emotion
        self.targets = {target_emotion: binary_label}
    
    def __repr__(self):
        return (f"BinaryTabularSample(subject={self.subject}, "
                f"recording={self.recording}, "
                f"emotion={self.target_emotion}, "
                f"label={self.targets[self.target_emotion]})")


def wrap_tabular_samples(
    samples: List,
    target_emotion: str,
    threshold: float = 0.0
) -> List[BinaryTabularSample]:
    """
    Convert list of TabularWindowSample to BinaryTabularSample.
    
    Args:
        samples: List of TabularWindowSample objects
        target_emotion: Name of emotion to predict
        threshold: Intensity threshold for binary classification
        
    Returns:
        List of BinaryTabularSample objects
    """
    binary_samples = []
    
    for sample in samples:
        try:
            binary_sample = BinaryTabularSample(sample, target_emotion, threshold)
            binary_samples.append(binary_sample)
        except ValueError as e:
            # Skip samples that don't have the target emotion
            print(f"Warning: Skipping sample - {e}")
            continue
    
    print(f"Converted {len(binary_samples)}/{len(samples)} samples to binary format")
    
    # Print class distribution
    labels = [s.targets[target_emotion] for s in binary_samples]
    n_positive = sum(labels)
    n_negative = len(labels) - n_positive
    pos_pct = 100 * n_positive / len(labels) if labels else 0
    print(f"Class distribution: {n_negative} negative (<=threshold), "
          f"{n_positive} positive (>threshold) [{pos_pct:.1f}%]")
    
    return binary_samples
