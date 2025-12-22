from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


class BaseSplitter(ABC):
    """Abstract base class for data splitting strategies."""
    
    def __init__(self, graphs: List, val_size: int = 1):
        """
        Args:
            graphs: List of graph objects with .subject and .recording attributes
            val_size: Number of subjects/recordings to use for validation
        """
        self.graphs = graphs
        self.val_size = val_size
        self._build_index()
    
    @abstractmethod
    def _build_index(self):
        """Build internal index mappings."""
        pass
    
    @abstractmethod
    def split(self):
        """Yield (train_idx, val_idx, test_idx) tuples."""
        pass
    
    def _get_indices_by_subject(self, subject):
        """Get all graph indices for a given subject."""
        return [i for i, g in enumerate(self.graphs) if g.subject == subject]
    
    def _get_indices_by_recording(self, recording):
        """Get all graph indices for a given recording."""
        return [i for i, g in enumerate(self.graphs) if g.recording == recording]
    
    def _get_indices_by_subject_and_recording(self, subject, recording):
        """Get graph indices for specific subject-recording pair."""
        return [i for i, g in enumerate(self.graphs) 
                if g.subject == subject and g.recording == recording]
    
    def _build_indices_from_subjects(self, subjects):
        """Build flat list of graph indices from list of subjects."""
        indices = []
        for s in subjects:
            indices.extend(self._get_indices_by_subject(s))
        return np.array(indices)
    
    def _build_indices_from_recordings(self, recordings):
        """Build flat list of graph indices from list of recordings."""
        indices = []
        for r in recordings:
            indices.extend(self._get_indices_by_recording(r))
        return np.array(indices)


class SubjectLOOSplitter(BaseSplitter):
    """Leave-one-subject-out cross-validation."""
    
    def __init__(self, graphs: List, val_size: int = 1, random_state: Optional[int] = None):
        self.random_state = random_state
        super().__init__(graphs, val_size)
    
    def _build_index(self):
        self.subjects = sorted(set(g.subject for g in self.graphs))
        self.rng = np.random.RandomState(self.random_state)
        print("Total subjects:", len(self.subjects))
        print("subjects:", self.subjects)
    
    def split(self):
        """Yield splits leaving out one subject at a time."""
        for test_subject in self.subjects:
            # Get test indices
            test_idx = self._get_indices_by_subject(test_subject)
            
            # Get remaining subjects for train/val
            available_subjects = [s for s in self.subjects if s != test_subject]
            
            # Randomly select val_size subjects for validation
            available_subjects = available_subjects.copy()
            self.rng.shuffle(available_subjects)
            val_subjects = available_subjects[:self.val_size]
            train_subjects = available_subjects[self.val_size:]
            
            # Build train and val indices
            train_idx = self._build_indices_from_subjects(train_subjects)
            val_idx = self._build_indices_from_subjects(val_subjects)
            
            yield (train_idx, val_idx, np.array(test_idx))


class RecordingLOOSplitter(BaseSplitter):
    """Leave-one-recording-out cross-validation."""
    
    def __init__(self, graphs: List, val_size: int = 1, random_state: Optional[int] = None):
        self.random_state = random_state
        super().__init__(graphs, val_size)
    
    def _build_index(self):
        self.recordings = sorted(set(g.recording for g in self.graphs))
        self.rng = np.random.RandomState(self.random_state)
    
    def split(self):
        """Yield splits leaving out one recording at a time."""
        for test_recording in self.recordings:
            # Get test indices
            test_idx = self._get_indices_by_recording(test_recording)
            
            # Get remaining recordings for train/val
            available_recordings = [r for r in self.recordings if r != test_recording]
            
            # Randomly select val_size recordings for validation
            available_recordings = available_recordings.copy()
            self.rng.shuffle(available_recordings)
            val_recordings = available_recordings[:self.val_size]
            train_recordings = available_recordings[self.val_size:]
            
            # Build train and val indices
            train_idx = self._build_indices_from_recordings(train_recordings)
            val_idx = self._build_indices_from_recordings(val_recordings)
            
            yield (train_idx, val_idx, np.array(test_idx))


class CombinedLOOSplitter(BaseSplitter):
    """Leave out both a subject and a recording.
    
    Test set: graphs from (subject=S, recording=R) - the intersection
    Excluded: all graphs from subject=S AND all graphs from recording=R
    Train set: remaining graphs, with val_size subjects held out for validation
    """
    
    def __init__(self, graphs: List, val_size: int = 1, random_state: Optional[int] = None):
        self.random_state = random_state
        super().__init__(graphs, val_size)
    
    def _build_index(self):
        self.subjects = sorted(set(g.subject for g in self.graphs))
        self.recordings = sorted(set(g.recording for g in self.graphs))
        self.rng = np.random.RandomState(self.random_state)
    
    def split(self):
        """Yield splits for each subject-recording combination."""
        for test_subject in self.subjects:
            for test_recording in self.recordings:
                # Test: only (subject=test_subject, recording=test_recording)
                test_idx = self._get_indices_by_subject_and_recording(
                    test_subject, test_recording
                )
                
                # Skip if no data for this combination
                if len(test_idx) == 0:
                    continue
                
                # Exclude: all from test_subject OR test_recording
                excluded_idx = set(self._get_indices_by_subject(test_subject))
                excluded_idx.update(self._get_indices_by_recording(test_recording))
                
                # Available subjects for train/val (excluding test_subject)
                available_subjects = [s for s in self.subjects if s != test_subject]
                
                # Randomly select val_size subjects for validation
                available_subjects = available_subjects.copy()
                self.rng.shuffle(available_subjects)
                val_subjects = available_subjects[:self.val_size]
                train_subjects = available_subjects[self.val_size:]
                
                # Build train indices (exclude test_recording and val_subjects)
                train_idx = []
                for s in train_subjects:
                    for i in self._get_indices_by_subject(s):
                        if i not in excluded_idx:
                            train_idx.append(i)
                
                # Build val indices (exclude test_recording)
                val_idx = []
                for s in val_subjects:
                    for i in self._get_indices_by_subject(s):
                        if i not in excluded_idx:
                            val_idx.append(i)
                
                yield (np.array(train_idx), np.array(val_idx), np.array(test_idx))


class SubjectKFoldSplitter(BaseSplitter):
    """K-fold cross-validation on subjects."""
    
    def __init__(self, graphs: List, n_splits: int = 5, val_size: int = 1, 
                 shuffle: bool = True, random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        super().__init__(graphs, val_size)
    
    def _build_index(self):
        self.subjects = sorted(set(g.subject for g in self.graphs))
        
        # Try stratification if labels available
        if hasattr(self.graphs[0], 'y') and self.graphs[0].y is not None:
            # Get representative label for each subject (e.g., most common)
            subject_labels = {}
            for subj in self.subjects:
                subj_graphs = [g for g in self.graphs if g.subject == subj]
                # Use first graph's label as representative
                subject_labels[subj] = subj_graphs[0].y.argmax().item() if len(subj_graphs[0].y.shape) > 0 else subj_graphs[0].y.item()
            
            y = np.array([subject_labels[s] for s in self.subjects])
            self.kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, 
                                         random_state=self.random_state)
            self.subject_array = np.array(self.subjects)
            self.y = y
        else:
            self.kfold = KFold(n_splits=self.n_splits, shuffle=self.shuffle, 
                              random_state=self.random_state)
            self.subject_array = np.array(self.subjects)
            self.y = None
    
    def split(self):
        """Yield k-fold splits on subjects."""
        rng = np.random.RandomState(self.random_state)
        
        if self.y is not None:
            splits = self.kfold.split(self.subject_array, self.y)
        else:
            splits = self.kfold.split(self.subject_array)
        
        for train_val_subj_idx, test_subj_idx in splits:
            test_subjects = self.subject_array[test_subj_idx]
            train_val_subjects = self.subject_array[train_val_subj_idx].copy()
            
            # Randomly shuffle and split train_val into train and val
            rng.shuffle(train_val_subjects)
            val_subjects = train_val_subjects[:self.val_size]
            train_subjects = train_val_subjects[self.val_size:]
            
            # Build indices
            train_idx = self._build_indices_from_subjects(train_subjects)
            val_idx = self._build_indices_from_subjects(val_subjects)
            test_idx = self._build_indices_from_subjects(test_subjects)
            
            yield (train_idx, val_idx, test_idx)


class RecordingKFoldSplitter(BaseSplitter):
    """K-fold cross-validation on recordings."""
    
    def __init__(self, graphs: List, n_splits: int = 5, val_size: int = 1,
                 shuffle: bool = True, random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        super().__init__(graphs, val_size)
    
    def _build_index(self):
        self.recordings = sorted(set(g.recording for g in self.graphs))
        self.kfold = KFold(n_splits=self.n_splits, shuffle=self.shuffle, 
                          random_state=self.random_state)
        self.recording_array = np.array(self.recordings)
    
    def split(self):
        """Yield k-fold splits on recordings."""
        rng = np.random.RandomState(self.random_state)
        
        for train_val_rec_idx, test_rec_idx in self.kfold.split(self.recording_array):
            test_recordings = self.recording_array[test_rec_idx].copy()
            train_val_recordings = self.recording_array[train_val_rec_idx].copy()
            
            # Randomly shuffle and split train_val into train and val
            rng.shuffle(train_val_recordings)
            val_recordings = train_val_recordings[:self.val_size]
            train_recordings = train_val_recordings[self.val_size:]
            
            # Build indices
            train_idx = self._build_indices_from_recordings(train_recordings)
            val_idx = self._build_indices_from_recordings(val_recordings)
            test_idx = self._build_indices_from_recordings(test_recordings)
            
            yield (train_idx, val_idx, test_idx)
