"""
Binary classification baseline models for emotion recognition.

Extends baseline models for binary classification using scikit-learn classifiers.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from emotions.binary.metrics_binary import evaluate_binary_classification


class BinaryBaselineModel:
    """Base class for binary classification baseline models."""

    def __init__(self, name):
        self.name = name
        self.model = None

    def fit(self, X_train, y_train):
        """Train model."""
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        """Predict probabilities."""
        if hasattr(self.model, 'predict_proba'):
            # Return probability of positive class (class 1)
            return self.model.predict_proba(X)[:, 1]
        else:
            # Fallback to binary predictions
            return self.model.predict(X).astype(float)

    def evaluate(self, X, y, emotion_names=None, metadata=None, 
                 threshold=0.5, pair_aggregation_fn=np.mean):
        """
        Compute comprehensive binary classification metrics.

        Args:
            X: Input features (DataFrame or array)
            y: Target binary labels (DataFrame or array)
            emotion_names: Optional list with single emotion name
            metadata: Optional dict with 'subjects' and 'recordings' lists
            threshold: Decision threshold for classification
            pair_aggregation_fn: Function to aggregate per-pair metrics

        Returns:
            Dictionary with standard and aggregated metrics
        """
        y_pred = self.predict_proba(X)
        return evaluate_binary_classification(
            y_pred, y,
            emotion_names=emotion_names,
            metadata=metadata,
            threshold=threshold,
            pair_aggregation_fn=pair_aggregation_fn
        )


class MeanClassifier(BinaryBaselineModel):
    """Predict mean probability of training targets."""
    """
    A baseline classifier that predicts the mean probability of the positive class.
    This classifier computes the mean probability of the positive class from the training
    targets and uses this constant probability for all predictions. It serves as a simple
    baseline to evaluate the performance of more sophisticated models.
    Note: This is in general NOT the same as a majority classifier but in our case IT IS IF we use threshold=0.5 for classification. 
    The majority classifier predicts the most frequent class label, while MeanClassifier predicts the mean probability of
    the positive class, which can be any value between 0 and 1. They differ in that:
    - Majority classifier: outputs discrete class labels (0 or 1)
    - MeanClassifier: outputs continuous probability values (e.g., 0.45)
    Attributes:
        mean_prob_ (float): The mean probability of the positive class computed during fit().
    Methods:
        fit(X_train, y_train): Computes and stores the mean probability of the positive class.
        predict_proba(X): Returns the mean probability for all samples in X.
    """
    
    def __init__(self):
        super().__init__("Mean")
    
    def fit(self, X_train, y_train):
        y_array = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
        y_array = y_array.flatten()
        # Mean probability of positive class
        self.mean_prob_ = float(y_array.mean())
    
    def predict_proba(self, X):
        X_len = len(X) if hasattr(X, '__len__') else X.shape[0]
        return np.full(X_len, self.mean_prob_)


class SVMBinaryBaseline(BinaryBaselineModel):
    """SVM classifier with RBF kernel and feature normalization."""
    
    def __init__(self, n_jobs=-1, probability=True, **kwargs):
        super().__init__("SVM")
        self.scaler = StandardScaler()
        self.model = SVC(
            kernel='rbf',
            C=1.0,
            probability=probability,  # Enable probability estimates
            random_state=42
        )
        self.n_jobs = n_jobs
    
    def fit(self, X_train, y_train):
        """Train model with normalized features."""
        X_array = X_train.values if hasattr(X_train, 'values') else np.array(X_train)
        y_array = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
        y_array = y_array.flatten().astype(int)
        
        X_scaled = self.scaler.fit_transform(X_array)
        self.model.fit(X_scaled, y_array)
    
    def predict_proba(self, X):
        """Predict probabilities on normalized data."""
        X_array = X.values if hasattr(X, 'values') else np.array(X)
        X_scaled = self.scaler.transform(X_array)
        return self.model.predict_proba(X_scaled)[:, 1]


class LGBMBinaryBaseline(BinaryBaselineModel):
    """LightGBM binary classifier."""
    
    def __init__(self, n_estimators=100, n_jobs=-1, verbose=-1, **kwargs):
        super().__init__("LightGBM")
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=5,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=42
        )
    
    def fit(self, X_train, y_train):
        """Train binary classifier."""
        y_array = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
        y_array = y_array.flatten().astype(int)
        self.model.fit(X_train, y_array)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)[:, 1]


class MLPBinaryBaseline(BinaryBaselineModel):
    """Simple 2-layer MLP classifier with feature normalization."""
    
    def __init__(self, hidden_layer_size=64, max_iter=200, random_state=42, 
                 early_stopping=False, **kwargs):
        super().__init__("MLP")
        self.hidden_layer_size = hidden_layer_size
        self.max_iter = max_iter
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.scaler = StandardScaler()
        
        mlp_kwargs = {
            'hidden_layer_sizes': (hidden_layer_size, hidden_layer_size),
            'activation': 'relu',
            'solver': 'adam',
            'max_iter': max_iter,
            'random_state': random_state
        }
        
        if early_stopping:
            mlp_kwargs['early_stopping'] = True
            mlp_kwargs['validation_fraction'] = 0.1
            mlp_kwargs['n_iter_no_change'] = 10
        
        self.model = MLPClassifier(**mlp_kwargs)
    
    def fit(self, X_train, y_train):
        """Train model with normalized features."""
        X_array = X_train.values if hasattr(X_train, 'values') else np.array(X_train)
        y_array = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
        y_array = y_array.flatten().astype(int)
        
        X_scaled = self.scaler.fit_transform(X_array)
        self.model.fit(X_scaled, y_array)
    
    def predict_proba(self, X):
        """Predict probabilities on normalized data."""
        X_array = X.values if hasattr(X, 'values') else np.array(X)
        X_scaled = self.scaler.transform(X_array)
        return self.model.predict_proba(X_scaled)[:, 1]


def get_binary_baseline_by_name(model_name: str, **hyperparams):
    """
    Get a single binary classification baseline model by name.
    
    Args:
        model_name: Name of the model ('Mean', 'SVM', 'LightGBM', 'MLP')
        **hyperparams: Hyperparameters for the model
        
    Returns:
        Binary baseline model instance
        
    Raises:
        ValueError: If model_name is not supported
    """
    models = {
        'Mean': MeanClassifier,
        'SVM': SVMBinaryBaseline,
        'LightGBM': LGBMBinaryBaseline,
        'MLP': MLPBinaryBaseline
    }
    
    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: {', '.join(models.keys())}"
        )
    
    return models[model_name](**hyperparams)
