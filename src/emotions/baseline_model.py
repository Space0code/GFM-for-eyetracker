"""
Baseline models for emotion prediction.

Supported models: Mean, SVM, LightGBM
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from emotions.metrics import compute_metrics


class BaselineModel:
    """Base class for baseline models."""

    def __init__(self, name):
        self.name = name
        self.model = None

    def fit(self, X_train, y_train):
        """Train model."""
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Predict on data."""
        return self.model.predict(X)

    def evaluate(self, X, y, emotion_names=None, metadata=None, pair_aggregation_fn=np.mean):
        """Compute comprehensive evaluation metrics.

        Args:
            X: Input features (DataFrame or array)
            y: Target values (DataFrame or array)
            emotion_names: Optional list of emotion names
            metadata: Optional list of (subject, recording) tuples
            pair_aggregation_fn: Function to aggregate per-pair metrics (default: np.mean)

        Returns:
            dict: Dictionary containing 'standard' and 'per_pair_aggregated' metrics
        """
        y_pred = self.predict(X)
        return compute_metrics(
            y_pred, y,
            emotion_names=emotion_names,
            metadata=metadata,
            pair_aggregation_fn=pair_aggregation_fn
        )


class MeanEstimator(BaselineModel):
    """Predict mean of training targets."""
    
    def __init__(self):
        super().__init__("Mean")
    
    def fit(self, X_train, y_train):
        y_array = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
        self.mean_ = np.array(y_array.mean(axis=0)).flatten()
    
    def predict(self, X):
        X_len = len(X) if hasattr(X, '__len__') else X.shape[0]
        return np.tile(self.mean_, (X_len, 1))


class SVMBaseline(BaselineModel):
    """SVM regressor with RBF kernel and feature normalization."""
    
    def __init__(self, n_jobs=-1, **kwargs):
        super().__init__("SVM")
        self.scaler = StandardScaler()
        self.model = MultiOutputRegressor(
            SVR(kernel='rbf', C=1.0, epsilon=0.1),
            n_jobs=n_jobs
        )
        self.n_jobs = n_jobs
    
    def fit(self, X_train, y_train):
        """Train model with normalized features."""
        X_array = X_train.values if hasattr(X_train, 'values') else np.array(X_train)
        y_array = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
        X_scaled = self.scaler.fit_transform(X_array)
        self.model.fit(X_scaled, y_array)
    
    def predict(self, X):
        """Predict on normalized data."""
        X_array = X.values if hasattr(X, 'values') else np.array(X)
        X_scaled = self.scaler.transform(X_array)
        return self.model.predict(X_scaled)


class LGBMBaseline(BaselineModel):
    """LightGBM gradient boosting."""
    
    def __init__(self, n_estimators=100, n_jobs=-1, verbose=-1, **kwargs):
        super().__init__("LightGBM")
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.models = []
    
    def fit(self, X_train, y_train):
        """Train separate model for each target."""
        self.models = []
        X_array = X_train.values if hasattr(X_train, 'values') else np.array(X_train)
        y_array = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
        for i in range(y_array.shape[1]):
            model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=0.05,
                max_depth=5,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
            model.fit(X_array, y_array[:, i])
            self.models.append(model)
    
    def predict(self, X):
        """Predict with all models."""
        X_array = X.values if hasattr(X, 'values') else np.array(X)
        predictions = np.zeros((len(X_array), len(self.models)))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X_array)
        return predictions


def get_all_baselines(**hyperparams):
    """Return list of all baseline models with optional hyperparameters.
    
    Args:
        **hyperparams: Dict of model_name -> dict of hyperparameters
        
    Returns:
        List of baseline model instances
    """
    models = {
        'Mean': MeanEstimator,
        'SVM': SVMBaseline,
        'LightGBM': LGBMBaseline
    }
    
    return [
        model_class(**hyperparams.get(name, {}))
        for name, model_class in models.items()
    ]


def get_baseline_by_name(model_name: str, **hyperparams):
    """Get a single baseline model by name.
    
    Args:
        model_name: Name of the model ('Mean', 'SVM', 'LightGBM')
        **hyperparams: Hyperparameters for the model
        
    Returns:
        Baseline model instance
        
    Raises:
        ValueError: If model_name is not supported
    """
    models = {
        'Mean': MeanEstimator,
        'SVM': SVMBaseline,
        'LightGBM': LGBMBaseline
    }
    
    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: {', '.join(models.keys())}"
        )
    
    return models[model_name](**hyperparams)