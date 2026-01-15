"""
Baseline models for emotion prediction.
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import lightgbm as lgb


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
    
    def evaluate(self, X, y, emotion_names=None):
        """Compute comprehensive evaluation metrics (aggregated and per-emotion).
        
        Args:
            X: Input features
            y: Target values (can be DataFrame with column names or array)
            emotion_names: Optional list of emotion names (extracted from y if DataFrame)
        
        Returns:
            dict: Dictionary containing aggregated and per-emotion metrics
        """
        y_pred = self.predict(X)
        y_array = y.values if hasattr(y, 'values') else y
        y_pred_array = y_pred.values if hasattr(y_pred, 'values') else y_pred
        
        # Extract emotion names from DataFrame columns if available
        if emotion_names is None and hasattr(y, 'columns'):
            emotion_names = list(y.columns)
        elif emotion_names is None:
            emotion_names = [f'emotion_{i}' for i in range(y_array.shape[1] if len(y_array.shape) > 1 else 1)]
        
        # Aggregated metrics (flatten all emotions)
        y_flat = y_array.flatten()
        y_pred_flat = y_pred_array.flatten()
        
        aggregated = {
            'mse': float(mean_squared_error(y_flat, y_pred_flat)),
            'mae': float(mean_absolute_error(y_flat, y_pred_flat)),
            'sd_error': float(np.std(y_flat - y_pred_flat)),
        }
        
        # Pearson correlation for aggregated
        if np.std(y_flat) > 1e-8 and np.std(y_pred_flat) > 1e-8:
            aggregated['pearson_r'] = float(pearsonr(y_flat, y_pred_flat)[0])
        else:
            aggregated['pearson_r'] = 0.0
        
        # Per-emotion metrics
        per_emotion = {}
        num_emotions = y_array.shape[1] if len(y_array.shape) > 1 else 1
        
        for i, emo_name in enumerate(emotion_names[:num_emotions]):
            y_emo = y_array[:, i] if len(y_array.shape) > 1 else y_array
            y_pred_emo = y_pred_array[:, i] if len(y_pred_array.shape) > 1 else y_pred_array
            
            per_emotion[emo_name] = {
                'mse': float(mean_squared_error(y_emo, y_pred_emo)),
                'mae': float(mean_absolute_error(y_emo, y_pred_emo)),
                'sd_error': float(np.std(y_emo - y_pred_emo)),
            }
            
            # Pearson correlation per emotion
            if np.std(y_emo) > 1e-8 and np.std(y_pred_emo) > 1e-8:
                per_emotion[emo_name]['pearson_r'] = float(pearsonr(y_emo, y_pred_emo)[0])
            else:
                per_emotion[emo_name]['pearson_r'] = 0.0
        
        return {
            'aggregated': aggregated,
            'per_emotion': per_emotion
        }


class MeanEstimator(BaselineModel):
    """Predict mean of training targets."""
    
    def __init__(self):
        super().__init__("MeanEstimator")
    
    def fit(self, X_train, y_train):
        y_array = y_train.values if hasattr(y_train, 'values') else y_train
        self.mean_ = y_array.mean(axis=0)
    
    def predict(self, X):
        return np.tile(self.mean_, (len(X), 1))


class SVMBaseline(BaselineModel):
    """SVM regressor with RBF kernel and feature normalization."""
    
    def __init__(self):
        super().__init__("SVM")
        self.scaler = StandardScaler()
        self.model = MultiOutputRegressor(SVR(kernel='rbf', C=1.0, epsilon=0.1))
    
    def fit(self, X_train, y_train):
        """Train model with normalized features."""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
    
    def predict(self, X):
        """Predict on normalized data."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class GaussianNBBaseline(BaselineModel):
    """Gaussian Naive Bayes (discretized for regression)."""
    
    def __init__(self):
        super().__init__("GaussianNB")
        self.model = MultiOutputRegressor(GaussianNB())


class MLPBaseline(BaselineModel):
    """2-layer Multi-Layer Perceptron."""
    
    def __init__(self, hidden_layer_sizes=(512, 128)):
        super().__init__("MLP")
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False
        )
    
    def fit(self, X_train, y_train):
        """Train model with normalized features."""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
    
    def predict(self, X):
        """Predict on normalized data."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class LGBMBaseline(BaselineModel):
    """LightGBM gradient boosting."""
    
    def __init__(self, n_estimators=100):
        super().__init__("LightGBM")
        self.n_estimators = n_estimators
        self.models = []
    
    def fit(self, X_train, y_train):
        """Train separate model for each target."""
        self.models = []
        y_array = y_train.values if hasattr(y_train, 'values') else y_train
        for i in range(y_array.shape[1]):
            model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=0.05,
                max_depth=5,
                verbose=-1
            )
            model.fit(X_train, y_array[:, i])
            self.models.append(model)
    
    def predict(self, X):
        """Predict with all models."""
        predictions = np.zeros((len(X), len(self.models)))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        return predictions


def get_all_baselines():
    """Return list of all baseline models."""
    return [
        MeanEstimator(),
        SVMBaseline(),
        GaussianNBBaseline(),
        MLPBaseline(),
        LGBMBaseline()
    ]
