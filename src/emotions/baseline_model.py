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
    
    def evaluate(self, X, y):
        """Compute comprehensive evaluation metrics.
        
        Returns:
            dict: Dictionary containing MSE, MAE, SD, R², D², and Pearson R
        """
        y_pred = self.predict(X)
        y_array = y.values if hasattr(y, 'values') else y
        y_pred_array = y_pred.values if hasattr(y_pred, 'values') else y_pred
        
        # Flatten for overall metrics
        y_flat = y_array.flatten()
        y_pred_flat = y_pred_array.flatten()
        
        # MSE
        mse = mean_squared_error(y_flat, y_pred_flat)
        
        # MAE
        mae = mean_absolute_error(y_flat, y_pred_flat)
        
        # Standard deviation of error
        errors = y_flat - y_pred_flat
        sd_error = np.std(errors)
        
        # R² (coefficient of determination)
        r2 = r2_score(y_flat, y_pred_flat)
        
        # D² (fraction of deviance explained)
        ss_res = np.sum((y_flat - y_pred_flat) ** 2)
        ss_tot = np.sum((y_flat - np.mean(y_flat)) ** 2)
        d2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Pearson correlation coefficient
        if np.std(y_flat) > 0 and np.std(y_pred_flat) > 0:
            pearson_r, _ = pearsonr(y_flat, y_pred_flat)
        else:
            pearson_r = 0.0
        
        return {
            'mse': mse,
            'mae': mae,
            'sd_error': sd_error,
            'r2': r2,
            'd2': d2,
            'pearson_r': pearson_r
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
