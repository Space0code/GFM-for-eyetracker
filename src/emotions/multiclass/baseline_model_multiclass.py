"""Baseline models for multiclass emotion classification."""

from __future__ import annotations

from typing import Any, Dict, List

import lightgbm as lgb
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from emotions.multiclass.metrics_multiclass import evaluate_multiclass_classification


def _as_1d_labels(y: Any) -> np.ndarray:
    if hasattr(y, "values"):
        values = np.asarray(y.values)
    else:
        values = np.asarray(y)

    if values.ndim > 1:
        values = values.reshape(-1)
    return values.astype(int)


def _align_probabilities(
    proba: np.ndarray,
    model_classes: np.ndarray,
    all_classes: np.ndarray,
) -> np.ndarray:
    aligned = np.zeros((proba.shape[0], len(all_classes)), dtype=float)
    class_to_idx = {int(label): idx for idx, label in enumerate(all_classes.tolist())}
    for src_col, label in enumerate(model_classes.tolist()):
        label_idx = class_to_idx.get(int(label))
        if label_idx is not None:
            aligned[:, label_idx] = proba[:, src_col]

    row_sums = aligned.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        aligned = np.divide(aligned, row_sums, out=np.zeros_like(aligned), where=row_sums != 0)
    return aligned


class MulticlassBaselineModel:
    """Base wrapper for multiclass baseline estimators."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.model = None
        self._constant_class: int | None = None
        self._classes_train: np.ndarray | None = None

    def fit(self, X_train: Any, y_train: Any) -> None:
        raise NotImplementedError

    def predict_proba(self, X: Any, all_classes: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def evaluate(
        self,
        X: Any,
        y: Any,
        all_classes: np.ndarray,
        metadata: Dict[str, List[str]] | None = None,
    ) -> Dict[str, Any]:
        y_true = _as_1d_labels(y)
        y_pred_proba = self.predict_proba(X, all_classes=all_classes)
        return evaluate_multiclass_classification(
            y_pred_proba=y_pred_proba,
            y_true=y_true,
            class_labels=all_classes.tolist(),
            metadata=metadata,
        )


class MeanMulticlassClassifier(MulticlassBaselineModel):
    """Predict class priors estimated on training labels."""

    def __init__(self) -> None:
        super().__init__("Mean")
        self.class_priors_: Dict[int, float] = {}

    def fit(self, X_train: Any, y_train: Any) -> None:
        labels = _as_1d_labels(y_train)
        classes, counts = np.unique(labels, return_counts=True)
        total = float(counts.sum())
        self.class_priors_ = {int(cls): float(cnt / total) for cls, cnt in zip(classes, counts)}

    def predict_proba(self, X: Any, all_classes: np.ndarray) -> np.ndarray:
        n_samples = len(X)
        probs = np.zeros((n_samples, len(all_classes)), dtype=float)
        for col_idx, class_value in enumerate(all_classes.tolist()):
            probs[:, col_idx] = self.class_priors_.get(int(class_value), 0.0)

        row_sums = probs.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            probs = np.divide(probs, row_sums, out=np.zeros_like(probs), where=row_sums != 0)
        return probs


class RandomMulticlassClassifier(MulticlassBaselineModel):
    """Predict uniformly random classes with a reproducible random seed."""

    def __init__(self, random_state: int = 42, **_: Any) -> None:
        super().__init__("Random")
        self.random_state = random_state

    def fit(self, X_train: Any, y_train: Any) -> None:
        labels = _as_1d_labels(y_train)
        self._classes_train = np.unique(labels)

    def predict_proba(self, X: Any, all_classes: np.ndarray) -> np.ndarray:
        n_samples = len(X)
        probs = np.zeros((n_samples, len(all_classes)), dtype=float)
        if n_samples == 0 or len(all_classes) == 0:
            return probs

        rng = np.random.default_rng(self.random_state)
        sampled_classes = rng.choice(all_classes, size=n_samples, replace=True)
        class_to_idx = {int(label): idx for idx, label in enumerate(all_classes.tolist())}
        for row_idx, label in enumerate(sampled_classes.tolist()):
            probs[row_idx, class_to_idx[int(label)]] = 1.0
        return probs


class MajorityMulticlassClassifier(MulticlassBaselineModel):
    """Always predict the most frequent training class."""

    def __init__(self, **_: Any) -> None:
        super().__init__("Majority")

    def fit(self, X_train: Any, y_train: Any) -> None:
        labels = _as_1d_labels(y_train)
        classes, counts = np.unique(labels, return_counts=True)
        self._constant_class = int(classes[np.argmax(counts)])
        self._classes_train = classes

    def predict_proba(self, X: Any, all_classes: np.ndarray) -> np.ndarray:
        probs = np.zeros((len(X), len(all_classes)), dtype=float)
        if self._constant_class is None:
            return probs
        col = np.where(all_classes == self._constant_class)[0]
        if len(col) > 0:
            probs[:, int(col[0])] = 1.0
        return probs


class SVMMulticlassBaseline(MulticlassBaselineModel):
    """SVM multiclass classifier with feature standardization."""

    def __init__(self, probability: bool = True, random_state: int = 42, **_: Any) -> None:
        super().__init__("SVM")
        self.scaler = StandardScaler()
        self.model = SVC(
            kernel="rbf",
            C=1.0,
            probability=probability,
            decision_function_shape="ovr",
            random_state=random_state,
        )

    def fit(self, X_train: Any, y_train: Any) -> None:
        labels = _as_1d_labels(y_train)
        self._classes_train = np.unique(labels)
        if len(self._classes_train) < 2:
            self._constant_class = int(self._classes_train[0])
            return

        X_array = np.asarray(X_train.values if hasattr(X_train, "values") else X_train, dtype=float)
        X_scaled = self.scaler.fit_transform(X_array)
        self.model.fit(X_scaled, labels)

    def predict_proba(self, X: Any, all_classes: np.ndarray) -> np.ndarray:
        if self._constant_class is not None:
            probs = np.zeros((len(X), len(all_classes)), dtype=float)
            col = np.where(all_classes == self._constant_class)[0]
            if len(col) > 0:
                probs[:, int(col[0])] = 1.0
            return probs

        X_array = np.asarray(X.values if hasattr(X, "values") else X, dtype=float)
        X_scaled = self.scaler.transform(X_array)
        raw = self.model.predict_proba(X_scaled)
        return _align_probabilities(raw, self.model.classes_, all_classes)


class LGBMMulticlassBaseline(MulticlassBaselineModel):
    """LightGBM multiclass classifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        n_jobs: int = -1,
        verbose: int = -1,
        random_state: int = 42,
        **_: Any,
    ) -> None:
        super().__init__("LightGBM")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X_train: Any, y_train: Any) -> None:
        labels = _as_1d_labels(y_train)
        self._classes_train = np.unique(labels)
        if len(self._classes_train) < 2:
            self._constant_class = int(self._classes_train[0])
            return

        objective = "multiclass" if len(self._classes_train) > 2 else "binary"
        self.model = lgb.LGBMClassifier(
            objective=objective,
            num_class=int(len(self._classes_train)) if objective == "multiclass" else None,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
        )
        self.model.fit(X_train, labels)

    def predict_proba(self, X: Any, all_classes: np.ndarray) -> np.ndarray:
        if self._constant_class is not None:
            probs = np.zeros((len(X), len(all_classes)), dtype=float)
            col = np.where(all_classes == self._constant_class)[0]
            if len(col) > 0:
                probs[:, int(col[0])] = 1.0
            return probs

        raw = self.model.predict_proba(X)
        if raw.ndim == 1:
            # binary fallback shape [n_samples], convert to [n_samples, 2]
            raw = np.column_stack([1.0 - raw, raw])
        return _align_probabilities(np.asarray(raw), np.asarray(self.model.classes_), all_classes)


class MLPMulticlassBaseline(MulticlassBaselineModel):
    """MLP multiclass classifier with feature standardization."""

    def __init__(
        self,
        hidden_layer_size: int = 64,
        max_iter: int = 200,
        random_state: int = 42,
        early_stopping: bool = False,
        **_: Any,
    ) -> None:
        super().__init__("MLP")
        self.scaler = StandardScaler()

        params = {
            "hidden_layer_sizes": (hidden_layer_size, hidden_layer_size),
            "activation": "relu",
            "solver": "adam",
            "max_iter": max_iter,
            "random_state": random_state,
        }
        if early_stopping:
            params["early_stopping"] = True
            params["validation_fraction"] = 0.1
            params["n_iter_no_change"] = 10

        self.model = MLPClassifier(**params)

    def fit(self, X_train: Any, y_train: Any) -> None:
        labels = _as_1d_labels(y_train)
        self._classes_train = np.unique(labels)
        if len(self._classes_train) < 2:
            self._constant_class = int(self._classes_train[0])
            return

        X_array = np.asarray(X_train.values if hasattr(X_train, "values") else X_train, dtype=float)
        X_scaled = self.scaler.fit_transform(X_array)
        self.model.fit(X_scaled, labels)

    def predict_proba(self, X: Any, all_classes: np.ndarray) -> np.ndarray:
        if self._constant_class is not None:
            probs = np.zeros((len(X), len(all_classes)), dtype=float)
            col = np.where(all_classes == self._constant_class)[0]
            if len(col) > 0:
                probs[:, int(col[0])] = 1.0
            return probs

        X_array = np.asarray(X.values if hasattr(X, "values") else X, dtype=float)
        X_scaled = self.scaler.transform(X_array)
        raw = self.model.predict_proba(X_scaled)
        return _align_probabilities(np.asarray(raw), np.asarray(self.model.classes_), all_classes)


def get_multiclass_baseline_by_name(model_name: str, **hyperparams: Any) -> MulticlassBaselineModel:
    """Factory for multiclass baseline estimators."""
    models = {
        "Random": RandomMulticlassClassifier,
        "Majority": MajorityMulticlassClassifier,
        "Mean": MeanMulticlassClassifier,
        "SVM": SVMMulticlassBaseline,
        "LightGBM": LGBMMulticlassBaseline,
        "MLP": MLPMulticlassBaseline,
    }
    if model_name not in models:
        raise ValueError(
            f"Unknown multiclass baseline '{model_name}'. Supported: {sorted(models.keys())}"
        )
    return models[model_name](**hyperparams)
