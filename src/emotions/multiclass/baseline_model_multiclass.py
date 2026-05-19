"""Baseline models for multiclass emotion classification."""

from __future__ import annotations

from typing import Any, Dict, List

import lightgbm as lgb
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from emotions.gazemae_baseline import GAZEMAE_MODEL_NAME
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
    """PyTorch MLP classifier with explicit validation-loss early stopping."""

    def __init__(
        self,
        hidden_layer_size: int = 64,
        max_iter: int = 200,
        random_state: int = 42,
        early_stopping: bool = True,
        early_stopping_patience: int = 15,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        dropout: float = 0.1,
        batch_size: int = 128,
        device: str = "auto",
        **_: Any,
    ) -> None:
        super().__init__("MLP")
        self.scaler = StandardScaler()
        self.hidden_layer_size = int(hidden_layer_size)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.early_stopping = bool(early_stopping)
        self.early_stopping_patience = int(early_stopping_patience)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.device_arg = str(device)
        self.device = torch.device("cuda" if self.device_arg == "auto" and torch.cuda.is_available() else "cpu")
        if self.device_arg != "auto":
            self.device = torch.device(self.device_arg)
        self.training_history_: List[Dict[str, float]] = []
        self.class_indices_: np.ndarray | None = None

    def fit(self, X_train: Any, y_train: Any) -> None:
        self.fit_with_validation(X_train=X_train, y_train=y_train, X_val=None, y_val=None)

    def fit_with_validation(self, X_train: Any, y_train: Any, X_val: Any | None, y_val: Any | None) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.model = None
        self._constant_class = None
        self.class_indices_ = None
        self.training_history_ = []

        labels = _as_1d_labels(y_train)
        self._classes_train = np.unique(labels)
        if len(self._classes_train) < 2:
            self._constant_class = int(self._classes_train[0])
            return

        X_array = np.asarray(X_train.values if hasattr(X_train, "values") else X_train, dtype=float)
        X_scaled = self.scaler.fit_transform(X_array)

        n_classes = int(np.max(labels)) + 1
        has_val = X_val is not None and y_val is not None and len(X_val) > 0
        X_val_tensor: torch.Tensor | None = None
        y_val_tensor: torch.Tensor | None = None
        if has_val:
            val_labels = _as_1d_labels(y_val)
            if val_labels.size > 0:
                n_classes = max(n_classes, int(np.max(val_labels)) + 1)
                X_val_array = np.asarray(X_val.values if hasattr(X_val, "values") else X_val, dtype=float)
                X_val_scaled = self.scaler.transform(X_val_array)
                X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32, device=self.device)
                y_val_tensor = torch.tensor(val_labels, dtype=torch.long, device=self.device)
            else:
                has_val = False
        self.class_indices_ = np.arange(n_classes, dtype=int)

        self.model = _TorchMLPHead(
            in_features=int(X_scaled.shape[1]),
            hidden_size=self.hidden_layer_size,
            n_classes=n_classes,
            dropout=self.dropout,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        train_dataset = TensorDataset(
            torch.tensor(X_scaled, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        train_loader = DataLoader(
            train_dataset,
            batch_size=max(1, self.batch_size),
            shuffle=True,
            generator=generator,
        )

        best_state: Dict[str, torch.Tensor] | None = None
        best_val = float("inf")
        stale_epochs = 0

        for epoch in range(1, self.max_iter + 1):
            self.model.train()
            total_loss = 0.0
            total_count = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * int(batch_x.shape[0])
                total_count += int(batch_x.shape[0])

            train_loss = total_loss / max(1, total_count)
            val_loss = float("nan")
            if has_val and X_val_tensor is not None and y_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss = float(criterion(self.model(X_val_tensor), y_val_tensor).item())

                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                    stale_epochs = 0
                else:
                    stale_epochs += 1

            self.training_history_.append(
                {
                    "epoch": float(epoch),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                }
            )
            if self.early_stopping and has_val and stale_epochs >= self.early_stopping_patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict_proba(self, X: Any, all_classes: np.ndarray) -> np.ndarray:
        if self._constant_class is not None:
            probs = np.zeros((len(X), len(all_classes)), dtype=float)
            col = np.where(all_classes == self._constant_class)[0]
            if len(col) > 0:
                probs[:, int(col[0])] = 1.0
            return probs

        X_array = np.asarray(X.values if hasattr(X, "values") else X, dtype=float)
        X_scaled = self.scaler.transform(X_array)
        if self.model is None or self.class_indices_ is None:
            raise RuntimeError("MLP has not been fitted.")
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X_scaled, dtype=torch.float32, device=self.device))
            raw = torch.softmax(logits, dim=1).detach().cpu().numpy()
        return _align_probabilities(raw, self.class_indices_, all_classes)

    def save_checkpoint(self, output_path: str) -> None:
        if self.model is None:
            return
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "class_indices": self.class_indices_,
                "hidden_layer_size": self.hidden_layer_size,
                "dropout": self.dropout,
            },
            output_path,
        )

    def move_to_cpu(self) -> None:
        """Move the fitted classifier to CPU before pickling for portable artifacts."""
        if self.model is not None:
            self.model.to("cpu")
        self.device = torch.device("cpu")


class _TorchMLPHead(nn.Module):
    """Small classifier head used for frozen embedding baselines."""

    def __init__(self, in_features: int, hidden_size: int, n_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GazeMAEMLPMulticlassBaseline(MulticlassBaselineModel):
    """PyTorch MLP head trained on frozen GazeMAE window embeddings."""

    def __init__(
        self,
        hidden_layer_size: int = 128,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        dropout: float = 0.2,
        early_stopping_patience: int = 15,
        batch_size: int = 128,
        random_state: int = 42,
        device: str = "auto",
        **_: Any,
    ) -> None:
        super().__init__(GAZEMAE_MODEL_NAME)
        self.hidden_layer_size = int(hidden_layer_size)
        self.num_epochs = int(num_epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.dropout = float(dropout)
        self.early_stopping_patience = int(early_stopping_patience)
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)
        self.device_arg = str(device)
        self.device = torch.device("cuda" if self.device_arg == "auto" and torch.cuda.is_available() else "cpu")
        if self.device_arg != "auto":
            self.device = torch.device(self.device_arg)
        self.training_history_: List[Dict[str, float]] = []
        self.class_indices_: np.ndarray | None = None

    @staticmethod
    def _as_float_array(X: Any) -> np.ndarray:
        return np.asarray(X.values if hasattr(X, "values") else X, dtype=np.float32)

    def fit(self, X_train: Any, y_train: Any) -> None:
        self.fit_with_validation(X_train=X_train, y_train=y_train, X_val=None, y_val=None)

    def fit_with_validation(self, X_train: Any, y_train: Any, X_val: Any | None, y_val: Any | None) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.model = None
        self._constant_class = None
        self.class_indices_ = None
        self.training_history_ = []

        X_train_np = self._as_float_array(X_train)
        y_train_np = _as_1d_labels(y_train)
        self._classes_train = np.unique(y_train_np)
        if len(self._classes_train) < 2:
            self._constant_class = int(self._classes_train[0])
            return

        n_classes = int(np.max(y_train_np)) + 1
        if y_val is not None:
            y_val_np = _as_1d_labels(y_val)
            if y_val_np.size > 0:
                n_classes = max(n_classes, int(np.max(y_val_np)) + 1)
        self.class_indices_ = np.arange(n_classes, dtype=int)

        self.model = _TorchMLPHead(
            in_features=int(X_train_np.shape[1]),
            hidden_size=self.hidden_layer_size,
            n_classes=n_classes,
            dropout=self.dropout,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        train_dataset = TensorDataset(
            torch.tensor(X_train_np, dtype=torch.float32),
            torch.tensor(y_train_np, dtype=torch.long),
        )
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        train_loader = DataLoader(
            train_dataset,
            batch_size=max(1, self.batch_size),
            shuffle=True,
            generator=generator,
        )

        has_val = X_val is not None and y_val is not None and len(X_val) > 0
        X_val_tensor: torch.Tensor | None = None
        y_val_tensor: torch.Tensor | None = None
        if has_val:
            X_val_tensor = torch.tensor(self._as_float_array(X_val), dtype=torch.float32, device=self.device)
            y_val_tensor = torch.tensor(_as_1d_labels(y_val), dtype=torch.long, device=self.device)

        best_state: Dict[str, torch.Tensor] | None = None
        best_val = float("inf")
        stale_epochs = 0

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_count = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * int(batch_x.shape[0])
                total_count += int(batch_x.shape[0])

            train_loss = total_loss / max(1, total_count)
            val_loss = float("nan")
            if has_val and X_val_tensor is not None and y_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss = float(criterion(self.model(X_val_tensor), y_val_tensor).item())

                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                    stale_epochs = 0
                else:
                    stale_epochs += 1

            self.training_history_.append(
                {
                    "epoch": float(epoch),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                }
            )
            if has_val and stale_epochs >= self.early_stopping_patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict_proba(self, X: Any, all_classes: np.ndarray) -> np.ndarray:
        if self._constant_class is not None:
            probs = np.zeros((len(X), len(all_classes)), dtype=float)
            col = np.where(all_classes == self._constant_class)[0]
            if len(col) > 0:
                probs[:, int(col[0])] = 1.0
            return probs
        if self.model is None or self.class_indices_ is None:
            raise RuntimeError("GazeMAE_MLP has not been fitted.")

        X_np = self._as_float_array(X)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X_np, dtype=torch.float32, device=self.device))
            raw = torch.softmax(logits, dim=1).detach().cpu().numpy()
        return _align_probabilities(raw, self.class_indices_, all_classes)

    def save_checkpoint(self, output_path: str) -> None:
        if self.model is None:
            return
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "class_indices": self.class_indices_,
                "hidden_layer_size": self.hidden_layer_size,
                "dropout": self.dropout,
            },
            output_path,
        )

    def move_to_cpu(self) -> None:
        """Move the fitted head to CPU before pickling for portable artifacts."""
        if self.model is not None:
            self.model.to("cpu")
        self.device = torch.device("cpu")


def get_multiclass_baseline_by_name(model_name: str, **hyperparams: Any) -> MulticlassBaselineModel:
    """Factory for multiclass baseline estimators."""
    models = {
        "Random": RandomMulticlassClassifier,
        "Majority": MajorityMulticlassClassifier,
        "Mean": MeanMulticlassClassifier,
        "SVM": SVMMulticlassBaseline,
        "LightGBM": LGBMMulticlassBaseline,
        "MLP": MLPMulticlassBaseline,
        GAZEMAE_MODEL_NAME: GazeMAEMLPMulticlassBaseline,
    }
    if model_name not in models:
        raise ValueError(
            f"Unknown multiclass baseline '{model_name}'. Supported: {sorted(models.keys())}"
        )
    return models[model_name](**hyperparams)
