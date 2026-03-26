"""Compatibility shim for regression task training entrypoints."""

from emotions.regression.train_regression import run_training_from_config

__all__ = ["run_training_from_config"]
