"""Compatibility shim for multiclass task training entrypoints."""

from emotions.multiclass.train_multiclass import run_training_from_config

__all__ = ["run_training_from_config"]
