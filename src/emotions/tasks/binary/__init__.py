"""Compatibility shim for binary task training entrypoints."""

from emotions.binary.train_binary import run_training_from_config

__all__ = ["run_training_from_config"]
