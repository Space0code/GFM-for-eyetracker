from __future__ import annotations

import data
import emotions
import gnext

from emotions.binary.train_binary import run_training_from_config as legacy_binary
from emotions.multiclass.train_multiclass import run_training_from_config as legacy_multiclass
from emotions.regression.train_regression import run_training_from_config as legacy_regression
from emotions.tasks.binary import run_training_from_config as shim_binary
from emotions.tasks.multiclass import run_training_from_config as shim_multiclass
from emotions.tasks.regression import run_training_from_config as shim_regression
from gnext.train_utils import prepare_data, prepare_data_tabular


def test_root_packages_are_importable() -> None:
    assert data.__doc__ is not None
    assert emotions.__doc__ is not None
    assert gnext.__doc__ is not None


def test_task_shims_point_to_legacy_entrypoints() -> None:
    assert shim_binary is legacy_binary
    assert shim_multiclass is legacy_multiclass
    assert shim_regression is legacy_regression


def test_gnext_train_utils_exports_single_prepare_api() -> None:
    assert callable(prepare_data)
    assert callable(prepare_data_tabular)
