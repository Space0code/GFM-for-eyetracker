"""Deep config merge helpers for suite wrapper.

This module implements deterministic deep-merge semantics where:
- Nested dictionaries are merged recursively.
- Non-dict values from override replace base values.
- `null`/`None` in the override removes the inherited key.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


_DELETE = object()


def _merge_node(base_value: Any, override_value: Any) -> Any:
    """Merge one config node and return merged value or delete sentinel."""
    if override_value is None:
        return _DELETE

    if isinstance(base_value, dict) and isinstance(override_value, dict):
        merged: Dict[str, Any] = deepcopy(base_value)
        for key, value in override_value.items():
            if value is None:
                merged.pop(key, None)
                continue

            if key in merged:
                child = _merge_node(merged[key], value)
                if child is _DELETE:
                    merged.pop(key, None)
                else:
                    merged[key] = child
            else:
                merged[key] = deepcopy(value)
        return merged

    if isinstance(override_value, dict):
        # Override introduces a dict where base may be non-dict or missing.
        merged: Dict[str, Any] = {}
        for key, value in override_value.items():
            if value is None:
                continue
            merged[key] = deepcopy(value)
        return merged

    return deepcopy(override_value)


def deep_merge_with_null(base: Dict[str, Any], override: Dict[str, Any] | None) -> Dict[str, Any]:
    """Deep-merge two dictionaries with `None`-as-delete semantics.

    Args:
        base: Base dictionary.
        override: Override dictionary. If `None`, a deep copy of `base` is returned.

    Returns:
        Deep merged dictionary.
    """
    if override is None:
        return deepcopy(base)

    merged = _merge_node(base, override)
    if merged is _DELETE:
        return {}
    if not isinstance(merged, dict):
        raise ValueError("Top-level merged config must be a dictionary.")
    return merged


def merge_many(base: Dict[str, Any], *overrides: Dict[str, Any] | None) -> Dict[str, Any]:
    """Apply multiple overrides in order with deep null-aware merge semantics."""
    result = deepcopy(base)
    for override in overrides:
        result = deep_merge_with_null(result, override)
    return result
