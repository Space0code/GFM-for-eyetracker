"""Cross-validation helpers shared by task-specific training pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Sequence, Tuple

import numpy as np

CombinedIdStyle = Literal["pipe", "underscore"]


@dataclass(frozen=True)
class FoldIdentity:
    """Fold identifier details used for artifact paths and summaries."""

    test_id: str
    test_name: str
    fold_key: Tuple[str, Tuple[str, ...]]


def _subject_recording_token(dataset: Sequence[Any], index: int) -> Tuple[str, str]:
    sample = dataset[index]
    subject = str(getattr(sample, "subject"))
    recording = str(getattr(sample, "recording"))
    return subject, recording


def validate_non_empty_train_splits(
    splits: List[tuple[np.ndarray, np.ndarray, np.ndarray]],
    strategy: str,
    dataset_label: str,
) -> None:
    """Fail fast if a splitter produced one or more empty train folds."""
    bad_folds = [i for i, (train_idx, _, _) in enumerate(splits) if len(train_idx) == 0]
    if bad_folds:
        fold_text = ", ".join(str(i) for i in bad_folds)
        raise ValueError(
            f"{dataset_label} split(s) {fold_text} from strategy '{strategy}' have empty train sets. "
            "Adjust `cross_validation.strategies`, reduce `cross_validation.val_size`, "
            "or include more distinct subjects/recordings."
        )


def split_group_tokens(
    strategy: str,
    dataset: Sequence[Any],
    indices: np.ndarray,
    combined_id_style: CombinedIdStyle = "pipe",
) -> Tuple[str, ...]:
    """Build canonical strategy-dependent tokens used for split signatures."""
    if strategy in {"subject_loo", "subject_kfold"}:
        return tuple(sorted({_subject_recording_token(dataset, int(i))[0] for i in indices}))

    if strategy in {"recording_loo", "recording_kfold"}:
        return tuple(sorted({_subject_recording_token(dataset, int(i))[1] for i in indices}))

    if strategy == "combined_loo":
        pair_tokens: List[str] = []
        for i in indices:
            subject, recording = _subject_recording_token(dataset, int(i))
            if combined_id_style == "pipe":
                pair_tokens.append(f"{subject}|{recording}")
            else:
                pair_tokens.append(f"{subject}_{recording}")
        return tuple(sorted(set(pair_tokens)))

    raise ValueError(f"Unsupported strategy '{strategy}' for split tokenization.")


def describe_fold(
    strategy: str,
    dataset: Sequence[Any],
    test_idx: np.ndarray,
    fold_num: int,
    combined_id_style: CombinedIdStyle = "pipe",
) -> FoldIdentity:
    """Generate stable fold id/name metadata for one CV split."""
    if strategy == "subject_loo":
        subjects = split_group_tokens(strategy=strategy, dataset=dataset, indices=test_idx)
        return FoldIdentity(
            test_id=f"s_{'_'.join(subjects)}",
            test_name=f"Subjects {', '.join(subjects)}",
            fold_key=("subject_loo", subjects),
        )

    if strategy == "subject_kfold":
        subjects = split_group_tokens(strategy=strategy, dataset=dataset, indices=test_idx)
        return FoldIdentity(
            test_id=f"skf_{fold_num}_{'_'.join(subjects)}",
            test_name=f"SubjectKFold {fold_num} | Test subjects {', '.join(subjects)}",
            fold_key=("subject_kfold", subjects),
        )

    if strategy == "recording_loo":
        recordings = split_group_tokens(strategy=strategy, dataset=dataset, indices=test_idx)
        return FoldIdentity(
            test_id=f"r_{'_'.join(recordings)}",
            test_name=f"Recordings {', '.join(recordings)}",
            fold_key=("recording_loo", recordings),
        )

    if strategy == "recording_kfold":
        recordings = split_group_tokens(strategy=strategy, dataset=dataset, indices=test_idx)
        safe_recordings = [recording.replace("/", "_") for recording in recordings]
        return FoldIdentity(
            test_id=f"rkf_{fold_num}_{'_'.join(safe_recordings)}",
            test_name=f"RecordingKFold {fold_num} | Test recordings {', '.join(recordings)}",
            fold_key=("recording_kfold", recordings),
        )

    if strategy == "combined_loo":
        pair_tokens = split_group_tokens(
            strategy=strategy,
            dataset=dataset,
            indices=test_idx,
            combined_id_style=combined_id_style,
        )
        if combined_id_style == "pipe":
            pretty_pairs = ", ".join(f"({token.replace('|', ', ')})" for token in pair_tokens)
        else:
            pretty_parts: List[str] = []
            for token in pair_tokens:
                left, right = token.split("_", 1) if "_" in token else (token, "")
                pretty_parts.append(f"({left}, {right})")
            pretty_pairs = ", ".join(pretty_parts)
        return FoldIdentity(
            test_id=f"sr_{'_'.join(pair_tokens)}",
            test_name=f"Pairs {pretty_pairs}",
            fold_key=("combined_loo", pair_tokens),
        )

    return FoldIdentity(
        test_id=f"fold_{fold_num}",
        test_name=f"Fold {fold_num}",
        fold_key=(str(strategy), (str(fold_num),)),
    )


def _assert_pairwise_disjoint(
    a: Tuple[str, ...],
    b: Tuple[str, ...],
    a_name: str,
    b_name: str,
    *,
    strategy: str,
    fold_num: int,
    dataset_label: str,
    token_label: str,
) -> None:
    overlap = sorted(set(a).intersection(b))
    if overlap:
        overlap_text = ", ".join(overlap[:10])
        suffix = " ..." if len(overlap) > 10 else ""
        raise ValueError(
            f"{dataset_label} fold {fold_num} with strategy '{strategy}' has overlapping "
            f"{token_label} between {a_name} and {b_name}: {overlap_text}{suffix}. "
            "Expected disjoint groups across train/val/test."
        )


def validate_kfold_group_disjointness(
    splits: List[tuple[np.ndarray, np.ndarray, np.ndarray]],
    strategy: str,
    dataset: Sequence[Any],
    dataset_label: str,
    combined_id_style: CombinedIdStyle = "pipe",
) -> None:
    """Fail fast when k-fold train/val/test sets overlap on split-defining groups."""
    if strategy not in {"subject_kfold", "recording_kfold"}:
        return

    token_label = "subjects" if strategy == "subject_kfold" else "recordings"

    for fold_num, (train_idx, val_idx, test_idx) in enumerate(splits):
        train_tokens = split_group_tokens(
            strategy=strategy,
            dataset=dataset,
            indices=train_idx,
            combined_id_style=combined_id_style,
        )
        val_tokens = split_group_tokens(
            strategy=strategy,
            dataset=dataset,
            indices=val_idx,
            combined_id_style=combined_id_style,
        )
        test_tokens = split_group_tokens(
            strategy=strategy,
            dataset=dataset,
            indices=test_idx,
            combined_id_style=combined_id_style,
        )

        _assert_pairwise_disjoint(
            train_tokens,
            val_tokens,
            "train",
            "val",
            strategy=strategy,
            fold_num=fold_num,
            dataset_label=dataset_label,
            token_label=token_label,
        )
        _assert_pairwise_disjoint(
            train_tokens,
            test_tokens,
            "train",
            "test",
            strategy=strategy,
            fold_num=fold_num,
            dataset_label=dataset_label,
            token_label=token_label,
        )
        _assert_pairwise_disjoint(
            val_tokens,
            test_tokens,
            "val",
            "test",
            strategy=strategy,
            fold_num=fold_num,
            dataset_label=dataset_label,
            token_label=token_label,
        )


def build_split_entries(
    strategy: str,
    dataset: Sequence[Any],
    splits: List[tuple[np.ndarray, np.ndarray, np.ndarray]],
    combined_id_style: CombinedIdStyle = "pipe",
) -> List[Dict[str, Any]]:
    """Materialize split tuples with stable IDs and split signatures."""
    entries: List[Dict[str, Any]] = []
    for fold_num, (train_idx, val_idx, test_idx) in enumerate(splits):
        fold = describe_fold(
            strategy=strategy,
            dataset=dataset,
            test_idx=test_idx,
            fold_num=fold_num,
            combined_id_style=combined_id_style,
        )
        split_signature = (
            split_group_tokens(
                strategy=strategy,
                dataset=dataset,
                indices=train_idx,
                combined_id_style=combined_id_style,
            ),
            split_group_tokens(
                strategy=strategy,
                dataset=dataset,
                indices=val_idx,
                combined_id_style=combined_id_style,
            ),
            split_group_tokens(
                strategy=strategy,
                dataset=dataset,
                indices=test_idx,
                combined_id_style=combined_id_style,
            ),
        )
        entries.append(
            {
                "fold_num": fold_num,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
                "test_id": fold.test_id,
                "test_name": fold.test_name,
                "fold_key": fold.fold_key,
                "split_signature": split_signature,
            }
        )
    return entries
