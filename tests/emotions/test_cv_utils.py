from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emotions.common.cv_utils import build_split_entries, describe_fold


@dataclass(frozen=True)
class _Sample:
    subject: str
    recording: str


def _samples() -> list[_Sample]:
    return [
        _Sample(subject="P2", recording="rec/2"),
        _Sample(subject="P1", recording="rec/1"),
        _Sample(subject="P1", recording="rec/3"),
    ]


def test_describe_fold_subject_loo_is_stable_and_sorted() -> None:
    fold = describe_fold(
        strategy="subject_loo",
        dataset=_samples(),
        test_idx=np.asarray([0, 1], dtype=int),
        fold_num=0,
    )

    assert fold.test_id == "s_P1_P2"
    assert fold.test_name == "Subjects P1, P2"
    assert fold.fold_key == ("subject_loo", ("P1", "P2"))


def test_describe_fold_recording_kfold_sanitizes_path_separators() -> None:
    fold = describe_fold(
        strategy="recording_kfold",
        dataset=_samples(),
        test_idx=np.asarray([0], dtype=int),
        fold_num=3,
    )

    assert fold.test_id == "rkf_3_rec_2"
    assert fold.fold_key == ("recording_kfold", ("rec/2",))


def test_build_split_entries_generates_split_signatures() -> None:
    splits = [
        (
            np.asarray([0, 2], dtype=int),
            np.asarray([1], dtype=int),
            np.asarray([0], dtype=int),
        )
    ]

    entries = build_split_entries(
        strategy="combined_loo",
        dataset=_samples(),
        splits=splits,
        combined_id_style="pipe",
    )

    assert len(entries) == 1
    assert entries[0]["test_id"] == "sr_P2|rec/2"
    assert entries[0]["split_signature"][0] == ("P1|rec/3", "P2|rec/2")
