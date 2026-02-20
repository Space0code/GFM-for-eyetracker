"""
Usage:
python src/data/data_conversion/hci_tagging_conversion.py \
    --spec specifications/hci_to_GFM_spec.yaml \
    --input data/raw/hci-tagging \
    --output data/raw-one-format/hci-tagging
"""

import argparse
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

PATTERN = re.compile(r".*-All-Data-.*\.tsv$")
FILENAME_PATTERN = re.compile(
    r"(?P<subject>P\d+)-(?P<recording>Rec\d+)-All-Data-.*_Section_(?P<section>\d+)\.tsv$"
)

DEFAULT_EXPERIMENT_TYPE_MAP = {
    "emotion elicitation": "emotion-elicitation",
    "video tagging": "video-tagging",
    "image tagging 1": "image-tagging-1",
    "image tagging 2": "image-tagging-2",
}

DEFAULT_SHARED_OUTPUT_COLUMNS = [
    "subject",
    "recording",
    "section",
    "session-id",
    "experiment-type",
    "is-stimulus",
    "media-file",
    "time-rel-seconds",
    "time-abs-seconds",
    "x-left",
    "y-left",
    "x-right",
    "y-right",
    "x-avg",
    "y-avg",
    "distance-left",
    "distance-right",
    "pupil-size-left-avg",
    "pupil-size-right-avg",
    "fixation-index",
    "fixation-duration",
    "fixation",
    "raw-validity-gaze-left",
    "raw-validity-gaze-right",
    "raw-validity-pupil-left",
    "raw-validity-pupil-right",
    "confidence-gaze-left",
    "confidence-gaze-right",
    "confidence-pupil-left",
    "confidence-pupil-right",
    "event",
    "event-type",
    "stimulus-id",
]

DEFAULT_LABEL_COLUMNS_BY_EXPERIMENT_TYPE = {
    "emotion-elicitation": [
        "emotion-id",
        "emotion-arousal",
        "emotion-valence",
        "emotion-control",
        "emotion-predictability",
        "emotion-source",
        "emotion-derivation-status",
    ],
    "video-tagging": [
        "tag-valid",
        "tag-agree",
        "tag-source",
        "tag-derivation-status",
    ],
    "image-tagging-1": [
        "tag-valid",
        "tag-agree",
        "tag-source",
        "tag-derivation-status",
    ],
    "image-tagging-2": [
        "tag-valid",
        "tag-agree",
        "tag-source",
        "tag-derivation-status",
    ],
}

EMOTION_KEY_TO_ID = {
    # Mapping from participant key press in "Emotion keyword" to feltEmo id (manual Table 5).
    1: 5,   # Sadness
    2: 4,   # Joy, happiness
    3: 2,   # Disgust
    4: 0,   # Neutral
    5: 11,  # Amusement
    6: 1,   # Anger
    7: 3,   # Fear
    8: 6,   # Surprise
    9: 12,  # Anxiety
}

QUESTION_COLUMNS = {
    "Emotion keyword": "emotion-id",
    "Arousal assessment": "emotion-arousal",
    "Valence assessment": "emotion-valence",
    "Dominance assessment": "emotion-control",
    "Predictability assessment": "emotion-predictability",
}


@dataclass
class SessionMetadata:
    """Parsed subset of session.xml fields relevant for conversion."""

    session_id: Optional[str]
    subject_id: Optional[str]
    experiment_type: Optional[str]
    is_stimulus: Optional[bool]
    media_file: Optional[str]
    aud_begin_sample: Optional[float]
    felt_emo: Optional[int]
    felt_arousal: Optional[int]
    felt_valence: Optional[int]
    felt_control: Optional[int]
    felt_predictability: Optional[int]
    tag_valid: Optional[int]
    tag_agree: Optional[int]
    guide_cut_filename: Optional[str]


def load_spec(spec_path: str) -> Dict[str, Any]:
    """Load conversion spec and fill default configuration sections."""
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}

    mappings = spec.get("mappings", {})
    experiment_type_map = {
        str(k).strip().lower(): str(v).strip()
        for k, v in (spec.get("experiment_type_map") or DEFAULT_EXPERIMENT_TYPE_MAP).items()
    }
    shared_output_columns = list(spec.get("shared_output_columns") or DEFAULT_SHARED_OUTPUT_COLUMNS)
    label_columns_by_experiment_type = {
        str(k).strip(): list(v)
        for k, v in (
            spec.get("label_columns_by_experiment_type") or DEFAULT_LABEL_COLUMNS_BY_EXPERIMENT_TYPE
        ).items()
    }
    label_status_values = spec.get("label_status_values", {})

    return {
        "mappings": mappings,
        "experiment_type_map": experiment_type_map,
        "shared_output_columns": shared_output_columns,
        "label_columns_by_experiment_type": label_columns_by_experiment_type,
        "label_status_values": label_status_values,
    }


def _parse_int(value: Any) -> Optional[int]:
    """Parse optional integer, returning None for empty or malformed values."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _parse_float(value: Any) -> Optional[float]:
    """Parse optional float, returning None for empty or malformed values."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_bool_from_stim(value: Any) -> Optional[bool]:
    """Parse XML `isStim` field into bool."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "1":
        return True
    if text == "0":
        return False
    return None


def _normalize_experiment_type(
    raw_experiment_type: Optional[str], experiment_type_map: Dict[str, str]
) -> Optional[str]:
    """Map raw experimentType from XML to normalized slug used in folder and column values."""
    if raw_experiment_type is None:
        return None
    key = raw_experiment_type.strip().lower()
    return experiment_type_map.get(key)


def _find_header_row(input_tsv: str, header_prefix: str = "Timestamp\t") -> int:
    """Return the 0-based row index where the exported table header starts."""
    with open(input_tsv, "r", encoding="utf-8-sig", errors="replace") as f:
        for idx, line in enumerate(f):
            if line.startswith(header_prefix):
                return idx
    raise ValueError(f"Header row starting with '{header_prefix}' not found in {input_tsv}")


def _parse_tsv_preamble(input_tsv: str, header_row_idx: int) -> Dict[str, str]:
    """Extract simple key-value metadata lines from the preamble before the table header."""
    metadata: Dict[str, str] = {}
    with open(input_tsv, "r", encoding="utf-8-sig", errors="replace") as f:
        for idx, line in enumerate(f):
            if idx >= header_row_idx:
                break
            if "\t" not in line:
                continue
            key, value = line.split("\t", 1)
            key = key.strip().rstrip(":")
            value = value.strip()
            if key and value:
                metadata[key] = value
    return metadata


def _parse_filename_metadata(input_tsv: str) -> Dict[str, Optional[str]]:
    """Parse subject, recording and section from the TSV filename."""
    filename = os.path.basename(input_tsv)
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return {"subject": None, "recording": None, "section": None}
    return {
        "subject": match.group("subject"),
        "recording": match.group("recording"),
        "section": match.group("section"),
    }


def _parse_session_xml(session_xml_path: str) -> Optional[SessionMetadata]:
    """Parse session.xml metadata required for conversion and labels."""
    if not os.path.exists(session_xml_path):
        return None
    try:
        tree = ET.parse(session_xml_path)
    except ET.ParseError:
        print(f"Warning: failed to parse XML {session_xml_path}.")
        return None

    root = tree.getroot()
    subject_node = root.find("subject")
    subject_id = subject_node.get("id") if subject_node is not None else None

    guide_cut_filename = None
    for annotation in root.findall(".//annotation"):
        if annotation.get("type") == "GuideCut":
            guide_cut_filename = annotation.get("filename")
            if guide_cut_filename:
                break

    return SessionMetadata(
        session_id=root.get("sessionId"),
        subject_id=subject_id,
        experiment_type=root.get("experimentType"),
        is_stimulus=_parse_bool_from_stim(root.get("isStim")),
        media_file=root.get("mediaFile"),
        aud_begin_sample=_parse_float(root.get("audBeginSmp")),
        felt_emo=_parse_int(root.get("feltEmo")),
        felt_arousal=_parse_int(root.get("feltArsl")),
        felt_valence=_parse_int(root.get("feltVlnc")),
        felt_control=_parse_int(root.get("feltCtrl")),
        felt_predictability=_parse_int(root.get("feltPred")),
        tag_valid=_parse_int(root.get("tagValid")),
        tag_agree=_parse_int(root.get("tagAgree")),
        guide_cut_filename=guide_cut_filename,
    )


def _find_guide_cut_path(tsv_dir: str, session_meta: Optional[SessionMetadata]) -> Optional[str]:
    """Find companion Guide-Cut TSV path for a session."""
    if session_meta and session_meta.guide_cut_filename:
        candidate = os.path.join(tsv_dir, session_meta.guide_cut_filename)
        if os.path.exists(candidate):
            return candidate
    candidates = [f for f in os.listdir(tsv_dir) if f.endswith("Guide-Cut.tsv")]
    if not candidates:
        return None
    candidates.sort()
    return os.path.join(tsv_dir, candidates[0])


def _find_guide_segment(
    guide_df: pd.DataFrame, media_file: str, aud_begin_sample: float, tolerance: float = 2.0
) -> Optional[pd.DataFrame]:
    """Return the questionnaire segment after the target movie in Guide-Cut data."""
    starts = guide_df[
        (guide_df["Action"] == "MovieStart")
        & (guide_df["Name"] == media_file)
        & guide_df["AudioSampleCut"].notna()
    ].copy()
    if starts.empty:
        return None

    starts["sample_diff"] = (starts["AudioSampleCut"] - aud_begin_sample).abs()
    starts = starts.sort_values("sample_diff")
    if float(starts.iloc[0]["sample_diff"]) > tolerance:
        return None

    start_idx = int(starts.index[0])

    ends = guide_df[
        (guide_df.index > start_idx)
        & (guide_df["Action"] == "MovieEnd")
        & (guide_df["Name"] == media_file)
    ]
    if ends.empty:
        return None
    end_idx = int(ends.index[0])

    next_start = guide_df[(guide_df.index > end_idx) & (guide_df["Action"] == "MovieStart")]
    next_start_idx = int(next_start.index[0]) if not next_start.empty else None

    if next_start_idx is None:
        return guide_df.loc[end_idx + 1 :].copy()
    return guide_df.loc[end_idx + 1 : next_start_idx - 1].copy()


def _parse_guide_question_answers(
    segment_df: pd.DataFrame,
) -> Tuple[Dict[str, Optional[int]], List[str]]:
    """Parse key press answers from a guide questionnaire segment with strict labeling."""
    answers: Dict[str, Optional[int]] = {q: None for q in QUESTION_COLUMNS.keys()}
    active_question: Optional[str] = None

    for _, row in segment_df.iterrows():
        action = str(row.get("Action", "")).strip()
        name = str(row.get("Name", "")).strip()

        if action == "InstructionStart":
            active_question = name if name in QUESTION_COLUMNS else None
            continue

        if action == "InstructionEnd":
            active_question = None
            continue

        if action == "KeyPress" and active_question is not None and answers[active_question] is None:
            match = re.fullmatch(r"D(\d)", name)
            if match:
                answers[active_question] = int(match.group(1))

    missing_questions = [q for q, value in answers.items() if value is None]
    return answers, missing_questions


def _derive_emotion_from_guide(
    guide_tsv_path: str, session_meta: SessionMetadata
) -> Tuple[Optional[Dict[str, Optional[int]]], str]:
    """Derive emotion ratings from Guide-Cut for sessions without felt* in XML."""
    if not session_meta.media_file or session_meta.aud_begin_sample is None:
        return None, "guide-cut-missing-session-metadata"

    guide_df = pd.read_csv(guide_tsv_path, sep="\t", dtype=str, encoding="utf-8-sig")
    for numeric_col in ["AudioSampleNumber", "AudioSampleCut"]:
        if numeric_col in guide_df.columns:
            guide_df[numeric_col] = pd.to_numeric(guide_df[numeric_col], errors="coerce")
        else:
            guide_df[numeric_col] = pd.NA

    segment_df = _find_guide_segment(
        guide_df=guide_df,
        media_file=session_meta.media_file,
        aud_begin_sample=session_meta.aud_begin_sample,
    )
    if segment_df is None:
        return None, "guide-cut-match-failed"

    answers, missing_questions = _parse_guide_question_answers(segment_df)
    if missing_questions:
        return None, "guide-cut-missing-questions"

    emotion_key = answers["Emotion keyword"]
    emotion_id = EMOTION_KEY_TO_ID.get(int(emotion_key)) if emotion_key is not None else None
    if emotion_id is None:
        return None, "guide-cut-invalid-emotion-key"

    values = {
        "emotion-id": emotion_id,
        "emotion-arousal": answers["Arousal assessment"],
        "emotion-valence": answers["Valence assessment"],
        "emotion-control": answers["Dominance assessment"],
        "emotion-predictability": answers["Predictability assessment"],
    }
    return values, "ok"


def _map_validity_to_confidence(
    raw_validity: pd.Series, side_name: str
) -> Tuple[pd.Series, int]:
    """Map Tobii validity codes into binary confidence values, warning on unknown codes."""
    numeric = pd.to_numeric(raw_validity, errors="coerce")
    confidence = pd.Series(pd.NA, index=raw_validity.index, dtype="Int8")

    confidence.loc[numeric.isin([0, 1])] = 1
    confidence.loc[numeric.isin([2, 3, 4])] = 0

    invalid_mask = numeric.notna() & ~numeric.isin([0, 1, 2, 3, 4])
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        print(
            "Warning: encountered "
            f"{invalid_count} unexpected validity code(s) on {side_name}; "
            "mapped confidence to NaN for those rows."
        )

    return confidence, invalid_count


def _initialize_emotion_columns(df: pd.DataFrame) -> None:
    """Ensure all emotion-related columns exist before filling values."""
    for col in [
        "emotion-id",
        "emotion-arousal",
        "emotion-valence",
        "emotion-control",
        "emotion-predictability",
    ]:
        df[col] = pd.NA


def _parse_binary_label(value: Optional[int], label_name: str, input_tsv: str) -> Tuple[Optional[int], bool]:
    """Validate binary label from XML and keep only 0/1 values."""
    if value is None:
        return None, False
    if value in (0, 1):
        return value, False
    print(
        f"Warning: unexpected {label_name}={value} in {input_tsv}; "
        "setting label to NaN."
    )
    return None, True


def convert_tsv(
    input_tsv: str,
    mappings: Dict[str, str],
    output_csv: str,
    session_meta: Optional[SessionMetadata],
    experiment_type_slug: str,
    shared_output_columns: List[str],
    label_columns_by_experiment_type: Dict[str, List[str]],
) -> bool:
    """Convert one HCI-tagging TSV export into the common CSV schema."""
    print(f"Converting {input_tsv}")
    header_row_idx = _find_header_row(input_tsv)
    preamble = _parse_tsv_preamble(input_tsv, header_row_idx)
    file_meta = _parse_filename_metadata(input_tsv)

    bad_line_count = 0
    bad_line_samples: List[str] = []

    def _handle_bad_line(fields: List[str]) -> Optional[List[str]]:
        """Skip malformed rows but keep a short sample for debugging."""
        nonlocal bad_line_count
        bad_line_count += 1
        if len(bad_line_samples) < 3:
            bad_line_samples.append(f"{len(fields)} fields")
        return None

    raw_df = pd.read_csv(
        input_tsv,
        sep="\t",
        header=0,
        skiprows=header_row_idx,
        dtype=str,
        encoding="utf-8-sig",
        engine="python",
        on_bad_lines=_handle_bad_line,
    )
    if bad_line_count:
        print(
            f"Warning: skipped {bad_line_count} malformed lines while reading "
            f"{input_tsv}. Sample sizes: {', '.join(bad_line_samples)}"
        )

    rename_map = {old: new for old, new in mappings.items() if new}
    df = raw_df.rename(columns=rename_map)
    target_cols = [c for c in rename_map.values() if c in df.columns]
    df = df[target_cols].copy()

    for col in [
        "time-rel-seconds",
        "time-abs-seconds",
        "raw-validity-gaze-left",
        "raw-validity-gaze-right",
        "fixation-index",
        "fixation-duration",
        "event",
        "event-type",
        "stimulus-id",
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    df["time-rel-seconds"] = pd.to_numeric(df["time-rel-seconds"], errors="coerce") / 1e6
    df["time-abs-seconds"] = pd.to_numeric(df["time-abs-seconds"], errors="coerce") / 1e6
    if "Timestamp" in raw_df.columns:
        raw_timestamp_seconds = pd.to_numeric(raw_df["Timestamp"], errors="coerce") / 1000.0
        df["time-rel-seconds"] = df["time-rel-seconds"].fillna(raw_timestamp_seconds)

    offset_series = (df["time-abs-seconds"] - df["time-rel-seconds"]).dropna()
    if not offset_series.empty:
        abs_offset = float(offset_series.median())
        df["time-abs-seconds"] = df["time-abs-seconds"].fillna(df["time-rel-seconds"] + abs_offset)

    subject = file_meta["subject"]
    recording = file_meta["recording"]
    section = file_meta["section"]
    if subject is None:
        participant = preamble.get("Participant")
        if participant:
            subject = participant
        elif session_meta and session_meta.subject_id:
            subject = f"P{session_meta.subject_id}"
    if recording is None:
        recording = preamble.get("Recording name")
    if section is None and session_meta and session_meta.session_id:
        section = session_meta.session_id

    df["subject"] = subject
    df["recording"] = recording
    df["section"] = section
    df["session-id"] = session_meta.session_id if session_meta is not None else pd.NA
    df["experiment-type"] = experiment_type_slug
    df["is-stimulus"] = session_meta.is_stimulus if session_meta is not None else pd.NA
    df["media-file"] = session_meta.media_file if session_meta is not None else pd.NA

    df["raw-validity-pupil-left"] = df["raw-validity-gaze-left"]
    df["raw-validity-pupil-right"] = df["raw-validity-gaze-right"]
    df["confidence-gaze-left"], _ = _map_validity_to_confidence(
        df["raw-validity-gaze-left"], "left eye"
    )
    df["confidence-gaze-right"], _ = _map_validity_to_confidence(
        df["raw-validity-gaze-right"], "right eye"
    )
    df["confidence-pupil-left"] = df["confidence-gaze-left"]
    df["confidence-pupil-right"] = df["confidence-gaze-right"]

    fixation_index_numeric = pd.to_numeric(df["fixation-index"], errors="coerce")
    fixation_duration_numeric = pd.to_numeric(df["fixation-duration"], errors="coerce")
    df["fixation"] = (fixation_index_numeric.notna() & (fixation_duration_numeric > 0)).astype("boolean")

    if experiment_type_slug == "emotion-elicitation":
        _initialize_emotion_columns(df)
        emotion_source = "none"
        derivation_status = "not-reported"

        xml_values = None
        if session_meta is not None:
            xml_values = {
                "emotion-id": session_meta.felt_emo,
                "emotion-arousal": session_meta.felt_arousal,
                "emotion-valence": session_meta.felt_valence,
                "emotion-control": session_meta.felt_control,
                "emotion-predictability": session_meta.felt_predictability,
            }

        if xml_values is None:
            derivation_status = "session-metadata-missing"
        elif all(value is not None for value in xml_values.values()):
            for col, value in xml_values.items():
                df[col] = value
            emotion_source = "xml"
            derivation_status = "ok"
        else:
            guide_values = None
            guide_status = None
            should_try_guide = (
                session_meta is not None
                and session_meta.is_stimulus is True
                and any(value is None for value in xml_values.values())
            )
            if should_try_guide:
                guide_tsv_path = _find_guide_cut_path(os.path.dirname(input_tsv), session_meta)
                if guide_tsv_path is not None:
                    guide_values, guide_status = _derive_emotion_from_guide(guide_tsv_path, session_meta)
                else:
                    guide_status = "guide-cut-file-missing"

            filled_values = dict(xml_values)
            if guide_values is not None:
                for col in filled_values:
                    if filled_values[col] is None:
                        filled_values[col] = guide_values[col]
                emotion_source = (
                    "xml+guide-cut"
                    if any(v is not None for v in xml_values.values())
                    else "guide-cut"
                )
            elif any(v is not None for v in xml_values.values()):
                emotion_source = "xml"

            for col, value in filled_values.items():
                df[col] = value

            if all(value is not None for value in filled_values.values()):
                derivation_status = "ok"
            else:
                derivation_status = guide_status or "not-reported"

        df["emotion-source"] = emotion_source
        df["emotion-derivation-status"] = derivation_status

    if experiment_type_slug in {"video-tagging", "image-tagging-1", "image-tagging-2"}:
        tag_valid_value = None
        tag_agree_value = None
        invalid_values_found = False
        tag_source = "none"
        tag_status = "not-reported"

        if session_meta is None:
            tag_status = "session-metadata-missing"
        else:
            tag_source = "xml"
            tag_valid_value, invalid_valid = _parse_binary_label(
                session_meta.tag_valid, "tagValid", input_tsv
            )
            tag_agree_value, invalid_agree = _parse_binary_label(
                session_meta.tag_agree, "tagAgree", input_tsv
            )
            invalid_values_found = invalid_valid or invalid_agree

            if invalid_values_found:
                tag_status = "invalid-tag-values"
            elif tag_valid_value is not None and tag_agree_value is not None:
                tag_status = "ok"
            elif tag_valid_value is None and tag_agree_value is None:
                tag_status = "not-reported"
            else:
                tag_status = "partial"

        df["tag-valid"] = pd.Series([tag_valid_value] * len(df), dtype="Int8")
        df["tag-agree"] = pd.Series([tag_agree_value] * len(df), dtype="Int8")
        df["tag-source"] = tag_source
        df["tag-derivation-status"] = tag_status

    label_columns = label_columns_by_experiment_type.get(experiment_type_slug, [])
    output_columns = list(dict.fromkeys(shared_output_columns + label_columns))
    for col in output_columns:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[output_columns]

    df.to_csv(output_csv, index=False)
    return True


def process_folder(input_root: str, output_root: str, spec: Dict[str, Any]) -> None:
    """Traverse input directory and convert all matching HCI TSV files."""
    mappings: Dict[str, str] = spec["mappings"]
    experiment_type_map: Dict[str, str] = spec["experiment_type_map"]
    shared_output_columns: List[str] = spec["shared_output_columns"]
    label_columns_by_experiment_type: Dict[str, List[str]] = spec["label_columns_by_experiment_type"]

    converted_count = 0
    skipped_count = 0

    for dirpath, _, filenames in os.walk(input_root):
        session_meta = _parse_session_xml(os.path.join(dirpath, "session.xml"))
        experiment_type_slug = _normalize_experiment_type(
            session_meta.experiment_type if session_meta else None,
            experiment_type_map,
        )

        for fname in filenames:
            if not PATTERN.match(fname):
                continue
            if not fname.endswith(".tsv"):
                print(f"Skipping {fname}, not a TSV file.")
                continue
            if experiment_type_slug is None:
                print(
                    "Warning: skipping file due to missing/unknown experiment type "
                    f"in session metadata: {os.path.join(dirpath, fname)}"
                )
                skipped_count += 1
                continue

            in_path = os.path.join(dirpath, fname)
            out_fname = os.path.splitext(fname)[0] + ".csv"
            out_dir = os.path.join(output_root, "Sessions", experiment_type_slug)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, out_fname)

            was_converted = convert_tsv(
                input_tsv=in_path,
                mappings=mappings,
                output_csv=out_path,
                session_meta=session_meta,
                experiment_type_slug=experiment_type_slug,
                shared_output_columns=shared_output_columns,
                label_columns_by_experiment_type=label_columns_by_experiment_type,
            )
            if was_converted:
                converted_count += 1
            else:
                skipped_count += 1

    print(
        f"Done. Converted files: {converted_count}. "
        f"Skipped files: {skipped_count}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch convert Tobii TSV eye-tracking data to GFM CSV format"
    )
    parser.add_argument("--spec", required=True, help="Path to TOBII_to_GFM_spec.yaml")
    parser.add_argument("--input", required=True, help="Path to input folder")
    parser.add_argument("--output", required=True, help="Path to output folder")
    args = parser.parse_args()

    spec = load_spec(args.spec)
    process_folder(args.input, args.output, spec)
