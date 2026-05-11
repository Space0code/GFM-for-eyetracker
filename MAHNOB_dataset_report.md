# MAHNOB-HCI / HCI-Tagging Dataset Report

**Date:** 2026-05-11  
**Project:** GFM-for-eyetracker  
**Prepared from:** local data in `data/`, local processing code in `src/`, and the local paper copy `docs/hci-tagging dataset paper.pdf`

## 1. Purpose of this report

This document summarizes:

- what the MAHNOB-HCI / HCI-Tagging dataset contains,
- what part of it is available in this repository,
- how the local data was converted and cleaned,
- what is currently used for model training,
- what is uncertain or cannot be verified anymore.

This report is intentionally explicit about unknowns, because the original dataset website is no longer accessible to us (`404`) and the current local copy is not a complete original package. The local copy was manually reduced to eye-tracking-related data.

## 2. Important caveat about what we can and cannot verify

We no longer have access to the original dataset download page, and we do not have the complete original multimodal package locally.

What this means:

- We can verify the current local copy very accurately.
- We can verify what the paper says about the protocol and excluded subjects.
- We cannot fully verify whether the current local eye-tracking subset is identical to the original full release.
- We cannot fully verify whether a participant excluded in the paper is unusable for eye tracking specifically, or only for some other modality, unless the paper states that explicitly.

## 3. What the paper says

From the local paper copy:

- 30 participants were recruited.
- 27 participants were used in the paper analysis.
- Participants `P9`, `P12`, and `P15` were not analyzed due to technical problems and unfinished data collection.
- The paper further states:
  - participants `9` and `15` are not complete due to technical problems,
  - physiological responses of participant `12` are missing due to recording difficulties.

Interpretation:

- For `P9` and `P15`, the paper suggests incomplete recordings.
- For `P12`, the paper explicitly mentions missing physiological signals, not broken eye tracking.
- Therefore, the paper does **not** prove that `P12` eye-tracking data is unusable.
- The paper also does **not** clearly state whether `P9` eye-tracking data is usable despite incomplete overall data.

## 4. Why we now exclude `P9`, `P12`, `P15` by default

Because we only have a reduced local eye-tracking copy and cannot re-check the full original package, the safest default is to align conservatively with the paper and exclude the participants that the paper itself excluded from analysis.

Current decision in this repository:

- `P9`, `P12`, and `P15` are excluded by default in the HCI training configs via `dataset.exclude_subjects`.
- `P15` is already absent from the local emotion eye-tracking copy, so this exclusion has no practical effect for `P15`.
- The effective exclusions in practice are therefore `P9` and `P12`.

Rationale:

- This keeps our default pipeline closer to the published analysis protocol.
- It avoids quietly relying on data that the paper itself treated as problematic.
- It is conservative in the presence of uncertainty.

This is a default, not a permanent scientific claim. If we later recover the original full dataset or find stronger evidence that `P9` and/or `P12` eye tracking is valid for our ET-only task, we can revisit this decision.

## 5. Where the default exclusion is configured

The repository now supports explicit subject exclusion through:

- `dataset.exclude_subjects`

This is implemented in:

- graph dataset loading,
- tabular window building,
- suite snapshot building,
- cache key construction,
- HCI training configs.

The main HCI configs now default to:

```yaml
exclude_subjects: [P9, P12, P15]
```

Relevant config files updated:

- `src/emotions/multiclass/configs/train_multiclass_hci_tagging.yaml`
- `src/emotions/regression/configs/train_regression_hci_tagging.yaml`
- `src/emotions/binary/configs/train_binary_hci_tagging.yaml`
- `src/emotions/binary/configs/train_binary_hci_tagging_valence.yaml`
- `src/emotions/binary/configs/train_binary_hci_tagging_arousal.yaml`
- `src/emotions/binary/configs/train_binary_hci_tagging_control.yaml`
- `src/emotions/binary/configs/train_binary_hci_tagging_predictability.yaml`
- `src/emotions/suite/configs/run_hci_experiment_suite.yaml`
- `src/emotions/suite/configs/run_hci_experiment_suite_table6_3class.yaml`

To disable the exclusion in a future experiment, set:

```yaml
exclude_subjects: null
```

or provide a different exclusion list.

## 6. High-level description of the dataset

MAHNOB-HCI contains two broad experiment families:

1. emotion recognition from responses to emotional videos,
2. implicit tagging experiments.

In the local copy under `data/raw/hci-tagging/Sessions`, we observe four experiment types:

- `emotion elicitation`
- `video tagging`
- `image tagging 1`
- `image tagging 2`

## 7. Local dataset inventory

### 7.1 Entire local `Sessions/` tree

| Metric | Value |
|---|---:|
| Total session folders | 3,193 |
| Unique subjects overall | 25 |
| Emotion elicitation sessions | 943 |
| Video tagging sessions | 750 |
| Image tagging 1 sessions | 750 |
| Image tagging 2 sessions | 750 |

Unique subject IDs present anywhere in the local copy:

`1, 2, 4, 5, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30`

Missing IDs from the nominal `1..30` range in the local copy:

`3, 6, 11, 15, 25`

Important note:

- subject `26` appears in the local dataset overall, but not in local emotion elicitation.

### 7.2 Local emotion-elicitation subset before default exclusion

| Metric | Value |
|---|---:|
| Raw emotion sessions from `session.xml` | 943 |
| Unique local emotion subjects | 24 |
| Unique emotion recordings (`mediaFile`) | 24 |
| Converted emotion session IDs in CSV cache | 942 |

One raw emotion session is missing from the converted CSVs:

- `session_id = 1984`
- `subject = 16`
- `mediaFile = None`
- `isStim = 1`

Interpretation:

- this raw session is incomplete in metadata,
- it does not survive conversion into the common CSV format.

## 8. Local emotion-elicitation subset after the new default exclusion

Default exclusion:

- `P9`
- `P12`
- `P15` (already absent locally)

### 8.1 Current default local emotion data footprint

| Metric | Value |
|---|---:|
| Rows in processed emotion cache after exclusion | 3,535,842 |
| Subjects after exclusion | 22 |
| Recordings after exclusion | 24 |
| Unique converted session IDs after exclusion | 873 |
| Labeled `ok` session IDs after exclusion | 436 |
| `not-reported` session IDs after exclusion | 437 |
| Unique `(subject, recording)` pairs after exclusion | 524 |
| Labeled `(subject, recording)` pairs after exclusion | 436 |
| Unlabeled `(subject, recording)` pairs after exclusion | 88 |

Rows removed by the default exclusion:

| Removed by exclusion | Rows |
|---|---:|
| `P9` + `P12` + `P15` total | 261,323 |
| `P12`, labeled `ok` | 128,530 |
| `P12`, `not-reported` | 24,438 |
| `P9`, labeled `ok` | 89,838 |
| `P9`, `not-reported` | 18,517 |

`P15` contributes `0` rows locally because that subject is not present in the local copy.

## 9. Session, recording, section, and trial terminology

These terms are easy to confuse and should not be treated as synonyms.

| Term | Meaning in this repository |
|---|---|
| `recording` | the clip name / media file, for example `111.avi` or `funny_f.avi` |
| `session-id` | one concrete MAHNOB raw trial folder from `Sessions/<id>/session.xml` |
| `section` | local Tobii export section number used in filenames; not globally unique |
| `(subject, recording)` pair | the main logical trial unit used by our training pipeline |

Important consequences:

- multiple raw sessions can map to the same `(subject, recording)` pair,
- this happens mainly for baseline / neutral clips,
- our current training pipeline groups by `(subject, recording)`, not by `session-id`.

Before default subject exclusion:

- `942` converted session IDs collapse to `566` unique `(subject, recording)` pairs,
- `470` of those pairs are labeled emotion trials,
- `96` are unlabeled baseline pairs.

After default subject exclusion:

- `873` converted session IDs collapse to `524` unique `(subject, recording)` pairs,
- `436` of those pairs are labeled emotion trials,
- `88` are unlabeled baseline pairs.

## 10. Recordings in the emotion subset

The local emotion subset contains 24 unique recording names:

`107.avi, 111.avi, 138.avi, 146.avi, 30.avi, 52.avi, 53.avi, 55.avi, 58.avi, 69.avi, 73.avi, 79.avi, 80.avi, 90.avi, cats_f.avi, colorbars_Final.avi, dallas_f.avi, detroit_f.avi, earworm_f.avi, funny_f.avi, newyork_f.avi, seagulls_Final.avi, sticks_Final.avi, waves_Final.avi`

The 4 clips treated as unlabeled baseline / neutral material in the current local pipeline are:

- `colorbars_Final.avi`
- `seagulls_Final.avi`
- `sticks_Final.avi`
- `waves_Final.avi`

These remain in the processed cache, but they are not used for emotion training because they are `not-reported`.

## 11. Labeling status

### 11.1 Before default subject exclusion

| Status | Rows |
|---|---:|
| `ok` | 3,214,000 |
| `not-reported` | 583,165 |

### 11.2 After default subject exclusion

| Status | Rows |
|---|---:|
| `ok` | 2,995,632 |
| `not-reported` | 540,210 |

Interpretation:

- `ok` rows belong to emotion-labeled stimulus trials,
- `not-reported` rows belong to unlabeled baseline / neutral periods in the local cache.

## 12. How labels are obtained in the local pipeline

Conversion logic is implemented in `src/data/data_conversion/hci_tagging_conversion.py`.

For emotion elicitation, labels are taken from:

1. `session.xml` when all relevant fields are present,
2. `Guide-Cut` fallback when XML labels are incomplete and the trial is a stimulus trial.

Emotion fields in the common CSV format:

- `emotion-id`
- `emotion-arousal`
- `emotion-valence`
- `emotion-control`
- `emotion-predictability`
- `emotion-source`
- `emotion-derivation-status`

Observed label sources before default exclusion:

| Source | Rows |
|---|---:|
| `xml` | 3,085,470 |
| `guide-cut` | 128,530 |
| `none` | 583,165 |

Interpretation:

- most labeled rows come directly from XML,
- `guide-cut` fills some stimulus sessions that lacked XML labels,
- `none` corresponds to unlabeled baseline rows.

## 13. Row counts across pipeline stages

This section distinguishes between:

- the full local converted emotion subset,
- the current default training setup after excluding `P9/P12/P15`.

### 13.1 Full local converted subset before default exclusion

| Stage | Rows | Notes |
|---|---:|---|
| `raw-one-format` emotion CSVs | 3,814,946 | converted from raw TSV/XML before preprocessing trim |
| `processed` merged emotion cache | 3,797,165 | after confidence trimming and preprocessing preparation |
| labeled `ok` rows | 3,214,000 | rows with valid derived emotion labels |
| training-ready rows | 3,018,447 | after dropping missing core gaze/pupil signals |
| suite rows after q01-q99 outlier filter | 2,831,104 | current suite-style extra signal filter |

### 13.2 Current default training subset after excluding `P9/P12/P15`

| Stage | Rows | Notes |
|---|---:|---|
| processed emotion cache after exclusion | 3,535,842 | current default starting point |
| labeled `ok` rows after exclusion | 2,995,632 | labeled rows only |
| training-ready rows after exclusion | 2,814,005 | after dropping missing core gaze/pupil signals |
| suite rows after exclusion + q01-q99 filter | 2,639,048 | current suite-style default |

## 14. Why rows disappear

### 14.1 Preprocessing trim

Before default subject exclusion:

| Transition | Row change | Main reason |
|---|---:|---|
| `raw-one-format` -> `processed` | `-17,781` | trim each file to the first and last valid-confidence segment |

### 14.2 Label filtering

Before default subject exclusion:

| Transition | Row change | Main reason |
|---|---:|---|
| `processed` -> `ok` only | `-583,165` | baseline rows are `not-reported` and therefore unlabeled |

After default subject exclusion:

| Transition | Row change | Main reason |
|---|---:|---|
| cache after exclusion -> `ok` only | `-540,210` | same reason, but after removing `P9/P12/P15` |

### 14.3 Missing signal filtering

Before default subject exclusion:

| Transition | Row change | Main reason |
|---|---:|---|
| `ok` -> training-ready | `-195,553` | missing core gaze/pupil signal columns after confidence gating and bounded interpolation |

After default subject exclusion:

| Transition | Row change | Main reason |
|---|---:|---|
| `ok` -> training-ready | `-181,627` | same logic after removing `P9/P12/P15` |

### 14.4 Additional suite outlier filter

Before default subject exclusion:

| Transition | Row change | Main reason |
|---|---:|---|
| training-ready -> suite rows | `-187,343` | q01-q99 clipping on four core signals |

After default subject exclusion:

| Transition | Row change | Main reason |
|---|---:|---|
| training-ready -> suite rows | `-174,957` | same q01-q99 signal filter |

## 15. Cleaning and filtering steps in code

Current cleaning logic comes from:

- `src/data/data_conversion/hci_tagging_conversion.py`
- `src/data/data_preprocess.py`
- `src/data/data.py`
- `src/emotions/suite/data_snapshot.py`

In practical order, the local pipeline does the following:

1. Convert raw Tobii TSV exports plus `session.xml` metadata into a common CSV schema.
2. Normalize experiment type names.
3. Map Tobii validity codes into binary confidence indicators.
4. For HCI data, require both left and right gaze confidence to be valid.
5. Replace non-protected signal columns with `NaN` when confidence is bad.
6. Trim each file to the first and last valid-confidence row.
7. Re-anchor `time-rel-seconds` so each sequence starts at zero.
8. Interpolate numeric signal columns across only short gaps.
9. Smooth pupil channels with a short rolling mean.
10. Rebuild merged cache `data/processed/cached_hci_tagging_emotion.csv`.
11. Keep only `allowed_experiment_types = ["emotion-elicitation"]`.
12. Keep only `allowed_label_quality_values = ["ok"]` for emotion training.
13. Exclude default problematic subjects via `exclude_subjects = [P9, P12, P15]`.
14. Drop rows with missing core columns:
    - `time-rel-seconds`
    - `x-avg`
    - `y-avg`
    - `pupil-size-left-avg`
    - `pupil-size-right-avg`
    - `subject`
    - `recording`
    - target column(s)
15. In suite configs, additionally remove signal outliers using q01-q99 bounds on:
    - `x-avg`
    - `y-avg`
    - `pupil-size-left-avg`
    - `pupil-size-right-avg`

## 16. Windowing and graph construction

The current default graph/tabular training setup uses:

- `window_length = 10` seconds
- `window_overlap = 0`
- `kt = 2`
- `ks = 2`
- `min_samples_per_window = max(kt, ks) + 1 = 3`
- feature columns:
  - `x-avg`
  - `y-avg`
  - `pupil-size-left-avg`
  - `pupil-size-right-avg`

Each usable window becomes:

- one tabular sample for baseline models, or
- one spatio-temporal graph for the GNN.

## 17. Current window counts

### 17.1 Before default subject exclusion

| Metric | Value |
|---|---:|
| Non-empty 10-second windows | 5,553 |
| Usable windows (`>= 3` samples) | 5,551 |
| Mean nodes per usable window | 543.766 |

### 17.2 After default subject exclusion

| Metric | Value |
|---|---:|
| Non-empty 10-second windows | 5,179 |
| Usable windows (`>= 3` samples) | 5,177 |
| Mean nodes per usable window | 543.559 |

### 17.3 Current suite default after default subject exclusion

After q01-q99 signal outlier filtering:

| Metric | Value |
|---|---:|
| Non-empty windows | 5,158 |
| Usable windows (`>= 3` samples) | 5,158 |

## 18. What we currently use for training

Current default HCI emotion training assumptions in this repository:

- use only `emotion-elicitation`,
- use only rows with `emotion-derivation-status = ok`,
- exclude `P9`, `P12`, `P15` by default,
- use 10-second windows,
- use four node features:
  - `x-avg`
  - `y-avg`
  - `pupil-size-left-avg`
  - `pupil-size-right-avg`,
- use the current suite signal outlier filter unless a config overrides it.

Effective default training footprint after the new exclusion:

| Metric | Value |
|---|---:|
| Subjects used | 22 |
| Labeled `(subject, recording)` groups used | 436 |
| Emotional recordings used | 20 |
| Training-ready rows | 2,814,005 |
| Suite-default rows | 2,639,048 |
| Usable 10-second windows | 5,158 |

## 19. What remains uncertain

The following points are important and currently unresolved:

1. We cannot verify the full original MAHNOB download package anymore because the original website is unavailable to us.
2. We cannot prove that the current local eye-tracking subset matches the original release exactly.
3. We cannot prove that `P12` eye tracking is bad, because the paper only explicitly says that physiological responses are missing.
4. We cannot prove that `P9` eye tracking is unusable, although the paper says the participant data is incomplete due to technical problems.
5. We cannot infer with certainty why the local copy contains 24 emotion subjects while the paper reports analysis on 27 participants.
6. We cannot reconstruct missing local subject IDs `3, 6, 11, 15, 25` from the current local copy alone.
7. We cannot fully explain whether any local deletions happened before the current repository state beyond the fact that only eye-tracking-relevant data was intentionally kept locally.

## 20. Recommended wording for thesis/report writing

If a precise, careful sentence is needed:

> The original MAHNOB-HCI paper reports 30 recruited participants and 27 analyzed participants, excluding P9, P12, and P15 due to technical problems or incomplete data collection. Because the original dataset package is no longer accessible to us and the local repository contains only a reduced eye-tracking-focused copy, we cannot verify modality-specific usability for those participants with certainty. Therefore, the current pipeline adopts a conservative default and excludes P9, P12, and P15 from training.

## 21. Bottom line

The repository currently uses a local eye-tracking-focused MAHNOB-HCI emotion subset, not a fully verifiable copy of the original full multimodal release. To stay close to the paper and avoid relying on questionable participant data, the pipeline now excludes `P9`, `P12`, and `P15` by default. Under this default, current emotion training uses:

- `22` subjects,
- `436` labeled `(subject, recording)` trials,
- `20` emotional recordings,
- `2,814,005` training-ready rows before suite outlier filtering,
- `2,639,048` rows after suite-default outlier filtering,
- `5,158` usable 10-second windows.
