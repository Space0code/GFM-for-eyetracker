# eSEED_v2_conversion.py
"""
Dataset structure:
    - sample_{id}, id = 01 to 48
        - gaze_{i}.csv, i = 1 to 10 (10 recordings per sample)
        - pupil_{i}.csv, i = 1 to 10
        - blinks_{i}.csv, i = 1 to 10
        - annotation_{i}.csv, i = 1 to 10
        - questionnaires.csv
        - subject_info.csv
"""

