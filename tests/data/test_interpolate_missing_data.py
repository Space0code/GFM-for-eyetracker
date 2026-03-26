from __future__ import annotations

import pandas as pd

from data.data import interpolate_missing_data


def test_interpolate_missing_data_single_row_is_stable() -> None:
    df = pd.DataFrame(
        {
            "time-rel-seconds": [0.4],
            "x-avg": [1.2],
            "y-avg": [2.1],
            "pupil-size-left-avg": [3.1],
            "pupil-size-right-avg": [3.3],
        }
    )

    out = interpolate_missing_data(
        df.copy(),
        interpolation_columns=["x-avg", "y-avg", "pupil-size-left-avg", "pupil-size-right-avg"],
    )

    assert len(out) == 1
    assert out.iloc[0]["time-rel-seconds"] == 0.4
