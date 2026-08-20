import numpy as np
import pandas as pd

from src.data.cleaner import CleaningConfig, IQRClipper, build_preprocessor


def test_preprocessor_imputes_and_encodes_mixed_data():
    frame = pd.DataFrame(
        {
            "size": [900.0, np.nan, 1400.0, 1100.0],
            "location": ["North", "South", None, "North"],
        }
    )
    transformed = build_preprocessor(frame, CleaningConfig()).fit_transform(frame)
    assert transformed.shape[0] == len(frame)
    assert transformed.shape[1] >= 3
    assert np.isfinite(transformed).all()


def test_iqr_clipper_learns_bounds_from_fit_data():
    train = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])
    clipper = IQRClipper().fit(train)
    transformed = clipper.transform(np.array([[1000.0]]))
    assert transformed[0, 0] == clipper.upper_[0]
