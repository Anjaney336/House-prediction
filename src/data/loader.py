from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd


class DataLoadError(ValueError):
    """Raised when a CSV cannot be safely loaded."""


def load_csv(source: str | Path | bytes | BinaryIO, max_size_mb: int = 100) -> pd.DataFrame:
    """Load and minimally validate a CSV from a path, bytes, or upload object."""
    try:
        if isinstance(source, bytes):
            if len(source) > max_size_mb * 1024 * 1024:
                raise DataLoadError(f"CSV exceeds the {max_size_mb} MB upload limit.")
            stream = BytesIO(source)
            df = pd.read_csv(stream)
        else:
            df = pd.read_csv(source)
    except (UnicodeDecodeError, pd.errors.ParserError, OSError) as exc:
        raise DataLoadError(f"Could not read this CSV: {exc}") from exc
    if df.empty:
        raise DataLoadError("The CSV has no data rows.")
    if len(df.columns) < 2:
        raise DataLoadError("The CSV must contain at least two columns.")
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise DataLoadError(f"Duplicate column names are not supported: {duplicates}")
    return df


def dataset_hash(df: pd.DataFrame) -> str:
    """Create a stable short key from schema and values."""
    digest = hashlib.sha256()
    digest.update("|".join(map(str, df.columns)).encode())
    digest.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return digest.hexdigest()[:12]
