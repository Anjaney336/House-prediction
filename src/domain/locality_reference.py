from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.utils.config import ROOT_DIR


DEFAULT_REFERENCE = ROOT_DIR / "data" / "reference" / "localities.csv"


def locality_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold().replace("sector", "sec").replace("gurgaon", "gurugram"))


def load_locality_reference(path: str | Path = DEFAULT_REFERENCE) -> pd.DataFrame:
    columns = ["normalized_name", "alias", "city", "state", "country", "latitude", "longitude", "source", "verified_at"]
    reference = pd.read_csv(path)
    missing = set(columns) - set(reference.columns)
    if missing:
        raise ValueError(f"Locality reference is missing columns: {', '.join(sorted(missing))}.")
    reference = reference[columns].copy()
    reference["lookup_key"] = reference["alias"].map(locality_key)
    conflicting = reference.groupby("lookup_key")["normalized_name"].nunique()
    if (conflicting > 1).any():
        raise ValueError("A locality alias maps to more than one normalized locality.")
    return reference.drop_duplicates("lookup_key", keep="first")


def resolve_locality(value: str, reference: pd.DataFrame | None = None) -> dict | None:
    table = load_locality_reference() if reference is None else reference
    match = table.loc[table["lookup_key"] == locality_key(value)]
    if match.empty:
        return None
    row = match.iloc[0].drop(labels=["lookup_key"])
    return {key: (None if pd.isna(value) else value) for key, value in row.items()}
