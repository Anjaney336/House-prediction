"""Reproducible synthetic-market and AutoML validation laboratory."""

from src.benchmark.generators import GENERATORS, MarketConfig, SyntheticDataset, generate_dataset

__all__ = ["GENERATORS", "MarketConfig", "SyntheticDataset", "generate_dataset"]
