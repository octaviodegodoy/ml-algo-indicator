"""
protocols.py — Structural typing contracts (PEP 544) for the main abstractions.

Using Protocol enables dependency inversion: high-level modules
(ml_signal_generator, trade) can depend on these interfaces instead of
concrete MT5 implementations, satisfying both ISP and DIP.
"""

from typing import Optional, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class BarFetcherProtocol(Protocol):
    """Narrow interface for anything that can supply OHLCV bar data."""

    def fetch_bars(self, symbol: str, timeframe: int, n: int) -> Optional[pd.DataFrame]:
        """Return a DataFrame with columns Open/High/Low/Close/Volume, or None."""
        ...

    def fetch_htf_bars(self, symbol: str) -> Optional[pd.DataFrame]:
        """Return higher-timeframe (H1) bars for context features, or None."""
        ...


@runtime_checkable
class MicrostructureProtocol(Protocol):
    """Narrow interface for real-time microstructure data collection."""

    def append_dom_snapshot(self, symbol: str, slug: str) -> bool:
        """Capture one DOM snapshot and append it to the per-slug CSV. Returns True on success."""
        ...

    def fetch_and_aggregate_ticks(self, symbol: str, slug: str) -> int:
        """Fetch new trade ticks since the last call and append bar-aggregations. Returns row count."""
        ...
