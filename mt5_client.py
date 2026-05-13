"""
mt5_client.py — MT5 lifecycle, bar fetching, DOM snapshots, and tick aggregation.
"""

import os
import time
import atexit
import threading
from typing import Optional

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

from config import TARGETS, DOM_LEVELS, DOM_SAMPLE_SECS, DOM_MIN_LEVEL_LOTS, DOM_LARGE_ORDER_LOTS, TF_SECONDS, dom_path, ticks_path

# ── MT5 lifecycle ─────────────────────────────────────────────────────────────
_subscribed_books:  list          = []
_dom_thread:        Optional[threading.Thread] = None
_dom_thread_stop:   threading.Event            = threading.Event()

# Tracks which DOM CSV paths have already been schema-validated this session
_dom_schema_validated: set = set()

# Symbols for which the broker does not provide tick history (detected at startup)
_no_tick_symbols: set = set()


def _ensure_dom_schema(path: str, snap: dict) -> None:
    """Delete DOM CSV if its column schema no longer matches the current snapshot keys.

    Runs at most once per file path per process lifetime to avoid repeated I/O.
    """
    if path in _dom_schema_validated:
        return
    _dom_schema_validated.add(path)
    if not os.path.exists(path):
        return
    try:
        with open(path, 'r') as _f:
            header_cols = _f.readline().strip().split(',')
        if set(header_cols) != set(snap.keys()):
            os.remove(path)
            print(f"DOM schema mismatch — stale file removed: {path}")
    except Exception:
        try:
            os.remove(path)
        except Exception:
            pass


def mt5_setup() -> None:
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    seen: set = set()
    for t in TARGETS:
        s = t['symbol']
        if s in seen:
            continue
        seen.add(s)
        if not mt5.symbol_select(s, True):
            print(f"Warning: could not select {s} in Market Watch")
        if mt5.market_book_add(s):
            _subscribed_books.append(s)
            print(f"DOM subscribed: {s}")
        else:
            print(f"Warning: market_book_add({s}) failed — DOM features unavailable for it")
    print(f"MT5 connected. Targets: {[t['symbol'] for t in TARGETS]}")
    _check_broker_capabilities()
    _start_dom_thread()


def _check_broker_capabilities() -> None:
    """One-time startup check: log which data sources are available per symbol."""
    import time as _time
    now = int(_time.time())
    print("── Broker capability check ──────────────────────────────")
    for t in TARGETS:
        s    = t['symbol']
        # DOM
        book = mt5.market_book_get(s)
        dom_ok = book is not None and len(book) > 0
        # Ticks (last 10 minutes)
        ticks = mt5.copy_ticks_range(
            s,
            pd.to_datetime(now - 600, unit='s'),
            pd.to_datetime(now,       unit='s'),
            mt5.COPY_TICKS_TRADE,
        )
        tick_ok = ticks is not None and len(ticks) > 0
        print(f"  {s}:  DOM={'YES (' + str(len(book)) + ' levels)' if dom_ok else 'NO — L2 not provided by broker'}  |  "
              f"Ticks={'YES (' + str(len(ticks)) + ')' if tick_ok else 'NO — tick history not provided by broker'}")
        if not tick_ok:
            # Mark symbol so fetch_and_aggregate_ticks skips it silently
            _no_tick_symbols.add(s)
    print("─────────────────────────────────────────────────────────")


def mt5_teardown() -> None:
    _dom_thread_stop.set()
    if _dom_thread is not None:
        _dom_thread.join(timeout=10)
    for s in _subscribed_books:
        try:
            mt5.market_book_release(s)
        except Exception:
            pass
    mt5.shutdown()
    print("MT5 disconnected.")


atexit.register(mt5_teardown)


# ── Bar fetch ─────────────────────────────────────────────────────────────────
def fetch_bars(symbol: str, timeframe, n: int) -> Optional[pd.DataFrame]:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                       'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()


# ── DOM snapshot collector ────────────────────────────────────────────────────
# Per-symbol state for refresh-rate and absorption tracking
_dom_prev_bid: dict = {}   # symbol → best_bid at last snapshot
_dom_prev_ask: dict = {}   # symbol → best_ask at last snapshot
_dom_prev_mid: dict = {}   # symbol → weighted mid at last snapshot


def _snapshot_dom(symbol: str) -> Optional[dict]:
    book = mt5.market_book_get(symbol)
    if book is None or len(book) == 0:
        return None

    bids = [(e.price, e.volume) for e in book if e.type == mt5.BOOK_TYPE_BUY]
    asks = [(e.price, e.volume) for e in book if e.type == mt5.BOOK_TYPE_SELL]
    if not bids or not asks:
        return None

    bids.sort(key=lambda x: -x[0])
    asks.sort(key=lambda x:  x[0])

    # Filter noise: drop levels below minimum lot threshold
    bids_f = [(p, v) for p, v in bids if v >= DOM_MIN_LEVEL_LOTS]
    asks_f = [(p, v) for p, v in asks if v >= DOM_MIN_LEVEL_LOTS]
    # Fall back to unfiltered if too thin (e.g. pre-market)
    if not bids_f or not asks_f:
        bids_f, asks_f = bids, asks

    best_bid, best_bid_vol = bids_f[0]
    best_ask, best_ask_vol = asks_f[0]
    mid    = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid

    bid_vol_n = sum(v for _, v in bids_f[:DOM_LEVELS])
    ask_vol_n = sum(v for _, v in asks_f[:DOM_LEVELS])
    total     = bid_vol_n + ask_vol_n

    # Large / iceberg order detection: levels ≥ DOM_LARGE_ORDER_LOTS
    large_bids = [(p, v) for p, v in bids_f[:DOM_LEVELS] if v >= DOM_LARGE_ORDER_LOTS]
    large_asks = [(p, v) for p, v in asks_f[:DOM_LEVELS] if v >= DOM_LARGE_ORDER_LOTS]
    large_bid_levels = len(large_bids)
    large_ask_levels = len(large_asks)
    large_bid_vol    = sum(v for _, v in large_bids)
    large_ask_vol    = sum(v for _, v in large_asks)
    large_total      = large_bid_vol + large_ask_vol
    large_imbalance  = (large_bid_vol - large_ask_vol) / large_total if large_total > 0 else 0.0

    # Weighted mid-price: fairer fair-value estimate than simple mid
    w_mid = (ask_vol_n * best_bid + bid_vol_n * best_ask) / total if total > 0 else mid

    # Book refresh rate: did the best quotes move since last snapshot?
    prev_bid      = _dom_prev_bid.get(symbol, best_bid)
    prev_ask      = _dom_prev_ask.get(symbol, best_ask)
    book_refreshed = int(best_bid != prev_bid or best_ask != prev_ask)

    # Weighted mid drift from last snapshot (directional pressure)
    prev_mid   = _dom_prev_mid.get(symbol, w_mid)
    wmid_drift = w_mid - prev_mid

    _dom_prev_bid[symbol] = best_bid
    _dom_prev_ask[symbol] = best_ask
    _dom_prev_mid[symbol] = w_mid

    return {
        'ts':               int(time.time()),
        'spread':           spread,
        'spread_bps':       (spread / mid * 1e4) if mid > 0 else np.nan,
        'top_imbalance':    (best_bid_vol - best_ask_vol) / (best_bid_vol + best_ask_vol)
                            if (best_bid_vol + best_ask_vol) > 0 else 0.0,
        'depth_imbalance':  (bid_vol_n - ask_vol_n) / total if total > 0 else 0.0,
        'bid_vol_top':      bid_vol_n,
        'ask_vol_top':      ask_vol_n,
        'best_bid':         best_bid,
        'best_ask':         best_ask,
        'weighted_mid':     w_mid,
        'wmid_drift':       wmid_drift,
        'book_refreshed':   book_refreshed,
        # Large-order / iceberg fields
        'large_bid_levels': large_bid_levels,
        'large_ask_levels': large_ask_levels,
        'large_bid_vol':    large_bid_vol,
        'large_ask_vol':    large_ask_vol,
        'large_imbalance':  large_imbalance,   # +1 = all big lots on bid, -1 = all on ask
    }


def append_dom_snapshot(symbol: str, slug: str) -> bool:
    """Called from main loop to confirm DOM is live; actual high-freq sampling is on background thread."""
    snap = _snapshot_dom(symbol)
    if snap is None:
        return False
    path = dom_path(slug)
    _ensure_dom_schema(path, snap)
    pd.DataFrame([snap]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)
    return True


def _dom_sampling_thread() -> None:
    """Background thread: samples DOM every DOM_SAMPLE_SECS for all subscribed symbols."""
    slug_map = {t['symbol']: t['slug'] for t in TARGETS}
    while not _dom_thread_stop.is_set():
        for symbol in _subscribed_books:
            slug = slug_map.get(symbol)
            if slug is None:
                continue
            snap = _snapshot_dom(symbol)
            if snap is None:
                continue
            path = dom_path(slug)
            try:
                _ensure_dom_schema(path, snap)
                pd.DataFrame([snap]).to_csv(
                    path, mode='a', header=not os.path.exists(path), index=False
                )
            except Exception:
                pass
        _dom_thread_stop.wait(timeout=DOM_SAMPLE_SECS)


def _start_dom_thread() -> None:
    global _dom_thread
    if not _subscribed_books:
        return
    _dom_thread_stop.clear()
    _dom_thread = threading.Thread(target=_dom_sampling_thread, name='dom_sampler', daemon=True)
    _dom_thread.start()
    print(f"DOM background sampler started (every {DOM_SAMPLE_SECS}s)")


# ── Trade-tick aggregator ─────────────────────────────────────────────────────
_last_tick_ts: dict = {}  # per-symbol incremental cursor


def fetch_and_aggregate_ticks(symbol: str, slug: str) -> int:
    if symbol in _no_tick_symbols:
        return 0
    now   = int(time.time())
    start = _last_tick_ts.get(symbol, 0) or (now - 3600)
    ticks = mt5.copy_ticks_range(
        symbol,
        pd.to_datetime(start, unit='s'),
        pd.to_datetime(now,   unit='s'),
        mt5.COPY_TICKS_TRADE,
    )
    _last_tick_ts[symbol] = now
    if ticks is None or len(ticks) == 0:
        return 0

    df = pd.DataFrame(ticks)
    if 'time_msc' in df.columns:
        df['time'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True)
    else:
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)

    flags         = df.get('flags', pd.Series(0, index=df.index))
    df['is_buy']  = ((flags & 4) > 0).astype(int)
    df['is_sell'] = ((flags & 8) > 0).astype(int)
    df['vol']     = df.get('volume', pd.Series(0, index=df.index)).astype(float)

    df['bar'] = df['time'].dt.floor(f'{TF_SECONDS}s')
    agg = df.groupby('bar').agg(
        trade_count=('vol', 'size'),
        buy_vol    =('vol', lambda v: v[df.loc[v.index, 'is_buy']  > 0].sum()),
        sell_vol   =('vol', lambda v: v[df.loc[v.index, 'is_sell'] > 0].sum()),
        avg_price  =('price', 'mean'),
        last_price =('price', 'last'),
        first_price=('price', 'first'),
        max_price  =('price', 'max'),
        min_price  =('price', 'min'),
        price_std  =('price', 'std'),
    ).reset_index()
    agg['delta_vol'] = agg['buy_vol'] - agg['sell_vol']
    total_vol        = agg['buy_vol'] + agg['sell_vol']

    # Absorption ratio: heavy volume relative to price movement within the bar
    # High ratio = large trades absorbed without proportional price move → hidden resistance/support
    price_range = (agg['max_price'] - agg['min_price']).replace(0, np.nan)
    agg['absorption_ratio'] = total_vol / price_range   # lots per point moved

    # Price efficiency: how much of the bar range was actually used (tight = contested, wide = breakout)
    agg['price_efficiency'] = (agg['last_price'] - agg['first_price']).abs() / price_range.replace(np.nan, 1)
    agg['price_efficiency'] = agg['price_efficiency'].fillna(0)

    agg['ts'] = agg['bar'].astype('int64') // 10**9
    agg = agg[['ts', 'trade_count', 'buy_vol', 'sell_vol', 'delta_vol',
               'avg_price', 'last_price', 'absorption_ratio', 'price_efficiency', 'price_std']]

    path = ticks_path(slug)
    agg.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
    return len(agg)
