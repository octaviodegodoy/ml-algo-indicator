"""
microstructure.py — DOM snapshot collection, background sampling thread, and
tick aggregation.  Single responsibility: real-time microstructure data capture.

Called by mt5_client.setup() / teardown() to integrate with the MT5 lifecycle.
"""

import os
import time
import threading
from typing import Optional

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

from config import (
    TARGETS,
    DOM_LEVELS, DOM_SAMPLE_SECS, DOM_MIN_LEVEL_LOTS, DOM_LARGE_ORDER_LOTS,
    TF_SECONDS, dom_path, ticks_path,
)

# ── Module-level state ────────────────────────────────────────────────────────
_subscribed_books:     list             = []
_dom_thread:           Optional[threading.Thread] = None
_dom_thread_stop:      threading.Event            = threading.Event()
_dom_schema_validated: set              = set()
_dom_csv_lock                           = threading.Lock()
_no_tick_symbols:      set              = set()

# Per-symbol state for refresh-rate tracking
_dom_prev_bid: dict = {}
_dom_prev_ask: dict = {}
_dom_prev_mid: dict = {}

_last_tick_ts: dict = {}   # per-symbol incremental cursor


# ── Lifecycle ─────────────────────────────────────────────────────────────────
def setup(targets: list) -> None:
    """Subscribe to DOM books for all targets, run broker capability check, start sampler."""
    for t in targets:
        s = t['symbol']
        if mt5.market_book_add(s):
            _subscribed_books.append(s)
            print(f"DOM subscribed: {s}")
        else:
            print(f"Warning: market_book_add({s}) failed — DOM features unavailable for it")
    _check_broker_capabilities(targets)
    _start_dom_thread()


def teardown() -> None:
    """Stop the background sampler and release all DOM subscriptions."""
    _dom_thread_stop.set()
    if _dom_thread is not None:
        _dom_thread.join(timeout=10)
    for s in _subscribed_books:
        try:
            mt5.market_book_release(s)
        except Exception:
            pass


# ── Broker capability check ───────────────────────────────────────────────────
def _check_broker_capabilities(targets: list) -> None:
    """One-time startup check: log which data sources are available per symbol."""
    now = int(time.time())
    print("── Broker capability check ──────────────────────────────")
    for t in targets:
        s      = t['symbol']
        book   = mt5.market_book_get(s)
        dom_ok = book is not None and len(book) > 0
        ticks  = mt5.copy_ticks_range(
            s,
            pd.to_datetime(now - 600, unit='s'),
            pd.to_datetime(now,       unit='s'),
            mt5.COPY_TICKS_TRADE,
        )
        tick_ok = ticks is not None and len(ticks) > 0
        print(
            f"  {s}:  DOM={'YES (' + str(len(book)) + ' levels)' if dom_ok else 'NO — L2 not provided by broker'}"
            f"  |  Ticks={'YES (' + str(len(ticks)) + ')' if tick_ok else 'NO — tick history not provided by broker'}"
        )
        if not tick_ok:
            _no_tick_symbols.add(s)
    print("─────────────────────────────────────────────────────────")


# ── DOM snapshot ──────────────────────────────────────────────────────────────
def _ensure_dom_schema(path: str, snap: dict) -> None:
    """Delete DOM CSV if its column schema no longer matches the current snapshot keys."""
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

    bids_f = [(p, v) for p, v in bids if v >= DOM_MIN_LEVEL_LOTS]
    asks_f = [(p, v) for p, v in asks if v >= DOM_MIN_LEVEL_LOTS]
    if not bids_f or not asks_f:
        bids_f, asks_f = bids, asks

    best_bid, best_bid_vol = bids_f[0]
    best_ask, best_ask_vol = asks_f[0]
    mid    = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid

    bid_vol_n = sum(v for _, v in bids_f[:DOM_LEVELS])
    ask_vol_n = sum(v for _, v in asks_f[:DOM_LEVELS])
    total     = bid_vol_n + ask_vol_n

    large_bids       = [(p, v) for p, v in bids_f[:DOM_LEVELS] if v >= DOM_LARGE_ORDER_LOTS]
    large_asks       = [(p, v) for p, v in asks_f[:DOM_LEVELS] if v >= DOM_LARGE_ORDER_LOTS]
    large_bid_levels = len(large_bids)
    large_ask_levels = len(large_asks)
    large_bid_vol    = sum(v for _, v in large_bids)
    large_ask_vol    = sum(v for _, v in large_asks)
    large_total      = large_bid_vol + large_ask_vol
    large_imbalance  = (large_bid_vol - large_ask_vol) / large_total if large_total > 0 else 0.0

    w_mid = (ask_vol_n * best_bid + bid_vol_n * best_ask) / total if total > 0 else mid

    prev_bid       = _dom_prev_bid.get(symbol, best_bid)
    prev_ask       = _dom_prev_ask.get(symbol, best_ask)
    book_refreshed = int(best_bid != prev_bid or best_ask != prev_ask)
    prev_mid       = _dom_prev_mid.get(symbol, w_mid)
    wmid_drift     = w_mid - prev_mid

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
        'large_bid_levels': large_bid_levels,
        'large_ask_levels': large_ask_levels,
        'large_bid_vol':    large_bid_vol,
        'large_ask_vol':    large_ask_vol,
        'large_imbalance':  large_imbalance,
    }


def append_dom_snapshot(symbol: str, slug: str) -> bool:
    """Capture one DOM snapshot and append it to the per-slug CSV. Returns True on success."""
    snap = _snapshot_dom(symbol)
    if snap is None:
        return False
    path = dom_path(slug)
    with _dom_csv_lock:
        _ensure_dom_schema(path, snap)
        pd.DataFrame([snap]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)
    return True


# ── Background DOM sampler ────────────────────────────────────────────────────
def _dom_sampling_thread() -> None:
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
                with _dom_csv_lock:
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
def fetch_and_aggregate_ticks(symbol: str, slug: str) -> int:
    """Fetch new trade ticks since the last call and append bar-aggregations. Returns row count."""
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

    price_range = (agg['max_price'] - agg['min_price']).replace(0, np.nan)
    agg['absorption_ratio'] = total_vol / price_range
    agg['price_efficiency'] = (agg['last_price'] - agg['first_price']).abs() / price_range.replace(np.nan, 1)
    agg['price_efficiency'] = agg['price_efficiency'].fillna(0)

    agg['ts'] = agg['bar'].astype('int64') // 10**9
    agg = agg[['ts', 'trade_count', 'buy_vol', 'sell_vol', 'delta_vol',
               'avg_price', 'last_price', 'absorption_ratio', 'price_efficiency', 'price_std']]

    path = ticks_path(slug)
    agg.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
    return len(agg)
