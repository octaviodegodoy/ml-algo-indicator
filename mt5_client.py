"""
mt5_client.py — MT5 lifecycle, bar fetching, DOM snapshots, and tick aggregation.
"""

import os
import time
import atexit
from typing import Optional

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

from config import TARGETS, DOM_LEVELS, TF_SECONDS, dom_path, ticks_path

# ── MT5 lifecycle ─────────────────────────────────────────────────────────────
_subscribed_books: list = []


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


def mt5_teardown() -> None:
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

    best_bid, best_bid_vol = bids[0]
    best_ask, best_ask_vol = asks[0]
    mid    = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid

    bid_vol_n = sum(v for _, v in bids[:DOM_LEVELS])
    ask_vol_n = sum(v for _, v in asks[:DOM_LEVELS])
    total     = bid_vol_n + ask_vol_n

    return {
        'ts':              int(time.time()),
        'spread':          spread,
        'spread_bps':      (spread / mid * 1e4) if mid > 0 else np.nan,
        'top_imbalance':   (best_bid_vol - best_ask_vol) / (best_bid_vol + best_ask_vol)
                           if (best_bid_vol + best_ask_vol) > 0 else 0.0,
        'depth_imbalance': (bid_vol_n - ask_vol_n) / total if total > 0 else 0.0,
        'bid_vol_top':     bid_vol_n,
        'ask_vol_top':     ask_vol_n,
        'best_bid':        best_bid,
        'best_ask':        best_ask,
    }


def append_dom_snapshot(symbol: str, slug: str) -> bool:
    snap = _snapshot_dom(symbol)
    if snap is None:
        return False
    path = dom_path(slug)
    pd.DataFrame([snap]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)
    return True


# ── Trade-tick aggregator ─────────────────────────────────────────────────────
_last_tick_ts: dict = {}  # per-symbol incremental cursor


def fetch_and_aggregate_ticks(symbol: str, slug: str) -> int:
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
    ).reset_index()
    agg['delta_vol'] = agg['buy_vol'] - agg['sell_vol']
    agg['ts']        = agg['bar'].astype('int64') // 10**9
    agg = agg[['ts', 'trade_count', 'buy_vol', 'sell_vol', 'delta_vol', 'avg_price', 'last_price']]

    path = ticks_path(slug)
    agg.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
    return len(agg)
