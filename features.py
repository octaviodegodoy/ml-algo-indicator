"""
features.py — Feature engineering: technicals, candlestick patterns, regime,
VWAP, higher-timeframe context, microstructure loaders, and merge helpers.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

from config import TF_SECONDS, HTF_BARS, MIN_MICRO_ROWS, dom_path, ticks_path


# ── Low-level indicator helpers ───────────────────────────────────────────────
def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([
        (df['High'] - df['Low']),
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low']  - df['Close'].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ── Core OHLCV technicals ─────────────────────────────────────────────────────
def make_features(df: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for w in (1, 3, 5, 10, 20):
        out[f'{prefix}ret_{w}']    = df['Close'].pct_change(w)
        out[f'{prefix}logret_{w}'] = np.log(df['Close'] / df['Close'].shift(w))
    r1 = df['Close'].pct_change()
    for w in (5, 10, 20):
        out[f'{prefix}vol_{w}'] = r1.rolling(w).std()
    atr14 = _atr(df, 14)
    tr = pd.concat([
        (df['High'] - df['Low']),
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low']  - df['Close'].shift()).abs(),
    ], axis=1).max(axis=1)
    out[f'{prefix}atr_14']    = atr14
    out[f'{prefix}atr_ratio'] = tr / atr14
    out[f'{prefix}rsi_14']    = _rsi(df['Close'], 14)
    ma20 = df['Close'].rolling(20).mean()
    ma50 = df['Close'].rolling(50).mean()
    out[f'{prefix}ma20_dist']      = (df['Close'] - ma20) / ma20
    out[f'{prefix}ma50_dist']      = (df['Close'] - ma50) / ma50
    out[f'{prefix}ma20_50_spread'] = (ma20 - ma50) / ma50
    std20 = df['Close'].rolling(20).std()
    out[f'{prefix}bb_width'] = (4 * std20) / ma20
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        vol_ma20 = df['Volume'].rolling(20).mean()
        vol_std  = df['Volume'].rolling(20).std().replace(0, np.nan)
        out[f'{prefix}vol_z']     = (df['Volume'] - vol_ma20) / vol_std
        out[f'{prefix}vol_ratio'] = df['Volume'] / vol_ma20
    out = pd.concat([out, make_candle_features(df, prefix=prefix)], axis=1)
    out = add_regime_features(out, df, n=14)
    out = add_vwap_features(out, df, atr14)
    return out


def add_time_features(out: pd.DataFrame) -> pd.DataFrame:
    minutes        = out.index.hour * 60 + out.index.minute
    out['tod_sin'] = np.sin(2 * np.pi * minutes / (24 * 60))
    out['tod_cos'] = np.cos(2 * np.pi * minutes / (24 * 60))
    out['dow']     = out.index.dayofweek
    return out


# ── Candlestick pattern features ──────────────────────────────────────────────
def make_candle_features(df: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
    """Body/shadow ratios, directional streak, and basic pattern flags."""
    out    = pd.DataFrame(index=df.index)
    range_ = (df['High'] - df['Low']).replace(0, np.nan)
    body   = df['Close'] - df['Open']
    abs_b  = body.abs()
    norm_b = (abs_b / range_).fillna(0)

    out[f'{prefix}body_ratio']   = body / range_
    out[f'{prefix}upper_shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / range_
    out[f'{prefix}lower_shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low'])  / range_
    out[f'{prefix}candle_dir']   = np.sign(body)

    dirs: list = np.sign(body.fillna(0)).astype(int).tolist()
    streak: list = []
    s = 0
    for d in dirs:
        if   d > 0: s = s + 1 if s > 0 else 1
        elif d < 0: s = s - 1 if s < 0 else -1
        else:       s = 0
        streak.append(s)
    out[f'{prefix}dir_streak'] = streak

    lower_sh = out[f'{prefix}lower_shadow']
    upper_sh = out[f'{prefix}upper_shadow']
    out[f'{prefix}is_doji']       = (norm_b < 0.1).astype(float)
    out[f'{prefix}is_hammer']     = ((lower_sh > 2 * norm_b) & (upper_sh < norm_b) & (body < 0)).astype(float)
    out[f'{prefix}is_inv_hammer'] = ((upper_sh > 2 * norm_b) & (lower_sh < norm_b) & (body > 0)).astype(float)

    prev_body = body.shift(1)
    out[f'{prefix}bull_engulf'] = (
        (body > 0) & (prev_body < 0) &
        (df['Open'] < df['Close'].shift(1)) & (df['Close'] > df['Open'].shift(1))
    ).astype(float)
    out[f'{prefix}bear_engulf'] = (
        (body < 0) & (prev_body > 0) &
        (df['Open'] > df['Close'].shift(1)) & (df['Close'] < df['Open'].shift(1))
    ).astype(float)
    return out


# ── ADX / regime features ─────────────────────────────────────────────────────
def add_regime_features(out: pd.DataFrame, df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """ADX trend strength, DI difference, and price-channel position."""
    high, low, close = df['High'], df['Low'], df['Close']
    prev_c = close.shift(1)

    tr        = pd.concat([(high - low), (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    up_move   = (high - high.shift(1)).fillna(0)
    down_move = (low.shift(1) - low).fillna(0)

    plus_dm  = pd.Series(np.where((up_move > down_move)   & (up_move > 0),   up_move,   0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move)   & (down_move > 0), down_move, 0.0), index=df.index)

    atr_n    = tr.rolling(n).mean().replace(0, np.nan)
    plus_di  = 100 * plus_dm.rolling(n).mean()  / atr_n
    minus_di = 100 * minus_dm.rolling(n).mean() / atr_n
    out['adx_14']     = (100 * (plus_di - minus_di).abs() /
                         (plus_di + minus_di).replace(0, np.nan)).rolling(n).mean()
    out['di_diff_14'] = plus_di - minus_di

    for w in (20, 50):
        h_max = high.rolling(w).max()
        l_min = low.rolling(w).min()
        out[f'pct_range_{w}'] = (close - l_min) / (h_max - l_min).replace(0, np.nan)
    return out


# ── Intraday VWAP features ────────────────────────────────────────────────────
def add_vwap_features(out: pd.DataFrame, df: pd.DataFrame, atr14: pd.Series) -> pd.DataFrame:
    """Distance from daily VWAP normalised by ATR(14)."""
    if 'Volume' not in df.columns or df['Volume'].sum() == 0:
        return out
    typical  = (df['High'] + df['Low'] + df['Close']) / 3
    date_key = df.index.normalize()
    tpv      = (typical * df['Volume']).groupby(date_key).cumsum()
    cum_vol  = df['Volume'].groupby(date_key).cumsum().replace(0, np.nan)
    vwap     = tpv / cum_vol
    out['vwap_dist'] = (df['Close'] - vwap) / atr14.replace(0, np.nan)
    return out


# ── Higher-timeframe (H1) context features ────────────────────────────────────
def make_htf_features(symbol: str, bars_m5: pd.DataFrame) -> pd.DataFrame:
    """Fetch H1 bars, compute context features, and forward-fill onto the M5 index."""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, HTF_BARS)
    if rates is None or len(rates) == 0:
        return pd.DataFrame(index=bars_m5.index)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                       'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    if df.empty:
        return pd.DataFrame(index=bars_m5.index)

    htf = pd.DataFrame(index=df.index)
    htf['h1_ret_1']      = df['Close'].pct_change(1)
    htf['h1_ret_4']      = df['Close'].pct_change(4)
    htf['h1_rsi_14']     = _rsi(df['Close'], 14)
    htf['h1_ma20_dist']  = (df['Close'] - df['Close'].rolling(20).mean()) / \
                            df['Close'].rolling(20).mean().replace(0, np.nan)
    htf['h1_atr_norm']   = _atr(df, 14) / df['Close'].replace(0, np.nan)
    htf['h1_body_ratio'] = (df['Close'] - df['Open']) / \
                            (df['High'] - df['Low']).replace(0, np.nan)
    return htf.reindex(bars_m5.index, method='ffill')


# ── Microstructure feature loaders ───────────────────────────────────────────
def load_dom_features(slug: str) -> pd.DataFrame:
    path = dom_path(slug)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['ts']  = pd.to_datetime(df['ts'], unit='s', utc=True)
    df['bar'] = df['ts'].dt.floor(f'{TF_SECONDS}s')
    g = df.groupby('bar').agg(
        dom_spread_bps_mean  =('spread_bps',      'mean'),
        dom_spread_bps_max   =('spread_bps',      'max'),
        dom_top_imb_mean     =('top_imbalance',   'mean'),
        dom_top_imb_last     =('top_imbalance',   'last'),
        dom_depth_imb_mean   =('depth_imbalance', 'mean'),
        dom_depth_imb_last   =('depth_imbalance', 'last'),
        dom_bid_vol_mean     =('bid_vol_top',     'mean'),
        dom_ask_vol_mean     =('ask_vol_top',     'mean'),
        dom_snap_count       =('spread',          'size'),
        dom_wmid_drift_sum   =('wmid_drift',      'sum'),    # net weighted-mid pressure over the bar
        dom_wmid_drift_last  =('wmid_drift',      'last'),   # most recent directional push
        dom_book_refresh_rate=('book_refreshed',  'mean'),   # fraction of snapshots that saw a quote change
    )
    g.index.name = None
    # Normalised side volume ratio (bid pressure vs ask pressure, bounded [-1,+1])
    vol_sum = (g['dom_bid_vol_mean'] + g['dom_ask_vol_mean']).replace(0, np.nan)
    g['dom_vol_ratio'] = (g['dom_bid_vol_mean'] - g['dom_ask_vol_mean']) / vol_sum
    return g


def load_tick_features(slug: str) -> pd.DataFrame:
    path = ticks_path(slug)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['bar'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df = df.drop(columns=['ts'])
    g = df.groupby('bar').agg(
        tick_trade_count   =('trade_count',      'sum'),
        tick_buy_vol       =('buy_vol',          'sum'),
        tick_sell_vol      =('sell_vol',         'sum'),
        tick_delta_vol     =('delta_vol',        'sum'),
        tick_avg_price     =('avg_price',        'mean'),
        tick_last_price    =('last_price',       'last'),
        tick_absorption    =('absorption_ratio', 'mean'),   # lots per point moved
        tick_price_eff     =('price_efficiency', 'mean'),   # how directional the bar was
        tick_price_std     =('price_std',        'mean'),   # intra-bar price volatility
    )
    g['tick_buy_ratio'] = g['tick_buy_vol'] / (g['tick_buy_vol'] + g['tick_sell_vol']).replace(0, np.nan)
    g.index.name = None

    # CVD (Cumulative Volume Delta) — resets each calendar day
    g = g.sort_index()
    day_key = g.index.normalize()
    g['cvd']          = g.groupby(day_key)['tick_delta_vol'].cumsum()
    g['cvd_change_5'] = g['cvd'].diff(5)
    g['cvd_ma_10']    = g['tick_delta_vol'].rolling(10).mean()

    cvd_roll_mean      = g['cvd'].rolling(20).mean()
    cvd_roll_std       = g['cvd'].rolling(20).std().replace(0, np.nan)
    g['cvd_zscore_20'] = (g['cvd'] - cvd_roll_mean) / cvd_roll_std

    # CVD magnitude bins: replace binary divergence flag with signed strength (3 levels each side)
    price_dir = np.sign(g['tick_last_price'].diff(5))
    cvd_dir   = np.sign(g['cvd_change_5'])
    g['cvd_price_div'] = price_dir * cvd_dir   # -1 = divergence, +1 = confirmation

    # CVD magnitude: how large was the delta relative to total volume?
    total_vol = (g['tick_buy_vol'] + g['tick_sell_vol']).replace(0, np.nan)
    g['cvd_magnitude'] = g['tick_delta_vol'].abs() / total_vol   # 0=balanced, 1=all one side
    g['cvd_signed_mag'] = g['tick_delta_vol'] / total_vol         # signed: +1=all buy, -1=all sell

    # Absorption rolling z-score (normalise across session)
    if 'tick_absorption' in g.columns:
        abs_mean = g['tick_absorption'].rolling(20).mean()
        abs_std  = g['tick_absorption'].rolling(20).std().replace(0, np.nan)
        g['absorption_zscore'] = (g['tick_absorption'] - abs_mean) / abs_std

    return g


# ── Microstructure merge helper ───────────────────────────────────────────────
def merge_microstructure(
    feats: pd.DataFrame,
    micro: pd.DataFrame,
    prefix: str,
) -> Tuple[pd.DataFrame, int]:
    """Lag micro features by 1 bar and join onto feats. Returns (merged, n_rows_with_data)."""
    if micro.empty:
        return feats, 0
    micro    = micro.shift(1).reindex(feats.index)
    has_data = (~micro.isna().all(axis=1)).astype(int)
    n_rows   = int(has_data.sum())
    if n_rows < MIN_MICRO_ROWS:
        return feats, n_rows
    micro[f'{prefix}_has_data'] = has_data
    return pd.concat([feats, micro], axis=1), n_rows
