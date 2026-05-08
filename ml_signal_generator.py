# ML signal generator for WIN$N (mini Ibovespa future)
#
# Trains a model on WIN bars and outputs a signals CSV into MQL5/Files,
# so PlotMLSignals.mq5 can be loaded on the WIN chart to read it.
#
# Features:
#   - OHLCV technicals (multi-window returns, vol, ATR, RSI, MAs, BB, volume)
#   - Time-of-day & day-of-week
#   - Triple-barrier labels with embargoed walk-forward CV
#   - Order book (DOM) snapshots — top-of-book imbalance & spread
#   - Trade tick aggregates — buy/sell volume, delta, trade count
#
# Microstructure features (DOM + ticks) are LIVE-ONLY: they accumulate into
# per-symbol sidecar CSVs. Historical bars get NaN, median-imputed; a
# `*_has_data` marker lets the model distinguish missing from neutral.
#
# Requirements: MetaTrader5, pandas, numpy, scikit-learn
# pip install MetaTrader5 pandas numpy scikit-learn

import os
import time
import atexit
from typing import Optional
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ── Config ────────────────────────────────────────────────────────────────────
# Each target produces one CSV. `slug` controls output filenames
# (must match SignalFile input on the indicator).
TARGETS = [
    {'symbol': 'WINM26', 'slug': 'win'},
]

TIMEFRAME         = mt5.TIMEFRAME_M5
TF_SECONDS        = 5 * 60
N_BARS            = 5000

# Triple-barrier params (in ATR multiples)
TB_MAX_BARS       = 12           # vertical barrier: ~1 hour on M5
TB_PT_MULT        = 1.5          # profit target = 1.5 × ATR(14)
TB_SL_MULT        = 1.0          # stop loss    = 1.0 × ATR(14)

PROB_THRESHOLD    = 0.50
INTERVAL_SECONDS  = 60
N_SPLITS_CV       = 5

DOM_LEVELS        = 5            # top-N book levels to aggregate
MIN_MICRO_ROWS    = 50           # need ≥ this many bars with micro data before using features
RECENCY_DECAY     = 2.0          # exponential recency weighting: newest bars ~7x heavier than oldest
HTF_BARS          = 500          # H1 bars to fetch for higher-timeframe context

# Freeze history: once a bar is older than the current forming bar, its signal
# is locked in the CSV and never overwritten on subsequent runs. Only the
# current (still-forming) bar gets re-scored each iteration.
FREEZE_HISTORY    = True

# ── Trade execution config ────────────────────────────────────────────────────
# SAFETY: keep TRADE_ENABLED = False until you have verified signals visually.
TRADE_ENABLED     = False   # set True to allow real orders
TRADE_BOTH_SIDES  = False   # True  = signal 1 → long, signal 0 → short
                            # False = signal 1 → long, signal 0 → close long only
RISK_PCT          = 1.0     # % of account balance risked per trade
MAX_SLIPPAGE      = 10      # maximum allowed slippage in points
MAGIC_NUMBER      = 20260507  # unique tag for orders placed by this script

# ── Paths ─────────────────────────────────────────────────────────────────────
_files_dir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Files')
)
os.makedirs(_files_dir, exist_ok=True)


def out_path(slug: str)   -> str: return os.path.join(_files_dir, f'{slug}_ml_signals.csv')
def dom_path(slug: str)   -> str: return os.path.join(_files_dir, f'{slug}_dom_snapshots.csv')
def ticks_path(slug: str) -> str: return os.path.join(_files_dir, f'{slug}_tick_agg.csv')


# ── MT5 lifecycle (initialize ONCE, keep open) ────────────────────────────────
_subscribed_books: list = []

def mt5_setup():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    seen = set()
    for t in TARGETS:
        s = t['symbol']
        if s in seen:
            continue
        seen.add(s)
        if not mt5.symbol_select(s, True):
            print(f"Warning: could not select {s} in Market Watch")
        if mt5.market_book_add(t['symbol']):
            _subscribed_books.append(t['symbol'])
            print(f"DOM subscribed: {t['symbol']}")
        else:
            print(f"Warning: market_book_add({t['symbol']}) failed — DOM features unavailable for it")
    print(f"MT5 connected. Targets: {[t['symbol'] for t in TARGETS]}")


def mt5_teardown():
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
def snapshot_dom(symbol: str) -> Optional[dict]:
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
    snap = snapshot_dom(symbol)
    if snap is None:
        return False
    path = dom_path(slug)
    pd.DataFrame([snap]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)
    return True


# ── Trade-tick aggregator ─────────────────────────────────────────────────────
_last_tick_ts: dict = {}  # per-symbol incremental cursor


def fetch_and_aggregate_ticks(symbol: str, slug: str) -> int:
    now = int(time.time())
    start = _last_tick_ts.get(symbol, 0) or (now - 3600)
    ticks = mt5.copy_ticks_range(symbol,
                                 pd.to_datetime(start, unit='s'),
                                 pd.to_datetime(now,   unit='s'),
                                 mt5.COPY_TICKS_TRADE)
    _last_tick_ts[symbol] = now
    if ticks is None or len(ticks) == 0:
        return 0

    df = pd.DataFrame(ticks)
    if 'time_msc' in df.columns:
        df['time'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True)
    else:
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)

    flags = df.get('flags', pd.Series(0, index=df.index))
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
    agg['ts'] = agg['bar'].astype('int64') // 10**9
    agg = agg[['ts', 'trade_count', 'buy_vol', 'sell_vol', 'delta_vol',
               'avg_price', 'last_price']]

    path = ticks_path(slug)
    agg.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
    return len(agg)


# ── Microstructure feature loaders ────────────────────────────────────────────
def load_dom_features(slug: str) -> pd.DataFrame:
    path = dom_path(slug)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['ts'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df['bar'] = df['ts'].dt.floor(f'{TF_SECONDS}s')
    g = df.groupby('bar').agg(
        dom_spread_bps_mean=('spread_bps', 'mean'),
        dom_spread_bps_max =('spread_bps', 'max'),
        dom_top_imb_mean   =('top_imbalance', 'mean'),
        dom_top_imb_last   =('top_imbalance', 'last'),
        dom_depth_imb_mean =('depth_imbalance', 'mean'),
        dom_depth_imb_last =('depth_imbalance', 'last'),
        dom_bid_vol_mean   =('bid_vol_top', 'mean'),
        dom_ask_vol_mean   =('ask_vol_top', 'mean'),
        dom_snap_count     =('spread', 'size'),
    )
    g.index.name = None
    return g


def load_tick_features(slug: str) -> pd.DataFrame:
    path = ticks_path(slug)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['bar'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df = df.drop(columns=['ts'])
    g = df.groupby('bar').agg(
        tick_trade_count=('trade_count', 'sum'),
        tick_buy_vol    =('buy_vol',     'sum'),
        tick_sell_vol   =('sell_vol',    'sum'),
        tick_delta_vol  =('delta_vol',   'sum'),
        tick_avg_price  =('avg_price',   'mean'),
        tick_last_price =('last_price',  'last'),
    )
    g['tick_buy_ratio'] = g['tick_buy_vol'] / (g['tick_buy_vol'] + g['tick_sell_vol']).replace(0, np.nan)
    g.index.name = None

    # ── CVD (Cumulative Volume Delta) ──────────────────────────────────────────
    # Resets each calendar day so sessions are comparable
    g = g.sort_index()
    day_key = g.index.normalize()
    g['cvd'] = g.groupby(day_key)['tick_delta_vol'].cumsum()

    # 5-bar change in CVD: short-term delta momentum
    g['cvd_change_5'] = g['cvd'].diff(5)

    # 10-bar smoothed delta (noise reduction)
    g['cvd_ma_10'] = g['tick_delta_vol'].rolling(10).mean()

    # Rolling z-score of CVD over 20 bars: normalises magnitude across sessions
    cvd_roll_mean = g['cvd'].rolling(20).mean()
    cvd_roll_std  = g['cvd'].rolling(20).std().replace(0, np.nan)
    g['cvd_zscore_20'] = (g['cvd'] - cvd_roll_mean) / cvd_roll_std

    # Divergence: compare 5-bar price direction vs 5-bar CVD direction
    # +1 = confirming (both agree), -1 = diverging (price vs delta mismatch)
    price_dir = np.sign(g['tick_last_price'].diff(5))
    cvd_dir   = np.sign(g['cvd_change_5'])
    g['cvd_price_div'] = price_dir * cvd_dir   # -1 = divergence signal

    return g


# ── Standard feature engineering ──────────────────────────────────────────────
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
    out[f'{prefix}rsi_14'] = _rsi(df['Close'], 14)
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
    # Candlestick, regime and VWAP features enrich the bar description
    out = pd.concat([out, make_candle_features(df, prefix=prefix)], axis=1)
    out = add_regime_features(out, df, n=14)
    out = add_vwap_features(out, df, atr14)
    return out


def add_time_features(out: pd.DataFrame) -> pd.DataFrame:
    minutes = out.index.hour * 60 + out.index.minute
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

    # Consecutive bullish / bearish streak (+N = N bullish in a row, -N = N bearish)
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

    tr = pd.concat([(high - low), (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    up_move   = (high - high.shift(1)).fillna(0)
    down_move = (low.shift(1) - low).fillna(0)

    plus_dm  = pd.Series(np.where((up_move > down_move)   & (up_move > 0),   up_move,   0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move)   & (down_move > 0), down_move, 0.0), index=df.index)

    atr_n    = tr.rolling(n).mean().replace(0, np.nan)
    plus_di  = 100 * plus_dm.rolling(n).mean()  / atr_n
    minus_di = 100 * minus_dm.rolling(n).mean() / atr_n
    out['adx_14']     = (100 * (plus_di - minus_di).abs() /
                         (plus_di + minus_di).replace(0, np.nan)).rolling(n).mean()
    out['di_diff_14'] = plus_di - minus_di        # positive = bullish bias

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
    # Forward-fill H1 values onto M5 bar timestamps (last known H1 bar carries forward)
    return htf.reindex(bars_m5.index, method='ffill')


# ── Triple-barrier labeling ───────────────────────────────────────────────────
def triple_barrier_labels(df: pd.DataFrame, atr: pd.Series,
                          max_bars: int, pt_mult: float, sl_mult: float) -> pd.Series:
    close = df['Close'].values
    high  = df['High'].values
    low   = df['Low'].values
    a     = atr.values
    n     = len(close)
    labels = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if np.isnan(a[i]) or a[i] <= 0:
            continue
        upper = close[i] + pt_mult * a[i]
        lower = close[i] - sl_mult * a[i]
        end = min(i + 1 + max_bars, n)
        hit_tp = -1
        hit_sl = -1
        for j in range(i + 1, end):
            if high[j] >= upper and hit_tp == -1:
                hit_tp = j
            if low[j] <= lower and hit_sl == -1:
                hit_sl = j
            if hit_tp != -1 or hit_sl != -1:
                break
        if hit_tp != -1 and (hit_sl == -1 or hit_tp < hit_sl):
            labels[i] = 1
    labels = labels.astype(float)
    labels[-max_bars:] = np.nan
    return pd.Series(labels, index=df.index, name='y')


# ── Walk-forward CV ───────────────────────────────────────────────────────────
def compute_recency_weights(n: int, decay: float = RECENCY_DECAY) -> np.ndarray:
    """Exponential recency weights so that the newest bar has e^decay × more weight than the oldest."""
    t = np.arange(n, dtype=float)
    w = np.exp(decay * (t - (n - 1)) / n)
    return (w / w.mean()).astype(np.float32)


def evaluate_walkforward(X: pd.DataFrame, y: pd.Series, n_splits: int,
                         embargo: int,
                         weights: Optional[np.ndarray] = None) -> dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs, precs, recs = [], [], []
    for train_idx, test_idx in tscv.split(X):
        if embargo > 0 and len(train_idx) > embargo:
            train_idx = train_idx[:-embargo]
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        if ytr.nunique() < 2:
            continue
        sw = weights[train_idx] if weights is not None else None
        m = Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('gb',  LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                   class_weight='balanced', random_state=42, verbosity=-1)),
        ])
        m.fit(Xtr, ytr, gb__sample_weight=sw)
        pred = m.predict(Xte)
        accs.append(accuracy_score(yte, pred))
        precs.append(precision_score(yte, pred, zero_division=0))
        recs.append(recall_score(yte, pred, zero_division=0))
    if not accs:
        return {'accuracy': float('nan'), 'precision': float('nan'),
                'recall': float('nan'), 'baseline_rate': float(y.mean())}
    return {
        'accuracy':      float(np.mean(accs)),
        'precision':     float(np.mean(precs)),
        'recall':        float(np.mean(recs)),
        'baseline_rate': float(y.mean()),
    }


def merge_microstructure(feats: pd.DataFrame, micro: pd.DataFrame, prefix: str):
    """Add micro features lagged 1 bar. Returns (merged_feats, n_rows_with_data)."""
    if micro.empty:
        return feats, 0
    micro = micro.shift(1).reindex(feats.index)
    has_data = (~micro.isna().all(axis=1)).astype(int)
    n_rows = int(has_data.sum())
    if n_rows < MIN_MICRO_ROWS:
        return feats, n_rows
    micro[f'{prefix}_has_data'] = has_data
    return pd.concat([feats, micro], axis=1), n_rows


def compute_sl_points(signal_series: pd.Series, bars: pd.DataFrame,
                      atr14: pd.Series, sl_mult: float, max_bars: int) -> pd.Series:
    """
    Computes a fixed SL distance using the median ATR(14) over the entire training
    lookback (all N_BARS), scaled by sl_mult.  This is stable across runs because it
    uses thousands of bars rather than the small number of signal transitions.
    Every transition bar (buy or sell) receives the same fixed value.
    Non-transition bars receive 0 (not displayed on chart).
    """
    # Full-history median ATR — one stable number for the whole session
    atr_full = atr14.reindex(signal_series.index).values
    fixed_sl = float(np.nanmedian(atr_full) * sl_mult) if np.any(~np.isnan(atr_full)) else 0.0
    print(f"  SL (fixed median-ATR×{sl_mult}) = {fixed_sl:.0f}p  "
          f"(computed over {np.sum(~np.isnan(atr_full))} bars)")

    sig = signal_series.values
    n   = len(sig)
    sl  = np.zeros(n, dtype=float)
    for i in range(1, n):
        prev, curr = int(sig[i - 1]), int(sig[i])
        if (curr == 1 and prev == 0) or (curr == 0 and prev == 1):
            sl[i] = fixed_sl

    return pd.Series(np.round(sl, 0).astype(int), index=signal_series.index, name='SL_Points')


# ── Trade execution ──────────────────────────────────────────────────────────
_last_exec_signal: dict = {}   # symbol → last signal that triggered an order


def _get_lot_size(symbol: str, sl_price_units: float) -> float:
    """Risk RISK_PCT% of account balance.  Falls back to minimum lot on errors."""
    info    = mt5.symbol_info(symbol)
    account = mt5.account_info()
    if info is None or account is None or sl_price_units <= 0:
        return info.volume_min if info else 1.0
    tick_size  = info.trade_tick_size  if info.trade_tick_size  > 0 else 1.0
    tick_value = info.trade_tick_value if info.trade_tick_value > 0 else 0.20
    point_value_per_lot = tick_value / tick_size   # account-currency per price-unit per lot
    risk_amount = account.balance * RISK_PCT / 100.0
    raw_lot     = risk_amount / (sl_price_units * point_value_per_lot)
    step        = info.volume_step if info.volume_step > 0 else 0.01
    lot = round(raw_lot / step) * step
    lot = max(info.volume_min, min(info.volume_max, lot))
    return round(lot, 2)


def _close_symbol_position(symbol: str) -> bool:
    """Close all open positions for symbol tagged with MAGIC_NUMBER."""
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return True
    closed_all = True
    for pos in positions:
        if pos.magic != MAGIC_NUMBER:
            continue
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            closed_all = False
            continue
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price      = tick.bid              if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        request = {
            'action':       mt5.TRADE_ACTION_DEAL,
            'symbol':       symbol,
            'volume':       pos.volume,
            'type':         close_type,
            'position':     pos.ticket,
            'price':        price,
            'deviation':    MAX_SLIPPAGE,
            'magic':        MAGIC_NUMBER,
            'comment':      'ml_close',
            'type_time':    mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err = result.retcode if result else mt5.last_error()
            print(f"  [trade] close failed for #{pos.ticket}: retcode={err}")
            closed_all = False
        else:
            print(f"  [trade] closed #{pos.ticket} {pos.volume} lots @ {price:.2f}")
    return closed_all


def _validate_stops(price: float, sl: float, tp: float, order_type: int,
                    info) -> tuple:
    """
    Ensure SL and TP satisfy:
      1. Correct side of price (SL below entry for buys, above for sells; TP opposite)
      2. At least `min_dist` away from price (broker stops_level + freeze_level, in price units)
      3. Snapped to the nearest tick_size

    Returns (sl, tp, warnings: list[str]).
    Any adjustment is logged so the caller can audit it.
    """
    warnings_out: list = []
    point     = info.point          if info.point          > 0 else 0.01
    tick_size = info.trade_tick_size if info.trade_tick_size > 0 else point
    digits    = info.digits

    # Broker minimum stop distance in price units
    stops_lvl   = int(getattr(info, 'trade_stops_level',  0) or 0)
    freeze_lvl  = int(getattr(info, 'trade_freeze_level', 0) or 0)
    min_dist    = (stops_lvl + freeze_lvl) * point
    # Add one extra tick as safety margin
    min_dist   += tick_size

    is_buy = (order_type == mt5.ORDER_TYPE_BUY)

    # ── Snap to tick size ────────────────────────────────────────────────────
    def snap(v: float) -> float:
        return round(round(v / tick_size) * tick_size, digits)

    sl = snap(sl)
    tp = snap(tp)

    # ── Direction sanity ─────────────────────────────────────────────────────
    if is_buy:
        if sl >= price:
            new_sl = snap(price - min_dist)
            warnings_out.append(f"SL {sl:.{digits}f} >= price {price:.{digits}f}; clamped to {new_sl:.{digits}f}")
            sl = new_sl
        if tp <= price:
            new_tp = snap(price + min_dist * 2)
            warnings_out.append(f"TP {tp:.{digits}f} <= price {price:.{digits}f}; clamped to {new_tp:.{digits}f}")
            tp = new_tp
    else:
        if sl <= price:
            new_sl = snap(price + min_dist)
            warnings_out.append(f"SL {sl:.{digits}f} <= price {price:.{digits}f}; clamped to {new_sl:.{digits}f}")
            sl = new_sl
        if tp >= price:
            new_tp = snap(price - min_dist * 2)
            warnings_out.append(f"TP {tp:.{digits}f} >= price {price:.{digits}f}; clamped to {new_tp:.{digits}f}")
            tp = new_tp

    # ── Minimum distance from price ──────────────────────────────────────────
    if is_buy:
        if (price - sl) < min_dist:
            new_sl = snap(price - min_dist)
            warnings_out.append(f"SL too close ({price-sl:.{digits}f} < {min_dist:.{digits}f}); widened to {new_sl:.{digits}f}")
            sl = new_sl
        if (tp - price) < min_dist:
            new_tp = snap(price + min_dist * 2)
            warnings_out.append(f"TP too close ({tp-price:.{digits}f} < {min_dist:.{digits}f}); widened to {new_tp:.{digits}f}")
            tp = new_tp
    else:
        if (sl - price) < min_dist:
            new_sl = snap(price + min_dist)
            warnings_out.append(f"SL too close ({sl-price:.{digits}f} < {min_dist:.{digits}f}); widened to {new_sl:.{digits}f}")
            sl = new_sl
        if (price - tp) < min_dist:
            new_tp = snap(price - min_dist * 2)
            warnings_out.append(f"TP too close ({price-tp:.{digits}f} < {min_dist:.{digits}f}); widened to {new_tp:.{digits}f}")
            tp = new_tp

    return sl, tp, warnings_out


def execute_trade(symbol: str, current_signal: int, sl_price_units: float) -> None:
    """Send an order when signal transitions; respects TRADE_ENABLED flag."""
    if not TRADE_ENABLED:
        return
    prev_signal = _last_exec_signal.get(symbol, -1)
    if current_signal == prev_signal:
        return   # no transition — nothing to do

    _last_exec_signal[symbol] = current_signal
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if tick is None or info is None:
        print(f"  [trade] symbol info unavailable for {symbol}")
        return

    tp_mult = TB_PT_MULT / TB_SL_MULT   # e.g. 1.5 / 1.0 = 1.5× the SL distance

    if current_signal == 1:   # ── BUY signal ──────────────────────────────────
        _close_symbol_position(symbol)   # close any prior short
        lot   = _get_lot_size(symbol, sl_price_units)
        price = tick.ask
        sl_raw = price - sl_price_units
        tp_raw = price + sl_price_units * tp_mult
        sl, tp, warns = _validate_stops(price, sl_raw, tp_raw, mt5.ORDER_TYPE_BUY, info)
        for w in warns:
            print(f"  [trade] stop-adjust BUY: {w}")
        request = {
            'action':       mt5.TRADE_ACTION_DEAL,
            'symbol':       symbol,
            'volume':       lot,
            'type':         mt5.ORDER_TYPE_BUY,
            'price':        price,
            'sl':           sl,
            'tp':           tp,
            'deviation':    MAX_SLIPPAGE,
            'magic':        MAGIC_NUMBER,
            'comment':      'ml_buy',
            'type_time':    mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err = result.retcode if result else mt5.last_error()
            print(f"  [trade] BUY failed: retcode={err}  price={price}  SL={sl}  TP={tp}")
        else:
            print(f"  [trade] BUY  {lot} lots {symbol} @ {price:.{info.digits}f}  SL={sl:.{info.digits}f}  TP={tp:.{info.digits}f}")

    elif current_signal == 0:  # ── EXIT / SELL signal ───────────────────────────
        _close_symbol_position(symbol)   # always close existing long
        if TRADE_BOTH_SIDES:
            lot   = _get_lot_size(symbol, sl_price_units)
            price = tick.bid
            sl_raw = price + sl_price_units
            tp_raw = price - sl_price_units * tp_mult
            sl, tp, warns = _validate_stops(price, sl_raw, tp_raw, mt5.ORDER_TYPE_SELL, info)
            for w in warns:
                print(f"  [trade] stop-adjust SELL: {w}")
            request = {
                'action':       mt5.TRADE_ACTION_DEAL,
                'symbol':       symbol,
                'volume':       lot,
                'type':         mt5.ORDER_TYPE_SELL,
                'price':        price,
                'sl':           sl,
                'tp':           tp,
                'deviation':    MAX_SLIPPAGE,
                'magic':        MAGIC_NUMBER,
                'comment':      'ml_sell',
                'type_time':    mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                err = result.retcode if result else mt5.last_error()
                print(f"  [trade] SELL failed: retcode={err}  price={price}  SL={sl}  TP={tp}")
            else:
                print(f"  [trade] SELL {lot} lots {symbol} @ {price:.{info.digits}f}  SL={sl:.{info.digits}f}  TP={tp:.{info.digits}f}")


# ── Per-target processing ─────────────────────────────────────────────────────
def process_target(target: dict) -> str:
    symbol  = target['symbol']
    slug    = target['slug']

    # 1. Live microstructure snapshot for this symbol
    dom_ok      = append_dom_snapshot(symbol, slug)
    n_tick_rows = fetch_and_aggregate_ticks(symbol, slug)

    # 2. Bars
    bars = fetch_bars(symbol, TIMEFRAME, N_BARS)
    if bars is None:
        return f"[{symbol}] no bars"

    # 3. Features
    feats = make_features(bars)
    feats = add_time_features(feats)

    feats, n_dom_used  = merge_microstructure(feats, load_dom_features(slug),  prefix='dom')
    feats, n_tick_used = merge_microstructure(feats, load_tick_features(slug), prefix='tick')

    # Higher-timeframe (H1) context — gives the model "last days" candle perspective
    htf_feats = make_htf_features(symbol, bars)
    if not htf_feats.empty:
        feats = pd.concat([feats, htf_feats], axis=1)

    # 4. Labels
    atr14  = _atr(bars, 14)
    target_y = triple_barrier_labels(bars, atr14, TB_MAX_BARS, TB_PT_MULT, TB_SL_MULT)

    # 5. Align
    aligned = pd.concat([feats, target_y], axis=1).dropna(subset=['y'])
    aligned = aligned.dropna(subset=feats.columns.tolist(),
                             thresh=int(len(feats.columns) * 0.6))
    if len(aligned) < 500:
        return f"[{symbol}] not enough rows: {len(aligned)}"

    X = aligned.drop(columns=['y'])
    y = aligned['y'].astype(int)

    buy_rate = float(y.mean())
    print(f"[{symbol}] label distribution: buy={buy_rate:.1%}  sell={1-buy_rate:.1%}  n={len(y)}")

    # Recency weights: exponential decay — recent bars influence training more
    rec_weights = compute_recency_weights(len(aligned))
    metrics = evaluate_walkforward(X, y, N_SPLITS_CV, embargo=TB_MAX_BARS,
                                   weights=rec_weights)

    model = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('gb',  LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                               class_weight='balanced', random_state=42, verbosity=-1)),
    ])
    model.fit(X, y, gb__sample_weight=rec_weights)

    proba  = model.predict_proba(X)[:, 1]
    print(f"[{symbol}] proba stats: min={proba.min():.3f} max={proba.max():.3f} mean={proba.mean():.3f} pct>threshold={( proba > PROB_THRESHOLD).mean():.1%}")
    signal = (proba > PROB_THRESHOLD).astype(int)

    sig_series = pd.Series(signal, index=X.index)
    sl_pts     = compute_sl_points(sig_series, bars, atr14, TB_SL_MULT, TB_MAX_BARS)

    # Fixed SL value — same for every transition (read from first non-zero entry)
    _nonzero = sl_pts[sl_pts > 0]
    avg_sl_val = int(_nonzero.iloc[0]) if len(_nonzero) > 0 else 0

    new_df = pd.DataFrame({
        'Timestamp': X.index.astype('int64') // 10**9,
        'ML_Signal': signal,
        'SL_Points': sl_pts.values,
    })

    # Score the unlabeled tail (last TB_MAX_BARS bars that lacked future bars
    # for triple-barrier labeling) so today's live bars appear in the CSV.
    tail_feats = feats[~feats.index.isin(aligned.index)]
    tail_feats = tail_feats.dropna(thresh=int(len(feats.columns) * 0.6))
    if len(tail_feats) > 0:
        tail_proba  = model.predict_proba(tail_feats)[:, 1]
        tail_signal = (tail_proba > PROB_THRESHOLD).astype(int)
        # Use the same fixed SL value for tail transition bars
        prev_sig = int(signal[-1]) if len(signal) > 0 else 0
        tail_sl = []
        for k, ts in enumerate(tail_signal):
            curr_sig = int(ts)
            if (curr_sig == 1 and prev_sig == 0) or (curr_sig == 0 and prev_sig == 1):
                tail_sl.append(avg_sl_val)
            else:
                tail_sl.append(0)
            prev_sig = curr_sig
        tail_df = pd.DataFrame({
            'Timestamp': tail_feats.index.astype('int64') // 10**9,
            'ML_Signal': tail_signal,
            'SL_Points': tail_sl,
        })
        new_df = pd.concat([new_df, tail_df], ignore_index=True)
        print(f"[{symbol}] tail bars scored: {len(tail_df)}  buys={int(tail_signal.sum())}")

    csv_path = out_path(slug)
    # If the existing CSV contains no buy signals at all, it is stale/corrupt;
    # delete it so we regenerate fully instead of inheriting all-zero frozen rows.
    if os.path.exists(csv_path):
        try:
            _old_check = pd.read_csv(csv_path)
            if (_old_check['ML_Signal'] == 1).sum() == 0:
                os.remove(csv_path)
                print(f"[{symbol}] stale all-zero CSV removed — regenerating from scratch")
        except Exception:
            pass
    n_frozen = 0
    if FREEZE_HISTORY and os.path.exists(csv_path):
        try:
            old_df = pd.read_csv(csv_path)
            # Ensure legacy CSVs without SL_Points don't corrupt column alignment
            if 'SL_Points' not in old_df.columns:
                old_df['SL_Points'] = 0
            # Lock in every closed bar from the previous CSV. Only the LAST
            # row of the old CSV is considered the (then-)forming bar and gets
            # overwritten by the fresh prediction.
            if len(old_df) > 1:
                frozen = old_df.iloc[:-1]
                # Keep new rows whose Timestamp is NOT already frozen.
                new_df = new_df[~new_df['Timestamp'].isin(frozen['Timestamp'])]
                merged = pd.concat([frozen, new_df], ignore_index=True)
                merged = merged.drop_duplicates(subset='Timestamp', keep='last')
                merged = merged.sort_values('Timestamp')
                merged['SL_Points'] = merged['SL_Points'].fillna(0).astype(int)
                n_frozen = len(frozen)
                merged.to_csv(csv_path, index=False)
            else:
                new_df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"[{symbol}] freeze-merge failed ({e}); rewriting full CSV")
            new_df.to_csv(csv_path, index=False)
    else:
        new_df.to_csv(csv_path, index=False)

    # Execute trade on confirmed signal transition
    latest_signal = int(new_df.sort_values('Timestamp').iloc[-1]['ML_Signal'])
    execute_trade(symbol, latest_signal, avg_sl_val)

    extras = []
    if n_dom_used  >= MIN_MICRO_ROWS:  extras.append(f'+dom({n_dom_used})')
    if n_tick_used >= MIN_MICRO_ROWS:  extras.append(f'+tick({n_tick_used})')
    edge = metrics['precision'] - metrics['baseline_rate']
    trade_tag = f"trade={'ON' if TRADE_ENABLED else 'OFF'}(sig={latest_signal})"
    freeze_tag = f"frozen={n_frozen}" if FREEZE_HISTORY else "freeze=off"
    return (
        f"[{symbol}] {' '.join(extras)} rows={len(new_df)} "
        f"buys={int(signal.sum())} ({signal.mean():.1%}) "
        f"acc={metrics['accuracy']:.3f} prec={metrics['precision']:.3f} "
        f"rec={metrics['recall']:.3f} base={metrics['baseline_rate']:.3f} "
        f"edge={edge:+.3f} | dom_snap={'OK' if dom_ok else 'no'} "
        f"new_tick_bars={n_tick_rows} {freeze_tag} {trade_tag}"
    )


# ── Main loop ─────────────────────────────────────────────────────────────────
def run_once():
    now = pd.Timestamp.now().strftime('%H:%M:%S')
    for t in TARGETS:
        try:
            line = process_target(t)
        except Exception as e:
            line = f"[{t['symbol']}] ERROR: {e}"
        print(f"[{now}] {line}")


def _latest_bar_time(symbol: str) -> Optional[int]:
    """Return the open-time (unix seconds) of the most recent closed bar, or None."""
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 1, 1)
    if rates is None or len(rates) == 0:
        return None
    return int(rates[0]['time'])


if __name__ == '__main__':
    mt5_setup()
    print("Multi-symbol ML signal generator started — fires on every candle close. Ctrl+C to stop.")
    print(f"Microstructure features need ≥{MIN_MICRO_ROWS} bars of collected data before activation.")

    # Track last closed-bar time per symbol to detect bar close events.
    last_bar = {t['symbol']: None for t in TARGETS}  # type: dict
    POLL_SECONDS = 1   # how often to check for a new bar (lightweight)

    while True:
        try:
            new_bar_detected = False
            for t in TARGETS:
                sym = t['symbol']
                current = _latest_bar_time(sym)
                if current is not None and current != last_bar[sym]:
                    last_bar[sym] = current
                    new_bar_detected = True

            if new_bar_detected:
                run_once()
        except Exception as e:
            print(f"Loop error: {e}")
        time.sleep(POLL_SECONDS)
