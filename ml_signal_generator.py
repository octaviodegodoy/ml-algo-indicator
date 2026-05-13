# ML signal generator for WIN$N (mini Ibovespa future)
#
# Trains a model on WIN bars and outputs a signals CSV into MQL5/Files,
# so PlotMLSignals.mq5 can be loaded on the WIN chart to read it.
#
# Module layout:
#   config.py       - constants and output-path helpers
#   mt5_client.py   - MT5 lifecycle, bar/DOM/tick fetching
#   features.py     - feature engineering and microstructure loaders
#   model.py        - triple-barrier labels, walk-forward CV, SL computation
#   trade.py        - order execution, trailing stop, Fibonacci grid
#   ml_signal_generator.py (this file) - orchestrator + main loop
#
# Requirements: MetaTrader5, pandas, numpy, scikit-learn, lightgbm
# pip install MetaTrader5 pandas numpy scikit-learn lightgbm

import os
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from config import (
    TARGETS, TIMEFRAME, N_BARS,
    TB_MAX_BARS, TB_PT_MULT, TB_SL_MULT,
    PROB_THRESHOLD, N_SPLITS_CV, MIN_MICRO_ROWS,
    FREEZE_HISTORY, TRADE_ENABLED, TRADE_SESSIONS, out_path,
)
from mt5_client import (
    mt5_setup, fetch_bars, append_dom_snapshot, fetch_and_aggregate_ticks,
)
from features import (
    _atr, make_features, add_time_features,
    load_dom_features, load_tick_features, make_htf_features, merge_microstructure,
)
from model import (
    triple_barrier_labels, compute_recency_weights,
    evaluate_walkforward, compute_sl_points,
)
from trade import execute_trade, manage_trailing_stops, manage_grid_orders, init_signal_state


# ── Signal persistence buffer (per symbol, last 2 evaluations) ───────────────
_signal_buffer: dict = {}   # symbol → deque(maxlen=2)

_BRT = timezone(timedelta(hours=-3))


def _in_trade_session() -> bool:
    """Return True if current BRT wall-clock time falls inside a TRADE_SESSIONS window."""
    now = datetime.now(_BRT)
    hm  = (now.hour, now.minute)
    for start, end in TRADE_SESSIONS:
        if start <= hm < end:
            return True
    return False


# ── Per-target processing ─────────────────────────────────────────────────────
def process_target(target: dict) -> None:
    symbol = target["symbol"]
    slug   = target["slug"]

    # 1. Live microstructure snapshot
    append_dom_snapshot(symbol, slug)
    fetch_and_aggregate_ticks(symbol, slug)

    # 2. Bars
    bars = fetch_bars(symbol, TIMEFRAME, N_BARS)
    if bars is None:
        print(f"[{symbol}] no bars")
        return

    # 3. Features
    feats = make_features(bars)
    feats = add_time_features(feats)
    feats, _  = merge_microstructure(feats, load_dom_features(slug),  prefix="dom")
    feats, _  = merge_microstructure(feats, load_tick_features(slug), prefix="tick")
    htf_feats = make_htf_features(symbol, bars)
    if not htf_feats.empty:
        feats = pd.concat([feats, htf_feats], axis=1)

    # 4. Labels
    atr14    = _atr(bars, 14)
    target_y = triple_barrier_labels(bars, atr14, TB_MAX_BARS, TB_PT_MULT, TB_SL_MULT)

    # 5. Align features and labels
    aligned = pd.concat([feats, target_y], axis=1).dropna(subset=["y"])
    aligned = aligned.dropna(subset=feats.columns.tolist(),
                              thresh=int(len(feats.columns) * 0.6))
    if len(aligned) < 500:
        print(f"[{symbol}] not enough rows: {len(aligned)}")
        return

    X = aligned.drop(columns=["y"])
    y = aligned["y"].astype(int)

    # 6. Train
    rec_weights = compute_recency_weights(len(aligned))
    metrics     = evaluate_walkforward(X, y, N_SPLITS_CV, embargo=TB_MAX_BARS,
                                       weights=rec_weights)

    model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("gb",  LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                               class_weight="balanced", random_state=42, verbosity=-1)),
    ])
    model.fit(X, y, gb__sample_weight=rec_weights)

    proba  = model.predict_proba(X)[:, 1]
    signal = (proba > PROB_THRESHOLD).astype(int)

    sig_series = pd.Series(signal, index=X.index)
    sl_pts     = compute_sl_points(sig_series, bars, atr14, TB_SL_MULT, TB_MAX_BARS)

    # Use current-bar ATR for SL sizing (avoids stale historical median on volatile days)
    _atr_last  = float(atr14.iloc[-1]) if not np.isnan(float(atr14.iloc[-1])) else float(np.nanmedian(atr14.values))
    avg_sl_val = int(round(_atr_last * TB_SL_MULT)) if _atr_last > 0 else 0

    new_df = pd.DataFrame({
        "Timestamp": X.index.astype("int64") // 10**9,
        "ML_Signal": signal,
        "SL_Points": sl_pts.values,
    })

    # 7. Score the unlabeled tail (live bars without future reference)
    latest_proba = float(proba[-1])
    tail_feats = feats[~feats.index.isin(aligned.index)]
    tail_feats = tail_feats.dropna(thresh=int(len(feats.columns) * 0.6))
    if len(tail_feats) > 0:
        tail_proba  = model.predict_proba(tail_feats)[:, 1]
        tail_signal = (tail_proba > PROB_THRESHOLD).astype(int)
        prev_sig    = int(signal[-1]) if len(signal) > 0 else 0
        tail_sl     = []
        for ts in tail_signal:
            curr_sig = int(ts)
            if (curr_sig == 1 and prev_sig == 0) or (curr_sig == 0 and prev_sig == 1):
                tail_sl.append(avg_sl_val)
            else:
                tail_sl.append(0)
            prev_sig = curr_sig
        tail_df = pd.DataFrame({
            "Timestamp": tail_feats.index.astype("int64") // 10**9,
            "ML_Signal": tail_signal,
            "SL_Points": tail_sl,
        })
        new_df = pd.concat([new_df, tail_df], ignore_index=True)
        latest_proba = float(tail_proba[-1])

    # 8. Write CSV (with optional history freeze)
    csv_path = out_path(slug)
    if os.path.exists(csv_path):
        try:
            _old = pd.read_csv(csv_path)
            if (_old["ML_Signal"] == 1).sum() == 0:
                os.remove(csv_path)
        except Exception:
            pass

    n_frozen = 0
    if FREEZE_HISTORY and os.path.exists(csv_path):
        try:
            old_df = pd.read_csv(csv_path)
            if "SL_Points" not in old_df.columns:
                old_df["SL_Points"] = 0
            if len(old_df) > 1:
                frozen   = old_df.iloc[:-1]
                new_df   = new_df[~new_df["Timestamp"].isin(frozen["Timestamp"])]
                merged   = pd.concat([frozen, new_df], ignore_index=True)
                merged   = merged.drop_duplicates(subset="Timestamp", keep="last")
                merged   = merged.sort_values("Timestamp")
                merged["SL_Points"] = merged["SL_Points"].fillna(0).astype(int)
                merged.to_csv(csv_path, index=False)
            else:
                new_df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"[{symbol}] freeze-merge failed ({e}); rewriting full CSV")
            new_df.to_csv(csv_path, index=False)
    else:
        new_df.to_csv(csv_path, index=False)

    # 9. Execute trade on latest signal (quality gate + 2-bar persistence filter)
    latest_signal = int(new_df.sort_values("Timestamp").iloc[-1]["ML_Signal"])
    _edge = metrics['precision'] - metrics['baseline_rate']
    signal_label  = "BUY" if latest_signal == 1 else "SELL/FLAT"
    trigger       = "TRIGGERED" if latest_proba > PROB_THRESHOLD else "below"
    print(f"[{symbol}] signal={signal_label}  proba={latest_proba:.3f} {'>' if latest_proba > PROB_THRESHOLD else '<'} threshold={PROB_THRESHOLD}  ({trigger})  edge={_edge:+.3f}  prec={metrics['precision']:.3f}  sl={avg_sl_val}p")
    # Quality gate: precision floor is 0.505 (barely above random); edge check is the main guard
    if metrics['precision'] >= 0.505 and _edge >= 0.02 and _in_trade_session():
        buf = _signal_buffer.setdefault(symbol, deque(maxlen=2))
        buf.append(latest_signal)
        if len(buf) >= 2 and len(set(buf)) == 1:
            execute_trade(symbol, latest_signal, avg_sl_val)

    return None


# ── Main loop ─────────────────────────────────────────────────────────────────
def run_once() -> None:
    now = pd.Timestamp.now().strftime("%H:%M:%S")
    for t in TARGETS:
        try:
            process_target(t)
        except Exception as e:
            sym = t["symbol"]
            print(f"[{now}] [{sym}] ERROR: {e}")


def _latest_bar_time(symbol: str) -> Optional[int]:
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 1, 1)
    if rates is None or len(rates) == 0:
        return None
    return int(rates[0]["time"])


if __name__ == "__main__":
    mt5_setup()
    init_signal_state()
    print("Multi-symbol ML signal generator started")
    print(f"Microstructure features need >={MIN_MICRO_ROWS} bars of collected data before activation.")

    last_bar: dict = {t["symbol"]: None for t in TARGETS}
    POLL_SECONDS = 1

    while True:
        try:
            manage_trailing_stops()
            manage_grid_orders()
            new_bar_detected = False
            for t in TARGETS:
                sym     = t["symbol"]
                current = _latest_bar_time(sym)
                if current is not None and current != last_bar[sym]:
                    last_bar[sym]    = current
                    new_bar_detected = True
            if new_bar_detected:
                run_once()
        except Exception as e:
            print(f"Loop error: {e}")
        time.sleep(POLL_SECONDS)
