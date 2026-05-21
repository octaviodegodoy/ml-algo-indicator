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
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from config import (
    TARGETS, TIMEFRAME, N_BARS,
    TB_MAX_BARS, TB_PT_MULT, TB_SL_MULT,
    PROB_THRESHOLD, N_SPLITS_CV, MIN_MICRO_ROWS,
    FREEZE_HISTORY, TRADE_ENABLED, TRADE_SESSIONS, out_path,
    MIN_AUC, COOLDOWN_BARS, DAILY_MAX_LOSS_PCT, MODEL_TYPE,
    REQUIRE_DI_CONFIRMATION, DI_CONFIRM_MIN_DIFF, SELL_PERSISTENCE_BARS,
)
from mt5_client import mt5_setup, fetch_bars, fetch_htf_bars
from microstructure import append_dom_snapshot, fetch_and_aggregate_ticks
from features import (
    _atr, make_features, add_time_features,
    load_dom_features, load_tick_features, make_htf_features, merge_microstructure,
)
from model import (
    triple_barrier_labels, compute_recency_weights,
    evaluate_walkforward, compute_sl_points, _make_classifier,
)
from trade import execute_trade, manage_trailing_stops, manage_grid_orders, init_signal_state


# ── Signal persistence buffer (per symbol, last 2 evaluations) ───────────────
_signal_buffer: dict = {}    # symbol → deque(maxlen=2)
_cooldown_counter: dict = {} # symbol → int (bars remaining before re-entry allowed)

_BRT = timezone(timedelta(hours=-3))


# ── Daily P&L helper ─────────────────────────────────────────────────────────
def _get_daily_pnl() -> float:
    """Return today's realized P&L from MT5 deal history (BRT day boundary)."""
    try:
        today     = datetime.now(_BRT).replace(hour=0, minute=0, second=0, microsecond=0)
        today_utc = today.astimezone(timezone.utc)
        now_utc   = datetime.now(timezone.utc)
        deals     = mt5.history_deals_get(today_utc, now_utc)
        if not deals:
            return 0.0
        return sum(d.profit for d in deals if d.entry == 1)
    except Exception:
        return 0.0


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
    htf_feats = make_htf_features(fetch_htf_bars(symbol), bars)
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

    # Drop columns that are 100% NaN (e.g., DOM features before snapshots accumulate)
    # — silences SimpleImputer "Skipping features without any observed values" warning.
    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)

    # 6. Train
    rec_weights = compute_recency_weights(len(aligned))
    metrics     = evaluate_walkforward(X, y, N_SPLITS_CV, embargo=TB_MAX_BARS + 1,
                                       weights=rec_weights)

    neg = float((y == 0).sum()); pos = float((y == 1).sum())
    spw = (neg / pos) if pos > 0 else 1.0
    model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("gb",  _make_classifier(scale_pos_weight=spw)),
    ])
    model.fit(X, y, gb__sample_weight=rec_weights)

    proba  = model.predict_proba(X)[:, 1]
    signal = (proba > PROB_THRESHOLD).astype(int)

    sig_series = pd.Series(signal, index=X.index)
    sl_pts     = compute_sl_points(sig_series, bars, atr14, TB_SL_MULT, TB_MAX_BARS)

    # Use current-bar ATR for SL sizing (avoids stale historical median on volatile days)
    _atr_last  = float(atr14.iloc[-1]) if not np.isnan(float(atr14.iloc[-1])) else float(np.nanmedian(atr14.values))
    avg_sl_val = int(round(_atr_last * TB_SL_MULT)) if _atr_last > 0 else 0

    # Model quality metric: ROC-AUC from walk-forward CV (threshold-independent)
    # Written to CSV columns so the indicator can filter low-quality model cycles.
    # Precision col = roc_auc (0.50–1.0);  Edge col = roc_auc − 0.50 (lift over random)
    _auc = round(metrics['roc_auc'], 4) if not np.isnan(metrics['roc_auc']) else 0.5

    new_df = pd.DataFrame({
        "Timestamp": X.index.astype("int64") // 10**9,
        "ML_Signal": signal,
        "SL_Points": sl_pts.values,
        "Prob":      proba.round(4),
    })

    # 7. Score the unlabeled tail (live bars without future reference)
    latest_proba = float(proba[-1])
    tail_feats = feats[~feats.index.isin(aligned.index)]
    tail_feats = tail_feats.dropna(thresh=int(len(feats.columns) * 0.6))
    if len(tail_feats) > 0:
        tail_feats  = tail_feats.reindex(columns=X.columns, fill_value=np.nan)
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
            "Prob":      tail_proba.round(4),
        })
        new_df = pd.concat([new_df, tail_df], ignore_index=True)
        latest_proba = float(tail_proba[-1])

    # Attach model-level quality metrics so the indicator can filter by them
    new_df["Precision"] = _auc              # stores roc_auc (0.50–1.0)
    new_df["Edge"]      = round(_auc - 0.5, 4)  # lift over random (0.0–0.5)

    # Capture latest signal BEFORE freeze-merge can filter new_df down to zero rows
    if not new_df.empty:
        latest_signal = int(new_df.sort_values("Timestamp").iloc[-1]["ML_Signal"])
    else:
        latest_signal = 0

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
                # Backfill Precision/Edge on ALL rows so the indicator can filter
                # by current model quality (frozen rows retain old-format values otherwise)
                merged["Precision"] = _auc
                merged["Edge"]      = round(_auc - 0.5, 4)
                merged.to_csv(csv_path, index=False)
            else:
                new_df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"[{symbol}] freeze-merge failed ({e}); rewriting full CSV")
            new_df.to_csv(csv_path, index=False)
    else:
        new_df.to_csv(csv_path, index=False)

    # 9. Execute trade on latest signal (quality gate + asymmetric persistence + DI gate)
    signal_label  = "BUY" if latest_signal == 1 else "SELL/FLAT"
    trigger       = "TRIGGERED" if latest_proba > PROB_THRESHOLD else "below"
    # Volatility gate: skip entry when current ATR > 2× the 20-bar ATR median (choppy/gapping bar)
    _atr_median  = float(np.nanmedian(atr14.iloc[-20:].values))
    _atr_current = float(atr14.iloc[-1]) if not np.isnan(float(atr14.iloc[-1])) else _atr_median
    _high_vol    = _atr_current > 2.0 * _atr_median if _atr_median > 0 else False

    # DI directional gate: SELL only when DI− leads DI+ by at least DI_CONFIRM_MIN_DIFF points
    # (prevents shorting into a pullback where trend hasn't actually confirmed down)
    _di_diff = float(feats['di_diff_14'].iloc[-1]) if 'di_diff_14' in feats.columns and not np.isnan(feats['di_diff_14'].iloc[-1]) else 0.0
    _di_ok   = True
    if REQUIRE_DI_CONFIRMATION:
        if latest_signal == 0 and _di_diff > -DI_CONFIRM_MIN_DIFF:
            _di_ok = False   # SELL blocked: DI not confirming downtrend
        elif latest_signal == 1 and _di_diff < DI_CONFIRM_MIN_DIFF:
            _di_ok = False   # BUY blocked: DI not confirming uptrend

    print(f"[{symbol}] signal={signal_label}  proba={latest_proba:.3f} {'>' if latest_proba > PROB_THRESHOLD else '<'} threshold={PROB_THRESHOLD}  ({trigger})  auc={_auc:.3f} ({'PASS' if _auc >= MIN_AUC else 'FAIL'})  sl={avg_sl_val}p  atr={_atr_current:.0f}  di_diff={_di_diff:.1f} ({'DI-OK' if _di_ok else 'DI-BLOCKED'}){'  [HIGH-VOL GATE]' if _high_vol else ''}")

    # Quality gate: AUC, session, volatility, and DI direction must all pass
    if _auc >= MIN_AUC and _in_trade_session() and not _high_vol and _di_ok:
        # Asymmetric persistence: SELL requires SELL_PERSISTENCE_BARS consecutive bars,
        # BUY requires only 2.  Prevents confirming a short at reversal extremes.
        _sell_persist = SELL_PERSISTENCE_BARS
        buf = _signal_buffer.setdefault(symbol, deque(maxlen=max(2, _sell_persist)))
        buf.append(latest_signal)
        _required = _sell_persist if latest_signal == 0 else 2
        _cooldown  = _cooldown_counter.get(symbol, 0)
        if _cooldown > 0:
            _cooldown_counter[symbol] = _cooldown - 1
            print(f"[{symbol}] trade HELD — cooldown ({_cooldown - 1} bars remaining)")
        elif len(buf) >= _required and len(set(list(buf)[-_required:])) == 1:
            execute_trade(symbol, latest_signal, avg_sl_val)
            # Reset cooldown on flat signal (signal=0 means close long / go flat)
            # so the next long entry waits COOLDOWN_BARS bars before re-entering
            if latest_signal == 0:
                _cooldown_counter[symbol] = COOLDOWN_BARS
        else:
            print(f"[{symbol}] trade HELD — persistence waiting ({len(buf)}/{_required} bars, buf={list(buf)[-_required:]})")

    return None


# ── Main loop ─────────────────────────────────────────────────────────────────
def run_once() -> None:
    now = pd.Timestamp.now().strftime("%H:%M:%S")
    # Daily loss circuit breaker — halt new entries for the rest of the day
    daily_pnl = _get_daily_pnl()
    _acct = mt5.account_info()
    _equity = _acct.equity if _acct else 0.0
    _daily_loss_limit = (DAILY_MAX_LOSS_PCT / 100.0) * _equity
    if _equity > 0 and daily_pnl <= _daily_loss_limit:
        print(f"[{now}] Daily loss circuit breaker: P&L={daily_pnl:.2f} ≤ {_daily_loss_limit:.2f} ({DAILY_MAX_LOSS_PCT}% of equity {_equity:.2f}). No new entries today.")
        return
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
