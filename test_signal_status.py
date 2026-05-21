"""
test_signal_status.py
─────────────────────
Prints a diagnostic table comparing every order-triggering parameter
against its current live value.  Run once; no trades are placed.

Usage:
    python test_signal_status.py
"""

from datetime import datetime, timezone, timedelta

import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from config import (
    TARGETS, TIMEFRAME, N_BARS,
    TB_MAX_BARS, TB_PT_MULT, TB_SL_MULT,
    PROB_THRESHOLD, N_SPLITS_CV, RECENCY_DECAY,
    TRADE_ENABLED, TRADE_BOTH_SIDES, TRADE_SESSIONS,
    RISK_PCT, MAX_SLIPPAGE, MAGIC_NUMBER,
    TRAIL_ACTIVATE_PCT,
    GRID_ENABLED, GRID_MAX_LEVELS, GRID_STEP_MULT, GRID_PORTFOLIO_SL_MULT,
    MIN_MICRO_ROWS,
)
from mt5_client import mt5_setup, fetch_bars, fetch_htf_bars
from microstructure import append_dom_snapshot, fetch_and_aggregate_ticks
from features import (
    _atr, make_features, add_time_features,
    load_dom_features, load_tick_features, make_htf_features, merge_microstructure,
)
from model import (
    triple_barrier_labels, compute_recency_weights,
    evaluate_walkforward,
)

_BRT = timezone(timedelta(hours=-3))

SEP  = "─" * 72
SEP2 = "═" * 72

def _pass_fail(condition: bool) -> str:
    return "✓  PASS" if condition else "✗  FAIL"

def _in_session(sessions) -> bool:
    now = datetime.now(_BRT)
    hm  = (now.hour, now.minute)
    for start, end in sessions:
        if start <= hm < end:
            return True
    return False


def run_diagnostics(target: dict) -> None:
    symbol = target["symbol"]
    slug   = target["slug"]

    print(f"\n{SEP2}")
    print(f"  SIGNAL DIAGNOSTICS — {symbol}  ({datetime.now(_BRT).strftime('%Y-%m-%d %H:%M:%S')} BRT)")
    print(SEP2)

    # ── 1. Fetch bars ──────────────────────────────────────────────────────────
    bars = fetch_bars(symbol, TIMEFRAME, N_BARS)
    if bars is None:
        print("  ERROR: could not fetch bars from MT5")
        return
    print(f"\n  Bars loaded        : {len(bars)}")

    # ── 2. DOM availability check ──────────────────────────────────────────────
    from mt5_client import _subscribed_books
    dom_subscribed = symbol in _subscribed_books
    dom_snap_live  = mt5.market_book_get(symbol)
    dom_snap_ok    = dom_snap_live is not None and len(dom_snap_live) > 0
    print(f"  DOM subscribed     : {dom_subscribed}")
    print(f"  DOM live snapshot  : {'OK (' + str(len(dom_snap_live)) + ' levels)' if dom_snap_ok else 'NONE — broker may not provide L2 for this symbol'}")
    if not dom_snap_ok:
        print("  NOTE: DOM features will be INACTIVE (0 rows). This is normal if the")
        print("        broker does not publish Level 2 order book for futures (WINM26).")

    # ── 3. Features ────────────────────────────────────────────────────────────
    append_dom_snapshot(symbol, slug)
    fetch_and_aggregate_ticks(symbol, slug)

    feats = make_features(bars)
    feats = add_time_features(feats)

    dom_raw  = load_dom_features(slug)
    tick_raw = load_tick_features(slug)

    # Pre-merge DOM diagnostics
    if not dom_raw.empty:
        dom_bar_count = len(dom_raw)
        dom_overlap   = len(dom_raw.index.intersection(feats.index))
        feats_sample  = feats.index[-1]
        dom_sample    = dom_raw.index[-1] if len(dom_raw) else None
        print(f"\n  DOM CSV bars       : {dom_bar_count}  (unique M5 bars after groupby)")
        print(f"  DOM / feats overlap: {dom_overlap} bars")
        print(f"  feats last bar tz  : {feats_sample}")
        print(f"  DOM   last bar tz  : {dom_sample}")
        if dom_overlap == 0:
            print("  WARNING: no timestamp overlap — timezone or flooring mismatch in DOM CSV")

    if not tick_raw.empty:
        tick_bar_count = len(tick_raw)
        tick_overlap   = len(tick_raw.index.intersection(feats.index))
        print(f"  Tick CSV bars      : {tick_bar_count}  overlap={tick_overlap}")

    feats, n_dom  = merge_microstructure(feats, dom_raw,  prefix="dom")
    feats, n_tick = merge_microstructure(feats, tick_raw, prefix="tick")
    htf = make_htf_features(fetch_htf_bars(symbol), bars)
    if not htf.empty:
        feats = pd.concat([feats, htf], axis=1)

    # ── 3. Labels & alignment ──────────────────────────────────────────────────
    atr14    = _atr(bars, 14)
    target_y = triple_barrier_labels(bars, atr14, TB_MAX_BARS, TB_PT_MULT, TB_SL_MULT)
    aligned  = pd.concat([feats, target_y], axis=1).dropna(subset=["y"])
    aligned  = aligned.dropna(subset=feats.columns.tolist(),
                               thresh=int(len(feats.columns) * 0.6))
    X = aligned.drop(columns=["y"])
    y = aligned["y"].astype(int)

    # ── 4. Walk-forward metrics ────────────────────────────────────────────────
    rec_weights = compute_recency_weights(len(aligned), decay=RECENCY_DECAY)
    metrics     = evaluate_walkforward(X, y, N_SPLITS_CV, embargo=TB_MAX_BARS,
                                       weights=rec_weights)

    # ── 5. Full-model proba ────────────────────────────────────────────────────
    model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("gb",  LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                               class_weight="balanced", random_state=42, verbosity=-1)),
    ])
    model.fit(X, y, gb__sample_weight=rec_weights)
    proba_train = model.predict_proba(X)[:, 1]

    # Score live tail (unlabelled bars)
    tail_feats = feats[~feats.index.isin(aligned.index)]
    tail_feats = tail_feats.dropna(thresh=int(len(feats.columns) * 0.6))
    if len(tail_feats) > 0:
        tail_proba  = model.predict_proba(tail_feats)[:, 1]
        latest_proba = float(tail_proba[-1])
    else:
        latest_proba = float(proba_train[-1])

    latest_signal = 1 if latest_proba > PROB_THRESHOLD else 0
    signal_label  = "BUY" if latest_signal == 1 else "SELL/FLAT"

    # ── 6. Derived values ─────────────────────────────────────────────────────
    precision = metrics["precision"]
    baseline  = metrics["baseline_rate"]
    edge      = precision - baseline
    atr_last  = float(atr14.iloc[-1]) if not np.isnan(float(atr14.iloc[-1])) else float(np.nanmedian(atr14.values))
    sl_pts    = int(round(atr_last * TB_SL_MULT)) if atr_last > 0 else 0

    tick      = mt5.symbol_info_tick(symbol)
    account   = mt5.account_info()
    in_session = _in_session(TRADE_SESSIONS)
    brt_now   = datetime.now(_BRT).strftime("%H:%M")

    # ── 7. Hidden gates (not shown in original table) ─────────────────────────
    # 2-bar persistence: previous bar must carry the same signal
    if len(tail_feats) >= 2:
        prev_bar_signal = 1 if float(tail_proba[-2]) > PROB_THRESHOLD else 0
    elif len(tail_feats) == 1:
        prev_bar_signal = 1 if float(proba_train[-1]) > PROB_THRESHOLD else 0
    elif len(proba_train) >= 2:
        prev_bar_signal = 1 if float(proba_train[-2]) > PROB_THRESHOLD else 0
    else:
        prev_bar_signal = latest_signal  # insufficient history; assume consistent
    persistence_ok = (prev_bar_signal == latest_signal)

    # Signal transition: execute_trade skips if signal == last executed signal
    # Infer last executed signal from any open position tagged with MAGIC_NUMBER
    open_positions = mt5.positions_get(symbol=symbol) or []
    last_exec_signal = -1
    for pos in open_positions:
        if pos.magic == MAGIC_NUMBER:
            last_exec_signal = 1 if pos.type == mt5.ORDER_TYPE_BUY else 0
            break
    transition_ok = (latest_signal != last_exec_signal)

    # ── Print table ────────────────────────────────────────────────────────────
    print(f"\n{'PARAMETER':<35} {'REQUIRED / CONFIG':<22} {'CURRENT VALUE':<18} STATUS")
    print(SEP)

    # -- Probability threshold
    req_proba = f"> {PROB_THRESHOLD}"
    cur_proba = f"{latest_proba:.4f}"
    print(f"  {'Proba (latest bar)':<33} {req_proba:<22} {cur_proba:<18} {_pass_fail(latest_proba > PROB_THRESHOLD)}")

    # -- Precision floor
    req_prec = ">= 0.505"
    cur_prec = f"{precision:.4f}"
    print(f"  {'Model precision (CV)':<33} {req_prec:<22} {cur_prec:<18} {_pass_fail(precision >= 0.505)}")

    # -- Edge
    req_edge = ">= 0.02"
    cur_edge = f"{edge:+.4f}"
    print(f"  {'Edge (prec - baseline)':<33} {req_edge:<22} {cur_edge:<18} {_pass_fail(edge >= 0.02)}")

    # -- Trade session
    sessions_str = f"{TRADE_SESSIONS[0][0]}-{TRADE_SESSIONS[-1][1]}"
    print(f"  {'Trade session (BRT)':<33} {'inside window':<22} {brt_now:<18} {_pass_fail(in_session)}")

    # -- Trade enabled
    print(f"  {'TRADE_ENABLED':<33} {'True':<22} {str(TRADE_ENABLED):<18} {_pass_fail(TRADE_ENABLED)}")

    # -- 2-bar persistence filter
    prev_label = "BUY" if prev_bar_signal == 1 else "SELL/FLAT"
    print(f"  {'2-bar persistence (prev bar)':<33} {'prev==current':<22} {prev_label:<18} {_pass_fail(persistence_ok)}")

    # -- Signal transition guard
    last_exec_label = "BUY" if last_exec_signal == 1 else ("SELL/FLAT" if last_exec_signal == 0 else "none (no pos)")
    print(f"  {'Signal transition':<33} {'current!=last exec':<22} {last_exec_label:<18} {_pass_fail(transition_ok)}")

    print(SEP)

    # ── Signal & order summary ────────────────────────────────────────────────
    all_gates = (latest_proba > PROB_THRESHOLD and precision >= 0.505
                 and edge >= 0.02 and in_session and TRADE_ENABLED
                 and persistence_ok and transition_ok)
    print(f"\n  Current signal     : {signal_label}")
    print(f"  All gates pass     : {_pass_fail(all_gates)}")
    if not all_gates:
        blockers = []
        if not (latest_proba > PROB_THRESHOLD): blockers.append("proba below threshold")
        if not (precision >= 0.505):            blockers.append("precision too low")
        if not (edge >= 0.02):                  blockers.append("edge too low")
        if not in_session:                      blockers.append("outside trade session")
        if not TRADE_ENABLED:                   blockers.append("TRADE_ENABLED=False")
        if not persistence_ok:                  blockers.append("2-bar persistence: prev bar signal differs")
        if not transition_ok:                   blockers.append(f"signal transition: already in {last_exec_label} position")
        print(f"  Blocked by         : {'; '.join(blockers)}")
    print(f"  → Order would be   : {'SENT (' + signal_label + ')' if all_gates else 'BLOCKED'}")

    # ── Supporting info ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  {'SUPPORTING INFO'}")
    print(SEP)
    print(f"  {'ATR(14) last bar':<33} {atr_last:.1f} pts")
    print(f"  {'SL distance (ATR × SL_MULT)':<33} {sl_pts} pts  (TB_SL_MULT={TB_SL_MULT})")
    print(f"  {'Profit target mult (TB_PT_MULT)':<33} {TB_PT_MULT}× ATR")
    print(f"  {'Vertical barrier (TB_MAX_BARS)':<33} {TB_MAX_BARS} bars (~{TB_MAX_BARS * 5} min)")
    print(f"  {'Model recall (CV)':<33} {metrics['recall']:.4f}")
    print(f"  {'Baseline buy rate':<33} {baseline:.4f}")
    from config import dom_path, ticks_path
    _dom_csv   = dom_path(slug)
    _tick_csv  = ticks_path(slug)
    _dom_file_rows  = len(pd.read_csv(_dom_csv))  if os.path.exists(_dom_csv)  else 0
    _tick_file_rows = len(pd.read_csv(_tick_csv)) if os.path.exists(_tick_csv) else 0
    _dom_status  = f"{n_dom} merged bars  ({_dom_file_rows} snapshots on disk)"  + ("" if n_dom >= MIN_MICRO_ROWS else f"  ← need {MIN_MICRO_ROWS} bars — run main loop to accumulate")
    _tick_status = f"{n_tick} merged bars  ({_tick_file_rows} tick-bars on disk)" + ("" if n_tick >= MIN_MICRO_ROWS else f"  ← need {MIN_MICRO_ROWS} bars — run main loop to accumulate")
    print(f"  {'DOM microstructure':<33} {_dom_status}")
    print(f"  {'Tick microstructure':<33} {_tick_status}")
    if tick:
        spread = round(tick.ask - tick.bid)
        print(f"  {'Current bid/ask':<33} {tick.bid} / {tick.ask}  (spread={spread}p)")
    if account:
        print(f"  {'Account balance':<33} {account.balance:.2f} {account.currency}")
        print(f"  {'Risk per trade (RISK_PCT)':<33} {RISK_PCT}%  ≈ {account.balance * RISK_PCT / 100:.2f} {account.currency}")
    print(f"  {'MAX_SLIPPAGE':<33} {MAX_SLIPPAGE} pts")
    print(f"  {'TRADE_BOTH_SIDES':<33} {TRADE_BOTH_SIDES}")
    print(f"  {'GRID_ENABLED':<33} {GRID_ENABLED}  (max {GRID_MAX_LEVELS} levels, step {GRID_STEP_MULT}×SL)")
    print(f"  {'TRAIL_ACTIVATE_PCT':<33} {TRAIL_ACTIVATE_PCT * 100:.0f}% of SL distance")
    print(f"  {'MAGIC_NUMBER':<33} {MAGIC_NUMBER}")
    print(SEP)


if __name__ == "__main__":
    mt5_setup()
    for t in TARGETS:
        try:
            run_diagnostics(t)
        except Exception as e:
            print(f"[{t['symbol']}] ERROR: {e}")
            raise
    mt5.shutdown()
