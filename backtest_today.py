"""
backtest_today.py — Side-by-side simulation of OLD vs NEW parameters on today's
M5 bars.  Uses the same feature/model pipeline as ml_signal_generator.py.

Methodology
-----------
  1. Fetch 5000 M5 bars from MT5 (requires MT5 to be running).
  2. Split: bars BEFORE today → training set; bars FROM today → test set.
  3. Train one LGBMClassifier per parameter set on the training data.
  4. Score today's bars and apply signal filters (persistence / quality gate).
  5. Walk through today's bars chronologically:
       - Signal flip  → close current position, open in new direction.
       - SL hit        → close at SL price.
       - End of day    → close at last close.
  6. Print comparison table and save chart to backtest_today.png.

Run:
    python backtest_today.py
"""

import sys
from collections import deque

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from features import _atr, add_time_features, make_features, make_htf_features
from model import (
    compute_recency_weights,
    evaluate_walkforward,
    triple_barrier_labels,
)
from mt5_client import fetch_bars, mt5_setup

# ── Simulation constants ──────────────────────────────────────────────────────
SYMBOL       = "WINM26"
TIMEFRAME    = mt5.TIMEFRAME_M5
N_TRAIN_BARS = 5000
LOTS         = 1       # fixed 1 mini lot per trade for a fair comparison
TICK_VALUE   = 0.20    # R$ per point per mini lot (WINM26 standard)
COMMISSION   = LOTS * 2.0  # R$ per round trip (1 × entry + 1 × exit fee)
N_SPLITS_CV  = 5
EMBARGO      = 12

# ── OLD vs NEW parameter sets ─────────────────────────────────────────────────
PARAMS = {
    "BASELINE": dict(
        prob_threshold   = 0.50,
        recency_decay    = 2.0,
        sl_mult          = 1.0,
        pt_mult          = 1.5,
        max_bars         = 12,
        use_persistence  = False,
        use_quality_gate = False,
    ),
    "OLD": dict(
        prob_threshold   = 0.50,
        recency_decay    = 2.0,
        sl_mult          = 1.0,
        pt_mult          = 1.5,
        max_bars         = 12,
        use_persistence  = False,
        use_quality_gate = False,
    ),
    "NEW": dict(
        prob_threshold   = 0.54,
        recency_decay    = 1.2,
        sl_mult          = 1.2,
        pt_mult          = 1.5,
        max_bars         = 12,
        use_persistence  = True,
        use_quality_gate = True,
    ),
}


# ── Train + score ─────────────────────────────────────────────────────────────
def train_and_score(bars_train: pd.DataFrame, bars_test: pd.DataFrame, p: dict):
    """
    Train on bars_train, score bars_test.
    Returns (signal_series, metrics_dict) or (None, metrics_dict) if gated.
    """
    atr14  = _atr(bars_train, 14)
    labels = triple_barrier_labels(
        bars_train, atr14, p["max_bars"], p["pt_mult"], p["sl_mult"]
    )

    feats_train = make_features(bars_train)
    feats_train = add_time_features(feats_train)
    htf_train   = make_htf_features(SYMBOL, bars_train)
    if not htf_train.empty:
        feats_train = pd.concat([feats_train, htf_train], axis=1)

    aligned = pd.concat([feats_train, labels], axis=1).dropna(subset=["y"])
    aligned = aligned.dropna(thresh=int(len(feats_train.columns) * 0.6))
    if len(aligned) < 500:
        print(f"    Not enough training rows: {len(aligned)}")
        return None, {}

    X_train = aligned.drop(columns=["y"])
    y_train = aligned["y"].astype(int)

    weights = compute_recency_weights(len(aligned), decay=p["recency_decay"])
    metrics = evaluate_walkforward(
        X_train, y_train, N_SPLITS_CV, embargo=EMBARGO, weights=weights
    )
    edge = metrics["precision"] - metrics["baseline_rate"]

    if p["use_quality_gate"] and (metrics["precision"] < 0.505 or edge < 0.02):
        print(
            f"    Quality gate REJECTED  "
            f"prec={metrics['precision']:.3f}  edge={edge:+.3f} → no trades"
        )
        return None, metrics

    model = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            (
                "gb",
                LGBMClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    class_weight="balanced",
                    random_state=42,
                    verbosity=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train, gb__sample_weight=weights)

    # Score test bars using the same feature columns as training
    feats_test = make_features(bars_test)
    feats_test = add_time_features(feats_test)
    htf_test   = make_htf_features(SYMBOL, bars_test)
    if not htf_test.empty:
        feats_test = pd.concat([feats_test, htf_test], axis=1)
    feats_test = feats_test.reindex(columns=X_train.columns, fill_value=np.nan)

    proba      = model.predict_proba(feats_test)[:, 1]
    raw_signal = (proba > p["prob_threshold"]).astype(int)
    # Proba diagnostics
    print(f"    proba stats: min={proba.min():.3f}  max={proba.max():.3f}  "
          f"mean={proba.mean():.3f}  pct>{p['prob_threshold']}={(proba > p['prob_threshold']).mean():.1%}  "
          f"pct>0.50={(proba > 0.50).mean():.1%}")
    print(f"    raw_signal: buy_bars={int(raw_signal.sum())}  sell_bars={int((raw_signal==0).sum())}  "
          f"consecutive_same={sum(1 for a,b in zip(raw_signal,raw_signal[1:]) if a==b)}")

    if p["use_persistence"]:
        # 2-bar confirmation: hold last confirmed signal; only change when 2 consecutive bars agree
        confirmed      = []
        buf            = deque(maxlen=2)
        last_confirmed = -1          # -1 = no signal yet (before first confirmation)
        for s in raw_signal:
            buf.append(s)
            if len(buf) == 2 and len(set(buf)) == 1:   # both bars agree
                last_confirmed = int(s)
            confirmed.append(last_confirmed)            # hold last confirmed direction
        signal_series = pd.Series(confirmed, index=feats_test.index, name="signal")
        n_confirmed = sum(1 for c, r in zip(confirmed, raw_signal) if c == r and c != -1)
        print(f"    persistence: confirmed_changes={n_confirmed}  "
              f"undecided_bars={confirmed.count(-1)}")
    else:
        signal_series = pd.Series(raw_signal, index=feats_test.index, name="signal")

    return signal_series, metrics


# ── P&L simulation ────────────────────────────────────────────────────────────
def simulate_pnl(bars_test: pd.DataFrame, signal_series: pd.Series, sl_mult: float):
    """
    Walk through today's bars and simulate trades.
    Entry / exit are at the close of the triggering bar (conservative).
    SL = sl_mult × ATR(14) computed from the test bars themselves.
    Returns (trades_df, summary_dict).
    """
    atr_test = _atr(bars_test, 14)
    close    = bars_test["Close"]
    high     = bars_test["High"]
    low      = bars_test["Low"]

    trades:   list = []
    in_trade: bool  = False
    direction: int  = -1     # 1=long, 0=short
    entry_price     = 0.0
    sl_price        = 0.0
    entry_time      = None
    prev_sig        = -1

    def _close_trade(exit_ts, exit_price, reason):
        pts = (exit_price - entry_price) if direction == 1 else (entry_price - exit_price)
        net = pts * LOTS * TICK_VALUE - COMMISSION
        trades.append(
            dict(
                entry_time  = entry_time,
                exit_time   = exit_ts,
                direction   = "LONG" if direction == 1 else "SHORT",
                entry       = entry_price,
                exit_price  = exit_price,
                pnl_pts     = pts,
                net_brl     = net,
                reason      = reason,
            )
        )

    for ts in signal_series.index:
        sig  = int(signal_series.loc[ts])
        c    = float(close.get(ts, np.nan))
        h    = float(high.get(ts,  np.nan))
        l    = float(low.get(ts,   np.nan))

        # Check SL on open position (bar's high/low)
        if in_trade and not (np.isnan(h) or np.isnan(l)):
            if direction == 1 and l <= sl_price:
                _close_trade(ts, sl_price, "SL")
                in_trade = False
            elif direction == 0 and h >= sl_price:
                _close_trade(ts, sl_price, "SL")
                in_trade = False

        if sig == -1:          # persistence filter: still waiting for confirmation
            prev_sig = sig
            continue

        if sig != prev_sig and prev_sig != -1:
            # Signal flip
            if in_trade and not np.isnan(c):
                _close_trade(ts, c, "SIGNAL")
                in_trade = False

            # Open new position
            atr_val = float(atr_test.get(ts, np.nan))
            if not np.isnan(c) and not np.isnan(atr_val) and atr_val > 0:
                entry_price = c
                entry_time  = ts
                direction   = sig
                sl_price    = (c - atr_val * sl_mult) if sig == 1 else (c + atr_val * sl_mult)
                in_trade    = True

        prev_sig = sig

    # Close any remaining position at end of day
    if in_trade:
        last_ts    = close.index[-1]
        last_price = float(close.iloc[-1])
        _close_trade(last_ts, last_price, "EOD")

    df = pd.DataFrame(trades)
    if df.empty:
        return df, dict(n_trades=0, net_brl=0.0, win_rate=0.0, avg_pnl=0.0, best=0.0, worst=0.0)

    n_win = (df["net_brl"] > 0).sum()
    summary = dict(
        n_trades = len(df),
        net_brl  = df["net_brl"].sum(),
        win_rate = n_win / len(df),
        avg_pnl  = df["net_brl"].mean(),
        best     = df["net_brl"].max(),
        worst    = df["net_brl"].min(),
    )
    return df, summary


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    mt5_setup()

    today_str = pd.Timestamp.now(tz="America/Sao_Paulo").date()
    print(f"\nBacktest  {SYMBOL}  M5  —  {today_str}")
    print(f"Fetching {N_TRAIN_BARS + 300} bars …")

    all_bars = fetch_bars(SYMBOL, TIMEFRAME, N_TRAIN_BARS + 300)
    if all_bars is None or all_bars.empty:
        print("ERROR: no bars returned from MT5.")
        mt5.shutdown()
        sys.exit(1)

    today_mask = all_bars.index.normalize() == pd.Timestamp(today_str, tz="UTC")
    bars_today = all_bars[today_mask]
    bars_train = all_bars[~today_mask].iloc[-N_TRAIN_BARS:]

    if bars_today.empty:
        # Fallback: try last calendar date present in data
        last_date  = all_bars.index.normalize().unique()[-1]
        today_mask = all_bars.index.normalize() == last_date
        bars_today = all_bars[today_mask]
        bars_train = all_bars[~today_mask].iloc[-N_TRAIN_BARS:]
        print(f"  (No bars for today in feed — using last available date: {last_date.date()})")

    print(f"Training bars : {len(bars_train)}")
    print(f"Today's bars  : {len(bars_today)}")

    results_map:   dict = {}
    trades_map:    dict = {}
    metrics_map:   dict = {}

    for label, p in PARAMS.items():
        print(f"\n{'─'*52}")
        print(f"  [{label}]  threshold={p['prob_threshold']}  decay={p['recency_decay']}  "
              f"sl_mult={p['sl_mult']}  persistence={p['use_persistence']}  "
              f"quality_gate={p['use_quality_gate']}")
        signals, metrics = train_and_score(bars_train, bars_today, p)
        metrics_map[label] = metrics
        if signals is None:
            results_map[label] = dict(n_trades=0, net_brl=0.0, win_rate=0.0,
                                       avg_pnl=0.0, best=0.0, worst=0.0)
            trades_map[label]  = pd.DataFrame()
            continue

        edge = metrics.get("precision", 0) - metrics.get("baseline_rate", 0)
        n_buy  = int((signals == 1).sum())
        n_sell = int((signals == 0).sum())
        print(
            f"    acc={metrics.get('accuracy', 0):.3f}  "
            f"prec={metrics.get('precision', 0):.3f}  "
            f"rec={metrics.get('recall', 0):.3f}  "
            f"edge={edge:+.3f}  "
            f"buy_bars={n_buy}  sell_bars={n_sell}  undecided={int((signals==-1).sum())}"
        )

        trades_df, summary = simulate_pnl(bars_today, signals, p["sl_mult"])
        results_map[label] = summary
        trades_map[label]  = trades_df
        print(
            f"    Trades={summary['n_trades']}  "
            f"Net=R${summary['net_brl']:.2f}  "
            f"Win={summary['win_rate']:.0%}  "
            f"Avg=R${summary['avg_pnl']:.2f}  "
            f"Best=R${summary['best']:.2f}  "
            f"Worst=R${summary['worst']:.2f}"
        )

    mt5.shutdown()

    # ── Summary table ──────────────────────────────────────────────────────────
    dbl  = "\u2550"
    dash = "-"
    arr  = "OLD\u2192NEW"
    print(f"\n{dbl*72}")
    print(f"  RESULTS  —  {today_str}  |  {SYMBOL} M5  |  {LOTS} lot  |  comm=R${COMMISSION:.2f}/RT")
    print(f"{dbl*72}")
    print(f"  {'Metric':<22} {'BASELINE':>12} {'OLD':>12} {'NEW':>12}  {arr:>10}")
    print(f"  {dash*68}")
    for k in ["n_trades", "net_brl", "win_rate", "avg_pnl", "best", "worst"]:
        bv = results_map["BASELINE"].get(k, 0)
        ov = results_map["OLD"].get(k, 0)
        nv = results_map["NEW"].get(k, 0)
        delta = nv - ov
        if k == "win_rate":
            print(f"  {k:<22} {bv:>11.1%} {ov:>11.1%} {nv:>11.1%}  {delta:>+10.1%}")
        elif k == "n_trades":
            print(f"  {k:<22} {int(bv):>12} {int(ov):>12} {int(nv):>12}  {int(delta):>+10}")
        else:
            print(f"  {k:<22} {bv:>12.2f} {ov:>12.2f} {nv:>12.2f}  {delta:>+10.2f}")
    print(f"{dbl*72}")
    print(f"  Actual today (CSV):   net=R$29.00  trades=21  (multi-lot, grid included)")

    # ── Trade-by-trade detail ──────────────────────────────────────────────────
    for label in ["OLD", "NEW"]:
        df = trades_map[label]
        if df.empty:
            continue
        print(f"\n  {label} trade log:")
        print(f"  {'#':<4} {'Dir':<6} {'Entry':>10} {'Exit':>10} {'Pts':>8} {'R$Net':>8} {'Reason'}")
        print(f"  {'-'*58}")
        for i, row in df.iterrows():
            print(
                f"  {i+1:<4} {row['direction']:<6} "
                f"{row['entry']:>10.0f} {row['exit_price']:>10.0f} "
                f"{row['pnl_pts']:>8.0f} {row['net_brl']:>8.2f}  {row['reason']}"
            )

    # ── Plot ───────────────────────────────────────────────────────────────────
    plot_labels = [k for k in PARAMS if k != "BASELINE"]
    n_panels    = len(plot_labels)
    fig, axes = plt.subplots(
        n_panels + 1, 1, figsize=(15, 4 + 3 * n_panels), sharex=True,
        gridspec_kw={"height_ratios": [4] + [1.5] * n_panels}
    )
    fig.suptitle(
        f"Backtest {today_str}  |  {SYMBOL} M5  |  OLD vs NEW parameters  |  {LOTS} lot",
        fontsize=12, fontweight="bold",
    )
    ax_price   = axes[0]
    panel_axes = {k: axes[i + 1] for i, k in enumerate(plot_labels)}

    # Price
    ax_price.plot(bars_today.index, bars_today["Close"],
                  color="steelblue", lw=1, label="Close")
    ax_price.set_ylabel("Price (R$)")
    ax_price.grid(alpha=0.3)
    ax_price.legend(loc="upper left", fontsize=8)

    color_map = {"OLD": "crimson", "NEW": "seagreen"}

    for label, ax in panel_axes.items():
        df  = trades_map.get(label, pd.DataFrame())
        col = color_map.get(label, "orange")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        if not df.empty:
            exit_ts = pd.to_datetime(df["exit_time"])
            cum_pnl = df["net_brl"].cumsum()
            ax.plot(exit_ts, cum_pnl.values, marker="o", ms=4, color=col, lw=1.5)
            for _, row in df.iterrows():
                clr = "lime" if row["net_brl"] > 0 else "salmon"
                ax_price.axvspan(
                    pd.to_datetime(row["entry_time"]),
                    pd.to_datetime(row["exit_time"]),
                    alpha=0.10, color=clr, label="_",
                )
        s = results_map.get(label, {})
        ax.set_title(
            f"{label}: {s.get('n_trades', 0)} trades  "
            f"win={s.get('win_rate', 0):.0%}  "
            f"net=R${s.get('net_brl', 0):.2f}  "
            f"avg=R${s.get('avg_pnl', 0):.2f}",
            fontsize=9, loc="left",
        )
        ax.set_ylabel("Cum P&L (R$)")
        ax.grid(alpha=0.3)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    out_path = "backtest_today.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n  Chart saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
