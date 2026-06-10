"""
renko_today.py — Build and plot a 35-tick R-type Renko chart for today's session.

Brick definition
----------------
  Brick size  : 35 ticks  (35 × symbol.trade_tick_size points)
  Type        : R-type with -1 tick reversal
                  • Continuation : price moves ≥ brick_size from last brick close
                                   in the same direction
                  • Reversal     : price moves ≥ (brick_size − 1 tick) from last
                                   brick close in the opposite direction
                The -1 tick makes reversals 1 tick more sensitive than a new
                continuation, matching MetaTrader's "Renko −1 tick" bar type.

Data source
-----------
  mt5.copy_ticks_range(..., COPY_TICKS_TRADE) — trade ticks only (no bid/ask noise).
  Uses the `price` field (last traded price).

Usage
-----
  python renko_today.py [SYMBOL]          # default symbol from config.py (WINM26)
  python renko_today.py WINM26
  python renko_today.py EURUSD
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from config import TARGETS

# ── Constants ─────────────────────────────────────────────────────────────────
BRICK_TICKS      = 35    # brick size in ticks
REVERSAL_OFFSET  = 1     # R-type: reversal threshold = (BRICK_TICKS - REVERSAL_OFFSET) ticks
OUTPUT_FILE      = "renko_today.png"


# ── MT5 helpers ───────────────────────────────────────────────────────────────
def _init_mt5(symbol: str) -> None:
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Could not select {symbol} in Market Watch")


def _get_tick_size(symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"symbol_info({symbol}) returned None")
    return info.trade_tick_size


def _fetch_today_ticks(symbol: str) -> pd.DataFrame:
    """Return all trade ticks for today (UTC midnight → now)."""
    now        = datetime.now(timezone.utc)
    day_start  = now.replace(hour=0, minute=0, second=0, microsecond=0)

    ticks = mt5.copy_ticks_range(
        symbol,
        day_start,
        now,
        mt5.COPY_TICKS_TRADE,
    )
    if ticks is None or len(ticks) == 0:
        raise RuntimeError(
            f"No trade ticks returned for {symbol} today. "
            "Ensure MT5 is connected and the market has traded today."
        )

    df = pd.DataFrame(ticks)
    if 'time_msc' in df.columns:
        df['dt'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True)
    else:
        df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)

    # `price` column contains the last traded price for COPY_TICKS_TRADE
    df = df[df['price'] > 0].copy()
    if df.empty:
        raise RuntimeError(f"All {symbol} trade ticks have price=0. Check symbol type.")
    return df.reset_index(drop=True)


# ── Renko builder ─────────────────────────────────────────────────────────────
def build_renko(prices: np.ndarray, brick_size: float, tick_size: float) -> pd.DataFrame:
    """
    Build R-type Renko bricks from a price array.

    Parameters
    ----------
    prices     : 1-D array of traded prices (chronological)
    brick_size : size of one brick in price units  (BRICK_TICKS × tick_size)
    tick_size  : minimum price increment for the symbol

    Returns
    -------
    DataFrame with columns: open, close, dir  (+1 = up, -1 = down)
    """
    reversal_size = brick_size - tick_size  # R-type: 34-tick reversal

    bricks: list[dict] = []
    direction = 0          # 0 = not yet set, +1 = up, -1 = down
    last_close = prices[0]

    for price in prices:
        if direction >= 0:   # uptrend or undecided
            # Continuation UP
            if price >= last_close + brick_size:
                while price >= last_close + brick_size:
                    bricks.append({'open': last_close,
                                   'close': last_close + brick_size,
                                   'dir': 1})
                    last_close += brick_size
                direction = 1
            # Reversal DOWN (R-type: -1 tick threshold)
            elif direction == 1 and price <= last_close - reversal_size:
                while price <= last_close - brick_size:
                    bricks.append({'open': last_close,
                                   'close': last_close - brick_size,
                                   'dir': -1})
                    last_close -= brick_size
                direction = -1
        else:                # downtrend
            # Continuation DOWN
            if price <= last_close - brick_size:
                while price <= last_close - brick_size:
                    bricks.append({'open': last_close,
                                   'close': last_close - brick_size,
                                   'dir': -1})
                    last_close -= brick_size
                direction = -1
            # Reversal UP (R-type: -1 tick threshold)
            elif price >= last_close + reversal_size:
                while price >= last_close + brick_size:
                    bricks.append({'open': last_close,
                                   'close': last_close + brick_size,
                                   'dir': 1})
                    last_close += brick_size
                direction = 1

    if not bricks:
        raise RuntimeError(
            f"No Renko bricks formed. The day's price range may be less than "
            f"one brick ({brick_size:.2f} pts = {BRICK_TICKS} ticks)."
        )
    return pd.DataFrame(bricks)


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_renko(renko: pd.DataFrame, symbol: str, brick_size: float,
               brick_ticks: int, tick_size: float) -> None:
    """Render Renko bricks as a bar chart and save / show."""
    n = len(renko)
    fig, ax = plt.subplots(figsize=(max(14, n * 0.25), 7))

    bar_width = 0.8
    for i, row in renko.iterrows():
        color      = '#26a69a' if row['dir'] == 1 else '#ef5350'  # teal / red
        edge_color = '#1a756e' if row['dir'] == 1 else '#b03a37'
        lo, hi     = sorted([row['open'], row['close']])
        rect = mpatches.FancyBboxPatch(
            (i - bar_width / 2, lo),
            bar_width,
            hi - lo,
            boxstyle="square,pad=0",
            linewidth=0.6,
            edgecolor=edge_color,
            facecolor=color,
        )
        ax.add_patch(rect)

    # Price axis limits
    price_lo = renko[['open', 'close']].min().min()
    price_hi = renko[['open', 'close']].max().max()
    margin   = brick_size * 2
    ax.set_xlim(-1, n)
    ax.set_ylim(price_lo - margin, price_hi + margin)

    # Tick labels: show every ~20 bricks on x-axis
    step = max(1, n // 20)
    ax.set_xticks(range(0, n, step))
    ax.set_xticklabels(range(0, n, step), fontsize=8)

    # Horizontal grid lines every brick_size
    grid_lo = int(price_lo / brick_size) * brick_size
    grid_hi = int(price_hi / brick_size + 1) * brick_size
    grid_prices = np.arange(grid_lo, grid_hi + brick_size, brick_size)
    for gp in grid_prices:
        ax.axhline(gp, color='#cccccc', linewidth=0.4, linestyle='--', zorder=0)

    # Direction-change markers
    prev_dir = renko['dir'].iloc[0]
    for i in range(1, n):
        cur_dir = renko['dir'].iloc[i]
        if cur_dir != prev_dir:
            label = '▲' if cur_dir == 1 else '▼'
            color  = '#26a69a' if cur_dir == 1 else '#ef5350'
            y_pos  = renko['close'].iloc[i]
            ax.text(i, y_pos + (brick_size * 0.3 if cur_dir == 1 else -brick_size * 0.8),
                    label, ha='center', va='bottom', fontsize=7, color=color)
        prev_dir = cur_dir

    # Last price label
    last_close = renko['close'].iloc[-1]
    last_dir   = renko['dir'].iloc[-1]
    ax.text(n - 0.5, last_close,
            f' {last_close:.0f}', va='center', fontsize=9,
            color='#26a69a' if last_dir == 1 else '#ef5350', fontweight='bold')

    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    ax.set_title(
        f"{symbol}  ·  Renko {brick_ticks}R  (brick = {brick_ticks} ticks = "
        f"{brick_size:.0f} pts, R-type −1 tick reversal)  ·  {today_str}",
        fontsize=11, pad=10
    )
    ax.set_xlabel("Brick index", fontsize=9)
    ax.set_ylabel("Price", fontsize=9)

    up_patch   = mpatches.Patch(color='#26a69a', label='Up brick')
    down_patch = mpatches.Patch(color='#ef5350', label='Down brick')
    ax.legend(handles=[up_patch, down_patch], loc='upper left', fontsize=9)

    ax.grid(axis='y', color='#eeeeee', linewidth=0.3)
    fig.tight_layout()

    fig.savefig(OUTPUT_FILE, dpi=150)
    print(f"Chart saved → {OUTPUT_FILE}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    symbol = sys.argv[1] if len(sys.argv) > 1 else TARGETS[0]['symbol']
    print(f"Symbol  : {symbol}")
    print(f"Bricks  : {BRICK_TICKS}R  (R-type, −1 tick reversal)")

    _init_mt5(symbol)
    try:
        tick_size  = _get_tick_size(symbol)
        brick_size = BRICK_TICKS * tick_size

        print(f"Tick size  : {tick_size}")
        print(f"Brick size : {brick_size:.4f}  ({BRICK_TICKS} ticks)")
        print(f"Reversal   : {brick_size - tick_size:.4f}  ({BRICK_TICKS - REVERSAL_OFFSET} ticks)")

        print("Fetching today's trade ticks …")
        ticks_df = _fetch_today_ticks(symbol)
        print(f"Ticks received : {len(ticks_df):,}")

        renko = build_renko(ticks_df['price'].values, brick_size, tick_size)
        print(f"Renko bricks   : {len(renko)}  "
              f"(↑ {(renko['dir'] == 1).sum()}  ↓ {(renko['dir'] == -1).sum()})")

        plot_renko(renko, symbol, brick_size, BRICK_TICKS, tick_size)
    finally:
        mt5.shutdown()


if __name__ == '__main__':
    main()
