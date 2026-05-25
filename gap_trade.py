"""
gap_trade.py — Opening Gap Fill strategy.

WIN futures (mini Ibovespa) regularly open with a price gap vs the previous
session's close.  This module fades that gap (counter-trend, mean-reversion)
expecting price to return to the previous close within the first 30 minutes.

Logic:
  - Gap Up  (today_open > prev_close): SELL at market, TP = prev_close, SL above open
  - Gap Down (today_open < prev_close): BUY  at market, TP = prev_close, SL below open

Filters:
  - Gap size must be in [GAP_MIN_PTS, GAP_MAX_PTS] (noise filter + trend filter)
  - Entry only inside GAP_ENTRY_START .. GAP_ENTRY_END window (first ~25 min)
  - Current close must still be on the gap side (gap not yet fully filled)
  - At most one trade per direction per calendar day

Architecture:
  - Uses GAP_MAGIC (separate from ORB_MAGIC and MAGIC_NUMBER)
  - Daily state resets automatically on each new BRT calendar date
  - Call process_gap(symbol, bars) each cycle from the main loop
  - Call close_gap_positions(symbol) for EOD cleanup when outside session
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

from config import (
    TRADE_ENABLED,
    GAP_ENABLED, GAP_MIN_PTS, GAP_MAX_PTS,
    GAP_SL_MULT, GAP_TP_MULT, GAP_RISK_PCT, GAP_MAGIC,
    GAP_ENTRY_START, GAP_ENTRY_END, GAP_TRADE_SHORT,
    MAX_SLIPPAGE,
)

_BRT = timezone(timedelta(hours=-3))

# ── Per-day state ─────────────────────────────────────────────────────────────
_state: dict = {}   # symbol → {'date': date, 'buy_done': bool, 'sell_done': bool}


def _get_state(symbol: str) -> dict:
    today = datetime.now(_BRT).date()
    s = _state.get(symbol)
    if s is None or s['date'] != today:
        _state[symbol] = {'date': today, 'buy_done': False, 'sell_done': False}
    return _state[symbol]


def _in_entry_window() -> bool:
    now = datetime.now(_BRT)
    hm  = (now.hour, now.minute)
    return GAP_ENTRY_START <= hm < GAP_ENTRY_END


def compute_gap(bars: pd.DataFrame) -> Optional[Tuple[float, float, float, int]]:
    """
    Return (gap_size, today_open, prev_close, gap_dir) or None.
      gap_dir: +1 = gap up (open above prev close), -1 = gap down.

    Strips the UTC timezone label from the bars index because the broker
    stores BRT timestamps tagged as UTC (same convention as the ORB module).
    """
    naive      = bars.index.tz_localize(None)
    today      = pd.Timestamp.now().normalize()
    today_bars = bars[naive.normalize() == today]
    prev_bars  = bars[naive.normalize() < today]

    if today_bars.empty or prev_bars.empty:
        return None

    today_open = float(today_bars['Open'].iloc[0])
    prev_close = float(prev_bars['Close'].iloc[-1])
    gap_size   = abs(today_open - prev_close)
    gap_dir    = 1 if today_open > prev_close else -1

    if gap_size <= 0:
        return None

    return gap_size, today_open, prev_close, gap_dir


# ── Lot sizing ────────────────────────────────────────────────────────────────
def _lot_size(symbol: str, sl_pts: float) -> float:
    """Risk GAP_RISK_PCT% of account balance.  Falls back to minimum lot on errors."""
    info    = mt5.symbol_info(symbol)
    account = mt5.account_info()
    if info is None or account is None or sl_pts <= 0:
        return info.volume_min if info else 1.0
    tick_size           = info.trade_tick_size  if info.trade_tick_size  > 0 else 1.0
    tick_value          = info.trade_tick_value if info.trade_tick_value > 0 else 0.20
    point_value_per_lot = tick_value / tick_size
    risk_amount         = account.balance * GAP_RISK_PCT / 100.0
    raw_lot             = risk_amount / (sl_pts * point_value_per_lot)
    step                = info.volume_step if info.volume_step > 0 else 0.01
    lot                 = round(raw_lot / step) * step
    return max(info.volume_min, min(info.volume_max, round(lot, 2)))


# ── Stop snapping ─────────────────────────────────────────────────────────────
def _snap(value: float, tick_size: float, digits: int) -> float:
    return round(round(value / tick_size) * tick_size, digits)


def _adjust_stops(
    order_type: int,
    price: float,
    sl: float,
    tp: float,
    info,
) -> Tuple[float, float]:
    """Snap SL/TP to tick grid and enforce broker minimum-distance requirement."""
    digits    = info.digits
    tick_size = info.trade_tick_size if info.trade_tick_size > 0 else info.point
    stops_lvl = int(getattr(info, 'trade_stops_level', 0) or 0)
    min_dist  = stops_lvl * info.point + tick_size
    is_buy    = order_type == mt5.ORDER_TYPE_BUY

    sl = _snap(sl, tick_size, digits)
    tp = _snap(tp, tick_size, digits)

    if is_buy:
        if price - sl < min_dist:
            sl = _snap(price - min_dist, tick_size, digits)
        if tp - price < min_dist:
            tp = _snap(price + min_dist, tick_size, digits)
    else:
        if sl - price < min_dist:
            sl = _snap(price + min_dist, tick_size, digits)
        if price - tp < min_dist:
            tp = _snap(price - min_dist, tick_size, digits)
    return sl, tp


# ── Order execution ───────────────────────────────────────────────────────────
def _send_gap_order(
    symbol: str,
    order_type: int,
    lot: float,
    price: float,
    sl: float,
    tp: float,
    info,
) -> bool:
    digits = info.digits
    sl, tp = _adjust_stops(order_type, price, sl, tp, info)
    is_buy = order_type == mt5.ORDER_TYPE_BUY
    side   = 'BUY' if is_buy else 'SELL'

    result = mt5.order_send({
        'action':       mt5.TRADE_ACTION_DEAL,
        'symbol':       symbol,
        'volume':       lot,
        'type':         order_type,
        'price':        price,
        'sl':           sl,
        'tp':           tp,
        'deviation':    MAX_SLIPPAGE,
        'magic':        GAP_MAGIC,
        'comment':      f'gap_{side.lower()}',
        'type_time':    mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC,
    })
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        err = result.retcode if result else mt5.last_error()
        print(f"  [GAP] {side} failed: retcode={err}  price={price:.{digits}f}  SL={sl:.{digits}f}  TP={tp:.{digits}f}")
        return False

    sl_dist = abs(price - sl)
    tp_dist = abs(tp - price)
    rr      = tp_dist / sl_dist if sl_dist > 0 else 0
    print(
        f"  [GAP] {side} {lot} lots {symbol} @ {price:.{digits}f}"
        f"  SL={sl:.{digits}f} ({sl_dist:.0f}p)"
        f"  TP={tp:.{digits}f} ({tp_dist:.0f}p)"
        f"  R:R={rr:.1f}"
    )
    return True


# ── EOD position close ────────────────────────────────────────────────────────
def close_gap_positions(symbol: str) -> None:
    """
    Close all open GAP positions for `symbol`.
    Called after GAP_ENTRY_END to avoid holding an unfilled gap overnight.
    Safe to call repeatedly — does nothing if no GAP positions are open.
    """
    positions = mt5.positions_get(symbol=symbol) or []
    for pos in positions:
        if pos.magic != GAP_MAGIC:
            continue
        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        if tick is None:
            continue
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price      = tick.bid              if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        result = mt5.order_send({
            'action':       mt5.TRADE_ACTION_DEAL,
            'symbol':       symbol,
            'volume':       pos.volume,
            'type':         close_type,
            'position':     pos.ticket,
            'price':        price,
            'deviation':    MAX_SLIPPAGE,
            'magic':        GAP_MAGIC,
            'comment':      'gap_eod',
            'type_time':    mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        })
        digits = info.digits if info else 2
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err = result.retcode if result else mt5.last_error()
            print(f"  [GAP] EOD close failed for #{pos.ticket}: retcode={err}")
        else:
            print(f"  [GAP] EOD closed #{pos.ticket} {pos.volume} lots @ {price:.{digits}f}")


# ── Main entry point ──────────────────────────────────────────────────────────
def process_gap(symbol: str, bars: pd.DataFrame) -> None:
    """
    Called each cycle with the latest M5 bars.
    Detects an opening gap and fades it at market if conditions are met.
    At most one trade per direction per day.

    Gates:
      - GAP_ENABLED and TRADE_ENABLED must be True
      - Must be inside GAP_ENTRY_START .. GAP_ENTRY_END window (BRT)
      - Gap must be in [GAP_MIN_PTS, GAP_MAX_PTS]
      - Current close must still show the gap (not already filled)
      - Not already traded that direction today
    """
    if not GAP_ENABLED or not TRADE_ENABLED:
        return

    state = _get_state(symbol)
    if state['buy_done'] and (state['sell_done'] or not GAP_TRADE_SHORT):
        return

    if not _in_entry_window():
        return

    result = compute_gap(bars)
    if result is None:
        return

    gap_size, today_open, prev_close, gap_dir = result

    if not (GAP_MIN_PTS <= gap_size <= GAP_MAX_PTS):
        return  # too small (noise) or too large (strong overnight trend, may not fill)

    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if info is None or tick is None:
        return

    current_close = float(bars['Close'].iloc[-1])
    sl_pts = gap_size * GAP_SL_MULT

    print(
        f"  [GAP] {symbol}  gap={gap_size:.0f}p  dir={'UP' if gap_dir == 1 else 'DOWN'}"
        f"  open={today_open:.0f}  prev_close={prev_close:.0f}"
        f"  close={current_close:.0f}"
        f"  buy={'done' if state['buy_done'] else 'open'}"
        f"  sell={'done' if state['sell_done'] else 'open'}"
    )

    # ── Gap Up → fade with SELL ───────────────────────────────────────────────
    # Price opened above prev_close; enter short expecting it to return to prev_close.
    # Only enter if current close is still above prev_close (gap not yet filled).
    if gap_dir == 1 and GAP_TRADE_SHORT and not state['sell_done']:
        if current_close > prev_close + GAP_MIN_PTS:
            price = tick.bid
            sl    = price + sl_pts      # SL above entry (we lose if price continues up)
            tp    = prev_close          # TP at gap fill level
            lot   = _lot_size(symbol, sl_pts)
            print(f"  [GAP] SELL gap-up fade  close={current_close:.0f} > prev_close={prev_close:.0f}")
            if _send_gap_order(symbol, mt5.ORDER_TYPE_SELL, lot, price, sl, tp, info):
                state['sell_done'] = True

    # ── Gap Down → fade with BUY ──────────────────────────────────────────────
    # Price opened below prev_close; enter long expecting it to return to prev_close.
    # Only enter if current close is still below prev_close (gap not yet filled).
    if gap_dir == -1 and not state['buy_done']:
        if current_close < prev_close - GAP_MIN_PTS:
            price = tick.ask
            sl    = price - sl_pts      # SL below entry (we lose if price continues down)
            tp    = prev_close          # TP at gap fill level
            lot   = _lot_size(symbol, sl_pts)
            print(f"  [GAP] BUY  gap-down fade  close={current_close:.0f} < prev_close={prev_close:.0f}")
            if _send_gap_order(symbol, mt5.ORDER_TYPE_BUY, lot, price, sl, tp, info):
                state['buy_done'] = True
