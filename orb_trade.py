"""
orb_trade.py — Opening Range Breakout (ORB) strategy.

Builds the opening range from the first ORB_BARS M5 bars each session
(default = first 3 bars = 09:00–09:14 BRT) and executes a market order
on the first confirmed breakout bar, with a fixed SL and TP based on the
ORB range.  At most one trade per direction per calendar day.

Architecture:
  - Uses ORB_MAGIC (separate from MAGIC_NUMBER) so positions are never
    confused with ML-driven trades.
  - TP is placed directly on the order; the broker handles the exit.
  - Daily state auto-resets on each new BRT calendar date.
  - Call process_orb(symbol, bars) each cycle from the main loop.
  - Call close_orb_positions(symbol) for EOD cleanup when outside session.
"""

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

from config import (
    TRADE_ENABLED,
    ORB_ENABLED, ORB_BARS, ORB_SESSION_START, ORB_SESSION_END,
    ORB_SL_MULT, ORB_TP_MULT, ORB_RISK_PCT, ORB_MAGIC, ORB_TRADE_SHORT,
    MAX_SLIPPAGE,
)

_BRT = timezone(timedelta(hours=-3))

# ── Per-day state ─────────────────────────────────────────────────────────────
# Resets automatically when the BRT calendar date changes.
_state: dict = {}   # symbol → {'date': date, 'buy_done': bool, 'sell_done': bool}


def _get_state(symbol: str) -> dict:
    today = datetime.now(_BRT).date()
    s = _state.get(symbol)
    if s is None or s['date'] != today:
        _state[symbol] = {'date': today, 'buy_done': False, 'sell_done': False}
    return _state[symbol]


def _in_orb_session() -> bool:
    now = datetime.now(_BRT)
    hm  = (now.hour, now.minute)
    return ORB_SESSION_START <= hm < ORB_SESSION_END


def compute_orb(bars: pd.DataFrame) -> Optional[Tuple[float, float, float]]:
    """
    Return (orb_high, orb_low, orb_range) from today's first ORB_BARS M5 bars.
    Returns None if fewer than ORB_BARS bars have formed today.

    Uses the raw bar-index date (broker clock = BRT) consistent with the rest of
    the codebase (pd.Timestamp.now().normalize() == broker date for BRT sessions).
    """
    today      = pd.Timestamp.now().normalize()
    today_bars = bars[bars.index.normalize() == today]
    if len(today_bars) < ORB_BARS:
        return None
    window    = today_bars.iloc[:ORB_BARS]
    orb_high  = float(window['High'].max())
    orb_low   = float(window['Low'].min())
    orb_range = orb_high - orb_low
    if orb_range <= 0:
        return None
    return orb_high, orb_low, orb_range


# ── Lot sizing ────────────────────────────────────────────────────────────────
def _lot_size(symbol: str, sl_pts: float) -> float:
    """Risk ORB_RISK_PCT% of account balance.  Falls back to minimum lot on errors."""
    info    = mt5.symbol_info(symbol)
    account = mt5.account_info()
    if info is None or account is None or sl_pts <= 0:
        return info.volume_min if info else 1.0
    tick_size           = info.trade_tick_size  if info.trade_tick_size  > 0 else 1.0
    tick_value          = info.trade_tick_value if info.trade_tick_value > 0 else 0.20
    point_value_per_lot = tick_value / tick_size
    risk_amount         = account.balance * ORB_RISK_PCT / 100.0
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
def _send_orb_order(
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
        'magic':        ORB_MAGIC,
        'comment':      f'orb_{side.lower()}',
        'type_time':    mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC,
    })
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        err = result.retcode if result else mt5.last_error()
        print(f"  [ORB] {side} failed: retcode={err}  price={price:.{digits}f}  SL={sl:.{digits}f}  TP={tp:.{digits}f}")
        return False

    sl_dist = abs(price - sl)
    tp_dist = abs(tp - price)
    print(
        f"  [ORB] {side} {lot} lots {symbol} @ {price:.{digits}f}"
        f"  SL={sl:.{digits}f} ({sl_dist:.0f}p)"
        f"  TP={tp:.{digits}f} ({tp_dist:.0f}p)"
        f"  R:R={tp_dist/sl_dist:.1f}"
    )
    return True


# ── EOD position close ────────────────────────────────────────────────────────
def close_orb_positions(symbol: str) -> None:
    """
    Close all open ORB positions for `symbol`.
    Called after ORB_SESSION_END to avoid holding overnight.
    Safe to call repeatedly — does nothing if no ORB positions are open.
    """
    positions = mt5.positions_get(symbol=symbol) or []
    for pos in positions:
        if pos.magic != ORB_MAGIC:
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
            'magic':        ORB_MAGIC,
            'comment':      'orb_eod',
            'type_time':    mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        })
        digits = info.digits if info else 2
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err = result.retcode if result else mt5.last_error()
            print(f"  [ORB] EOD close failed for #{pos.ticket}: retcode={err}")
        else:
            print(f"  [ORB] EOD closed #{pos.ticket} {pos.volume} lots @ {price:.{digits}f}")


# ── Main entry point ──────────────────────────────────────────────────────────
def process_orb(symbol: str, bars: pd.DataFrame) -> None:
    """
    Called each cycle with the latest M5 bars.
    Detects the first breakout bar above/below the ORB range and fires
    at most one trade per direction per day.

    Gates:
      - ORB_ENABLED and TRADE_ENABLED must be True
      - Must be inside ORB_SESSION_START .. ORB_SESSION_END (BRT)
      - ORB must be fully formed (≥ ORB_BARS bars today)
      - Current bar's close must break outside the ORB range
      - Not already traded that direction today
    """
    if not ORB_ENABLED or not TRADE_ENABLED:
        return

    state = _get_state(symbol)
    if state['buy_done'] and (state['sell_done'] or not ORB_TRADE_SHORT):
        return   # all available directions already traded today

    if not _in_orb_session():
        return

    orb = compute_orb(bars)
    if orb is None:
        return   # opening range not yet formed

    orb_high, orb_low, orb_range = orb
    current_close = float(bars['Close'].iloc[-1])

    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if info is None or tick is None:
        return

    sl_pts = orb_range * ORB_SL_MULT
    tp_pts = orb_range * ORB_TP_MULT

    print(
        f"  [ORB] {symbol}  close={current_close:.0f}"
        f"  range=[{orb_low:.0f}–{orb_high:.0f}] ({orb_range:.0f}p)"
        f"  SL={sl_pts:.0f}p  TP={tp_pts:.0f}p"
        f"  buy={'done' if state['buy_done'] else 'open'}"
        f"  sell={'done' if state['sell_done'] else 'open'}"
    )

    # ── BUY: close > ORB High ─────────────────────────────────────────────────
    if not state['buy_done'] and current_close > orb_high:
        price = tick.ask
        sl    = price - sl_pts
        tp    = price + tp_pts
        lot   = _lot_size(symbol, sl_pts)
        print(f"  [ORB] BUY breakout  close={current_close:.0f} > orb_high={orb_high:.0f}")
        if _send_orb_order(symbol, mt5.ORDER_TYPE_BUY, lot, price, sl, tp, info):
            state['buy_done'] = True

    # ── SELL: close < ORB Low ─────────────────────────────────────────────────
    if ORB_TRADE_SHORT and not state['sell_done'] and current_close < orb_low:
        price = tick.bid
        sl    = price + sl_pts
        tp    = price - tp_pts
        lot   = _lot_size(symbol, sl_pts)
        print(f"  [ORB] SELL breakout  close={current_close:.0f} < orb_low={orb_low:.0f}")
        if _send_orb_order(symbol, mt5.ORDER_TYPE_SELL, lot, price, sl, tp, info):
            state['sell_done'] = True
