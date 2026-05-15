"""
trade.py — Order execution, position management, trailing stop, and Fibonacci grid.
"""

import MetaTrader5 as mt5

from config import (
    TARGETS,
    TRADE_ENABLED, TRADE_BOTH_SIDES,
    RISK_PCT, MAX_SLIPPAGE, MAGIC_NUMBER,
    TRAIL_ACTIVATE_PCT,
    GRID_ENABLED, GRID_MAX_LEVELS, GRID_STEP_MULT, GRID_PORTFOLIO_SL_MULT,
)

# ── Module-level state ────────────────────────────────────────────────────────
_last_exec_signal: dict = {}          # symbol → last signal that triggered an order
_FIB: list = [1, 1, 2, 3, 5, 8, 13, 21]  # Fibonacci multipliers per grid level
_grid_state: dict = {}                # ticket → set of grid levels already placed


# ── Startup: restore signal state from open positions ─────────────────────────
def init_signal_state() -> None:
    """
    On script startup, infer _last_exec_signal from any already-open MT5
    positions so the first bar after a restart doesn't double-enter or
    prematurely reverse an existing trade.
    """
    for t in TARGETS:
        symbol    = t['symbol']
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            continue
        for pos in positions:
            if pos.magic != MAGIC_NUMBER:
                continue
            inferred = 1 if pos.type == mt5.ORDER_TYPE_BUY else 0
            _last_exec_signal[symbol] = inferred
            side = 'BUY' if inferred == 1 else 'SELL'
            print(f"  [trade] startup: restored {symbol} signal → {side} (ticket #{pos.ticket})")
            break  # one seed position per symbol is enough


# ── Lot sizing ────────────────────────────────────────────────────────────────
def _get_lot_size(symbol: str, sl_price_units: float) -> float:
    """Risk RISK_PCT% of account balance. Falls back to minimum lot on errors."""
    info    = mt5.symbol_info(symbol)
    account = mt5.account_info()
    if info is None or account is None or sl_price_units <= 0:
        return info.volume_min if info else 1.0
    tick_size           = info.trade_tick_size  if info.trade_tick_size  > 0 else 1.0
    tick_value          = info.trade_tick_value if info.trade_tick_value > 0 else 0.20
    point_value_per_lot = tick_value / tick_size
    risk_amount         = account.balance * RISK_PCT / 100.0
    raw_lot             = risk_amount / (sl_price_units * point_value_per_lot)
    step                = info.volume_step if info.volume_step > 0 else 0.01
    lot                 = round(raw_lot / step) * step
    lot                 = max(info.volume_min, min(info.volume_max, lot))
    return round(lot, 2)


def _fib_lot(base_lot: float, level: int, info) -> float:
    """Return base_lot × Fibonacci(level) clamped to broker lot constraints."""
    fib_mult = _FIB[min(level - 1, len(_FIB) - 1)]
    raw      = base_lot * fib_mult
    step     = info.volume_step if info.volume_step > 0 else 0.01
    lot      = round(raw / step) * step
    return max(info.volume_min, min(info.volume_max, round(lot, 2)))


# ── Position management ───────────────────────────────────────────────────────
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


# ── Stop validation ───────────────────────────────────────────────────────────
def _validate_stops(price: float, sl: float, tp: float, order_type: int, info) -> tuple:
    """
    Validate and snap SL to broker constraints.
    tp=0.0 means no take-profit — all TP checks are skipped.
    Returns (sl, tp, warnings: list[str]).
    """
    warnings_out: list = []
    point     = info.point           if info.point           > 0 else 0.01
    tick_size = info.trade_tick_size  if info.trade_tick_size  > 0 else point
    digits    = info.digits

    stops_lvl = int(getattr(info, 'trade_stops_level',  0) or 0)
    freeze_lvl = int(getattr(info, 'trade_freeze_level', 0) or 0)
    min_dist  = (stops_lvl + freeze_lvl) * point + tick_size   # one extra tick safety margin

    is_buy = (order_type == mt5.ORDER_TYPE_BUY)

    def snap(v: float) -> float:
        return round(round(v / tick_size) * tick_size, digits)

    sl = snap(sl)
    if tp != 0.0:
        tp = snap(tp)

    # Direction sanity
    if is_buy:
        if sl >= price:
            sl = snap(price - min_dist)
            warnings_out.append(f"SL clamped below price to {sl:.{digits}f}")
    else:
        if sl <= price:
            sl = snap(price + min_dist)
            warnings_out.append(f"SL clamped above price to {sl:.{digits}f}")

    # Minimum distance from price
    if is_buy:
        if (price - sl) < min_dist:
            sl = snap(price - min_dist)
            warnings_out.append(f"SL widened to satisfy min_dist → {sl:.{digits}f}")
    else:
        if (sl - price) < min_dist:
            sl = snap(price + min_dist)
            warnings_out.append(f"SL widened to satisfy min_dist → {sl:.{digits}f}")

    return sl, tp, warnings_out


# ── Signal-driven order execution ─────────────────────────────────────────────
def execute_trade(symbol: str, current_signal: int, sl_price_units: float) -> None:
    """Send an order when signal transitions; respects TRADE_ENABLED flag."""
    if not TRADE_ENABLED:
        return
    prev_signal = _last_exec_signal.get(symbol, -1)
    if current_signal == prev_signal:
        return

    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if tick is None or info is None:
        print(f"  [trade] symbol info unavailable for {symbol}")
        return

    open_positions = mt5.positions_get(symbol=symbol) or []
    strategy_positions = [p for p in open_positions if p.magic == MAGIC_NUMBER]
    if strategy_positions:
        if _close_symbol_position(symbol):
            print(f"  [trade] seed order deferred for {symbol}: existing positions were closed; waiting for flat re-entry")
        return

    if current_signal == 1:   # ── BUY ─────────────────────────────────────────
        lot    = _get_lot_size(symbol, sl_price_units)
        price  = tick.ask
        sl_raw = price - sl_price_units
        sl, _tp, warns = _validate_stops(price, sl_raw, 0.0, mt5.ORDER_TYPE_BUY, info)
        for w in warns:
            print(f"  [trade] stop-adjust BUY: {w}")
        result = mt5.order_send({
            'action':       mt5.TRADE_ACTION_DEAL,
            'symbol':       symbol,
            'volume':       lot,
            'type':         mt5.ORDER_TYPE_BUY,
            'price':        price,
            'sl':           sl,
            'tp':           0.0,
            'deviation':    MAX_SLIPPAGE,
            'magic':        MAGIC_NUMBER,
            'comment':      'ml_buy',
            'type_time':    mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        })
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err = result.retcode if result else mt5.last_error()
            print(f"  [trade] BUY failed: retcode={err}  price={price}  SL={sl}")
        else:
            _last_exec_signal[symbol] = current_signal
            print(f"  [trade] BUY  {lot} lots {symbol} @ {price:.{info.digits}f}  SL={sl:.{info.digits}f}  no-TP")

    elif current_signal == 0:  # ── SELL / EXIT ───────────────────────────────
        if not TRADE_BOTH_SIDES:
            _last_exec_signal[symbol] = current_signal
            return
        lot    = _get_lot_size(symbol, sl_price_units)
        price  = tick.bid
        sl_raw = price + sl_price_units
        sl, _tp, warns = _validate_stops(price, sl_raw, 0.0, mt5.ORDER_TYPE_SELL, info)
        for w in warns:
            print(f"  [trade] stop-adjust SELL: {w}")
        result = mt5.order_send({
            'action':       mt5.TRADE_ACTION_DEAL,
            'symbol':       symbol,
            'volume':       lot,
            'type':         mt5.ORDER_TYPE_SELL,
            'price':        price,
            'sl':           sl,
            'tp':           0.0,
            'deviation':    MAX_SLIPPAGE,
            'magic':        MAGIC_NUMBER,
            'comment':      'ml_sell',
            'type_time':    mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        })
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err = result.retcode if result else mt5.last_error()
            print(f"  [trade] SELL failed: retcode={err}  price={price}  SL={sl}")
        else:
            _last_exec_signal[symbol] = current_signal
            print(f"  [trade] SELL {lot} lots {symbol} @ {price:.{info.digits}f}  SL={sl:.{info.digits}f}  no-TP")


# ── Trailing stop manager ─────────────────────────────────────────────────────
def manage_trailing_stops() -> None:
    """
    Trail the SL of every open MAGIC_NUMBER position once unrealised profit
    reaches TRAIL_ACTIVATE_PCT × initial SL distance. The SL follows price at
    the same fixed distance, never moving against the trade.
    """
    if not TRADE_ENABLED:
        return
    for t in TARGETS:
        symbol    = t['symbol']
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            continue
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info is None or tick is None:
            continue
        tick_size = info.trade_tick_size if info.trade_tick_size > 0 else info.point
        digits    = info.digits

        for pos in positions:
            if pos.magic != MAGIC_NUMBER or pos.sl == 0.0:
                continue
            sl_dist       = abs(pos.price_open - pos.sl)
            if sl_dist <= 0:
                continue
            activate_dist = TRAIL_ACTIVATE_PCT * sl_dist

            if pos.type == mt5.ORDER_TYPE_BUY:
                current_price = tick.bid
                profit_dist   = current_price - pos.price_open
                if profit_dist < activate_dist:
                    continue
                trail_sl = round(round((current_price - sl_dist) / tick_size) * tick_size, digits)
                if trail_sl <= pos.sl:
                    continue
            else:
                current_price = tick.ask
                profit_dist   = pos.price_open - current_price
                if profit_dist < activate_dist:
                    continue
                trail_sl = round(round((current_price + sl_dist) / tick_size) * tick_size, digits)
                if trail_sl >= pos.sl:
                    continue

            result = mt5.order_send({
                'action':   mt5.TRADE_ACTION_SLTP,
                'position': pos.ticket,
                'symbol':   symbol,
                'sl':       trail_sl,
                'tp':       pos.tp,
            })
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                err = result.retcode if result else mt5.last_error()
                print(f"  [trail] SL modify failed #{pos.ticket}: retcode={err}")
            else:
                side = 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL'
                print(f"  [trail] #{pos.ticket} {side} SL "
                      f"{pos.sl:.{digits}f} → {trail_sl:.{digits}f}  "
                      f"profit={profit_dist:.{digits}f}  threshold={activate_dist:.{digits}f}")


# ── Portfolio-level equity stop ──────────────────────────────────────────────
def _portfolio_stop_loss(symbol: str, all_positions: list, seed, info) -> bool:
    """
    Computes total floating PnL (in account currency) across all MAGIC_NUMBER
    positions for the symbol. If the loss exceeds
        GRID_PORTFOLIO_SL_MULT × seed_sl_dist × point_value_per_lot × seed_lot
    every position is closed atomically and True is returned.

    This replaces the per-order SL logic for grid consistency: the whole grid
    is treated as one combined position with a single maximum-loss threshold.
    """
    total_profit = sum(p.profit for p in all_positions if p.magic == MAGIC_NUMBER)
    if total_profit >= 0:
        return False  # in profit or flat — nothing to do

    sl_dist = abs(seed.price_open - seed.sl)
    if sl_dist <= 0:
        return False

    tick_size           = info.trade_tick_size  if info.trade_tick_size  > 0 else info.point
    tick_value          = info.trade_tick_value if info.trade_tick_value > 0 else 0.20
    point_value_per_lot = tick_value / tick_size

    # Monetary value of one full seed SL hit (at seed lot size)
    seed_sl_value = sl_dist * point_value_per_lot * seed.volume
    max_loss      = GRID_PORTFOLIO_SL_MULT * seed_sl_value

    if abs(total_profit) >= max_loss:
        print(
            f"  [grid] PORTFOLIO STOP {symbol}: "
            f"loss={total_profit:.2f}  threshold=-{max_loss:.2f}  "
            f"({GRID_PORTFOLIO_SL_MULT}× seed SL value)"
        )
        _close_symbol_position(symbol)
        return True
    return False


# ── Fibonacci grid manager ────────────────────────────────────────────────────
def manage_grid_orders() -> None:
    """
    For each open seed position (ml_buy / ml_sell) with MAGIC_NUMBER, place
    additional orders in the loss direction as price moves against us.

    Grid levels are spaced GRID_STEP_MULT × SL_distance apart.
    Volume at each level = base_lot × Fibonacci(level):
        L1: ×1  L2: ×1  L3: ×2  L4: ×3  L5: ×5 …

    Grid orders carry the same MAGIC_NUMBER so _close_symbol_position() removes
    all of them on the next signal flip. Closed-position state is pruned each cycle.
    """
    if not TRADE_ENABLED or not GRID_ENABLED:
        return

    current_tickets: set = set()

    for t in TARGETS:
        symbol    = t['symbol']
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            continue
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info is None or tick is None:
            continue
        digits = info.digits

        seed_positions = [
            p for p in positions
            if p.magic == MAGIC_NUMBER
            and str(p.comment).startswith(('ml_buy', 'ml_sell'))
        ]

        for pos in seed_positions:
            current_tickets.add(pos.ticket)
            if pos.sl == 0.0:
                continue
            # Portfolio equity stop — close everything if total loss is too large
            if _portfolio_stop_loss(symbol, positions, pos, info):
                break  # positions are gone; skip remaining seeds this cycle
            sl_dist   = abs(pos.price_open - pos.sl)
            if sl_dist <= 0:
                continue
            grid_step = GRID_STEP_MULT * sl_dist
            base_lot  = pos.volume
            placed    = _grid_state.setdefault(pos.ticket, set())

            if pos.type == mt5.ORDER_TYPE_BUY:
                current_price = tick.ask
                for level in range(1, GRID_MAX_LEVELS + 1):
                    if level in placed:
                        continue
                    grid_trigger = pos.price_open - level * grid_step
                    if current_price > grid_trigger:
                        continue
                    lot    = _fib_lot(base_lot, level, info)
                    sl_raw = grid_trigger - sl_dist
                    sl, _tp, warns = _validate_stops(current_price, sl_raw, 0.0, mt5.ORDER_TYPE_BUY, info)
                    for w in warns:
                        print(f"  [grid] stop-adjust BUY L{level}: {w}")
                    result = mt5.order_send({
                        'action':       mt5.TRADE_ACTION_DEAL,
                        'symbol':       symbol,
                        'volume':       lot,
                        'type':         mt5.ORDER_TYPE_BUY,
                        'price':        current_price,
                        'sl':           sl,
                        'tp':           0.0,
                        'deviation':    MAX_SLIPPAGE,
                        'magic':        MAGIC_NUMBER,
                        'comment':      f'ml_grid_{level}',
                        'type_time':    mt5.ORDER_TIME_GTC,
                        'type_filling': mt5.ORDER_FILLING_IOC,
                    })
                    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                        err = result.retcode if result else mt5.last_error()
                        print(f"  [grid] BUY L{level} failed #{pos.ticket}: "
                              f"retcode={err}  trigger={grid_trigger:.{digits}f}")
                    else:
                        placed.add(level)
                        fib_mult = _FIB[min(level - 1, len(_FIB) - 1)]
                        print(f"  [grid] BUY L{level} #{pos.ticket} {lot}lots "
                              f"@ {current_price:.{digits}f}  fib×{fib_mult}  SL={sl:.{digits}f}  no-TP")

            else:  # SELL seed
                current_price = tick.bid
                for level in range(1, GRID_MAX_LEVELS + 1):
                    if level in placed:
                        continue
                    grid_trigger = pos.price_open + level * grid_step
                    if current_price < grid_trigger:
                        continue
                    lot    = _fib_lot(base_lot, level, info)
                    sl_raw = grid_trigger + sl_dist
                    sl, _tp, warns = _validate_stops(current_price, sl_raw, 0.0, mt5.ORDER_TYPE_SELL, info)
                    for w in warns:
                        print(f"  [grid] stop-adjust SELL L{level}: {w}")
                    result = mt5.order_send({
                        'action':       mt5.TRADE_ACTION_DEAL,
                        'symbol':       symbol,
                        'volume':       lot,
                        'type':         mt5.ORDER_TYPE_SELL,
                        'price':        current_price,
                        'sl':           sl,
                        'tp':           0.0,
                        'deviation':    MAX_SLIPPAGE,
                        'magic':        MAGIC_NUMBER,
                        'comment':      f'ml_grid_{level}',
                        'type_time':    mt5.ORDER_TIME_GTC,
                        'type_filling': mt5.ORDER_FILLING_IOC,
                    })
                    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                        err = result.retcode if result else mt5.last_error()
                        print(f"  [grid] SELL L{level} failed #{pos.ticket}: "
                              f"retcode={err}  trigger={grid_trigger:.{digits}f}")
                    else:
                        placed.add(level)
                        fib_mult = _FIB[min(level - 1, len(_FIB) - 1)]
                        print(f"  [grid] SELL L{level} #{pos.ticket} {lot}lots "
                              f"@ {current_price:.{digits}f}  fib×{fib_mult}  SL={sl:.{digits}f}  no-TP")

    # Prune state for positions that are no longer open
    for ticket in list(_grid_state.keys()):
        if ticket not in current_tickets:
            del _grid_state[ticket]
