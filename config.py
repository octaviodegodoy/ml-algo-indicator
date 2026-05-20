"""
config.py — All constants and output-path helpers.
Imported by every other module; keep it free of heavy logic.
"""

import os
import MetaTrader5 as mt5

# ── Targets ───────────────────────────────────────────────────────────────────
# Each target produces one CSV. `slug` controls output filenames
# (must match SignalFile input on the indicator).
TARGETS = [
    {'symbol': 'WINM26', 'slug': 'win'},
]

# ── Bar / timeframe ───────────────────────────────────────────────────────────
TIMEFRAME         = mt5.TIMEFRAME_M5
TF_SECONDS        = 5 * 60
N_BARS            = 15000

# ── Triple-barrier params (in ATR multiples) ──────────────────────────────────
TB_MAX_BARS       = 8           # vertical barrier: ~1 hour on M5
TB_PT_MULT        = 1.5          # profit target = 1.5 × ATR(14)
TB_SL_MULT        = 1.2          # stop loss    = 1.2 × ATR(14)  (widened to survive noise)

# ── Model ─────────────────────────────────────────────────────────────────────
# 'lightgbm' (default, faster) or 'xgboost' (level-wise, more conservative on noisy labels)
MODEL_TYPE        = 'xgboost'
PROB_THRESHOLD    = 0.45         # raised from 0.35 — fewer, higher-confidence signals to protect win-rate margin
MIN_AUC           = 0.46         # lowered: CV walk-forward AUC ≈ 0.47–0.48 while live win rate is 76%; CV metric underestimates live performance
DAILY_MAX_LOSS_PCT = -2.0           # stop new entries when realized day P&L drops below this % of equity
INTERVAL_SECONDS  = 60
N_SPLITS_CV       = 5
RECENCY_DECAY     = 1.2          # lowered from 2.0 — reduces oversensitivity to single volatile bars

# ── Cooldown after stop-out ───────────────────────────────────────────────────
COOLDOWN_BARS     = 6            # bars to skip re-entry after a stop-out (halved from 12 → 30 min buffer)

# ── Directional confirmation gate ─────────────────────────────────────────────
# When True, a SELL signal is only executed when DI− > DI+ (di_diff_14 < −DI_CONFIRM_MIN_DIFF)
# and a BUY signal is only executed when DI+ > DI− (di_diff_14 > +DI_CONFIRM_MIN_DIFF).
# Prevents entering counter-trend shorts/longs at reversal extremes (e.g. false sell on 2026-05-19).
REQUIRE_DI_CONFIRMATION = True
DI_CONFIRM_MIN_DIFF     = 3.0   # minimum |DI+ − DI−| to consider direction confirmed (filters noise near zero)

# Separate persistence for SELL transitions: SELL requires more bars than BUY to confirm
# because shorting into a bullish-trend pullback is the primary false-signal failure mode.
SELL_PERSISTENCE_BARS   = 3     # consecutive SELL bars required to enter SHORT (vs 2 for BUY)

# ── Microstructure ────────────────────────────────────────────────────────────
DOM_LEVELS            = 5     # top-N book levels to aggregate
DOM_SAMPLE_SECS       = 5     # background thread samples DOM every N seconds (was once per 60s cycle)
DOM_MIN_LEVEL_LOTS    = 5     # ignore book levels with fewer lots than this (noise / 1-lot algos)
DOM_LARGE_ORDER_LOTS  = 100   # threshold to classify a level as a resting large/iceberg order
MIN_MICRO_ROWS        = 50    # need ≥ this many bars with micro data before using features
HTF_BARS              = 500   # H1 bars to fetch for higher-timeframe context

# ── Trading session filter (BRT = UTC-3) ─────────────────────────────────────
# B3 WIN liquidity profile:
#   09:00-09:05  opening auction            → avoid (thin, volatile)
#   09:05-13:00  day → liquid  ✓
TRADE_SESSIONS = [
    ((10, 0), (15, 45)),   # skip noisy opening hour; 10:00–17:45 BRT has avg prob 13% higher than 9:xx
]

# ── Signal freeze ─────────────────────────────────────────────────────────────
# Once a bar is older than the current forming bar its signal is locked in the
# CSV and never overwritten. Only the current (still-forming) bar gets re-scored.
FREEZE_HISTORY    = True

# ── Trade execution ───────────────────────────────────────────────────────────
# SAFETY: keep TRADE_ENABLED = False until you have verified signals visually.
TRADE_ENABLED      = True   # set True to allow real orders
TRADE_BOTH_SIDES   = True   # True  = signal 1 → long, signal 0 → short
                              # False = signal 1 → long, signal 0 → close long only
RISK_PCT           = 2.0   # % of account balance risked per trade
MAX_SLIPPAGE       = 10     # maximum allowed slippage in points
MAGIC_NUMBER       = 20260507  # unique tag for orders placed by this script
TRAIL_ACTIVATE_PCT = 0.20   # activate trailing stop when profit >= 20% of SL distance (~50 pts)
MIN_FLIP_PROFIT_PTS = 0     # disabled — backtest showed no gain from suppressing signal-flip exits

# ── Grid (Fibonacci martingale) ───────────────────────────────────────────────
GRID_ENABLED           = True   # seed + grid total volume capped to RISK_PCT budget via _grid_divisor()
GRID_MAX_LEVELS        = 5      # maximum grid add-ons per seed position
GRID_STEP_MULT         = 0.50   # grid step = GRID_STEP_MULT × SL-distance in adverse direction
                                 # raised from 0.30 — less aggressive averaging-down
GRID_PORTFOLIO_SL_MULT = 3.50   # close entire grid when total floating loss exceeds
                                 # this multiple of (seed_sl_dist × point_value × seed_lot)
                                 # lowered from 5.00 — tighter total grid exposure cap

# ── Output paths ──────────────────────────────────────────────────────────────
_files_dir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Files')
)
os.makedirs(_files_dir, exist_ok=True)


def out_path(slug: str)   -> str: return os.path.join(_files_dir, f'{slug}_ml_signals.csv')
def dom_path(slug: str)   -> str: return os.path.join(_files_dir, f'{slug}_dom_snapshots.csv')
def ticks_path(slug: str) -> str: return os.path.join(_files_dir, f'{slug}_tick_agg.csv')


