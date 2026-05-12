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
N_BARS            = 5000

# ── Triple-barrier params (in ATR multiples) ──────────────────────────────────
TB_MAX_BARS       = 12           # vertical barrier: ~1 hour on M5
TB_PT_MULT        = 1.5          # profit target = 1.5 × ATR(14)
TB_SL_MULT        = 1.2          # stop loss    = 1.2 × ATR(14)  (widened to survive noise)

# ── Model ─────────────────────────────────────────────────────────────────────
PROB_THRESHOLD    = 0.54         # raised from 0.50; 0.57 was too strict for balanced-class LightGBM
INTERVAL_SECONDS  = 60
N_SPLITS_CV       = 5
RECENCY_DECAY     = 1.2          # lowered from 2.0 — reduces oversensitivity to single volatile bars

# ── Microstructure ────────────────────────────────────────────────────────────
DOM_LEVELS        = 5            # top-N book levels to aggregate
DOM_SAMPLE_SECS   = 5            # background thread samples DOM every N seconds (was once per 60s cycle)
MIN_MICRO_ROWS    = 50           # need ≥ this many bars with micro data before using features
HTF_BARS          = 500          # H1 bars to fetch for higher-timeframe context

# ── Trading session filter (BRT = UTC-3) ─────────────────────────────────────
# B3 WIN liquidity profile:
#   09:00-09:05  opening auction            → avoid (thin, volatile)
#   09:05-12:00  morning session            → liquid  ✓
#   12:00-13:30  lunch/dead zone            → avoid (wide spreads)
#   13:30-17:55  afternoon + US-open window → liquid  ✓  (peak 14:30-16:00)
#   17:55-18:00  close / final auction      → avoid
# Times are (hour, minute) tuples in BRT (UTC-3). Python datetime.now(tz) for BRT.
TRADE_SESSIONS = [
    ((9,  5), (12,  0)),   # morning session
    ((13, 30), (17, 55)),  # afternoon session
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
TRAIL_ACTIVATE_PCT = 0.50   # activate trailing stop when profit >= 50% of SL distance

# ── Grid (Fibonacci martingale) ───────────────────────────────────────────────
GRID_ENABLED           = True   # add Fibonacci-scaled orders when a position is in loss
GRID_MAX_LEVELS        = 5      # maximum grid add-ons per seed position
GRID_STEP_MULT         = 0.80   # grid step = GRID_STEP_MULT × SL-distance in adverse direction
                                 # raised from 0.30 — less aggressive averaging-down
GRID_PORTFOLIO_SL_MULT = 2.50   # close entire grid when total floating loss exceeds
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
