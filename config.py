"""
config.py — All constants and output-path helpers.
Imported by every other module; keep it free of heavy logic.
"""

import os

# ── Targets ───────────────────────────────────────────────────────────────────
# Each target produces one CSV. `slug` controls output filenames
# (must match SignalFile input on the indicator).
TARGETS = [
    {'symbol': 'WINM26', 'slug': 'win'},
]

# ── Bar / timeframe ───────────────────────────────────────────────────────────
# mt5.TIMEFRAME_M5 == 5  (raw integer from the MT5 enum — avoids importing MT5
# in config so the config module has no platform dependency).
TIMEFRAME         = 5
TF_SECONDS        = 5 * 60
N_BARS            = 15000

# ── Triple-barrier params (in ATR multiples) ──────────────────────────────────
TB_MAX_BARS       = 8           # vertical barrier: ~1 hour on M5
TB_PT_MULT        = 2.0          # profit target = 2.0 × ATR(14)  — improved R:R from 1.25:1 to 1.67:1
TB_SL_MULT        = 1.2          # stop loss    = 1.2 × ATR(14)  (widened to survive noise)

# ── Model ─────────────────────────────────────────────────────────────────────
# 'lightgbm' (default, faster) or 'xgboost' (level-wise, more conservative on noisy labels)
MODEL_TYPE        = 'lightgbm'
PROB_THRESHOLD    = 0.52         # only enter when model assigns >52% probability (above random)
MIN_AUC           = 0.52         # require walk-forward AUC above random (0.5) before trading
DAILY_MAX_LOSS_PCT = -2.0           # stop new entries when realized day P&L drops below this % of equity
INTERVAL_SECONDS  = 60
N_SPLITS_CV       = 5
RECENCY_DECAY     = 1.2          # lowered from 2.0 — reduces oversensitivity to single volatile bars

# ── Renko preprocessing for ML ───────────────────────────────────────────────
# Converts the M5 price series to Renko boxes before feature/label generation.
USE_RENKO_BARS     = True
RENKO_BOX_MODE     = 'atr'        # 'atr' or 'points'
RENKO_BOX_POINTS   = 35.0        # used when mode='points' and as fallback
RENKO_BOX_ATR_MULT = 1.0          # box = median(ATR14) * this multiplier when mode='atr'
RENKO_ATR_PERIOD   = 14
RENKO_MIN_POINTS   = 20.0         # floor for very quiet sessions

# ── Optional RNN sequence model (trained on Renko + indicators) ─────────────
USE_RNN_MODEL       = True
RNN_SEQ_LEN         = 32
RNN_HIDDEN_SIZE     = 32
RNN_NUM_LAYERS      = 1
RNN_DROPOUT         = 0.0
RNN_EPOCHS          = 4
RNN_LR              = 0.001
RNN_BATCH_SIZE      = 128
RNN_BLEND_WEIGHT    = 0.35   # final prob = (1-w)*GB + w*RNN where RNN is available

# ── RL execution overlay (hybrid: supervised signal + RL action policy) ─────
# The RL policy learns an execution decision from historical probabilities and
# market movement, then overrides only the latest live execution signal.
USE_RL_OVERLAY      = True
RL_TRAIN_WINDOW     = 2500    # recent rows used to fit Q-table each cycle
RL_N_EPISODES       = 12      # training passes over the window
RL_ALPHA            = 0.08    # Q-learning step size
RL_GAMMA            = 0.95    # discount factor
RL_EPSILON          = 0.05    # exploration during fitting
RL_COST_BPS         = 1.5     # round-turn equivalent switching cost
RL_HOLD0_PENALTY_BPS = 0.02   # tiny penalty for staying in signal=0 state
RL_PROBA_BINS       = 10
RL_DI_BINS          = 7
RL_VOL_BINS         = 5

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
TRADE_ENABLED      = True   # master switch — gates ALL order placing (ML + ORB)
ML_ENABLED         = True   # set False to disable ML trades and run ORB only
# ORB_ENABLED (below) is the equivalent switch for the ORB strategy.
TRADE_BOTH_SIDES   = True  # signal 1 → long, signal 0 → close long only (M5 shorts on WIN add noise)
RISK_PCT           = 2.0   # % of account balance risked per trade
MAX_SLIPPAGE       = 10     # maximum allowed slippage in points
MAGIC_NUMBER       = 20260507  # unique tag for orders placed by this script
TRAIL_ACTIVATE_PCT = 0.40   # activate trailing stop when profit >= 40% of SL distance (wider to avoid noise stop-out)
MIN_FLIP_PROFIT_PTS = 0     # disabled — backtest showed no gain from suppressing signal-flip exits

# ── Grid (Fibonacci martingale) ───────────────────────────────────────────────
GRID_ENABLED           = False  # DISABLED: Fibonacci martingale amplifies losses when model is wrong
GRID_MAX_LEVELS        = 5      # maximum grid add-ons per seed position
GRID_STEP_MULT         = 0.50   # grid step = GRID_STEP_MULT × SL-distance in adverse direction
                                 # raised from 0.30 — less aggressive averaging-down
GRID_PORTFOLIO_SL_MULT = 3.50   # close entire grid when total floating loss exceeds
                                 # this multiple of (seed_sl_dist × point_value × seed_lot)
                                 # lowered from 5.00 — tighter total grid exposure cap

# ── ORB (Opening Range Breakout) strategy ─────────────────────────────────────
# First ORB_BARS M5 bars each day define the range (09:00–09:14 BRT by default).
# After ORB_SESSION_START a close above ORB high triggers a BUY; a close below
# ORB low triggers a SELL (if ORB_TRADE_SHORT=True).  At most one trade per
# direction per calendar day.  TP is fixed on the order; no trailing stop needed.
ORB_ENABLED       = True
ORB_BARS          = 3           # number of M5 bars in the opening range (3 × 5 min = 15 min)
ORB_SESSION_START = (9, 15)     # BRT (h, m): earliest valid entry after ORB is formed
ORB_SESSION_END   = (14, 0)     # BRT (h, m): no new ORB entries after this; EOD close sweep
ORB_SL_MULT       = 0.5         # SL distance = ORB_SL_MULT × ORB range (placed from entry)
ORB_TP_MULT       = 1.0         # TP distance = ORB_TP_MULT × ORB range → 2:1 R:R (TP/SL = 2)
ORB_RISK_PCT      = 1.5         # % of account balance risked per ORB trade (separate budget)
ORB_MAGIC         = 20260522    # unique magic number — never share with MAGIC_NUMBER
ORB_TRADE_SHORT   = True        # True = also trade ORB downside breaks (SELL short)

# ── Gap Fill strategy ─────────────────────────────────────────────────────────
# Fades the overnight gap expecting price to return to the previous session's
# close within the first ~25 minutes.
#   Gap Up  (open > prev_close): SELL, TP = prev_close, SL = open + GAP_SL_MULT × gap
#   Gap Down (open < prev_close): BUY,  TP = prev_close, SL = open - GAP_SL_MULT × gap
GAP_ENABLED       = True
GAP_MIN_PTS       = 150         # minimum gap in points to trade (filters small noise gaps)
GAP_MAX_PTS       = 1500        # maximum gap — very large gaps may reflect strong trends that won't fill
GAP_SL_MULT       = 1.0         # SL distance = 1× gap beyond the entry price
GAP_TP_MULT       = 1.0         # TP = prev_close (full gap fill); R:R depends on entry price
GAP_RISK_PCT      = 1.0         # % of account balance risked per gap trade (conservative)
GAP_MAGIC         = 20260525    # unique magic number
GAP_TRADE_SHORT   = True        # True = also trade gap-up (SELL side)
GAP_ENTRY_START   = (9, 5)      # BRT: earliest entry — after first M5 bar closes (post-auction)
GAP_ENTRY_END     = (9, 30)     # BRT: stop entering new gap trades after 09:30

# ── Output paths ──────────────────────────────────────────────────────────────
_files_dir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Files')
)
os.makedirs(_files_dir, exist_ok=True)


def out_path(slug: str)   -> str: return os.path.join(_files_dir, f'{slug}_ml_signals.csv')
def dom_path(slug: str)   -> str: return os.path.join(_files_dir, f'{slug}_dom_snapshots.csv')
def ticks_path(slug: str) -> str: return os.path.join(_files_dir, f'{slug}_tick_agg.csv')
def orb_path(slug: str)   -> str: return os.path.join(_files_dir, f'{slug}_orb.csv')
def gap_path(slug: str)   -> str: return os.path.join(_files_dir, f'{slug}_gap.csv')


