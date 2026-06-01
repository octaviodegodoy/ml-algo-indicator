"""
rl_overlay.py - Lightweight Q-learning execution overlay.

Hybrid design:
- Supervised model estimates edge/probability per bar.
- RL overlay learns whether to execute signal=1 or signal=0 on the next bar,
  accounting for switching costs and short-term reward dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RLOverlayConfig:
    train_window: int = 2500
    n_episodes: int = 12
    alpha: float = 0.08
    gamma: float = 0.95
    epsilon: float = 0.05
    cost_bps: float = 1.5
    hold0_penalty_bps: float = 0.02
    proba_bins: int = 10
    di_bins: int = 7
    vol_bins: int = 5


def _digitize(values: np.ndarray, n_bins: int, lo: float, hi: float) -> np.ndarray:
    n_bins = max(2, int(n_bins))
    v = np.nan_to_num(values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    v = np.clip(v, lo, hi)
    edges = np.linspace(lo, hi, n_bins + 1)[1:-1]
    return np.digitize(v, edges, right=False).astype(np.int32)


def _safe_returns_bps(close: np.ndarray) -> np.ndarray:
    c = np.nan_to_num(close.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    ret = np.zeros_like(c, dtype=float)
    if len(c) < 2:
        return ret
    prev = c[:-1]
    nxt = c[1:]
    denom = np.where(np.abs(prev) > 1e-12, prev, 1.0)
    ret[:-1] = (nxt - prev) / denom * 10000.0
    return ret


def choose_latest_execution_signal(
    proba: np.ndarray,
    close: np.ndarray,
    di_diff: np.ndarray,
    atr_ratio: np.ndarray,
    cfg: RLOverlayConfig,
    trade_both_sides: bool,
) -> tuple[int, dict]:
    """Train a compact Q-table on recent history and return latest action signal.

    Action space intentionally matches existing trade executor:
    - 1 => BUY signal
    - 0 => SELL/FLAT signal (depends on TRADE_BOTH_SIDES)
    """
    p = np.asarray(proba, dtype=float)
    c = np.asarray(close, dtype=float)
    d = np.asarray(di_diff, dtype=float)
    v = np.asarray(atr_ratio, dtype=float)

    n = min(len(p), len(c), len(d), len(v))
    if n < 60:
        fallback = int(p[-1] > 0.5) if n > 0 else 0
        return fallback, {'reason': 'insufficient_rows', 'rows': n}

    p = p[-n:]
    c = c[-n:]
    d = d[-n:]
    v = v[-n:]

    # Train only on a recent window for faster adaptation.
    w = min(max(120, int(cfg.train_window)), n)
    p_tr = p[-w:]
    c_tr = c[-w:]
    d_tr = d[-w:]
    v_tr = v[-w:]

    # atr_ratio = true_range / ATR14 is strictly positive (~0..3, centered near 1).
    pb = _digitize(p_tr, cfg.proba_bins, 0.0, 1.0)
    db = _digitize(d_tr, cfg.di_bins, -60.0, 60.0)
    vb = _digitize(v_tr, cfg.vol_bins, 0.0, 3.0)
    rbps = _safe_returns_bps(c_tr)

    # Q[state_position, proba_bin, di_bin, vol_bin, action]
    q = np.zeros((2, cfg.proba_bins, cfg.di_bins, cfg.vol_bins, 2), dtype=np.float32)

    for _ in range(max(1, int(cfg.n_episodes))):
        pos = 0
        for t in range(0, w - 1):
            s0 = (pos, int(pb[t]), int(db[t]), int(vb[t]))

            if np.random.rand() < float(cfg.epsilon):
                a = int(np.random.randint(0, 2))
            else:
                a = int(np.argmax(q[s0]))

            # Reward from action a applied over [t, t+1].
            if a == 1:
                reward = float(rbps[t])
            else:
                reward = float(-rbps[t]) if trade_both_sides else float(-cfg.hold0_penalty_bps)

            if a != pos:
                reward -= float(cfg.cost_bps)

            pos_next = a
            s1 = (pos_next, int(pb[t + 1]), int(db[t + 1]), int(vb[t + 1]))

            td_target = reward + float(cfg.gamma) * float(np.max(q[s1]))
            td_error = td_target - float(q[s0 + (a,)])
            q[s0 + (a,)] += float(cfg.alpha) * td_error
            pos = pos_next

    # Greedy latest action from most recent state.
    p_last = int(_digitize(np.array([p[-1]]), cfg.proba_bins, 0.0, 1.0)[0])
    d_last = int(_digitize(np.array([d[-1]]), cfg.di_bins, -60.0, 60.0)[0])
    v_last = int(_digitize(np.array([v[-1]]), cfg.vol_bins, 0.0, 3.0)[0])

    # Use prior bar's supervised direction as current position proxy.
    prev_pos = int(p[-2] > 0.5) if n >= 2 else 0
    latest_action = int(np.argmax(q[(prev_pos, p_last, d_last, v_last)]))

    info = {
        'rows': int(w),
        'prev_pos': int(prev_pos),
        'latest_action': int(latest_action),
        'latest_proba': float(p[-1]),
    }
    return latest_action, info
