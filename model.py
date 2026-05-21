"""
model.py — Triple-barrier labeling, walk-forward cross-validation,
recency weighting, and stop-loss point computation.
"""

from typing import Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from config import RECENCY_DECAY, N_SPLITS_CV, MODEL_TYPE


# ── Classifier registry (OCP: add new model types without modifying callers) ──
def _lgbm_factory(scale_pos_weight: float) -> object:
    return LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        class_weight='balanced', random_state=42, verbosity=-1,
    )


def _xgb_factory(scale_pos_weight: float) -> object:
    if not _XGBOOST_AVAILABLE:
        print("WARNING: MODEL_TYPE='xgboost' but xgboost is not installed — falling back to LightGBM.")
        return _lgbm_factory(scale_pos_weight)
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc', verbosity=0, random_state=42,
    )


_CLASSIFIER_REGISTRY: dict = {
    'lightgbm': _lgbm_factory,
    'xgboost':  _xgb_factory,
}


def _make_classifier(scale_pos_weight: float = 1.0) -> object:
    """Return the configured classifier using the registry.
    To add a new model type, register a factory in _CLASSIFIER_REGISTRY."""
    factory = _CLASSIFIER_REGISTRY.get(MODEL_TYPE, _lgbm_factory)
    return factory(scale_pos_weight)


# ── Triple-barrier labeling ───────────────────────────────────────────────────
def triple_barrier_labels(
    df: pd.DataFrame,
    atr: pd.Series,
    max_bars: int,
    pt_mult: float,
    sl_mult: float,
) -> pd.Series:
    """
    Label each bar 1 (profit target hit first) or 0 (stop-loss hit first / timeout).
    Last `max_bars` rows are set to NaN because no future bars exist to evaluate them.
    """
    close  = df['Close'].values
    high   = df['High'].values
    low    = df['Low'].values
    a      = atr.values
    n      = len(close)
    labels = np.zeros(n, dtype=np.int8)

    for i in range(n):
        if np.isnan(a[i]) or a[i] <= 0:
            continue
        upper  = close[i] + pt_mult * a[i]
        lower  = close[i] - sl_mult * a[i]
        end    = min(i + 1 + max_bars, n)
        hit_tp = -1
        hit_sl = -1
        for j in range(i + 1, end):
            if high[j] >= upper and hit_tp == -1:
                hit_tp = j
            if low[j] <= lower and hit_sl == -1:
                hit_sl = j
            if hit_tp != -1 or hit_sl != -1:
                break
        if hit_tp != -1 and (hit_sl == -1 or hit_tp < hit_sl):
            labels[i] = 1

    labels = labels.astype(float)
    labels[-max_bars:] = np.nan
    return pd.Series(labels, index=df.index, name='y')


# ── Recency weighting ─────────────────────────────────────────────────────────
def compute_recency_weights(n: int, decay: float = RECENCY_DECAY) -> np.ndarray:
    """Exponential recency weights — newest bar has e^decay × more weight than oldest."""
    t = np.arange(n, dtype=float)
    w = np.exp(decay * (t - (n - 1)) / n)
    return (w / w.mean()).astype(np.float32)


# ── Walk-forward cross-validation ────────────────────────────────────────────
def evaluate_walkforward(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    embargo: int,
    weights: Optional[np.ndarray] = None,
) -> dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs, precs, recs, aucs = [], [], [], []

    for train_idx, test_idx in tscv.split(X):
        if embargo > 0 and len(train_idx) > embargo:
            train_idx = train_idx[:-embargo]
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        if ytr.nunique() < 2:
            continue
        sw = weights[train_idx] if weights is not None else None
        # XGBoost uses scale_pos_weight instead of class_weight='balanced'
        neg = float((ytr == 0).sum()); pos = float((ytr == 1).sum())
        spw = (neg / pos) if pos > 0 else 1.0
        clf_name = 'gb'
        m  = Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('gb',  _make_classifier(scale_pos_weight=spw)),
        ])
        m.fit(Xtr, ytr, gb__sample_weight=sw)
        pred  = m.predict(Xte)
        proba = m.predict_proba(Xte)[:, 1]
        accs.append(accuracy_score(yte, pred))
        precs.append(precision_score(yte, pred, zero_division=0))
        recs.append(recall_score(yte, pred, zero_division=0))
        try:
            aucs.append(roc_auc_score(yte, proba))
        except Exception:
            aucs.append(0.5)   # degenerate fold — treat as random

    if not accs:
        return {'accuracy': float('nan'), 'precision': float('nan'),
                'recall': float('nan'), 'roc_auc': float('nan'),
                'baseline_rate': float(y.mean())}
    return {
        'accuracy':      float(np.mean(accs)),
        'precision':     float(np.mean(precs)),
        'recall':        float(np.mean(recs)),
        'roc_auc':       float(np.mean(aucs)),
        'baseline_rate': float(y.mean()),
    }


# ── Stop-loss point computation ───────────────────────────────────────────────
def compute_sl_points(
    signal_series: pd.Series,
    bars: pd.DataFrame,
    atr14: pd.Series,
    sl_mult: float,
    max_bars: int,
) -> pd.Series:
    """
    ATR-normalised SL: at every signal-transition bar the SL distance is
    atr14[i] * sl_mult (current-bar volatility), rounded to the nearest integer.
    All non-transition bars receive 0.
    Falls back to the recent 20-bar median ATR if the current bar's ATR is missing or zero.
    """
    atr_vals = atr14.reindex(signal_series.index).values
    sig      = signal_series.values
    n        = len(sig)
    sl       = np.zeros(n, dtype=float)

    for i in range(1, n):
        prev, curr = int(sig[i - 1]), int(sig[i])
        if (curr == 1 and prev == 0) or (curr == 0 and prev == 1):
            bar_atr = atr_vals[i]
            if np.isnan(bar_atr) or bar_atr <= 0:
                # fall back to recent median of prior bars (exclude current bad bar)
                bar_atr = float(np.nanmedian(atr_vals[max(0, i - 20):i]))
            sl[i] = bar_atr * sl_mult

    return pd.Series(np.round(sl, 0).astype(int), index=signal_series.index, name='SL_Points')
