"""
rnn_model.py - Optional lightweight LSTM classifier for sequence modeling.

Design goals:
- Keep integration optional: if torch is unavailable, callers can skip gracefully.
- Train on rolling sequences of feature rows (Renko + engineered indicators).
- Return per-row probabilities aligned to input index (NaN for warmup rows).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class _LSTMClassifier(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
            super().__init__()
            rnn_dropout = dropout if num_layers > 1 else 0.0
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            logits = self.head(out[:, -1, :]).squeeze(-1)
            return logits

    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


@dataclass
class RNNBundle:
    model: object
    seq_len: int
    feature_columns: list[str]
    fill_values: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def is_rnn_available() -> bool:
    return _TORCH_AVAILABLE


def _build_sequences(X: np.ndarray, y: Optional[np.ndarray], seq_len: int):
    n = X.shape[0]
    if n < seq_len:
        return None, None
    xs = []
    ys = []
    for i in range(seq_len - 1, n):
        xs.append(X[i - seq_len + 1:i + 1])
        if y is not None:
            ys.append(y[i])
    Xs = np.asarray(xs, dtype=np.float32)
    Ys = None if y is None else np.asarray(ys, dtype=np.float32)
    return Xs, Ys


def _prepare_X(X: pd.DataFrame, fill_values: Optional[np.ndarray] = None):
    vals = X.values.astype(np.float32)
    if fill_values is None:
        fill_values = np.nanmedian(vals, axis=0)
        fill_values = np.where(np.isfinite(fill_values), fill_values, 0.0).astype(np.float32)
    vals = np.where(np.isnan(vals), fill_values, vals)
    mean = vals.mean(axis=0).astype(np.float32)
    std = vals.std(axis=0).astype(np.float32)
    std = np.where(std > 1e-8, std, 1.0).astype(np.float32)
    vals = (vals - mean) / std
    return vals, fill_values.astype(np.float32), mean, std


def fit_rnn_sequence_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    seq_len: int = 32,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    epochs: int = 4,
    lr: float = 1e-3,
    batch_size: int = 128,
) -> Optional[RNNBundle]:
    if not _TORCH_AVAILABLE:
        return None
    if len(X) < max(300, seq_len * 3):
        return None

    Xn, fill_values, mean, std = _prepare_X(X)
    yv = y.values.astype(np.float32)

    Xs, Ys = _build_sequences(Xn, yv, seq_len=seq_len)
    if Xs is None or Ys is None or len(Xs) < 100:
        return None

    device = torch.device('cpu')
    model = _LSTMClassifier(
        input_size=Xs.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    x_t = torch.from_numpy(Xs).to(device)
    y_t = torch.from_numpy(Ys).to(device)

    pos = float((Ys == 1.0).sum())
    neg = float((Ys == 0.0).sum())
    pos_weight = torch.tensor([(neg / pos) if pos > 0 else 1.0], dtype=torch.float32, device=device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    n = x_t.shape[0]
    idx = np.arange(n)

    model.train()
    for _ in range(max(1, int(epochs))):
        np.random.shuffle(idx)
        for i in range(0, n, batch_size):
            b = idx[i:i + batch_size]
            xb = x_t[b]
            yb = y_t[b]
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

    return RNNBundle(
        model=model,
        seq_len=seq_len,
        feature_columns=list(X.columns),
        fill_values=fill_values,
        mean=mean,
        std=std,
    )


def predict_rnn_proba(bundle: RNNBundle, X: pd.DataFrame) -> np.ndarray:
    if not _TORCH_AVAILABLE:
        return np.full(len(X), np.nan, dtype=float)

    X = X.reindex(columns=bundle.feature_columns, fill_value=np.nan)
    vals = X.values.astype(np.float32)
    vals = np.where(np.isnan(vals), bundle.fill_values, vals)
    vals = (vals - bundle.mean) / bundle.std

    Xs, _ = _build_sequences(vals, y=None, seq_len=bundle.seq_len)
    out = np.full(len(X), np.nan, dtype=float)
    if Xs is None or len(Xs) == 0:
        return out

    device = torch.device('cpu')
    bundle.model.eval()
    with torch.no_grad():
        logits = bundle.model(torch.from_numpy(Xs).to(device))
        probs = torch.sigmoid(logits).cpu().numpy().astype(float)

    out[bundle.seq_len - 1:] = probs
    return out
