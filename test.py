# test.py - Plot BOVA11 ML buy/sell signals from MT5 data
# Requirements: MetaTrader5, pandas, scikit-learn, matplotlib
# Install with: pip install MetaTrader5 pandas scikit-learn matplotlib

import MetaTrader5 as mt5
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ── 1. Fetch data from MT5 ────────────────────────────────────────────────────
if not mt5.initialize():
    raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

rates = mt5.copy_rates_from_pos('BOVA11', mt5.TIMEFRAME_M1, 0, 5000)
mt5.shutdown()

if rates is None or len(rates) == 0:
    raise RuntimeError("No data returned from MT5 for BOVA11")

data = pd.DataFrame(rates)
data['time'] = pd.to_datetime(data['time'], unit='s', utc=True)
data.set_index('time', inplace=True)
data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                     'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# ── 2. Feature engineering ────────────────────────────────────────────────────
data['Return'] = data['Close'].pct_change()
data['MA5']    = data['Close'].rolling(window=5).mean()
data['MA20']   = data['Close'].rolling(window=20).mean()
data['Signal'] = 0
data.loc[data.index[20:], 'Signal'] = (
    data['MA5'].iloc[20:] > data['MA20'].iloc[20:]
).astype(int)

data['Target'] = data['Signal'].shift(-1)
data = data.dropna()

# ── 3. Train ML model ─────────────────────────────────────────────────────────
features = ['Return', 'MA5', 'MA20', 'Signal']
X = data[features]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

data['ML_Signal'] = model.predict(X)

# ── 4. Derive buy/sell points ─────────────────────────────────────────────────
# Buy  : signal flips from 0 → 1
# Sell : signal flips from 1 → 0
prev = data['ML_Signal'].shift(1)
buy_signals  = data[(data['ML_Signal'] == 1) & (prev == 0)]
sell_signals = data[(data['ML_Signal'] == 0) & (prev == 1)]

# ── 5. Plot ───────────────────────────────────────────────────────────────────
# Show only the last 500 bars to keep the chart readable
plot_data = data.iloc[-500:]
plot_buy  = buy_signals[buy_signals.index >= plot_data.index[0]]
plot_sell = sell_signals[sell_signals.index >= plot_data.index[0]]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9),
                                gridspec_kw={'height_ratios': [3, 1]},
                                sharex=True)
fig.suptitle('BOVA11 – ML Buy / Sell Signals (last 500 M1 bars)', fontsize=13)

# Price + MAs + signals
ax1.plot(plot_data.index, plot_data['Close'], color='#1f77b4', linewidth=0.8, label='Close')
ax1.plot(plot_data.index, plot_data['MA5'],   color='#ff7f0e', linewidth=0.9, linestyle='--', label='MA5')
ax1.plot(plot_data.index, plot_data['MA20'],  color='#9467bd', linewidth=0.9, linestyle='--', label='MA20')

ax1.scatter(plot_buy.index,  plot_buy['Close'],
            marker='^', color='#2ca02c', zorder=5, s=60, label='Buy')
ax1.scatter(plot_sell.index, plot_sell['Close'],
            marker='v', color='#d62728', zorder=5, s=60, label='Sell')

ax1.set_ylabel('Price')
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

# ML signal line
ax2.step(plot_data.index, plot_data['ML_Signal'], color='#1f77b4', linewidth=0.8, where='post')
ax2.fill_between(plot_data.index, plot_data['ML_Signal'], step='post',
                 alpha=0.2, color='#1f77b4')
ax2.set_ylabel('ML Signal\n(1=Buy, 0=Sell)')
ax2.set_yticks([0, 1])
ax2.grid(True, alpha=0.3)

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
fig.autofmt_xdate(rotation=30)

plt.tight_layout()
plt.show()

print(f"Total bars : {len(data)}")
print(f"Buy  signals : {len(buy_signals)}")
print(f"Sell signals : {len(sell_signals)}")
