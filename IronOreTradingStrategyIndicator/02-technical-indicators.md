# Chapter 2: Technical Indicators

## Indicator Stack

Seven technical indicators organized in four functional layers.

---

## Layer 1: Trend Identification

### 1. Triple EMA (Exponential Moving Average)

**Purpose**: Identify short-term, medium-term, and long-term trends.

**Parameters**:
- EMA Fast: 12 periods
- EMA Medium: 26 periods
- EMA Slow: 50 periods

**Online Calculation**:
```python
alpha_12 = 2.0 / (12 + 1) = 0.1538
alpha_26 = 2.0 / (26 + 1) = 0.0741
alpha_50 = 2.0 / (50 + 1) = 0.0392

ema_12 = alpha_12 * close + (1 - alpha_12) * ema_12_prev
ema_26 = alpha_26 * close + (1 - alpha_26) * ema_26_prev
ema_50 = alpha_50 * close + (1 - alpha_50) * ema_50_prev
```

**Initialization**: `ema = close` (first bar)

**Trend Signals**:
- Uptrend: `ema_12 > ema_26 > ema_50`
- Downtrend: `ema_12 < ema_26 < ema_50`
- Mixed: All other configurations

### 2. MACD (Moving Average Convergence Divergence)

**Purpose**: Momentum confirmation for trend signals.

**Parameters**:
- Fast EMA: 12 periods (reuse from Triple EMA)
- Slow EMA: 26 periods (reuse from Triple EMA)
- Signal Line: 9-period EMA of MACD

**Online Calculation**:
```python
# MACD Line
macd = ema_12 - ema_26

# Signal Line
alpha_signal = 2.0 / (9 + 1) = 0.20
macd_signal = alpha_signal * macd + (1 - alpha_signal) * macd_signal_prev

# Histogram
macd_histogram = macd - macd_signal
```

**Momentum Signals**:
- Bullish: `macd > macd_signal` AND `macd_histogram > 0`
- Bearish: `macd < macd_signal` AND `macd_histogram < 0`

---

## Layer 2: Mean Reversion

### 3. RSI (Relative Strength Index)

**Purpose**: Identify overbought/oversold conditions.

**Parameters**:
- Period: 14
- Overbought threshold: 70
- Oversold threshold: 30

**Online Calculation**:
```python
# Price change
change = close - close_prev
gain = max(change, 0)
loss = max(-change, 0)

# Update gain/loss EMAs
alpha_rsi = 2.0 / (14 + 1) = 0.1333
gain_ema = alpha_rsi * gain + (1 - alpha_rsi) * gain_ema_prev
loss_ema = alpha_rsi * loss + (1 - alpha_rsi) * loss_ema_prev

# Calculate RSI
if loss_ema > 0:
    rs = gain_ema / loss_ema
    rsi = 100.0 - (100.0 / (1.0 + rs))
else:
    rsi = 100.0
```

**Initialization**: `gain_ema = 0`, `loss_ema = 0`, `rsi = 50`

**Mean Reversion Signals**:
- Oversold (Buy): `rsi < 30`
- Overbought (Sell): `rsi > 70`
- Neutral: `30 <= rsi <= 70`

### 4. Bollinger Bands

**Purpose**: Identify price extremes and volatility expansion/contraction.

**Parameters**:
- Period: 20
- Standard Deviations: 2

**Online Calculation**:

**Step 1 - Middle Band (SMA)**:
```python
# Use Welford's online algorithm for mean and variance
n = min(bar_index, 20)
delta = close - mean_prev
mean = mean_prev + delta / n
```

**Step 2 - Standard Deviation (Welford's Variance)**:
```python
delta2 = close - mean
variance = ((n - 1) * variance_prev + delta * delta2) / n
std_dev = sqrt(variance)
```

**Step 3 - Bands**:
```python
bb_middle = mean
bb_upper = mean + (2.0 * std_dev)
bb_lower = mean - (2.0 * std_dev)
bb_width = bb_upper - bb_lower
```

**For Rolling Window** (after 20 bars):
Use circular buffer and online mean/variance update formulas.

**Extremes Signals**:
- Lower Band Touch: `close <= bb_lower` (oversold)
- Upper Band Touch: `close >= bb_upper` (overbought)
- Band Width: `bb_width / bb_middle` (volatility measure)

---

## Layer 3: Volatility & Risk

### 5. ATR (Average True Range)

**Purpose**: Measure market volatility for risk-adjusted position sizing.

**Parameters**:
- Period: 14

**Online Calculation**:
```python
# True Range
tr1 = high - low
tr2 = abs(high - close_prev) if close_prev > 0 else 0
tr3 = abs(low - close_prev) if close_prev > 0 else 0
tr = max(tr1, tr2, tr3)

# ATR (EMA of True Range)
alpha_atr = 2.0 / (14 + 1) = 0.1333
atr = alpha_atr * tr + (1 - alpha_atr) * atr_prev
```

**Initialization**: `atr = high - low` (first bar)

**Volatility Levels**:
- Low: `atr < mean_atr * 0.8`
- Normal: `mean_atr * 0.8 <= atr <= mean_atr * 1.2`
- High: `atr > mean_atr * 1.2`

### 6. Bollinger Band Width

**Purpose**: Volatility confirmation (already calculated in BB).

**Calculation**:
```python
bb_width_pct = (bb_width / bb_middle) * 100
```

**Volatility Signals**:
- Squeeze: `bb_width_pct < 2.0` (low volatility)
- Normal: `2.0 <= bb_width_pct <= 4.0`
- Expansion: `bb_width_pct > 4.0` (high volatility)

---

## Layer 4: Liquidity Confirmation

### 7. Volume EMA

**Purpose**: Confirm signal validity with volume confirmation.

**Parameters**:
- Period: 20

**Online Calculation**:
```python
alpha_vol = 2.0 / (20 + 1) = 0.0952
volume_ema = alpha_vol * volume + (1 - alpha_vol) * volume_ema_prev
```

**Initialization**: `volume_ema = volume` (first bar)

**Volume Signals**:
- High Volume: `volume > volume_ema * 1.5` (strong confirmation)
- Normal Volume: `volume_ema * 0.8 < volume <= volume_ema * 1.5`
- Low Volume: `volume <= volume_ema * 0.8` (weak signal, avoid)

---

## State Variables Required

All indicators use online algorithms with O(1) memory:

```python
# EMA states
self.ema_12 = 0.0
self.ema_26 = 0.0
self.ema_50 = 0.0
self.volume_ema = 0.0

# MACD states
self.macd = 0.0
self.macd_signal = 0.0
self.macd_histogram = 0.0

# RSI states
self.rsi = 50.0
self.gain_ema = 0.0
self.loss_ema = 0.0
self.prev_close = 0.0

# Bollinger Band states (Welford's algorithm)
self.bb_mean = 0.0
self.bb_variance = 0.0
self.bb_upper = 0.0
self.bb_middle = 0.0
self.bb_lower = 0.0
self.bb_width = 0.0

# ATR states
self.atr = 0.0
self.prev_close_atr = 0.0

# Tracking
self.bar_index = 0
self.initialized = False
```

## Alpha Values Summary

```python
# Pre-computed smoothing factors
self.alpha_12 = 2.0 / 13.0   # 0.1538
self.alpha_26 = 2.0 / 27.0   # 0.0741
self.alpha_50 = 2.0 / 51.0   # 0.0392
self.alpha_9 = 2.0 / 10.0    # 0.2000
self.alpha_14 = 2.0 / 15.0   # 0.1333
self.alpha_20 = 2.0 / 21.0   # 0.0952
```

## Next Steps

Proceed to Chapter 3 for market regime classification rules.
