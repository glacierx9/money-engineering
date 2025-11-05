# Chapter 3: Market Regimes

## Regime Classification

Four distinct market regimes determined by trend strength, volatility, and momentum.

---

## Regime 1: Strong Uptrend

**Classification Criteria** (ALL must be true):

1. **Trend Alignment**: `ema_12 > ema_26 > ema_50`
2. **Momentum Confirmation**: `macd > macd_signal AND macd_histogram > 0`
3. **Volatility**: `atr <= mean_atr * 1.2` (not chaotic)
4. **Price Position**: `close > ema_26`

**Trading Strategy**:
- **Entry**: Buy on RSI oversold (<30) with volume confirmation
- **Position**: Momentum-based, hold until trend reversal
- **Stop Loss**: 3% below entry
- **Profit Target**: 5% (scale out 50%), 8% (full exit)

**Regime Code**: `1`

---

## Regime 2: Strong Downtrend

**Classification Criteria** (ALL must be true):

1. **Trend Alignment**: `ema_12 < ema_26 < ema_50`
2. **Momentum Confirmation**: `macd < macd_signal AND macd_histogram < 0`
3. **Volatility**: `atr <= mean_atr * 1.2` (not chaotic)
4. **Price Position**: `close < ema_26`

**Trading Strategy**:
- **Entry**: Sell on RSI overbought (>70) with volume confirmation
- **Position**: Momentum-based short, hold until trend reversal
- **Stop Loss**: 3% above entry
- **Profit Target**: 5% (scale out 50%), 8% (full exit)

**Regime Code**: `2`

---

## Regime 3: Sideways / Ranging

**Classification Criteria** (ANY can trigger):

1. **EMA Convergence**: `abs(ema_12 - ema_26) / ema_26 < 0.015` (1.5% separation)
2. **MACD Near Zero**: `abs(macd) < mean_atr * 0.5`
3. **Mixed Trend**: NOT Strong Uptrend AND NOT Strong Downtrend
4. **BB Squeeze**: `bb_width_pct < 2.5`

**Trading Strategy**:
- **Entry**: Mean reversion trades
  - Buy at BB lower band with RSI < 30
  - Sell at BB upper band with RSI > 70
- **Position**: Quick scalps, tight stops
- **Stop Loss**: 2% (tighter than trending regimes)
- **Profit Target**: 3% (scale out 50%), 5% (full exit)

**Regime Code**: `3`

---

## Regime 4: High Volatility Chaos

**Classification Criteria** (ANY can trigger):

1. **Extreme Volatility**: `atr > mean_atr * 1.5`
2. **BB Expansion**: `bb_width_pct > 5.0`
3. **Whipsaw**: EMA crossovers within last 3 bars
4. **Gap**: `abs(open - close_prev) / close_prev > 0.03` (3% gap)

**Trading Strategy**:
- **NO TRADING** - Capital preservation mode
- **Action**: Close existing positions at market
- **Risk**: Too high for reliable signal generation

**Regime Code**: `4`

---

## Regime Detection Logic

### Implementation Order

```python
def detect_regime(self):
    """Detect current market regime (1/2/3/4)"""

    # 1. Check for chaos FIRST (highest priority)
    if self._is_high_volatility_chaos():
        return 4

    # 2. Check for strong trends
    if self._is_strong_uptrend():
        return 1

    if self._is_strong_downtrend():
        return 2

    # 3. Default to ranging (everything else)
    return 3
```

### Helper Functions

```python
def _is_strong_uptrend(self):
    """Check Strong Uptrend criteria"""
    trend_aligned = (self.ema_12 > self.ema_26 > self.ema_50)
    momentum_bullish = (self.macd > self.macd_signal) and (self.macd_histogram > 0)
    price_above = self.close > self.ema_26
    volatility_normal = self.atr <= (self.mean_atr * 1.2)

    return trend_aligned and momentum_bullish and price_above and volatility_normal

def _is_strong_downtrend(self):
    """Check Strong Downtrend criteria"""
    trend_aligned = (self.ema_12 < self.ema_26 < self.ema_50)
    momentum_bearish = (self.macd < self.macd_signal) and (self.macd_histogram < 0)
    price_below = self.close < self.ema_26
    volatility_normal = self.atr <= (self.mean_atr * 1.2)

    return trend_aligned and momentum_bearish and price_below and volatility_normal

def _is_high_volatility_chaos(self):
    """Check High Volatility Chaos criteria"""
    extreme_volatility = self.atr > (self.mean_atr * 1.5)
    bb_expansion = self.bb_width_pct > 5.0

    # Check for recent whipsaw (EMA crossovers)
    # Track crossover_count in last 3 bars
    whipsaw = self.recent_crossover_count >= 2

    # Check for gap
    gap = abs(self.open - self.prev_close) / self.prev_close > 0.03

    return extreme_volatility or bb_expansion or whipsaw or gap
```

---

## Regime State Variables

Additional state tracking required:

```python
# Regime tracking
self.regime = 3  # Default to ranging
self.regime_prev = 3

# ATR tracking for mean calculation (online)
self.mean_atr = 0.0
self.atr_count = 0

# Whipsaw detection
self.recent_crossover_count = 0
self.crossover_window = []  # Last 3 bars

# Gap detection
self.open = 0.0
```

---

## Mean ATR Calculation (Online)

```python
def update_mean_atr(self):
    """Update mean ATR using online algorithm"""
    self.atr_count += 1
    delta = self.atr - self.mean_atr
    self.mean_atr += delta / min(self.atr_count, 100)  # Cap at 100 for rolling mean
```

---

## Regime Transition Handling

### Transition Rules

1. **Chaos → Any**: Exit chaos only after 2 consecutive bars of normal volatility
2. **Uptrend → Downtrend**: Require 1 bar confirmation (avoid whipsaw)
3. **Downtrend → Uptrend**: Require 1 bar confirmation
4. **Any → Ranging**: Immediate transition allowed

### Confirmation Period

```python
def confirm_regime_transition(self, new_regime):
    """Require confirmation for critical transitions"""

    # Exiting chaos requires 2 bars confirmation
    if self.regime == 4 and new_regime != 4:
        if self.regime_confirmation_count < 2:
            self.regime_confirmation_count += 1
            return self.regime  # Stay in chaos

    # Trend reversals require 1 bar confirmation
    if (self.regime == 1 and new_regime == 2) or \
       (self.regime == 2 and new_regime == 1):
        if self.regime_confirmation_count < 1:
            self.regime_confirmation_count += 1
            return self.regime  # Stay in current regime

    # Reset confirmation count
    self.regime_confirmation_count = 0
    return new_regime
```

---

## Regime Statistics Output

Track regime distribution for analysis:

```python
# Regime counters (for statistics)
self.regime_1_bars = 0  # Strong Uptrend
self.regime_2_bars = 0  # Strong Downtrend
self.regime_3_bars = 0  # Sideways
self.regime_4_bars = 0  # Chaos

def update_regime_stats(self):
    """Increment regime counter"""
    if self.regime == 1:
        self.regime_1_bars += 1
    elif self.regime == 2:
        self.regime_2_bars += 1
    elif self.regime == 3:
        self.regime_3_bars += 1
    elif self.regime == 4:
        self.regime_4_bars += 1
```

---

## Next Steps

Proceed to Chapter 4 for buy/sell signal generation logic.
