# Chapter 4: Signal Logic

## Signal Generation Framework

Multi-confirmation system requiring ALL indicators to agree before generating buy/sell signals.

---

## Position State Machine

**Core Principle**: Alternating buy/sell signals to prevent consecutive entries in same direction.

```python
# Position states
self.position_state = 0  # 0 = flat/out, 1 = in position

# Signal values
# 1 = BUY signal
# -1 = SELL signal
# 0 = NEUTRAL (no action)
```

**State Transitions**:
- `position_state = 0` → Can only generate BUY (signal = 1)
- `position_state = 1` → Can only generate SELL (signal = -1)
- After BUY: `position_state = 1`
- After SELL: `position_state = 0`

---

## Regime-Specific Signal Rules

### Regime 1: Strong Uptrend

**BUY Signal** (position_state must be 0):

```python
def check_uptrend_buy(self):
    """Generate BUY in uptrend regime"""

    # Regime check
    if self.regime != 1:
        return False

    # Position check
    if self.position_state != 0:
        return False

    # Multi-confirmation
    trend_ok = self.ema_12 > self.ema_26 > self.ema_50
    momentum_ok = self.macd > self.macd_signal
    oversold_ok = self.rsi < 30
    volume_ok = self.volume > (self.volume_ema * 1.5)
    price_pullback = self.close <= self.bb_lower  # Touch lower band

    return (trend_ok and momentum_ok and oversold_ok and
            volume_ok and price_pullback)
```

**Confidence Calculation**:
```python
# Confidence increases as RSI gets lower (more oversold)
confidence = (30.0 - self.rsi) / 30.0
# Range: [0.0, 1.0]
```

**SELL Signal** (position_state must be 1):

Exit uptrend position on:
1. RSI overbought (>70), OR
2. MACD crossover (macd < macd_signal), OR
3. Close below EMA_26, OR
4. Stop loss hit (3%)

---

### Regime 2: Strong Downtrend

**SELL Signal** (position_state must be 0):

```python
def check_downtrend_sell(self):
    """Generate SELL in downtrend regime"""

    # Regime check
    if self.regime != 2:
        return False

    # Position check (can enter short from flat)
    if self.position_state != 0:
        return False

    # Multi-confirmation
    trend_ok = self.ema_12 < self.ema_26 < self.ema_50
    momentum_ok = self.macd < self.macd_signal
    overbought_ok = self.rsi > 70
    volume_ok = self.volume > (self.volume_ema * 1.5)
    price_rally = self.close >= self.bb_upper  # Touch upper band

    return (trend_ok and momentum_ok and overbought_ok and
            volume_ok and price_rally)
```

**Confidence Calculation**:
```python
# Confidence increases as RSI gets higher (more overbought)
confidence = (self.rsi - 70.0) / 30.0
# Range: [0.0, 1.0]
```

**BUY Signal** (cover short, position_state must be 1):

Exit downtrend position on:
1. RSI oversold (<30), OR
2. MACD crossover (macd > macd_signal), OR
3. Close above EMA_26, OR
4. Stop loss hit (3%)

---

### Regime 3: Sideways / Ranging

**Mean Reversion Strategy**

**BUY Signal** (position_state must be 0):

```python
def check_ranging_buy(self):
    """Generate BUY in ranging regime"""

    # Regime check
    if self.regime != 3:
        return False

    # Position check
    if self.position_state != 0:
        return False

    # Mean reversion at lower band
    at_lower_band = self.close <= self.bb_lower
    oversold = self.rsi < 30
    volume_ok = self.volume > (self.volume_ema * 1.2)  # Lower volume requirement

    # Price bouncing off support
    price_bounce = self.close > self.low  # Not at extreme low of bar

    return at_lower_band and oversold and volume_ok and price_bounce
```

**Confidence Calculation**:
```python
# Distance from lower band (mean reversion potential)
bb_range = self.bb_upper - self.bb_lower
distance_from_mean = self.bb_middle - self.close
confidence = min(distance_from_mean / bb_range, 1.0)
# Range: [0.0, 1.0]
```

**SELL Signal** (position_state must be 1):

```python
def check_ranging_sell(self):
    """Generate SELL in ranging regime"""

    # Regime check
    if self.regime != 3:
        return False

    # Position check
    if self.position_state != 1:
        return False

    # Mean reversion at upper band
    at_upper_band = self.close >= self.bb_upper
    overbought = self.rsi > 70
    volume_ok = self.volume > (self.volume_ema * 1.2)

    # Price hitting resistance
    price_resistance = self.close < self.high  # Not at extreme high of bar

    return at_upper_band and overbought and volume_ok and price_resistance
```

**Alternative Exit**: Price returns to middle band (take profit at mean)
```python
if self.position_state == 1 and self.close >= self.bb_middle:
    # Exit at middle band for quick profit
    return True
```

---

### Regime 4: High Volatility Chaos

**NO TRADING**

```python
def check_chaos_signal(self):
    """Force exit all positions in chaos regime"""

    if self.regime != 4:
        return False

    # If in position, force exit immediately
    if self.position_state == 1:
        self.signal = -1  # Exit long
        self.confidence = 1.0  # High confidence to exit
        self.position_state = 0
        return True

    return False
```

---

## Master Signal Generation Function

```python
def generate_signal(self):
    """
    Master signal generation logic
    Returns: signal (-1, 0, 1), confidence (0.0-1.0)
    """

    # Priority 1: Check chaos regime (force exits)
    if self.regime == 4:
        if self.position_state == 1:
            self.signal = -1
            self.confidence = 1.0
            self.position_state = 0
            return
        else:
            self.signal = 0
            self.confidence = 0.0
            return

    # Priority 2: Generate regime-specific signals
    if self.regime == 1:  # Strong Uptrend
        if self.position_state == 0:
            # Check for BUY
            if self.check_uptrend_buy():
                self.signal = 1
                self.confidence = (30.0 - self.rsi) / 30.0
                self.position_state = 1
                return
        elif self.position_state == 1:
            # Check for exit
            if self.check_uptrend_exit():
                self.signal = -1
                self.confidence = 0.8
                self.position_state = 0
                return

    elif self.regime == 2:  # Strong Downtrend
        if self.position_state == 0:
            # Check for SELL
            if self.check_downtrend_sell():
                self.signal = -1
                self.confidence = (self.rsi - 70.0) / 30.0
                self.position_state = 1
                return
        elif self.position_state == 1:
            # Check for cover
            if self.check_downtrend_exit():
                self.signal = 1
                self.confidence = 0.8
                self.position_state = 0
                return

    elif self.regime == 3:  # Sideways/Ranging
        if self.position_state == 0:
            # Check for BUY
            if self.check_ranging_buy():
                self.signal = 1
                bb_range = self.bb_upper - self.bb_lower
                distance = self.bb_middle - self.close
                self.confidence = min(distance / bb_range, 1.0)
                self.position_state = 1
                return
        elif self.position_state == 1:
            # Check for SELL
            if self.check_ranging_sell():
                self.signal = -1
                bb_range = self.bb_upper - self.bb_lower
                distance = self.close - self.bb_middle
                self.confidence = min(distance / bb_range, 1.0)
                self.position_state = 0
                return

    # Default: No signal
    self.signal = 0
    self.confidence = 0.0
```

---

## Exit Condition Helpers

```python
def check_uptrend_exit(self):
    """Exit long position in uptrend"""
    overbought = self.rsi > 70
    macd_cross = self.macd < self.macd_signal
    below_ema = self.close < self.ema_26
    return overbought or macd_cross or below_ema

def check_downtrend_exit(self):
    """Exit short position in downtrend"""
    oversold = self.rsi < 30
    macd_cross = self.macd > self.macd_signal
    above_ema = self.close > self.ema_26
    return oversold or macd_cross or above_ema
```

---

## Signal Filtering

**Volume Filter**: Reject signals on low volume days
```python
def volume_filter(self):
    """Reject signal if volume too low"""
    if self.volume < (self.volume_ema * 0.8):
        return False
    return True
```

**Time Filter**: Avoid first/last bar of session (optional)
```python
def time_filter(self):
    """Reject signals at session boundaries"""
    # Check if first or last 15-min bar of day
    # Implementation depends on market hours
    pass
```

---

## Signal Output Structure

```python
# Output fields for each bar
self.signal = 0          # -1, 0, 1
self.confidence = 0.0    # 0.0 to 1.0
self.regime = 3          # 1, 2, 3, 4
self.position_state = 0  # 0 or 1

# Additional context for analysis
self.signal_reason = ""  # "uptrend_buy", "ranging_sell", etc.
```

---

## Logging Signal Generation

```python
if self.signal != 0:
    logger.info(
        f"Bar {self.bar_index}: "
        f"Signal={self.signal}, "
        f"Confidence={self.confidence:.3f}, "
        f"Regime={self.regime}, "
        f"RSI={self.rsi:.2f}, "
        f"MACD={self.macd:.4f}, "
        f"Close={self.close:.2f}, "
        f"BB_Lower={self.bb_lower:.2f}, "
        f"BB_Upper={self.bb_upper:.2f}"
    )
```

---

## Next Steps

Proceed to Chapter 5 for risk management and position sizing rules.
