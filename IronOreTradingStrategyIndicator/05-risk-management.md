# Chapter 5: Risk Management

## Capital Allocation

**Total Capital**: $1,000,000

**Cash Reserve Requirements**:
- Minimum 40% cash reserve at all times
- Maximum 60% in active positions
- Buffer for margin calls and drawdowns

---

## Position Sizing Strategy

### Progressive Scaling System

**Three-Tier Entry System**:

```
Tier 1 (Initial Entry): 20% of capital = $200,000
Tier 2 (Scale 1):       40% of capital = $400,000
Tier 3 (Scale 2):       60% of capital = $600,000
```

### Scaling Logic

**Entry Sequence**:
1. **First Signal**: Enter with 20% of capital
2. **Confirmation Signal** (within 5 bars): Scale to 40%
3. **Strong Trend** (within 10 bars): Scale to 60%

**Scaling Conditions**:

```python
def calculate_position_size(self):
    """Calculate position size based on signal confidence and regime"""

    # Base position (always start here)
    base_size = 0.20  # 20%

    # Scaling multipliers
    if self.consecutive_confirmations >= 2:
        # Strong signal confirmation
        position_size = 0.40  # 40%
    elif self.consecutive_confirmations >= 3:
        # Very strong trend
        position_size = 0.60  # 60%
    else:
        position_size = base_size

    # Adjust for confidence
    position_size *= self.confidence

    # Regime-based adjustment
    if self.regime == 1 or self.regime == 2:  # Trending
        position_size *= 1.0  # Full size
    elif self.regime == 3:  # Ranging
        position_size *= 0.75  # Reduced size for mean reversion
    elif self.regime == 4:  # Chaos
        position_size = 0.0  # No positions

    # Cap at 60%
    position_size = min(position_size, 0.60)

    return position_size
```

### Position Size State Variables

```python
self.current_position_size = 0.0  # Current position as % of capital
self.consecutive_confirmations = 0  # Count of confirming signals
self.bars_in_position = 0  # Bars since entry
```

---

## Stop Loss Rules

### Fixed Percentage Stop

**Stop Loss**: 3% from entry price

```python
def calculate_stop_loss(self, entry_price, direction):
    """
    Calculate stop loss price

    Args:
        entry_price: Entry price
        direction: 1 (long), -1 (short)

    Returns:
        Stop loss price
    """
    stop_pct = 0.03  # 3%

    if direction == 1:  # Long position
        stop_loss = entry_price * (1.0 - stop_pct)
    else:  # Short position
        stop_loss = entry_price * (1.0 + stop_pct)

    return stop_loss
```

### ATR-Based Trailing Stop (Optional)

For trending regimes, use ATR-based trailing stop after profit threshold:

```python
def calculate_trailing_stop(self, entry_price, current_price, direction):
    """ATR-based trailing stop (activate after 5% profit)"""

    profit_pct = abs(current_price - entry_price) / entry_price

    # Activate trailing stop after 5% profit
    if profit_pct < 0.05:
        return self.calculate_stop_loss(entry_price, direction)

    # Trail by 2x ATR
    atr_multiplier = 2.0

    if direction == 1:  # Long
        trailing_stop = current_price - (self.atr * atr_multiplier)
    else:  # Short
        trailing_stop = current_price + (self.atr * atr_multiplier)

    return trailing_stop
```

### Stop Loss State Variables

```python
self.entry_price = 0.0
self.stop_loss_price = 0.0
self.trailing_stop_active = False
self.highest_price_in_trade = 0.0  # For trailing stop
self.lowest_price_in_trade = 999999.0
```

---

## Profit Targets

### Two-Tier Take Profit System

**Target 1**: 5% profit → Exit 50% of position
**Target 2**: 8% profit → Exit remaining 50%

```python
def check_profit_targets(self, entry_price, current_price, direction):
    """
    Check profit targets and calculate exit size

    Returns:
        (exit_size, target_hit)
    """

    profit_pct = (current_price - entry_price) / entry_price * direction

    if profit_pct >= 0.08:  # 8% target
        return (1.0, 2)  # Exit 100% (remaining 50%)
    elif profit_pct >= 0.05:  # 5% target
        if not self.target_1_hit:
            self.target_1_hit = True
            return (0.5, 1)  # Exit 50%

    return (0.0, 0)  # No exit
```

### Regime-Specific Adjustments

**Ranging Regime** (tighter targets):
- Target 1: 3% → Exit 50%
- Target 2: 5% → Exit 50%

**Trending Regime** (wider targets):
- Target 1: 5% → Exit 50%
- Target 2: 8% → Exit 50%

```python
def get_profit_targets(self):
    """Get regime-adjusted profit targets"""

    if self.regime == 3:  # Ranging
        return (0.03, 0.05)
    else:  # Trending
        return (0.05, 0.08)
```

### Profit Target State Variables

```python
self.target_1_hit = False
self.target_2_hit = False
self.remaining_position_pct = 1.0  # 100% → 50% → 0%
```

---

## Risk Per Trade

### Maximum Risk

**Per Trade Risk**: 3% of capital = $30,000

**Position Size Calculation Based on Risk**:

```python
def calculate_risk_adjusted_position(self, entry_price, stop_loss_price):
    """
    Calculate position size based on fixed risk amount

    Args:
        entry_price: Entry price
        stop_loss_price: Stop loss price

    Returns:
        Position size in contracts
    """

    risk_per_trade = 30000  # $30,000 (3% of $1M)
    risk_per_contract = abs(entry_price - stop_loss_price)

    # Calculate number of contracts
    num_contracts = risk_per_trade / risk_per_contract

    # Calculate position value
    position_value = num_contracts * entry_price

    # Verify within capital limits
    max_position_value = 1000000 * 0.60  # 60% max

    if position_value > max_position_value:
        # Reduce to max allowed
        num_contracts = max_position_value / entry_price

    return int(num_contracts)
```

---

## Daily Loss Limit

**Maximum Daily Loss**: 5% of capital = $50,000

```python
# Daily tracking
self.daily_pnl = 0.0
self.daily_loss_limit = -50000
self.daily_trades = 0
self.current_tradeday = None

def check_daily_loss_limit(self):
    """Check if daily loss limit hit"""

    if self.daily_pnl <= self.daily_loss_limit:
        logger.warning(
            f"Daily loss limit hit: ${self.daily_pnl:.2f}. "
            "No new trades until next session."
        )
        return True

    return False

def reset_daily_stats(self, new_tradeday):
    """Reset daily tracking on new trading day"""

    if self.current_tradeday != new_tradeday:
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.current_tradeday = new_tradeday
```

---

## Maximum Drawdown Protection

**Maximum Drawdown Threshold**: 15% from peak equity

```python
self.peak_equity = 1000000  # Initial capital
self.current_equity = 1000000
self.max_drawdown_pct = 0.15

def check_max_drawdown(self):
    """Check maximum drawdown threshold"""

    # Update peak
    if self.current_equity > self.peak_equity:
        self.peak_equity = self.current_equity

    # Calculate drawdown
    drawdown = (self.peak_equity - self.current_equity) / self.peak_equity

    if drawdown >= self.max_drawdown_pct:
        logger.critical(
            f"Maximum drawdown reached: {drawdown*100:.2f}%. "
            "Halting all trading."
        )
        return True

    return False
```

---

## Position Limits

### Maximum Concurrent Positions

**Limit**: 1 position at a time (single instrument strategy)

```python
self.max_concurrent_positions = 1
self.active_positions = 0

def can_open_position(self):
    """Check if can open new position"""
    return self.active_positions < self.max_concurrent_positions
```

### Exposure Limits

**Maximum Gross Exposure**: 60% of capital
**Maximum Net Exposure**: 60% of capital (long or short)

```python
self.gross_exposure = 0.0  # Sum of all position values
self.net_exposure = 0.0    # Long value - Short value

def check_exposure_limits(self, new_position_value):
    """Check exposure limits before entry"""

    new_gross = self.gross_exposure + abs(new_position_value)
    max_exposure = 1000000 * 0.60

    if new_gross > max_exposure:
        logger.warning(
            f"Exposure limit exceeded: ${new_gross:.2f} > ${max_exposure:.2f}"
        )
        return False

    return True
```

---

## Emergency Exit Conditions

Force exit all positions immediately if:

1. **Regime 4** (High Volatility Chaos) detected
2. **Daily loss limit** hit
3. **Maximum drawdown** threshold reached
4. **Market circuit breaker** (exchange halt)

```python
def emergency_exit(self):
    """Force exit all positions"""

    if self.position_state == 1:
        logger.critical("Emergency exit triggered!")

        self.signal = -1  # Exit signal
        self.confidence = 1.0
        self.position_state = 0
        self.current_position_size = 0.0

        # Log exit reason
        logger.critical(
            f"Emergency exit: Regime={self.regime}, "
            f"Daily PnL=${self.daily_pnl:.2f}, "
            f"Drawdown={self.get_current_drawdown()*100:.2f}%"
        )
```

---

## Risk Metrics Tracking

```python
# Risk metrics (for analysis)
self.total_trades = 0
self.winning_trades = 0
self.losing_trades = 0
self.total_pnl = 0.0
self.max_win = 0.0
self.max_loss = 0.0
self.consecutive_losses = 0
self.max_consecutive_losses = 0

def update_trade_stats(self, trade_pnl):
    """Update risk metrics after trade close"""

    self.total_trades += 1
    self.total_pnl += trade_pnl

    if trade_pnl > 0:
        self.winning_trades += 1
        self.consecutive_losses = 0
        self.max_win = max(self.max_win, trade_pnl)
    else:
        self.losing_trades += 1
        self.consecutive_losses += 1
        self.max_consecutive_losses = max(
            self.max_consecutive_losses,
            self.consecutive_losses
        )
        self.max_loss = min(self.max_loss, trade_pnl)

    # Check for consecutive loss limit (optional)
    if self.consecutive_losses >= 3:
        logger.warning(
            f"3 consecutive losses. Consider pausing trading."
        )
```

---

## Next Steps

Proceed to Chapter 6 for implementation code structure and algorithms.
