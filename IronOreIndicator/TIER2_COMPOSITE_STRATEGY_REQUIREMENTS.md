# Tier-2 Composite Trading Strategy - Requirements Document

**Document Version**: 1.0
**Date**: 2025-11-06
**Target**: AI agents and developers
**Framework**: WOS (Wolverine Operating System)
**Strategy Type**: Multi-Instrument Portfolio Management (Tier-2)

---

## Executive Summary

This document specifies a complete Tier-2 composite strategy that manages a diversified portfolio of 3 futures instruments using signals from dedicated Tier-1 indicators. The strategy implements equal-weight capital allocation, strict risk management, and systematic rebalancing to achieve consistent positive P&L.

**Primary Objective**: Build a production-ready Tier-2 strategy that maintains positive P&L through diversified, risk-managed trading across multiple commodities.

---

## 1. Project Overview

### 1.1 Strategy Architecture

```
┌────────────────────────────────────────────────────┐
│  TIER 1: Three Independent Indicators              │
│  ─────────────────────────────────────────────────│
│  1. IronOreIndicator (DCE/i<00>)    [EXISTING]    │
│  2. CopperIndicator (SHFE/cu<00>)   [TO BUILD]    │
│  3. SoybeanIndicator (DCE/m<00>)    [TO BUILD]    │
│                                                    │
│  Each generates: signal, confidence, regime, etc. │
└──────────────────┬─────────────────────────────────┘
                   │
                   │ Individual Signals (3 streams)
                   ▼
┌────────────────────────────────────────────────────┐
│  TIER 2: COMPOSITE STRATEGY                        │
│  ─────────────────────────────────────────────────│
│  • Signal aggregation (3 instruments)              │
│  • Basket management (3 baskets)                   │
│  • Equal-weight capital allocation                 │
│  • Risk management (stops, drawdown limits)        │
│  • Portfolio rebalancing                           │
│  • Performance tracking                            │
└──────────────────┬─────────────────────────────────┘
                   │
                   │ Portfolio State & Trades
                   ▼
┌────────────────────────────────────────────────────┐
│  OUTPUT: Portfolio Performance                     │
│  • Position sizes per instrument                   │
│  • Net value (NV) and Portfolio value (PV)         │
│  • Drawdown tracking                               │
│  • Allocation percentages                          │
└────────────────────────────────────────────────────┘
```

### 1.2 Key Design Principles

1. **Diversification**: 3 instruments from different sectors reduce correlated risk
2. **Conservative Risk Management**: 10% max drawdown, 3% per-position stop-loss
3. **Equal-Weight Allocation**: Simple 25% base per instrument (75% total, 25% cash reserve)
4. **Independent Management**: Each instrument managed separately (Phase 1)
5. **Proven Indicators**: Use IronOre 7-indicator system for all 3 instruments
6. **Systematic Rebalancing**: Weekly portfolio rebalancing to maintain target allocations

### 1.3 Portfolio Specifications

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Initial Capital | ¥1,000,000 | Base portfolio size |
| Number of Instruments | 3 | Optimal diversification vs complexity |
| Base Allocation per Instrument | 25% (¥250,000) | Equal weight, conservative |
| Max Allocation per Instrument | 35% (¥350,000) | Allow 10% overweight |
| Cash Reserve (Minimum) | 15% (¥150,000) | Safety buffer |
| Max Total Exposure | 90% (¥900,000) | Conservative leverage |
| Max Portfolio Drawdown | 10% (¥100,000) | Risk tolerance limit |
| Max Daily Loss | 3% (¥30,000) | Circuit breaker |

### 1.4 Deliverables

**Location**: `Tier2CompositeStrategy/`

| File/Directory | Purpose |
|----------------|---------|
| `CopperIndicator/` | Tier-1 indicator for SHFE Copper |
| `CopperIndicator/CopperIndicator.py` | Main implementation |
| `CopperIndicator/uin.json` | Input config |
| `CopperIndicator/uout.json` | Output config |
| `CopperIndicator/test_resuming_mode.py` | Replay test |
| `SoybeanIndicator/` | Tier-1 indicator for DCE Soybean Meal |
| `SoybeanIndicator/SoybeanIndicator.py` | Main implementation |
| `SoybeanIndicator/uin.json` | Input config |
| `SoybeanIndicator/uout.json` | Output config |
| `SoybeanIndicator/test_resuming_mode.py` | Replay test |
| `CompositeStrategy/` | Tier-2 portfolio manager |
| `CompositeStrategy/CompositeStrategy.py` | Main strategy implementation |
| `CompositeStrategy/uin.json` | Import Tier-1 signals |
| `CompositeStrategy/uout.json` | Portfolio state output |
| `CompositeStrategy/analysis.ipynb` | P&L visualization |
| `CompositeStrategy/STRATEGY_GUIDE.md` | Strategy documentation |

---

## 2. Instrument Selection

### 2.1 Selected Instruments

| # | Market | Code | Instrument | Sector | Rationale |
|---|--------|------|------------|--------|-----------|
| 1 | DCE | i | Iron Ore | Industrial Metal | Existing indicator, proven performance |
| 2 | SHFE | cu | Copper | Base Metal | High liquidity, good trends, different market |
| 3 | DCE | m | Soybean Meal | Agriculture | Different sector, diversification |

### 2.2 Diversification Analysis

**Correlation Profile** (Expected):
- Iron Ore ↔ Copper: Moderate positive (0.4-0.6) - Both industrial metals
- Iron Ore ↔ Soybean: Low (0.1-0.3) - Different sectors
- Copper ↔ Soybean: Low (0.1-0.3) - Different sectors

**Sector Distribution**:
- Industrial Metals: 66.7% (Iron Ore + Copper)
- Agriculture: 33.3% (Soybean Meal)

**Market Distribution**:
- DCE: 66.7% (Iron Ore + Soybean)
- SHFE: 33.3% (Copper)

**Why These 3?**
1. ✅ Proven indicator exists for Iron Ore
2. ✅ Copper: Highly liquid, strong trends, different exchange
3. ✅ Soybean: Agricultural diversification, reduces metal correlation
4. ✅ All have 15-minute data available in WOS framework
5. ✅ Logical contracts available (i<00>, cu<00>, m<00>)

### 2.3 Liquidity Requirements

All 3 instruments meet minimum criteria:
- Average daily volume: >100,000 contracts
- Bid-ask spread: <0.1% of price
- Active trading: Full day coverage
- SampleQuote availability: 900s granularity

---

## 3. Tier-1 Indicator Specifications

### 3.1 Indicator Architecture (All 3 Instruments)

**Base Design**: IronOreIndicator 7-Indicator System

All three indicators use the same proven architecture:

**Technical Components**:
1. **Triple EMA** - Trend identification
2. **MACD** - Momentum confirmation
3. **RSI** - Mean reversion signals
4. **Bollinger Bands** - Price extremes & volatility
5. **ATR** - Volatility measurement
6. **BB Width** - Volatility confirmation
7. **Volume EMA** - Liquidity confirmation

**Market Regime Detection** (4 Regimes):
1. Strong Uptrend
2. Strong Downtrend
3. Sideways/Ranging
4. High Volatility Chaos

**Output Fields** (per indicator):
- `signal`: -1 (sell), 0 (neutral), 1 (buy)
- `confidence`: 0.0-1.0 (signal strength)
- `regime`: 1/2/3/4 (market regime)
- `signal_strength`: 0.0-1.0 (multi-factor score)
- `contracts_to_add`: 50/100 (position size suggestion)
- Technical indicator values (ema_12, ema_26, rsi, etc.)

### 3.2 CopperIndicator (SHFE/cu<00>) - NEW

**Container**: `CopperIndicator/`

**Parameter Tuning** (vs Iron Ore):
```python
# Copper is more reactive - use slightly faster EMAs
self.alpha_12 = 2.0 / 13.0   # 12-period (same)
self.alpha_26 = 2.0 / 25.0   # 24-period (faster than 26)
self.alpha_50 = 2.0 / 46.0   # 45-period (faster than 50)

# RSI - slightly more sensitive
# Use 12-period instead of 14
self.alpha_rsi = 2.0 / 13.0

# Other parameters same as IronOre
```

**Rationale**:
- Copper has higher liquidity → can use faster EMAs
- More intraday volatility → slightly more responsive indicators
- Base logic remains same (proven system)

**Implementation**:
- Copy IronOreIndicator.py → CopperIndicator.py
- Modify: market = b'SHFE', code = b'cu<00>'
- Adjust parameters as specified above
- Update meta_name = "CopperIndicator"

### 3.3 SoybeanIndicator (DCE/m<00>) - NEW

**Container**: `SoybeanIndicator/`

**Parameter Tuning** (vs Iron Ore):
```python
# Soybean is less volatile - use slightly slower EMAs
self.alpha_12 = 2.0 / 16.0   # 15-period (slower than 12)
self.alpha_26 = 2.0 / 31.0   # 30-period (slower than 26)
self.alpha_50 = 2.0 / 61.0   # 60-period (slower than 50)

# RSI - standard 14-period
self.alpha_rsi = 2.0 / 15.0  # (same as IronOre)

# BB period - slightly longer for noise reduction
# Use 22 instead of 20
self.bb_period = 22
```

**Rationale**:
- Agricultural commodity with different volatility profile
- More daily seasonality → slower EMAs reduce false signals
- Less intraday noise with longer periods
- Base logic remains same (proven system)

**Implementation**:
- Copy IronOreIndicator.py → SoybeanIndicator.py
- Modify: market = b'DCE', code = b'm<00>'
- Adjust parameters as specified above
- Update meta_name = "SoybeanIndicator"

### 3.4 IronOreIndicator (DCE/i<00>) - EXISTING

**Container**: `IronOreIndicator/` (already exists)

**Status**: No changes required - use as-is

**Parameters**: Keep existing parameters (proven through backtest)

**Note**: This indicator is already implemented and tested.

---

## 4. Tier-2 Composite Strategy Specification

### 4.1 CompositeStrategy Architecture

**Base Class**: `composite_strategyc3.composite_strategy`

**Container**: `CompositeStrategy/`

**Basket Configuration**:
```python
BASKET_COUNT = 3

BASKET_MAPPING = {
    0: ('DCE', b'i'),   # Iron Ore
    1: ('SHFE', b'cu'), # Copper
    2: ('DCE', b'm'),   # Soybean Meal
}
```

### 4.2 Capital Allocation Strategy

**Phase 1: Equal-Weight Allocation** (Initial Implementation)

```python
ALLOCATION_CONFIG = {
    'strategy_type': 'equal_weight',

    # Base allocations
    'base_allocation_pct': 0.25,      # 25% per instrument
    'max_allocation_pct': 0.35,       # 35% max per instrument
    'min_allocation_pct': 0.15,       # 15% min when active

    # Portfolio limits
    'max_total_exposure': 0.90,       # 90% max total
    'min_cash_reserve': 0.15,         # 15% min cash (conservative)

    # Rebalancing
    'rebalance_threshold': 0.10,      # 10% deviation triggers rebalance
    'rebalance_frequency': 96,        # Daily (96 × 15min bars)
}
```

**Allocation Logic**:
```python
def calculate_allocation(portfolio_value):
    """
    Simple equal-weight allocation

    Initial state:
    - Iron Ore: ¥250,000 (25%)
    - Copper: ¥250,000 (25%)
    - Soybean: ¥250,000 (25%)
    - Cash: ¥250,000 (25%)

    After positions taken (example):
    - Iron Ore: ¥300,000 (30%) - in position
    - Copper: ¥250,000 (25%) - in position
    - Soybean: ¥200,000 (20%) - in position
    - Cash: ¥250,000 (25%)
    Total Exposure: 75%
    """
    base_per_instrument = portfolio_value * 0.25

    return {
        'iron_ore': base_per_instrument,
        'copper': base_per_instrument,
        'soybean': base_per_instrument,
        'cash_reserve': base_per_instrument,
    }
```

### 4.3 Signal Aggregation Logic

**Strategy**: Independent Management (Phase 1)

Each instrument is managed independently based on its own Tier-1 signal:

```python
def process_signals(tier1_signals):
    """
    Process signals from 3 Tier-1 indicators independently

    Args:
        tier1_signals: {
            'iron_ore': {'signal': 1, 'confidence': 0.7, 'regime': 1, ...},
            'copper': {'signal': -1, 'confidence': 0.6, 'regime': 2, ...},
            'soybean': {'signal': 0, 'confidence': 0.4, 'regime': 3, ...},
        }

    Returns:
        actions: {
            'iron_ore': 'LONG',
            'copper': 'SHORT',
            'soybean': 'HOLD',
        }
    """
    actions = {}

    for instrument, signal_data in tier1_signals.items():
        # Check entry conditions
        if should_enter(signal_data):
            if signal_data['signal'] == 1:
                actions[instrument] = 'LONG'
            elif signal_data['signal'] == -1:
                actions[instrument] = 'SHORT'

        # Check exit conditions
        elif should_exit(signal_data, current_positions[instrument]):
            actions[instrument] = 'CLOSE'

        else:
            actions[instrument] = 'HOLD'

    return actions
```

**No Cross-Instrument Dependencies** (Phase 1):
- Each instrument traded independently
- No correlation adjustments
- No dynamic allocation between instruments
- Simpler to implement and debug

**Future Enhancement** (Phase 2):
- Add correlation-aware allocation
- Reduce exposure when instruments move in sync
- Dynamic reallocation based on relative performance

### 4.4 Position Sizing

**Method**: Capital-Based Sizing

```python
def calculate_position_size(allocated_capital, current_price, max_allocation_pct=0.35):
    """
    Calculate number of contracts based on allocated capital

    Example:
    - Allocated capital: ¥250,000 (base)
    - Current price: ¥800/contract
    - Max allocation: 35% (can go up to ¥350,000)

    Standard entry:
    - Use 100% of base allocation
    - Contracts = ¥250,000 / ¥800 = 312 contracts

    Conservative entry (low confidence):
    - Use 80% of base allocation
    - Contracts = ¥200,000 / ¥800 = 250 contracts

    Aggressive entry (high confidence):
    - Use 120% of base (up to max)
    - Contracts = ¥300,000 / ¥800 = 375 contracts
    """
    # Base calculation
    capital_to_use = allocated_capital

    # Confidence adjustment (simplified for Phase 1)
    # Just use base allocation - no scaling yet

    # Calculate contracts
    contracts = int(capital_to_use / current_price)

    # Ensure we don't exceed max allocation
    max_contracts = int((allocated_capital * max_allocation_pct / 0.25) / current_price)
    contracts = min(contracts, max_contracts)

    return max(contracts, 0)
```

**Contract Sizing Rules**:
- Minimum position: 50 contracts (below this, don't enter)
- Maximum per instrument: Based on 35% allocation cap
- Rounding: Always round down to integer contracts
- Cash check: Only enter if sufficient cash available

### 4.5 Entry Conditions

**Conservative Entry Filters** (For Positive P&L):

```python
def should_enter(signal_data, current_position=0):
    """
    Strict entry conditions for new positions

    Returns: True if should enter, False otherwise
    """
    # 1. Not already in position
    if current_position != 0:
        return False

    # 2. Strong signal (not neutral)
    if signal_data['signal'] == 0:
        return False

    # 3. High confidence threshold
    if signal_data['confidence'] < 0.60:  # 60% minimum
        return False

    # 4. Avoid chaos regime
    if signal_data['regime'] == 4:  # Chaos
        return False

    # 5. Signal strength check
    if signal_data['signal_strength'] < 0.50:  # 50% minimum
        return False

    # 6. Risk management checks (via RiskManager)
    if not risk_manager.can_enter(instrument, proposed_size):
        return False

    return True
```

**Entry Scenarios**:

| Scenario | Signal | Confidence | Regime | Action |
|----------|--------|------------|--------|--------|
| Strong Uptrend | 1 | 0.75 | 1 | ✅ ENTER LONG |
| Weak Uptrend | 1 | 0.45 | 1 | ❌ SKIP (low confidence) |
| Uptrend in Chaos | 1 | 0.70 | 4 | ❌ SKIP (chaos regime) |
| Strong Downtrend | -1 | 0.65 | 2 | ✅ ENTER SHORT |
| Ranging | 1 | 0.62 | 3 | ✅ ENTER LONG |
| Neutral Signal | 0 | 0.80 | 1 | ❌ SKIP (no signal) |

### 4.6 Exit Conditions

**Multi-Layered Exit Strategy**:

#### 4.6.1 Stop-Loss Exits (Highest Priority)

```python
def check_stop_loss(entry_price, current_price, position_type):
    """
    Hard stop-loss: 3% per position

    Returns: True if stop-loss hit
    """
    if position_type == 1:  # Long position
        loss_pct = (entry_price - current_price) / entry_price
    else:  # Short position
        loss_pct = (current_price - entry_price) / entry_price

    STOP_LOSS_PCT = 0.03  # 3%

    if loss_pct >= STOP_LOSS_PCT:
        logger.warning(f"STOP-LOSS HIT: {loss_pct*100:.2f}% loss")
        return True

    return False
```

#### 4.6.2 Profit-Taking Exits

```python
def check_profit_targets(entry_price, current_price, position_type, contracts_held):
    """
    Scaled profit taking to lock in gains

    Returns: (action, contracts_to_close)
    """
    if position_type == 1:  # Long
        profit_pct = (current_price - entry_price) / entry_price
    else:  # Short
        profit_pct = (entry_price - current_price) / entry_price

    # Profit targets
    TARGET_1 = 0.02  # 2% - take 50% off
    TARGET_2 = 0.05  # 5% - take another 30% off
    TARGET_3 = 0.08  # 8% - trailing stop for rest

    if profit_pct >= TARGET_3:
        # Trailing stop mode (advanced - Phase 2)
        return 'trailing_stop', 0

    elif profit_pct >= TARGET_2:
        # Take 30% more off (80% total closed)
        contracts_to_close = int(contracts_held * 0.30)
        return 'partial_close', contracts_to_close

    elif profit_pct >= TARGET_1:
        # Take 50% off table
        contracts_to_close = int(contracts_held * 0.50)
        return 'partial_close', contracts_to_close

    return 'hold', 0
```

#### 4.6.3 Signal-Based Exits

```python
def should_exit_on_signal(signal_data, current_position):
    """
    Exit based on Tier-1 signal reversal or weakness

    Returns: True if should exit
    """
    # 1. Signal reversal (opposite direction)
    if signal_data['signal'] * current_position < 0:
        logger.info("EXIT: Signal reversed")
        return True

    # 2. Confidence dropped significantly
    if signal_data['confidence'] < 0.30:  # 30% threshold
        logger.info("EXIT: Confidence dropped")
        return True

    # 3. Regime changed to chaos
    if signal_data['regime'] == 4:
        logger.info("EXIT: Chaos regime detected")
        return True

    # 4. Signal turned neutral with low confidence
    if signal_data['signal'] == 0 and signal_data['confidence'] < 0.40:
        logger.info("EXIT: Signal neutralized")
        return True

    return False
```

#### 4.6.4 Portfolio-Level Circuit Breaker

```python
def check_portfolio_circuit_breaker(portfolio_state):
    """
    Portfolio-level emergency exits

    Returns: True if should close ALL positions
    """
    # 1. Max drawdown exceeded
    if portfolio_state.drawdown >= 0.10:  # 10%
        logger.error("CIRCUIT BREAKER: Max drawdown exceeded")
        return True

    # 2. Daily loss limit hit
    if portfolio_state.daily_loss >= 0.03:  # 3%
        logger.error("CIRCUIT BREAKER: Daily loss limit hit")
        return True

    # 3. Multiple positions hitting stops simultaneously
    if portfolio_state.simultaneous_stops >= 2:
        logger.error("CIRCUIT BREAKER: Multiple stops hit")
        return True

    return False
```

**Exit Priority Order**:
1. Portfolio circuit breaker (closes ALL positions)
2. Per-position stop-loss (closes individual position immediately)
3. Profit-taking (partial closes to lock gains)
4. Signal-based exits (closes on reversal/weakness)

### 4.7 Portfolio Rebalancing

**Rebalancing Strategy**:

```python
class Rebalancer:
    def __init__(self):
        self.rebalance_frequency = 96  # Daily (96 bars × 15min)
        self.rebalance_threshold = 0.10  # 10% deviation
        self.last_rebalance_bar = 0

    def should_rebalance(self, current_bar, current_allocations, target_allocations):
        """
        Check if rebalancing needed

        Triggers:
        1. Time-based: Every 96 bars (daily)
        2. Deviation-based: >10% drift from target
        """
        # 1. Check frequency
        bars_since_rebalance = current_bar - self.last_rebalance_bar
        time_trigger = bars_since_rebalance >= self.rebalance_frequency

        # 2. Check allocation drift
        max_deviation = 0.0
        drifted_instrument = None

        for instrument in current_allocations:
            current_pct = current_allocations[instrument]
            target_pct = target_allocations[instrument]
            deviation = abs(current_pct - target_pct)

            if deviation > max_deviation:
                max_deviation = deviation
                drifted_instrument = instrument

        deviation_trigger = max_deviation > self.rebalance_threshold

        return (time_trigger or deviation_trigger), drifted_instrument, max_deviation

    def calculate_rebalance_actions(self, current_positions, target_allocations, portfolio_value):
        """
        Calculate trades needed to rebalance

        Example:
        Target: Each 25% (equal weight)
        Current:
          - Iron Ore: 35% (overweight +10%)
          - Copper: 20% (underweight -5%)
          - Soybean: 20% (underweight -5%)

        Actions:
          - Reduce Iron Ore by ¥100,000 (10% of ¥1M)
          - Increase Copper by ¥50,000 (5% of ¥1M)
          - Increase Soybean by ¥50,000 (5% of ¥1M)
        """
        actions = {}

        for instrument in current_positions:
            current_value = current_positions[instrument]
            current_pct = current_value / portfolio_value
            target_pct = target_allocations[instrument]

            deviation = current_pct - target_pct

            if abs(deviation) > self.rebalance_threshold:
                # Calculate adjustment needed
                adjustment_value = deviation * portfolio_value

                actions[instrument] = {
                    'type': 'reduce' if deviation > 0 else 'increase',
                    'value': abs(adjustment_value),
                    'deviation_pct': deviation * 100,
                }

        return actions
```

**Rebalancing Execution**:

```python
def execute_rebalance(actions, current_prices):
    """
    Execute rebalancing trades

    Order of operations:
    1. Close overweight positions first (raises cash)
    2. Open underweight positions (uses cash)
    """
    # Phase 1: Reduce overweight positions
    for instrument, action in actions.items():
        if action['type'] == 'reduce':
            contracts_to_close = int(action['value'] / current_prices[instrument])
            logger.info(f"REBALANCE: Reducing {instrument} by {contracts_to_close} contracts")
            close_position(instrument, contracts_to_close)

    # Phase 2: Increase underweight positions
    for instrument, action in actions.items():
        if action['type'] == 'increase':
            contracts_to_open = int(action['value'] / current_prices[instrument])
            logger.info(f"REBALANCE: Increasing {instrument} by {contracts_to_open} contracts")
            open_position(instrument, contracts_to_open)
```

**Rebalancing Constraints**:
- Never rebalance during active stop-loss scenarios
- Never rebalance if in portfolio drawdown >5%
- Respect per-instrument max allocation limits
- Maintain minimum cash reserve (15%)

---

## 5. Risk Management System

### 5.1 Risk Manager Implementation

```python
class RiskManager:
    """
    Centralized risk management for portfolio
    """

    def __init__(self, initial_capital=1000000.0):
        self.initial_capital = initial_capital

        # Per-position limits
        self.max_position_pct = 0.35  # 35% max per instrument
        self.max_position_loss = 0.03  # 3% stop-loss per position

        # Portfolio limits
        self.max_total_exposure = 0.90  # 90% max total
        self.min_cash_reserve = 0.15   # 15% min cash

        # Drawdown limits
        self.max_portfolio_drawdown = 0.10  # 10% max DD
        self.max_daily_loss = 0.03          # 3% daily limit

        # Position limits
        self.max_simultaneous_positions = 3  # All instruments can be active
        self.min_contracts_per_position = 50  # Minimum position size

        # Tracking
        self.peak_portfolio_value = initial_capital
        self.daily_start_value = initial_capital
        self.daily_loss = 0.0
        self.current_drawdown = 0.0

    def update_daily_tracking(self, portfolio_value):
        """Update daily P&L tracking (call on tradeday_begin)"""
        self.daily_start_value = portfolio_value
        self.daily_loss = 0.0

    def update_peak_tracking(self, portfolio_value):
        """Update peak and drawdown tracking (call every bar)"""
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value

        self.current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
        self.daily_loss = (self.daily_start_value - portfolio_value) / self.daily_start_value

    def can_enter_position(self, instrument, proposed_size, portfolio_state):
        """
        Pre-trade risk checks

        Returns: (can_enter: bool, reason: str)
        """
        # 1. Position size limit
        position_pct = proposed_size / portfolio_state.total_value
        if position_pct > self.max_position_pct:
            return False, f"Position size {position_pct*100:.1f}% exceeds limit {self.max_position_pct*100:.1f}%"

        # 2. Total exposure limit
        total_exposure = portfolio_state.total_exposure + proposed_size
        exposure_pct = total_exposure / portfolio_state.total_value
        if exposure_pct > self.max_total_exposure:
            return False, f"Total exposure {exposure_pct*100:.1f}% exceeds limit {self.max_total_exposure*100:.1f}%"

        # 3. Cash reserve check
        remaining_cash = portfolio_state.cash - proposed_size
        cash_pct = remaining_cash / portfolio_state.total_value
        if cash_pct < self.min_cash_reserve:
            return False, f"Cash reserve {cash_pct*100:.1f}% below minimum {self.min_cash_reserve*100:.1f}%"

        # 4. Drawdown check
        if self.current_drawdown >= self.max_portfolio_drawdown:
            return False, f"Portfolio drawdown {self.current_drawdown*100:.1f}% exceeds limit {self.max_portfolio_drawdown*100:.1f}%"

        # 5. Daily loss check
        if self.daily_loss >= self.max_daily_loss:
            return False, f"Daily loss {self.daily_loss*100:.1f}% exceeds limit {self.max_daily_loss*100:.1f}%"

        # 6. Minimum position size
        if proposed_size < self.min_contracts_per_position * portfolio_state.current_prices[instrument]:
            return False, f"Position size below minimum {self.min_contracts_per_position} contracts"

        return True, "OK"

    def get_risk_metrics(self):
        """Return current risk metrics for logging"""
        return {
            'current_drawdown': self.current_drawdown,
            'daily_loss': self.daily_loss,
            'peak_value': self.peak_portfolio_value,
        }
```

### 5.2 Risk Limits Summary

| Risk Parameter | Limit | Action When Exceeded |
|----------------|-------|----------------------|
| Per-position stop-loss | 3% | Close position immediately |
| Max drawdown | 10% | Close all positions (circuit breaker) |
| Daily loss limit | 3% | Stop new entries for day |
| Position size | 35% max | Reject new position entry |
| Total exposure | 90% max | Reject new entries |
| Cash reserve | 15% min | Reject new entries |
| Minimum position | 50 contracts | Don't enter if below minimum |

### 5.3 Risk Monitoring & Logging

```python
def log_risk_metrics(bar_index, portfolio_state, risk_manager):
    """
    Log risk metrics every bar for monitoring
    """
    metrics = risk_manager.get_risk_metrics()

    logger.info(
        f"[Bar {bar_index}] RISK METRICS: "
        f"PV=¥{portfolio_state.total_value:,.0f}, "
        f"DD={metrics['current_drawdown']*100:.2f}%, "
        f"DailyLoss={metrics['daily_loss']*100:.2f}%, "
        f"Exposure={portfolio_state.total_exposure_pct*100:.1f}%, "
        f"Cash={portfolio_state.cash_pct*100:.1f}%"
    )
```

---

## 6. Implementation Specifications

### 6.1 Module Structure

**File**: `CompositeStrategy/CompositeStrategy.py`

```python
#!/usr/bin/env python3
# coding=utf-8
"""
Tier-2 Composite Strategy - Multi-Instrument Portfolio Manager

Manages 3-instrument portfolio with equal-weight allocation:
- Iron Ore (DCE/i<00>)
- Copper (SHFE/cu<00>)
- Soybean Meal (DCE/m<00>)

Features:
- Equal-weight capital allocation (25% base per instrument)
- Conservative risk management (10% max DD, 3% stops)
- Systematic rebalancing
- Independent instrument management
"""

import math
from typing import Dict, List, Tuple
from collections import defaultdict

import pycaitlyn as pc
import pycaitlynutils3 as pcu3
import pycaitlynts3 as pcts3
import composite_strategyc3 as csc3

# Framework globals (REQUIRED)
use_raw = True
overwrite = False
granularity = 900
schema = None
max_workers = 1
worker_no = None
exports = {}
imports = {}
metas = {}
logger = pcu3.vanilla_logger()

# Configuration
BASKET_COUNT = 3
INITIAL_CAPITAL = 1000000.0

# Instrument mapping
BASKET_MAPPING = {
    0: (b'DCE', b'i'),   # Iron Ore
    1: (b'SHFE', b'cu'), # Copper
    2: (b'DCE', b'm'),   # Soybean Meal
}
```

### 6.2 Signal Parser Classes

```python
class IronOreSignalParser(pcts3.sv_object):
    """Parse IronOreIndicator signals"""
    def __init__(self):
        super().__init__()
        self.meta_name = "IronOreIndicatorRelaxed"
        self.namespace = pc.namespace_private
        self.revision = (1 << 32) - 1

        # Signal fields (from Tier-1)
        self.bar_index = 0
        self.signal = 0
        self.confidence = 0.0
        self.regime = 0
        self.signal_strength = 0.0
        self.contracts_to_add = 0

        # Technical indicators
        self.ema_12 = 0.0
        self.ema_26 = 0.0
        self.rsi = 0.0
        # ... (all output fields from Tier-1)

class CopperSignalParser(pcts3.sv_object):
    """Parse CopperIndicator signals"""
    def __init__(self):
        super().__init__()
        self.meta_name = "CopperIndicator"
        self.namespace = pc.namespace_private
        self.revision = (1 << 32) - 1

        # Same fields as IronOreSignalParser
        # ... (copy structure)

class SoybeanSignalParser(pcts3.sv_object):
    """Parse SoybeanIndicator signals"""
    def __init__(self):
        super().__init__()
        self.meta_name = "SoybeanIndicator"
        self.namespace = pc.namespace_private
        self.revision = (1 << 32) - 1

        # Same fields as IronOreSignalParser
        # ... (copy structure)
```

### 6.3 Composite Strategy Class

```python
class CompositeStrategy(csc3.composite_strategy):
    """
    Tier-2 Composite Strategy

    Manages 3 baskets (one per instrument) with equal-weight allocation
    """

    def __init__(self, initial_cash=INITIAL_CAPITAL):
        # Initialize critical attributes FIRST
        self.bar_index_this_run = -1
        self.latest_sv = None

        # Basket mapping
        self.basket_to_instrument = BASKET_MAPPING
        self.instrument_to_basket = {v: k for k, v in BASKET_MAPPING.items()}

        # Signal parsers
        self.parsers = {
            0: IronOreSignalParser(),
            1: CopperSignalParser(),
            2: SoybeanSignalParser(),
        }

        # Initialize parsers with metadata
        for basket_idx, parser in self.parsers.items():
            market, code = self.basket_to_instrument[basket_idx]
            parser.market = market
            parser.code = code + b'<00>'
            parser.granularity = granularity

        # Current signals from Tier-1
        self.tier1_signals = {}

        # Initialize composite strategy base class
        super().__init__(initial_cash, BASKET_COUNT)

        # Metadata
        self.namespace = pc.namespace_private
        self.granularity = granularity
        self.market = b'COMPOSITE'
        self.code = b'PORTFOLIO'
        self.meta_name = "CompositeStrategy"
        self.revision = (1 << 32) - 1
        self.timetag_ = None

        # Output fields
        self.bar_index = 0
        self.active_positions = 0
        self.total_signals_processed = 0
        self.portfolio_exposure_pct = 0.0
        self.cash_reserve_pct = 0.0

        # Allocation config (equal-weight)
        self.base_allocation_pct = 0.25
        self.max_allocation_pct = 0.35
        self.min_cash_reserve_pct = 0.15

        # Risk manager
        self.risk_manager = RiskManager(initial_cash)

        # Rebalancer
        self.rebalancer = Rebalancer()

        # Initialize baskets
        self._initialize_baskets(initial_cash)

        logger.info(f"Initialized CompositeStrategy with {BASKET_COUNT} baskets, "
                   f"initial capital ¥{initial_cash:,.0f}")

    def _initialize_baskets(self, initial_cash):
        """Allocate baskets to instruments"""
        base_capital = initial_cash * self.base_allocation_pct

        for basket_idx in range(BASKET_COUNT):
            market, code = self.basket_to_instrument[basket_idx]
            instrument_code = code + b'<00>'

            # Allocate basket
            self._allocate(basket_idx, market, instrument_code, base_capital, 1.0)

            logger.info(f"Basket {basket_idx}: {market.decode()}/{code.decode()} "
                       f"allocated ¥{base_capital:,.0f}")

    def on_bar(self, bar: pc.StructValue):
        """Main bar processing callback"""
        market = bar.get_market()
        code = bar.get_stock_code()
        tm = bar.get_time_tag()
        ns = bar.get_namespace()
        meta_id = bar.get_meta_id()

        # Initialize timetag
        if self.timetag is None:
            self.timetag = tm

        # Handle cycle boundaries
        if self.timetag < tm:
            self._on_cycle_pass(tm)

            results = []
            if self.bar_index > 0:
                results.append(self.sv_copy())

            self.timetag = tm
            self.bar_index += 1

            return results

        # Process Tier-1 signals
        self._process_tier1_signal(bar)

        return []

    def _process_tier1_signal(self, bar: pc.StructValue):
        """Parse incoming Tier-1 indicator signals"""
        ns = bar.get_namespace()
        meta_id = bar.get_meta_id()

        # Try each parser
        for basket_idx, parser in self.parsers.items():
            if ns == parser.namespace and meta_id == parser.meta_id:
                # Parse signal
                parser.from_sv(bar)

                # Store signal data
                self.tier1_signals[basket_idx] = {
                    'signal': parser.signal,
                    'confidence': parser.confidence,
                    'regime': parser.regime,
                    'signal_strength': parser.signal_strength,
                    'contracts_to_add': parser.contracts_to_add,
                    'ema_12': parser.ema_12,
                    'ema_26': parser.ema_26,
                    'rsi': parser.rsi,
                }

                self.total_signals_processed += 1
                break

    def _on_cycle_pass(self, time_tag):
        """Process end of cycle"""
        # Base class cycle processing
        super()._on_cycle_pass(time_tag)

        # Update risk tracking
        self.risk_manager.update_peak_tracking(self.pv)

        # Check circuit breakers
        if self._check_circuit_breakers():
            self._emergency_close_all_positions()
            return

        # Process trading signals
        self._process_trading_signals()

        # Check rebalancing
        self._check_and_rebalance()

        # Update portfolio metrics
        self._update_portfolio_metrics()

        # Sync state
        self._save()
        self._sync()

        # Logging
        self._log_portfolio_state()

    def _check_circuit_breakers(self):
        """Check portfolio-level circuit breakers"""
        metrics = self.risk_manager.get_risk_metrics()

        # Max drawdown
        if metrics['current_drawdown'] >= self.risk_manager.max_portfolio_drawdown:
            logger.error(f"CIRCUIT BREAKER: Max drawdown {metrics['current_drawdown']*100:.2f}% exceeded")
            return True

        # Daily loss limit
        if metrics['daily_loss'] >= self.risk_manager.max_daily_loss:
            logger.error(f"CIRCUIT BREAKER: Daily loss {metrics['daily_loss']*100:.2f}% exceeded")
            return True

        return False

    def _emergency_close_all_positions(self):
        """Emergency close all positions"""
        logger.error("EMERGENCY: Closing all positions")

        for basket_idx in range(BASKET_COUNT):
            basket = self.strategies[basket_idx]
            if basket.signal != 0:
                market, code = self.basket_to_instrument[basket_idx]
                logger.error(f"CLOSING: {market.decode()}/{code.decode()}")
                basket._signal(basket.price, basket.timetag, 0)

    def _process_trading_signals(self):
        """Execute trading signals for each basket"""
        for basket_idx in range(BASKET_COUNT):
            # Get Tier-1 signal
            if basket_idx not in self.tier1_signals:
                continue

            signal_data = self.tier1_signals[basket_idx]
            basket = self.strategies[basket_idx]

            # Check exit conditions first
            if basket.signal != 0:
                if self._should_exit(basket_idx, signal_data, basket):
                    self._execute_exit(basket_idx)
                    continue

            # Check entry conditions
            if basket.signal == 0:
                if self._should_enter(signal_data):
                    self._execute_entry(basket_idx, signal_data)

    def _should_enter(self, signal_data):
        """Entry condition checks"""
        # Strong signal
        if signal_data['signal'] == 0:
            return False

        # High confidence
        if signal_data['confidence'] < 0.60:
            return False

        # Avoid chaos
        if signal_data['regime'] == 4:
            return False

        # Signal strength
        if signal_data['signal_strength'] < 0.50:
            return False

        return True

    def _should_exit(self, basket_idx, signal_data, basket):
        """Exit condition checks"""
        # Stop-loss check
        entry_price = basket.entry_price  # Custom tracking needed
        current_price = basket.price
        if self._check_stop_loss(entry_price, current_price, basket.signal):
            logger.warning(f"STOP-LOSS: Basket {basket_idx}")
            return True

        # Signal reversal
        if signal_data['signal'] * basket.signal < 0:
            logger.info(f"EXIT: Signal reversed for basket {basket_idx}")
            return True

        # Confidence dropped
        if signal_data['confidence'] < 0.30:
            logger.info(f"EXIT: Confidence dropped for basket {basket_idx}")
            return True

        # Chaos regime
        if signal_data['regime'] == 4:
            logger.info(f"EXIT: Chaos regime for basket {basket_idx}")
            return True

        return False

    def _check_stop_loss(self, entry_price, current_price, position_type):
        """Check if stop-loss hit (3%)"""
        if entry_price == 0:
            return False

        if position_type == 1:  # Long
            loss_pct = (entry_price - current_price) / entry_price
        else:  # Short
            loss_pct = (current_price - entry_price) / entry_price

        return loss_pct >= 0.03

    def _execute_entry(self, basket_idx, signal_data):
        """Execute entry for basket"""
        basket = self.strategies[basket_idx]
        market, code = self.basket_to_instrument[basket_idx]

        # Calculate position size
        allocated_capital = self.pv * self.base_allocation_pct
        contracts = self._calculate_position_size(allocated_capital, basket.price)

        # Risk check
        proposed_size = contracts * basket.price
        can_enter, reason = self.risk_manager.can_enter_position(
            basket_idx, proposed_size, self._get_portfolio_state()
        )

        if not can_enter:
            logger.warning(f"ENTRY BLOCKED basket {basket_idx}: {reason}")
            return

        # Execute
        signal = signal_data['signal']
        logger.info(f"{'LONG' if signal == 1 else 'SHORT'} basket {basket_idx}: "
                   f"{market.decode()}/{code.decode()}, contracts={contracts}")

        basket._signal(basket.price, basket.timetag, signal)

    def _execute_exit(self, basket_idx):
        """Execute exit for basket"""
        basket = self.strategies[basket_idx]
        market, code = self.basket_to_instrument[basket_idx]

        logger.info(f"CLOSE basket {basket_idx}: {market.decode()}/{code.decode()}")
        basket._signal(basket.price, basket.timetag, 0)

    def _calculate_position_size(self, allocated_capital, current_price):
        """Calculate number of contracts"""
        if current_price == 0:
            return 0

        contracts = int(allocated_capital / current_price)
        return max(contracts, 0)

    def _check_and_rebalance(self):
        """Check if rebalancing needed and execute"""
        # Get current allocations
        current_allocations = self._get_current_allocations()
        target_allocations = {
            0: 0.25,
            1: 0.25,
            2: 0.25,
        }

        # Check if rebalancing needed
        should_rebalance, instrument, deviation = self.rebalancer.should_rebalance(
            self.bar_index, current_allocations, target_allocations
        )

        if should_rebalance:
            logger.info(f"REBALANCE triggered: {instrument} deviated by {deviation*100:.1f}%")
            self._execute_rebalance(current_allocations, target_allocations)
            self.rebalancer.last_rebalance_bar = self.bar_index

    def _get_current_allocations(self):
        """Calculate current allocation percentages"""
        allocations = {}
        for basket_idx in range(BASKET_COUNT):
            basket = self.strategies[basket_idx]
            allocations[basket_idx] = basket.pv / self.pv if self.pv > 0 else 0
        return allocations

    def _execute_rebalance(self, current_allocations, target_allocations):
        """Execute rebalancing trades"""
        # Calculate actions
        actions = self.rebalancer.calculate_rebalance_actions(
            current_allocations, target_allocations, self.pv
        )

        # Execute (simplified for Phase 1 - just log)
        for basket_idx, action in actions.items():
            market, code = self.basket_to_instrument[basket_idx]
            logger.info(f"REBALANCE: {action['type']} {market.decode()}/{code.decode()} "
                       f"by ¥{action['value']:,.0f} ({action['deviation_pct']:.1f}%)")

    def _update_portfolio_metrics(self):
        """Update portfolio-level metrics"""
        self.active_positions = sum(1 for s in self.strategies if s.signal != 0)

        total_exposure = sum(s.pv for s in self.strategies if s.signal != 0)
        self.portfolio_exposure_pct = total_exposure / self.pv if self.pv > 0 else 0

        # Cash calculation (simplified)
        self.cash_reserve_pct = 1.0 - self.portfolio_exposure_pct

    def _get_portfolio_state(self):
        """Get current portfolio state for risk checks"""
        class PortfolioState:
            def __init__(self):
                self.total_value = 0
                self.total_exposure = 0
                self.cash = 0
                self.current_prices = {}

        state = PortfolioState()
        state.total_value = self.pv
        state.total_exposure = sum(s.pv for s in self.strategies if s.signal != 0)
        state.cash = self.pv - state.total_exposure

        return state

    def _log_portfolio_state(self):
        """Log portfolio state"""
        metrics = self.risk_manager.get_risk_metrics()

        logger.info(
            f"[Bar {self.bar_index}] "
            f"PV=¥{self.pv:,.0f}, NV={self.nv:.4f}, "
            f"Active={self.active_positions}/3, "
            f"Exposure={self.portfolio_exposure_pct*100:.1f}%, "
            f"Cash={self.cash_reserve_pct*100:.1f}%, "
            f"DD={metrics['current_drawdown']*100:.2f}%"
        )

    def on_tradeday_begin(self, market, tradeday):
        """Called at start of trading day"""
        super().on_tradeday_begin(market, tradeday)
        self.risk_manager.update_daily_tracking(self.pv)
        logger.info(f"=== Trading Day Begin: {tradeday} ===")

    def on_tradeday_end(self, market, tradeday):
        """Called at end of trading day"""
        super().on_tradeday_end(market, tradeday)
        metrics = self.risk_manager.get_risk_metrics()
        logger.info(f"=== Trading Day End: {tradeday}, Daily Loss: {metrics['daily_loss']*100:.2f}% ===")

# Global strategy instance
strategy = CompositeStrategy()

# Framework callbacks (REQUIRED)
async def on_init():
    global strategy, imports, metas, worker_no
    if worker_no != 0 and metas and imports:
        strategy.load_def_from_dict(metas)
        for parser in strategy.parsers.values():
            parser.load_def_from_dict(metas)
            parser.set_global_imports(imports)
        logger.info("CompositeStrategy initialized")

async def on_ready():
    logger.info("CompositeStrategy ready")

async def on_bar(bar: pc.StructValue):
    global strategy, worker_no
    if worker_no != 1:
        return []
    return strategy.on_bar(bar)

async def on_market_open(market, tradeday, time_tag, time_string):
    pass

async def on_market_close(market, tradeday, timetag, timestring):
    pass

async def on_tradeday_begin(market, tradeday, time_tag, time_string):
    global strategy
    strategy.on_tradeday_begin(bytes(market, 'utf-8'), tradeday)

async def on_tradeday_end(market, tradeday, timetag, timestring):
    global strategy
    strategy.on_tradeday_end(bytes(market, 'utf-8'), tradeday)

async def on_reference(market, tradeday, data, timetag, timestring):
    pass

async def on_historical(params, records):
    pass
```

---

## 7. Configuration Files

### 7.1 CompositeStrategy/uin.json

```json
{
  "private": {
    "imports": {
      "IronOreIndicatorRelaxed": {
        "fields": [
          "_preserved_field",
          "bar_index",
          "signal",
          "confidence",
          "regime",
          "signal_strength",
          "contracts_to_add",
          "ema_12",
          "ema_26",
          "ema_50",
          "macd",
          "rsi",
          "bb_upper",
          "bb_lower",
          "atr"
        ],
        "granularities": [900],
        "markets": ["DCE"],
        "security_categories": [[1, 2, 3]],
        "securities": [["i"]]
      },
      "CopperIndicator": {
        "fields": [
          "_preserved_field",
          "bar_index",
          "signal",
          "confidence",
          "regime",
          "signal_strength",
          "contracts_to_add",
          "ema_12",
          "ema_26",
          "ema_50",
          "macd",
          "rsi",
          "bb_upper",
          "bb_lower",
          "atr"
        ],
        "granularities": [900],
        "markets": ["SHFE"],
        "security_categories": [[1, 2, 3]],
        "securities": [["cu"]]
      },
      "SoybeanIndicator": {
        "fields": [
          "_preserved_field",
          "bar_index",
          "signal",
          "confidence",
          "regime",
          "signal_strength",
          "contracts_to_add",
          "ema_12",
          "ema_26",
          "ema_50",
          "macd",
          "rsi",
          "bb_upper",
          "bb_lower",
          "atr"
        ],
        "granularities": [900],
        "markets": ["DCE"],
        "security_categories": [[1, 2, 3]],
        "securities": [["m"]]
      }
    }
  }
}
```

### 7.2 CompositeStrategy/uout.json

```json
{
  "private": {
    "markets": ["COMPOSITE"],
    "security_categories": [[1]],
    "securities": [["PORTFOLIO"]],
    "sample_granularities": {
      "type": "min",
      "cycles": [900],
      "cycle_lengths": [0]
    },
    "export": {
      "CompositeStrategy": {
        "fields": [
          "_preserved_field",
          "bar_index",
          "active_positions",
          "total_signals_processed",
          "portfolio_exposure_pct",
          "cash_reserve_pct",
          "nv",
          "pv"
        ],
        "defs": [
          {"field_type": "int64", "precision": 0},
          {"field_type": "integer", "precision": 0},
          {"field_type": "integer", "precision": 0},
          {"field_type": "integer", "precision": 0},
          {"field_type": "double", "precision": 6},
          {"field_type": "double", "precision": 6},
          {"field_type": "double", "precision": 6},
          {"field_type": "double", "precision": 6}
        ],
        "revision": -1
      }
    }
  }
}
```

---

## 8. Testing Requirements

### 8.1 Tier-1 Indicator Testing (3 Indicators)

**Test Each Indicator Separately**:

```bash
# 1. Test CopperIndicator (7 days)
cd CopperIndicator
python /home/wolverine/bin/running/calculator3_test.py \
    --testcase ${PWD}/ \
    --algoname CopperIndicator \
    --sourcefile CopperIndicator.py \
    --start 20241025000000 \
    --end 20241101000000 \
    --granularity 900 \
    --tm wss://${SVR_HOST}:4433/tm \
    --tm-master ${SVR_HOST}:6102 \
    --rails https://${SVR_HOST}:4433/private-api/ \
    --token ${SVR_TOKEN} \
    --category 1 \
    --is-managed 1 \
    --restore-length 864000000 \
    --multiproc 1

# 2. Test SoybeanIndicator (7 days)
cd ../SoybeanIndicator
python /home/wolverine/bin/running/calculator3_test.py \
    --testcase ${PWD}/ \
    --algoname SoybeanIndicator \
    --sourcefile SoybeanIndicator.py \
    --start 20241025000000 \
    --end 20241101000000 \
    --granularity 900 \
    --tm wss://${SVR_HOST}:4433/tm \
    --tm-master ${SVR_HOST}:6102 \
    --rails https://${SVR_HOST}:4433/private-api/ \
    --token ${SVR_TOKEN} \
    --category 1 \
    --is-managed 1 \
    --restore-length 864000000 \
    --multiproc 1

# 3. Replay consistency test for each
python test_resuming_mode.py  # In each indicator directory
```

### 8.2 Tier-2 Composite Strategy Testing

**Prerequisites**: All 3 Tier-1 indicators must be running and outputting signals

**Test Sequence**:

```bash
# 1. Quick test (7 days) - Validate integration
cd CompositeStrategy
python /home/wolverine/bin/running/calculator3_test.py \
    --testcase ${PWD}/ \
    --algoname CompositeStrategy \
    --sourcefile CompositeStrategy.py \
    --start 20241025000000 \
    --end 20241101000000 \
    --granularity 900 \
    --tm wss://${SVR_HOST}:4433/tm \
    --tm-master ${SVR_HOST}:6102 \
    --rails https://${SVR_HOST}:4433/private-api/ \
    --token ${SVR_TOKEN} \
    --category 1 \
    --is-managed 1 \
    --restore-length 864000000 \
    --multiproc 1

# 2. Medium test (1 month) - Validate risk management
python /home/wolverine/bin/running/calculator3_test.py \
    --testcase ${PWD}/ \
    --algoname CompositeStrategy \
    --sourcefile CompositeStrategy.py \
    --start 20241001000000 \
    --end 20241031000000 \
    --granularity 900 \
    --tm wss://${SVR_HOST}:4433/tm \
    --tm-master ${SVR_HOST}:6102 \
    --rails https://${SVR_HOST}:4433/private-api/ \
    --token ${SVR_TOKEN} \
    --category 1 \
    --is-managed 1 \
    --restore-length 864000000 \
    --multiproc 1

# 3. Full backtest (3 months) - Final validation
python /home/wolverine/bin/running/calculator3_test.py \
    --testcase ${PWD}/ \
    --algoname CompositeStrategy \
    --sourcefile CompositeStrategy.py \
    --start 20241001000000 \
    --end 20241231000000 \
    --granularity 900 \
    --tm wss://${SVR_HOST}:4433/tm \
    --tm-master ${SVR_HOST}:6102 \
    --rails https://${SVR_HOST}:4433/private-api/ \
    --token ${SVR_TOKEN} \
    --category 1 \
    --is-managed 1 \
    --restore-length 864000000 \
    --multiproc 1
```

### 8.3 Expected Results

**Tier-1 Indicators** (Each):
- Signals generated: 100-500 per instrument
- Memory: Bounded (O(1))
- Replay consistency: Pass

**Tier-2 Composite**:
- Total trades: 300-1500 (all instruments combined)
- Max drawdown: <10% (target)
- Portfolio value: Positive P&L (target)
- Risk violations: 0 (no circuit breakers triggered inappropriately)

---

## 9. Visualization and Analysis

### 9.1 Jupyter Notebook (CompositeStrategy/analysis.ipynb)

**Cell 1: Setup and Data Fetching**

```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import asyncio
import svr3
from datetime import datetime

%matplotlib inline
sns.set_style("darkgrid")

from dotenv import load_dotenv
load_dotenv()

SVR_HOST = os.getenv("SVR_HOST")
SVR_TOKEN = os.getenv("SVR_TOKEN")
RAILS_URL = f"https://{SVR_HOST}:4433/private-api/"
WS_URL = f"wss://{SVR_HOST}:4433/tm"
TM_MASTER = (SVR_HOST, 6102)

# Fetch composite strategy data
async def fetch_composite_data(start, end):
    reader = svr3.sv_reader(
        start, end,
        "CompositeStrategy",
        900,
        "private",
        "symbol",
        ["COMPOSITE"],
        ["PORTFOLIO"],
        False,
        RAILS_URL, WS_URL, "", "", TM_MASTER
    )
    reader.token = SVR_TOKEN
    await reader.login()
    await reader.connect()
    reader.ws_task = asyncio.create_task(reader.ws_loop())
    await reader.shakehand()
    ret = await reader.save_by_symbol()
    return pd.DataFrame(ret[1][1])

df_composite = await fetch_composite_data(20241001000000, 20241231000000)
print(f"Loaded {len(df_composite)} bars")
df_composite.head()
```

**Cell 2: Portfolio Performance Visualization**

```python
fig, axes = plt.subplots(4, 1, figsize=(16, 14))

# Panel 1: Portfolio Value (PV)
axes[0].plot(df_composite.index, df_composite['pv'], linewidth=2, color='blue')
axes[0].axhline(y=1000000, color='green', linestyle='--', alpha=0.5, label='Initial Capital')
axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Portfolio Value (¥)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Panel 2: Net Value (NV) - Performance Ratio
axes[1].plot(df_composite.index, df_composite['nv'], linewidth=2, color='purple')
axes[1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
axes[1].set_title('Net Value (Performance Ratio)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('NV (PV / Initial Capital)')
axes[1].grid(True, alpha=0.3)

# Panel 3: Active Positions
axes[2].plot(df_composite.index, df_composite['active_positions'], linewidth=2, color='orange')
axes[2].set_title('Number of Active Positions', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Active Positions (0-3)')
axes[2].set_ylim(-0.5, 3.5)
axes[2].grid(True, alpha=0.3)

# Panel 4: Portfolio Exposure
axes[3].plot(df_composite.index, df_composite['portfolio_exposure_pct'] * 100,
             linewidth=2, color='red', label='Exposure')
axes[3].plot(df_composite.index, df_composite['cash_reserve_pct'] * 100,
             linewidth=2, color='green', label='Cash Reserve')
axes[3].axhline(y=90, color='red', linestyle='--', alpha=0.5, label='Max Exposure (90%)')
axes[3].axhline(y=15, color='green', linestyle='--', alpha=0.5, label='Min Cash (15%)')
axes[3].set_title('Portfolio Allocation', fontsize=14, fontweight='bold')
axes[3].set_ylabel('Percentage (%)')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('composite_performance.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Cell 3: Performance Metrics**

```python
# Calculate metrics
initial_capital = 1000000
final_pv = df_composite['pv'].iloc[-1]
total_return = (final_pv - initial_capital) / initial_capital
total_return_pct = total_return * 100

# Drawdown calculation
cumulative_max = df_composite['pv'].cummax()
drawdown = (df_composite['pv'] - cumulative_max) / cumulative_max
max_drawdown = drawdown.min()
max_drawdown_pct = max_drawdown * 100

# Daily returns for Sharpe
df_composite['daily_return'] = df_composite['pv'].pct_change()
sharpe_ratio = (df_composite['daily_return'].mean() / df_composite['daily_return'].std()) * np.sqrt(252)

# Average exposure
avg_exposure = df_composite['portfolio_exposure_pct'].mean()
avg_cash = df_composite['cash_reserve_pct'].mean()
avg_active_positions = df_composite['active_positions'].mean()

metrics = {
    'Initial Capital': f"¥{initial_capital:,.0f}",
    'Final Portfolio Value': f"¥{final_pv:,.0f}",
    'Total Return': f"{total_return_pct:.2f}%",
    'Max Drawdown': f"{max_drawdown_pct:.2f}%",
    'Sharpe Ratio': f"{sharpe_ratio:.2f}",
    'Avg Exposure': f"{avg_exposure*100:.1f}%",
    'Avg Cash Reserve': f"{avg_cash*100:.1f}%",
    'Avg Active Positions': f"{avg_active_positions:.1f}",
    'Total Signals Processed': f"{df_composite['total_signals_processed'].iloc[-1]:,}",
}

pd.DataFrame([metrics]).T.rename(columns={0: 'Value'})
```

**Cell 4: Drawdown Analysis**

```python
fig, ax = plt.subplots(figsize=(14, 6))

drawdown_pct = drawdown * 100
ax.fill_between(df_composite.index, 0, drawdown_pct, color='red', alpha=0.3)
ax.plot(df_composite.index, drawdown_pct, color='darkred', linewidth=2)
ax.axhline(y=-10, color='red', linestyle='--', linewidth=2, label='Max Drawdown Limit (-10%)')
ax.set_title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
ax.set_ylabel('Drawdown (%)')
ax.set_xlabel('Time')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('drawdown_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Cell 5: Monthly Performance**

```python
# Monthly breakdown
df_composite['timestamp'] = pd.to_datetime(df_composite['timestamp'])
df_composite['month'] = df_composite['timestamp'].dt.to_period('M')

monthly_pv = df_composite.groupby('month')['pv'].last()
monthly_returns = monthly_pv.pct_change() * 100

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['green' if x > 0 else 'red' for x in monthly_returns]
monthly_returns.plot(kind='bar', ax=ax, color=colors)
ax.set_title('Monthly Returns (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('Return (%)')
ax.set_xlabel('Month')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('monthly_returns.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nMonthly Returns:")
print(monthly_returns)
```

### 9.2 Visualization Outputs

**Expected Charts**:
1. Portfolio value curve (PV over time)
2. Net value ratio (NV = PV / Initial Capital)
3. Active positions over time
4. Portfolio allocation (exposure vs cash)
5. Drawdown analysis
6. Monthly returns breakdown

**Files Generated**:
- `composite_performance.png` - Main 4-panel chart
- `drawdown_analysis.png` - Drawdown visualization
- `monthly_returns.png` - Monthly performance bars

---

## 10. Acceptance Criteria

### 10.1 Functional Requirements

**Tier-1 Indicators**:
- [ ] CopperIndicator generates signals for SHFE/cu<00>
- [ ] SoybeanIndicator generates signals for DCE/m<00>
- [ ] Both indicators pass replay consistency test
- [ ] Memory usage bounded (O(1) growth)

**Tier-2 Composite**:
- [ ] Receives signals from all 3 Tier-1 indicators
- [ ] Manages 3 independent baskets
- [ ] Implements equal-weight allocation (25% base per instrument)
- [ ] Executes entry/exit based on defined conditions
- [ ] Enforces risk limits (stops, drawdown, exposure)
- [ ] Rebalances portfolio when drift exceeds 10%

### 10.2 Performance Requirements

**Target Metrics** (3-month backtest):

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| Total Return | >0% | Must be positive |
| Max Drawdown | <10% | Must not exceed 10% |
| Sharpe Ratio | >1.0 | Preferred >1.0 |
| Portfolio Exposure | 60-80% avg | Never exceed 90% |
| Cash Reserve | 20-30% avg | Never below 15% |
| Stop-Loss Violations | 0 | Zero unexpected stops |
| Circuit Breakers Triggered | 0-2 | Only in extreme scenarios |

**Risk Compliance**:
- [ ] No position exceeds 35% of portfolio
- [ ] No position stop-loss >3% triggered unexpectedly
- [ ] Portfolio drawdown never exceeds 10%
- [ ] Daily loss limit never exceeds 3%
- [ ] Cash reserve maintained above 15%

### 10.3 Code Quality Requirements

- [ ] All WOS framework patterns followed (sv_object, cycle boundaries, online algorithms)
- [ ] All 5 doctrines applied correctly
- [ ] No unbounded memory growth
- [ ] Deterministic execution (replay consistent)
- [ ] Configuration files valid (uin.json, uout.json)
- [ ] Comprehensive logging at key decision points
- [ ] Error handling for edge cases

### 10.4 Documentation Requirements

- [ ] This REQUIREMENTS.md document complete and accurate
- [ ] STRATEGY_GUIDE.md explaining trading logic
- [ ] CLAUDE.md with development notes and decisions
- [ ] README.md for each component
- [ ] Inline code comments for complex logic

---

## 11. Development Workflow

### 11.1 Phase 1: Build Tier-1 Indicators (Week 1)

**Day 1-2: CopperIndicator**
```bash
# 1. Create project structure
mkdir CopperIndicator
cd CopperIndicator

# 2. Copy IronOreIndicator as template
cp ../IronOreIndicator/IronOreIndicator.py ./CopperIndicator.py
cp ../IronOreIndicator/uin.json .
cp ../IronOreIndicator/uout.json .
cp ../IronOreIndicator/test_resuming_mode.py .

# 3. Modify for Copper
# - Change market to b'SHFE'
# - Change code to b'cu<00>'
# - Update meta_name to "CopperIndicator"
# - Adjust EMA parameters (faster)
# - Update uin.json (SHFE, cu)
# - Update uout.json (CopperIndicator)

# 4. Test
python /home/wolverine/bin/running/calculator3_test.py \
    --testcase ${PWD}/ --algoname CopperIndicator \
    --sourcefile CopperIndicator.py \
    --start 20241025000000 --end 20241101000000 \
    --granularity 900 \
    --tm wss://${SVR_HOST}:4433/tm \
    --tm-master ${SVR_HOST}:6102 \
    --rails https://${SVR_HOST}:4433/private-api/ \
    --token ${SVR_TOKEN} \
    --category 1 --is-managed 1 \
    --restore-length 864000000 --multiproc 1

# 5. Replay consistency test
python test_resuming_mode.py
```

**Day 3-4: SoybeanIndicator**
```bash
# Same process as CopperIndicator
# - market = b'DCE'
# - code = b'm<00>'
# - meta_name = "SoybeanIndicator"
# - Adjust parameters (slower EMAs)
```

**Day 5: Validate All Tier-1**
```bash
# Run 1-month backtest on all 3 indicators
# Verify signals being generated
# Check memory usage
```

### 11.2 Phase 2: Build Tier-2 Composite (Week 2)

**Day 1-2: Core Structure**
```bash
# 1. Create project structure
mkdir CompositeStrategy
cd CompositeStrategy

# 2. Create files
touch CompositeStrategy.py
touch uin.json
touch uout.json

# 3. Implement core classes
# - RiskManager
# - Rebalancer
# - Signal parsers (3)
# - CompositeStrategy main class

# 4. Implement uin.json (import 3 Tier-1 signals)
# 5. Implement uout.json (export portfolio state)
```

**Day 3-4: Trading Logic**
```bash
# 1. Implement signal processing
# - _process_tier1_signal()
# - _should_enter()
# - _should_exit()

# 2. Implement risk management
# - _check_circuit_breakers()
# - Position sizing
# - Stop-loss checks

# 3. Implement rebalancing
# - _check_and_rebalance()
# - _execute_rebalance()
```

**Day 5: Integration Testing**
```bash
# 1. Quick test (7 days)
python /home/wolverine/bin/running/calculator3_test.py \
    --testcase ${PWD}/ --algoname CompositeStrategy \
    --sourcefile CompositeStrategy.py \
    --start 20241025000000 --end 20241101000000 \
    --granularity 900 \
    --tm wss://${SVR_HOST}:4433/tm \
    --tm-master ${SVR_HOST}:6102 \
    --rails https://${SVR_HOST}:4433/private-api/ \
    --token ${SVR_TOKEN} \
    --category 1 --is-managed 1 \
    --restore-length 864000000 --multiproc 1

# 2. Verify:
# - All 3 signals being received
# - Baskets being managed correctly
# - Risk limits enforced
# - Logs showing expected behavior
```

### 11.3 Phase 3: Testing & Validation (Week 3)

**Day 1-2: Medium Testing (1 month)**
```bash
# Run 1-month backtest
# Oct 1 - Oct 31, 2024
# Validate risk management working
# Check rebalancing triggers
```

**Day 3-4: Full Backtest (3 months)**
```bash
# Run 3-month backtest
# Oct 1 - Dec 31, 2024
# Final validation
# Performance analysis
```

**Day 5: Visualization & Documentation**
```bash
# Create analysis.ipynb
# Generate performance charts
# Write STRATEGY_GUIDE.md
# Update CLAUDE.md with learnings
```

### 11.4 Phase 4: Optimization (Week 4)

**Day 1-3: Parameter Tuning**
- Adjust entry thresholds based on results
- Fine-tune stop-loss levels
- Optimize rebalancing frequency
- Test different confidence thresholds

**Day 4-5: Final Validation**
- Re-run full backtest with optimized parameters
- Verify improvements
- Document final configuration
- Prepare for production

**Total Timeline**: 4 weeks

---

## 12. Risk Warnings and Limitations

### 12.1 Strategy Limitations

1. **Phase 1 Simplifications**:
   - Equal-weight allocation (no dynamic adjustment)
   - Independent management (no correlation awareness)
   - Simple entry/exit (no advanced order types)
   - Fixed parameters (no adaptive adjustment)

2. **Market Assumptions**:
   - Assumes liquid markets (may not hold during crisis)
   - No slippage modeling
   - No commission costs included
   - Perfect fill assumptions

3. **Data Dependencies**:
   - Requires all 3 Tier-1 indicators running correctly
   - Single indicator failure affects portfolio
   - No fallback logic if signals missing

### 12.2 Known Risks

1. **Correlation Risk**: Iron Ore and Copper are moderately correlated - both may drop simultaneously
2. **Regime Risk**: All instruments may enter chaos regime together
3. **Liquidity Risk**: Soybean may have lower liquidity than metals
4. **Stop-Loss Gaps**: Overnight gaps may exceed 3% stop-loss
5. **Rebalancing Costs**: Frequent rebalancing may incur transaction costs (not modeled)

### 12.3 Production Considerations

**Before going live**:
- [ ] Backtest with realistic slippage (0.05-0.1%)
- [ ] Add commission costs (¥5-10 per contract)
- [ ] Test with real order execution (not simulated)
- [ ] Validate with paper trading (1 month minimum)
- [ ] Review with risk management team
- [ ] Set up real-time monitoring and alerts

---

## 13. Future Enhancements (Phase 2+)

### 13.1 Dynamic Allocation (Phase 2)

```python
# Confidence-weighted allocation
def calculate_dynamic_allocation(signal_data, base_allocation):
    """
    Adjust allocation based on signal quality
    """
    confidence_mult = 0.5 + (signal_data['confidence'] * 1.0)
    regime_mult = 1.1 if signal_data['regime'] in [1, 2] else 0.9

    return base_allocation * confidence_mult * regime_mult
```

### 13.2 Correlation Awareness (Phase 2)

```python
# Reduce allocation when instruments correlated
def adjust_for_correlation(allocations, correlation_matrix, positions):
    """
    Reduce exposure when positions correlated and moving same direction
    """
    # If Iron Ore LONG and Copper LONG, reduce both to 20%
    # If Iron Ore LONG and Soybean LONG, keep at 25% (low correlation)
```

### 13.3 Profit-Taking (Phase 2)

```python
# Scaled profit taking
# 2% -> Close 50%
# 5% -> Close another 30%
# 8% -> Trailing stop on rest
```

### 13.4 Advanced Risk Management (Phase 3)

```python
# Value at Risk (VaR) calculation
# Conditional VaR (CVaR)
# Correlation-adjusted position limits
# Dynamic stop-loss based on ATR
```

### 13.5 Additional Instruments (Phase 3)

```python
# Expand to 5-6 instruments
# Add energy (crude oil, fuel oil)
# Add more agriculture (corn, wheat)
# Maintain <0.5 avg pairwise correlation
```

---

## 14. References

### 14.1 WOS Framework Documentation

**Required Reading**:
- `wos/01-overview.md` - Framework architecture
- `wos/02-uin-and-uout.md` - Configuration files
- `wos/03-programming-basics-and-cli.md` - Module structure
- `wos/04-structvalue-and-sv_object.md` - Data serialization
- `wos/05-stateless-design.md` - Online algorithms
- `wos/06-backtest.md` - Testing procedures
- `wos/07-tier1-indicator.md` - Tier-1 patterns
- `wos/08-tier2-composite.md` - Tier-2 patterns ⭐
- `wos/10-visualization.md` - Analysis and plotting

### 14.2 Existing Implementations

**Reference Code**:
- `IronOreIndicator/IronOreIndicator.py` - Proven Tier-1 implementation
- `IronOreIndicator/REQUIREMENTS.md` - Tier-1 specification example
- `templates/indicator.py.template` - Indicator template

### 14.3 External Resources

- **Composite Strategy Pattern**: WOS Chapter 8
- **Risk Management**: Industry best practices (3% stops, 10% max DD)
- **Portfolio Theory**: Modern Portfolio Theory (Markowitz)
- **Position Sizing**: Kelly Criterion (conservative 50% Kelly)

---

## 15. Glossary

| Term | Definition |
|------|------------|
| **Tier-1 Indicator** | Basic technical analysis indicator that generates signals for single instrument |
| **Tier-2 Composite** | Portfolio manager that aggregates Tier-1 signals across multiple instruments |
| **Basket** | Individual trading strategy instance within composite (one per instrument) |
| **Signal** | Trading recommendation: -1 (sell/short), 0 (neutral/hold), 1 (buy/long) |
| **Confidence** | Signal strength indicator (0.0-1.0), higher = stronger conviction |
| **Regime** | Market state classification (1=Uptrend, 2=Downtrend, 3=Ranging, 4=Chaos) |
| **Allocation** | Percentage of portfolio capital assigned to each instrument |
| **Exposure** | Total invested capital as percentage of portfolio (vs cash reserve) |
| **Drawdown** | Peak-to-trough decline in portfolio value (from historical peak) |
| **NV (Net Value)** | Performance ratio = Current PV / Initial Capital |
| **PV (Portfolio Value)** | Total portfolio worth (invested + cash) |
| **Rebalancing** | Adjusting positions to restore target allocations |
| **Stop-Loss** | Automatic exit when position loses specified percentage (3% here) |
| **Circuit Breaker** | Emergency shutdown when portfolio risk limits exceeded |
| **Logical Contract** | Continuous contract representation (e.g., i<00>, cu<00>) |
| **sv_object** | WOS framework base class for automatic state serialization |
| **Replay Consistency** | Property that indicator produces identical results when resumed from midpoint |

---

## 16. Summary

This document specifies a complete **Tier-2 Composite Trading Strategy** for managing a ¥1,000,000 portfolio across 3 diversified futures instruments.

**Key Design Decisions**:
1. ✅ **3 Instruments**: Iron Ore, Copper, Soybean Meal (diversified across sectors)
2. ✅ **IronOre-Style Indicators**: Proven 7-indicator system for all 3 instruments
3. ✅ **Equal-Weight Allocation**: Simple 25% base per instrument (Phase 1)
4. ✅ **Conservative Risk**: 10% max DD, 3% stops, 15% cash reserve
5. ✅ **Independent Management**: Each instrument managed separately (Phase 1)

**Deliverables**:
- CopperIndicator (Tier-1) - NEW
- SoybeanIndicator (Tier-1) - NEW
- CompositeStrategy (Tier-2) - NEW
- Complete configuration files
- Testing framework
- Visualization notebook

**Timeline**: 4 weeks
- Week 1: Build Tier-1 indicators
- Week 2: Build Tier-2 composite
- Week 3: Testing & validation
- Week 4: Optimization

**Success Criteria**:
- ✅ Positive P&L over 3-month backtest
- ✅ Max drawdown <10%
- ✅ All risk limits enforced
- ✅ Replay consistency passed

**Next Steps**:
1. Review and approve this requirements document
2. Set up development environment
3. Begin Week 1: Build CopperIndicator
4. Follow development workflow (Section 11)

---

**Document End**

This requirements document serves as the complete specification for building the Tier-2 Composite Trading Strategy. All implementation details, configurations, risk parameters, and testing procedures are defined above.

**For questions or clarifications, refer to**:
- WOS framework documentation: `wos/` directory
- Existing IronOreIndicator: `IronOreIndicator/` directory
- This document: Section-specific guidance

**Ready to begin implementation!** 🚀
