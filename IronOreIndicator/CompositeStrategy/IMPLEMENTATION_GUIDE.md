# Multi-Instrument Trading Strategy - Implementation Guide

**Architecture**: Tier-1 Signal Generators → Tier-2 Composite Portfolio Manager
**Objective**: Diversified trading across 3 futures instruments with risk management
**Markets**: DCE (Iron Ore, Soybean), SHFE (Copper)
**Framework**: WOS (Wolverine Operating System) for quantitative trading

---

## 1. System Architecture

### Overview
A two-tier system separating signal generation (Tier-1) from portfolio management (Tier-2):

```
┌─────────────────────────────────────────────────────────────┐
│                    TIER 1: Signal Generators                │
├─────────────────────────────────────────────────────────────┤
│  Iron Ore (DCE/i)  │  Copper (SHFE/cu)  │  Soybean (DCE/m) │
│  ┌──────────────┐  │  ┌──────────────┐  │  ┌──────────────┐│
│  │ 7 Indicators │  │  │ 7 Indicators │  │  │ 7 Indicators ││
│  │ RSI, MACD,   │  │  │ Faster EMAs  │  │  │ Standard     ││
│  │ EMA, BB, ATR │  │  │ (12/24/45)   │  │  │ (12/26/50)   ││
│  └──────┬───────┘  │  └──────┬───────┘  │  └──────┬───────┘│
│         │          │         │          │         │         │
│    signal=-1/0/1   │    signal=-1/0/1   │    signal=-1/0/1 │
│    confidence 0-1  │    confidence 0-1  │    confidence 0-1│
│    regime 1-4      │    regime 1-4      │    regime 1-4    │
└─────────┼──────────┴─────────┼──────────┴─────────┼─────────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
        ┌──────────────────────▼──────────────────────┐
        │        TIER 2: Composite Strategy           │
        ├─────────────────────────────────────────────┤
        │  • Receives signals from 3 indicators       │
        │  • Manages portfolio of 3 baskets           │
        │  • Executes LONG/SHORT positions            │
        │  • Enforces risk limits & stop-loss         │
        │  • Tracks PV, NV, drawdown, exposure        │
        └─────────────────────────────────────────────┘
```

### Key Principles
1. **Separation of Concerns**: Indicators only generate signals; strategy handles trading
2. **Pure Signals**: Indicators export direction + confidence, not trading commands
3. **Diversification**: 3 uncorrelated instruments reduce portfolio risk
4. **Risk Management**: Multiple layers (stop-loss, allocation limits, drawdown caps)
5. **Market Neutrality**: Both LONG and SHORT positions supported

---

## 2. Tier-1 Indicators (Signal Generators)

### Common Technical Framework (All 3 Indicators)

Each indicator uses a 7-component system to classify market regimes and generate signals:

**Technical Indicators:**
- **Triple EMA**: Trend identification (fast/medium/slow crossovers)
- **MACD**: Momentum confirmation & reversal detection
- **RSI**: Mean reversion & overbought/oversold conditions
- **Bollinger Bands**: Volatility & price extremes
- **ATR**: Volatility measurement
- **BB Width**: Volatility confirmation
- **Volume EMA**: Liquidity validation

**Market Regime Classification:**
1. **Regime 1 - Strong Uptrend**: EMA12 > EMA26 > EMA50, MACD bullish, normal volatility
2. **Regime 2 - Strong Downtrend**: EMA12 < EMA26 < EMA50, MACD bearish, normal volatility
3. **Regime 3 - Ranging/Sideways**: EMAs mixed, low volatility, no clear trend
4. **Regime 4 - High Volatility Chaos**: ATR > 1.5× mean, BB expansion > 5%

### 2.1 Iron Ore Indicator (DCE/i<00>)

**Market**: Dalian Commodity Exchange
**Security**: Iron Ore futures (logical contract i<00>)
**Granularity**: 900s (15 minutes)

**Parameters:**
```python
# EMA periods
alpha_12 = 2.0 / 13.0   # 12-period fast EMA
alpha_26 = 2.0 / 27.0   # 26-period medium EMA
alpha_50 = 2.0 / 51.0   # 50-period slow EMA

# Indicator periods
alpha_9 = 2.0 / 10.0    # MACD signal line
alpha_14 = 2.0 / 15.0   # RSI (14-period)
alpha_20 = 2.0 / 21.0   # Volume & BB (20-period)
```

**Signal Generation Logic:**

| Regime | LONG Signal (signal=1) | SHORT Signal (signal=-1) |
|--------|------------------------|--------------------------|
| **Uptrend** | RSI < 45 + MACD bullish | RSI > 55 + near BB upper |
| **Downtrend** | EMA12 > EMA26 + RSI < 45 (reversal) | RSI > 55 + MACD bearish |
| **Ranging** | Price near BB lower + RSI < 50 | Price near BB upper + RSI > 50 |
| **Chaos** | ATR calming + EMA12 > EMA26 + 25 < RSI < 50 | ATR extreme + MACD bearish |

**Output Fields:**
```json
{
  "signal": -1/0/1,           // Directional signal
  "confidence": 0.0-1.0,      // Signal strength
  "signal_strength": 0.0-1.0, // Conviction level
  "regime": 1/2/3/4,          // Market regime
  "close": float,             // Current price
  "ema_12": float,            // Technical indicators
  "ema_26": float,            // (for analysis)
  "rsi": float,
  "macd": float,
  "bb_upper": float,
  "bb_lower": float,
  "atr": float
}
```

### 2.2 Copper Indicator (SHFE/cu<00>)

**Market**: Shanghai Futures Exchange
**Security**: Copper futures (logical contract cu<00>)
**Granularity**: 900s (15 minutes)

**Key Differences from Iron Ore:**
- **Faster EMAs**: 12/24/45 (vs 12/26/50) - copper is more reactive
- **Faster RSI**: 12-period (vs 14-period) - more sensitive

**Parameters:**
```python
alpha_12 = 2.0 / 13.0   # 12-period (same)
alpha_26 = 2.0 / 25.0   # 24-period (FASTER)
alpha_50 = 2.0 / 46.0   # 45-period (FASTER)
alpha_14 = 2.0 / 13.0   # 12-period RSI (FASTER)
```

**Signal logic**: Same as Iron Ore, but with faster indicator response

### 2.3 Soybean Indicator (DCE/m<00>)

**Market**: Dalian Commodity Exchange
**Security**: Soybean Meal futures (logical contract m<00>)
**Granularity**: 900s (15 minutes)

**Parameters:** Standard (same as Iron Ore - 12/26/50, 14-period RSI)
**Signal logic:** Same as Iron Ore

### Implementation Pattern (All Indicators)

```python
class Indicator(pcts3.sv_object):
    def __init__(self):
        # Metadata
        self.meta_name = "IndicatorName"
        self.market = b'DCE'  # or b'SHFE'
        self.code = b'i<00>'  # or b'cu<00>' or b'm<00>'

        # State (auto-persisted)
        self.ema_12 = 0.0
        self.rsi = 50.0
        self.regime = 3

        # Output signals
        self.signal = 0          # -1/0/1
        self.confidence = 0.0    # 0-1
        self.signal_strength = 0.0

    def _on_cycle_pass(self, time_tag):
        # 1. Update indicators
        self._update_emas(self.close)
        self._update_macd()
        self._update_rsi(self.close)
        self._update_bollinger_bands(self.close)
        self._update_atr(self.high, self.low, self.close)
        self._update_volume_ema(self.volume)

        # 2. Detect regime
        self._detect_regime()

        # 3. Generate signal
        self._generate_signal()

    def _generate_signal(self):
        # Calculate signal strength from multiple factors
        self._calculate_signal_strength()

        # Default to neutral
        self.signal = 0
        self.confidence = 0.0

        # Regime-based signal generation
        if self.regime == 1:      # Uptrend
            self._check_uptrend_signals()
        elif self.regime == 2:    # Downtrend
            self._check_downtrend_signals()
        elif self.regime == 3:    # Ranging
            self._check_ranging_signals()
        elif self.regime == 4:    # Chaos
            self._check_chaos_signals()
```

---

## 3. Tier-2 Composite Strategy

### Portfolio Management

**Initial Capital**: ¥1,000,000
**Basket Structure**: 3 independent baskets (one per instrument)
**Base Allocation**: 25% per basket (¥250,000)

### Risk Parameters

| Parameter | Limit | Purpose |
|-----------|-------|---------|
| **Max Allocation/Basket** | 35% | Prevent over-concentration |
| **Min Cash Reserve** | 15% | Ensure liquidity |
| **Max Total Exposure** | 90% | Leave safety buffer |
| **Per-Position Stop-Loss** | 3% | Limit individual losses |
| **Max Portfolio Drawdown** | 10% | Circuit breaker |
| **Max Daily Loss** | 3% | Daily risk cap |

### Entry Conditions (RELAXED)

Signal must pass ALL checks:
```python
def _should_enter(signal_data):
    # 1. Directional signal present
    if signal_data['signal'] == 0:
        return False

    # 2. Moderate confidence (RELAXED from 0.50)
    if signal_data['confidence'] < 0.30:
        return False

    # 3. Signal strength (RELAXED from 0.40)
    if signal_data['signal_strength'] < 0.25:
        return False

    # 4. Risk checks pass
    if not risk_manager.can_enter():
        return False

    return True
```

**Note**: Chaos regime filter removed to increase trading opportunities.

### Exit Conditions

Trigger exit if ANY condition met:
1. **Stop-Loss**: Position down 3% from entry
2. **Signal Reversal**: Indicator flips direction (1 → -1 or vice versa)
3. **Confidence Drop**: Falls below 30%

### Position Sizing

```python
# Leverage based on confidence
confidence = signal_data['confidence']  # 0.3 to 1.0
leverage = 1.0 + (confidence * 1.5)     # Range: 1.3 to 2.5
leverage = min(leverage, 1.4)           # Cap at 1.4x

# Calculate position
allocated_capital = pv * 0.25           # 25% base allocation
position_size = allocated_capital * leverage
```

### Signal Processing Flow

```python
def on_bar(bar):
    # 1. Parse Tier-1 signals
    for basket_idx in [0, 1, 2]:  # Iron/Copper/Soybean
        signal_data = parse_tier1_signal(bar, basket_idx)

        # 2. Update basket price
        basket.price = signal_data['close']

        # 3. Check exit conditions (if in position)
        if basket.signal != 0:
            if should_exit(signal_data, basket):
                execute_exit(basket_idx)
                continue

        # 4. Check entry conditions (if flat)
        if basket.signal == 0:
            if should_enter(signal_data):
                execute_entry(basket_idx, signal_data)

    # 5. Update portfolio metrics
    update_portfolio_state()

    # 6. Serialize output
    return serialize_output()
```

### Trade Execution (WOS Framework Pattern)

```python
def _execute_entry(basket_idx, signal_data):
    basket = self.strategies[basket_idx]
    signal = signal_data['signal']  # 1 or -1
    price = signal_data['close']
    leverage = calculate_leverage(signal_data['confidence'])

    # WOS pattern: fit position, then signal (with inverted direction)
    basket._fit_position(leverage)
    basket._signal(price, basket.timetag, signal * -1)  # Note: inverted!

    # Track entry for stop-loss
    self.entry_prices[basket_idx] = price

def _execute_exit(basket_idx, signal_data):
    basket = self.strategies[basket_idx]
    price = signal_data['close']

    # Calculate P&L
    entry = self.entry_prices[basket_idx]
    if basket.signal == 1:  # Long
        pnl_pct = (price - entry) / entry * 100
    else:  # Short
        pnl_pct = (entry - price) / entry * 100

    # Exit position
    basket._signal(price, basket.timetag, 0)  # Signal=0 = flat
    self.entry_prices[basket_idx] = 0.0
```

### Portfolio Metrics

Tracked and exported every bar:
- **PV** (Portfolio Value): Cash + all position values
- **NV** (Net Value): PV / Initial Capital
- **Active Positions**: Number of baskets with open positions (0-3)
- **Portfolio Exposure**: % of PV in positions
- **Cash Reserve**: % of PV in cash
- **Drawdown**: Current decline from peak PV

---

## 4. Configuration Files

### 4.1 Indicator Output (uout.json)

Each indicator exports to **private namespace**:

```json
{
  "private": {
    "markets": ["DCE"],  // or ["SHFE"] for copper
    "securities": [["i"]],  // or [["cu"]] or [["m"]]
    "sample_granularities": {"type": "min", "cycles": [900]},
    "export": {
      "XXX": {
        "fields": [
          "_preserved_field", "bar_index", "close",
          "signal", "confidence", "regime", "signal_strength",
          "ema_12", "ema_26", "ema_50",
          "macd", "rsi", "bb_upper", "bb_lower", "atr"
        ]
      }
    }
  }
}
```

### 4.2 Composite Input (uin.json)

Imports from **3 indicators** + **market data**:

```json
{
  "global": {
    "imports": {
      "SampleQuote": {
        "fields": ["open", "high", "low", "close", "volume"],
        "markets": ["DCE", "SHFE"],
        "securities": [["i", "m"], ["cu"]]
      }
    }
  },
  "private": {
    "imports": {
      "IronOreIndicatorRelaxed": {
        "fields": ["close", "signal", "confidence", "regime", "signal_strength", ...],
        "markets": ["DCE"],
        "securities": [["i"]]
      },
      "CopperIndicator": {
        "fields": ["close", "signal", "confidence", "regime", "signal_strength", ...],
        "markets": ["SHFE"],
        "securities": [["cu"]]
      },
      "SoybeanIndicator": {
        "fields": ["close", "signal", "confidence", "regime", "signal_strength", ...],
        "markets": ["DCE"],
        "securities": [["m"]]
      }
    }
  }
}
```

### 4.3 Composite Output (uout.json)

Exports **portfolio state** to private namespace:

```json
{
  "private": {
    "markets": ["DCE"],
    "securities": [["COMPOSITE"]],  // Synthetic identifier
    "export": {
      "XXX": {
        "fields": [
          "_preserved_field", "bar_index",
          "pv", "nv",
          "active_positions",
          "total_signals_processed",
          "portfolio_exposure_pct",
          "cash_reserve_pct"
        ]
      }
    }
  }
}
```

---

## 5. Testing & Validation

### Quick Test (7 days)
```bash
python3 calculator3_test.py --testcase CompositeStrategy \
  --algoname CompositeStrategy --sourcefile CompositeStrategy.py \
  --start 20241025000000 --end 20241101000000 \
  --granularity 900 --multiproc 1 --category 1
```

### Replay Consistency Test
```bash
python test_resuming_mode.py
```
Ensures deterministic behavior (no random values, datetime.now(), etc.)

### Full Backtest (3 months)
```bash
# Run via VS Code launch.json or command line
# Analyze results with analysis.ipynb
```

---

## 6. Analysis & Visualization

### Performance Metrics
- Total Return %
- Sharpe Ratio (annualized)
- Max Drawdown %
- Win Rate (monthly)
- Average Exposure
- Total Signals Processed

### Visualizations (analysis.ipynb)
1. Portfolio Value Over Time
2. Net Value (Performance Ratio)
3. Active Positions (0-3)
4. Allocation (Exposure vs Cash)
5. Drawdown Analysis
6. Monthly Returns Breakdown

### Fetch Data from Server
```python
import svr3

reader = svr3.sv_reader(
    start_date, end_date,
    "CompositeStrategy",  # algoname
    900,                  # granularity
    "private",            # namespace
    "symbol",
    ["DCE"], ["COMPOSITE"],
    False, RAILS_URL, WS_URL, "", "", TM_MASTER
)
reader.token = SVR_TOKEN
await reader.login()
await reader.connect()
data = await reader.save_by_symbol()
```

---

## 7. Key Implementation Patterns

### Online Algorithms (Bounded Memory)
All indicators use **EMA-based calculations** (O(1) memory):

```python
# EMA update (not rolling window!)
self.ema = alpha * new_value + (1 - alpha) * self.ema

# RSI via gain/loss EMAs
gain = max(close - prev_close, 0)
loss = max(prev_close - close, 0)
self.gain_ema = alpha * gain + (1 - alpha) * self.gain_ema
self.loss_ema = alpha * loss + (1 - alpha) * self.loss_ema
self.rsi = 100 - (100 / (1 + gain_ema / loss_ema))
```

### State Persistence
WOS framework auto-persists `sv_object` attributes:

```python
class Indicator(pcts3.sv_object):
    def __init__(self):
        # These are automatically saved/restored
        self.ema_12 = 0.0
        self.rsi = 50.0
        self.bar_index = 0

        # NO need for manual serialization!
```

### Logical Contract Filtering
```python
def on_bar(self, bar):
    code = bar.get_stock_code()

    # Only process logical contracts (ending in <00>)
    if code.endswith(b'<00>'):
        # Process this contract
        pass
```

### Framework Callbacks (Required)
```python
async def on_init():
    """Initialize with metadata schemas"""
    indicator.initialize(imports, metas)

async def on_bar(bar):
    """Process incoming market data"""
    return indicator.on_bar(bar)

async def on_ready():
    """Called when framework ready"""
    logger.info("Strategy ready")
```

---

## 8. Common Pitfalls & Solutions

### Issue: No trading occurring
**Cause**: Entry thresholds too strict
**Solution**: Relax confidence (≥0.30) and signal_strength (≥0.25) thresholds

### Issue: Configuration mismatch errors
**Cause**: uin.json imports fields not in uout.json
**Solution**: Ensure field lists match exactly between indicator exports and strategy imports

### Issue: Portfolio value not updating
**Cause**: WOS framework handles PV automatically
**Solution**: Don't manually set `basket.pv` - framework updates it from positions × prices

### Issue: Replay test fails (non-deterministic)
**Cause**: Random values, system time, or external state
**Solution**: Use only EMA-based online algorithms, no `random()` or `datetime.now()`

---

## 9. Directory Structure

```
IronOreIndicator/
├── IronOreIndicator.py          # Tier-1: Iron Ore signals
├── uin.json                     # Input: SampleQuote
├── uout.json                    # Output: signals
├── CopperIndicator/
│   ├── CopperIndicator.py       # Tier-1: Copper signals
│   ├── uin.json
│   └── uout.json
├── SoybeanIndicator/
│   ├── SoybeanIndicator.py      # Tier-1: Soybean signals
│   ├── uin.json
│   └── uout.json
└── CompositeStrategy/
    ├── CompositeStrategy.py     # Tier-2: Portfolio manager
    ├── uin.json                 # Import: 3 indicators + market data
    ├── uout.json                # Output: portfolio metrics
    ├── analysis.ipynb           # Performance analysis
    └── IMPLEMENTATION_GUIDE.md  # This file
```

---

## 10. Next Steps

1. **Optimize Parameters**: Tune confidence/signal_strength thresholds based on backtest results
2. **Extended Backtest**: Run 6-12 month tests to validate robustness
3. **Add Instruments**: Expand to 5-7 instruments for better diversification
4. **Regime Adaptation**: Dynamic parameter adjustment based on detected regime
5. **Machine Learning**: Train signal confidence models on historical performance

---

**Last Updated**: 2025-11-07
**Status**: Production-ready with relaxed thresholds
**Framework**: WOS 3.x
