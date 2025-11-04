# Chapter 7: Tier 1 Indicator Development

**Learning objectives:**
- Build complete Tier 1 indicators from scratch
- Implement multi-commodity patterns
- Handle cycle boundaries correctly
- Serialize outputs properly
- Apply all framework best practices

**Previous:** [06 - Backtest](06-backtest.md) | **Next:** [08 - Tier 2 Composite Strategy](08-tier2-composite.md)

---

## Overview

Tier 1 indicators are the foundation of the Wolverine architecture. They consume raw market data and produce technical analysis signals that Tier 2 strategies aggregate into portfolio decisions.

## Tier 1 Architecture

```
┌──────────────────────────────────────┐
│   Raw Market Data (SampleQuote)     │
│   - OHLCV from exchanges             │
│   - Multiple granularities           │
│   - Multiple instruments             │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│   TIER 1 INDICATOR                   │
│   - Technical analysis               │
│   - Signal generation                │
│   - Regime detection                 │
│   - Multi-timeframe analysis         │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│   Indicator Outputs                  │
│   - Signal strength                  │
│   - Confidence levels                │
│   - Regime classifications           │
│   - Supporting metrics               │
└──────────────────────────────────────┘
```

## Building a Simple Indicator

### Step 1: Project Structure

```bash
mkdir MyIndicator
cd MyIndicator
touch MyIndicator.py
touch uin.json
touch uout.json
touch test_resuming_mode.py
mkdir .vscode
touch .vscode/launch.json
```

### Step 2: uin.json Configuration

```json
{
  "global": {
    "imports": {
      "SampleQuote": {
        "fields": ["open", "high", "low", "close", "volume", "turnover"],
        "granularities": [900],
        "revision": 4294967295,
        "markets": ["SHFE"],
        "security_categories": [[1, 2, 3]],
        "securities": [["cu", "al"]]
      }
    }
  }
}
```

### Step 3: uout.json Configuration

```json
{
  "private": {
    "markets": ["SHFE"],
    "security_categories": [[1, 2, 3]],
    "securities": [["cu", "al"]],
    "sample_granularities": {
      "type": "min",
      "cycles": [900],
      "cycle_lengths": [0]
    },
    "export": {
      "XXX": {
        "fields": [
          "_preserved_field",
          "bar_index",
          "ema_fast",
          "ema_slow",
          "signal",
          "confidence"
        ],
        "defs": [
          {"field_type": "int64", "precision": 0},
          {"field_type": "integer", "precision": 0},
          {"field_type": "double", "precision": 6},
          {"field_type": "double", "precision": 6},
          {"field_type": "integer", "precision": 0},
          {"field_type": "double", "precision": 6}
        ],
        "revision": -1
      }
    }
  }
}
```

### Step 4: Complete Indicator Implementation

```python
#!/usr/bin/env python3
# coding=utf-8
"""Simple EMA Crossover Indicator - Tier 1"""

import math
from collections import deque
import pycaitlyn as pc
import pycaitlynts3 as pcts3
import pycaitlynutils3 as pcu3

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

class SampleQuote(pcts3.sv_object):
    """Parse SampleQuote StructValue data."""
    def __init__(self):
        super().__init__()
        self.meta_name = "SampleQuote"
        self.namespace = pc.namespace_global
        self.revision = (1 << 32) - 1
        self.open = None
        self.high = None
        self.low = None
        self.close = None
        self.volume = None
        self.turnover = None

class CommodityIndicator(pcts3.sv_object):
    """EMA crossover indicator for a single commodity."""

    def __init__(self, commodity_code: bytes, market: bytes):
        super().__init__()

        # Metadata (CONSTANTS)
        self.meta_name = "MyIndicator"
        self.namespace = pc.namespace_private
        self.revision = (1 << 32) - 1
        self.granularity = 900
        self.market = market
        self.code = commodity_code + b'<00>'  # Logical contract

        # Data parser
        self.quote = SampleQuote()

        # State variables (automatically persisted)
        self.bar_index = 0
        self.timetag = None

        # Indicator state
        self.ema_fast = 0.0
        self.ema_slow = 0.0
        self.signal = 0  # -1, 0, 1
        self.confidence = 0.0

        # Algorithm parameters
        self.alpha_fast = 2.0 / 11.0  # 10-period EMA
        self.alpha_slow = 2.0 / 21.0  # 20-period EMA

        # Recent data for confidence calculation
        self.recent_prices = deque(maxlen=20)

        # Previous values
        self.prev_close = 0.0
        self.initialized = False
        self.persistent = True

        logger.info(f"Initialized indicator for {commodity_code.decode()}")

    def initialize(self, imports, metas):
        """Initialize schemas."""
        self.load_def_from_dict(metas)
        self.set_global_imports(imports)
        self.quote.load_def_from_dict(metas)
        self.quote.set_global_imports(imports)

    def on_bar(self, bar: pc.StructValue):
        """Process incoming bar data."""
        # Filter for our instrument and logical contract
        market = bar.get_market()
        code = bar.get_stock_code()

        if not (market == self.market and 
                code.startswith(self.code[:-4]) and  # Commodity prefix
                code.endswith(b'<00>')):  # Logical contract only
            return []

        # Parse quote data
        ns = bar.get_namespace()
        meta_id = bar.get_meta_id()

        if ns == self.quote.namespace and meta_id == self.quote.meta_id:
            self.quote.market = market
            self.quote.code = code
            self.quote.granularity = bar.get_granularity()
            self.quote.from_sv(bar)
        else:
            return []

        tm = bar.get_time_tag()

        # Initialize timetag
        if self.timetag is None:
            self.timetag = tm

        # Handle cycle boundary
        if self.timetag < tm:
            self._on_cycle_pass()

            # Output if ready
            results = []
            if self.bar_index > 0 and self.initialized:
                results.append(self.copy_to_sv())

            self.timetag = tm
            self.bar_index += 1

            return results

        return []

    def _on_cycle_pass(self):
        """Process end of cycle."""
        # Extract price data
        close = float(self.quote.close)
        high = float(self.quote.high)
        low = float(self.quote.low)

        # Initialize on first bar
        if not self.initialized:
            self.ema_fast = close
            self.ema_slow = close
            self.prev_close = close
            self.initialized = True
            return

        # Update EMAs (online algorithm)
        self.ema_fast = self.alpha_fast * close + (1 - self.alpha_fast) * self.ema_fast
        self.ema_slow = self.alpha_slow * close + (1 - self.alpha_slow) * self.ema_slow

        # Store recent prices for confidence
        self.recent_prices.append(close)

        # Generate signal
        self._generate_signal(close)

        # Update previous values
        self.prev_close = close

        logger.debug(f"Bar {self.bar_index}: "
                    f"EMA_fast={self.ema_fast:.2f}, "
                    f"EMA_slow={self.ema_slow:.2f}, "
                    f"Signal={self.signal}")

    def _generate_signal(self, close):
        """Generate trading signal."""
        # EMA crossover
        if self.ema_fast > self.ema_slow * 1.005:  # 0.5% buffer
            self.signal = 1  # Bullish
        elif self.ema_fast < self.ema_slow * 0.995:
            self.signal = -1  # Bearish
        else:
            self.signal = 0  # Neutral

        # Calculate confidence based on trend consistency
        if len(self.recent_prices) >= 10:
            recent = list(self.recent_prices)[-10:]
            # Linear regression slope
            n = len(recent)
            x_mean = (n - 1) / 2.0
            y_mean = sum(recent) / n

            numerator = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
            denominator = sum((i - x_mean) ** 2 for i in range(n))

            if denominator > 0:
                slope = numerator / denominator
                # Normalize slope to confidence [0, 1]
                trend_strength = abs(slope) / (y_mean / n) if y_mean > 0 else 0
                self.confidence = min(trend_strength, 1.0)
            else:
                self.confidence = 0.0
        else:
            self.confidence = 0.0

class MultiCommodityManager:
    """Manage indicators for multiple commodities."""

    def __init__(self):
        self.indicators = {}

        # Commodity to market mapping
        commodities = {
            b'SHFE': [b'cu', b'al'],
        }

        # Create indicator for each commodity
        for market, codes in commodities.items():
            for code in codes:
                key = (market, code)
                self.indicators[key] = CommodityIndicator(code, market)

        logger.info(f"Initialized {len(self.indicators)} commodity indicators")

    def initialize(self, imports, metas):
        """Initialize all indicators."""
        for indicator in self.indicators.values():
            indicator.initialize(imports, metas)

    def on_bar(self, bar: pc.StructValue):
        """Distribute bar to all indicators."""
        results = []

        for indicator in self.indicators.values():
            outputs = indicator.on_bar(bar)
            results.extend(outputs)

        return results

# Global manager
manager = MultiCommodityManager()

# Framework callbacks
async def on_init():
    global manager, imports, metas, worker_no
    if worker_no != 0 and metas and imports:
        manager.initialize(imports, metas)

async def on_ready():
    pass

async def on_bar(bar: pc.StructValue):
    global manager, worker_no
    if worker_no != 1:
        return []
    return manager.on_bar(bar)

async def on_market_open(market, tradeday, time_tag, time_string):
    pass

async def on_market_close(market, tradeday, timetag, timestring):
    pass

async def on_tradeday_begin(market, tradeday, time_tag, time_string):
    pass

async def on_tradeday_end(market, tradeday, timetag, timestring):
    pass

async def on_reference(market, tradeday, data, timetag, timestring):
    pass

async def on_historical(params, records):
    pass
```

## Multi-Commodity Pattern

**Critical Doctrine: Multiple Indicator Objects (DOCTRINE 1)**

Never reuse a single indicator for multiple commodities:

```python
# ❌ WRONG - Single object for multiple commodities
class BadIndicator:
    def on_bar(self, bar):
        code = bar.get_stock_code()
        if code.startswith(b'cu'):
            # Process copper
        elif code.startswith(b'al'):
            # Process aluminum - WRONG! State contamination!

# ✅ CORRECT - Separate objects per commodity
class GoodManager:
    def __init__(self):
        self.indicators = {
            b'cu': CommodityIndicator(b'cu', b'SHFE'),
            b'al': CommodityIndicator(b'al', b'SHFE'),
        }

    def on_bar(self, bar):
        results = []
        for indicator in self.indicators.values():
            outputs = indicator.on_bar(bar)
            results.extend(outputs)
        return results
```

## Cycle Boundary Handling

Proper pattern for cycle boundaries:

```python
def on_bar(self, bar):
    """Correct cycle boundary handling."""
    tm = bar.get_time_tag()

    # Initialize timetag
    if self.timetag is None:
        self.timetag = tm

    # Cycle boundary detected
    if self.timetag < tm:
        # 1. Process end of previous cycle
        self._on_cycle_pass()

        # 2. Generate output if ready
        results = []
        if self.ready_to_serialize():
            results.append(self.copy_to_sv())

        # 3. Update for new cycle
        self.timetag = tm
        self.bar_index += 1

        return results

    # Still in current cycle - accumulate data
    return []
```

## Output Serialization

Always output as list:

```python
def on_bar(self, bar):
    """ALWAYS return list."""
    # Process bar
    self._process(bar)

    # Cycle boundary
    if self.timetag < bar.get_time_tag():
        self._on_cycle_pass()

        # ✅ CORRECT - Always return list
        results = []
        if self.bar_index > 0:
            results.append(self.copy_to_sv())

        self.timetag = bar.get_time_tag()
        self.bar_index += 1

        return results  # List (possibly empty)

    return []  # Empty list if no output
```

## Best Practices

### 1. Filter Logical Contracts Only

```python
# DOCTRINE 4: Only process logical contracts
if (market == self.market and
    code.startswith(self.commodity_code) and
    code.endswith(b'<00>')):  # Logical contract filter
    # Process data
```

### 2. Trust Dependency Data

```python
# DOCTRINE 2: No fallback logic for dependency data
# ❌ WRONG
close = float(quote.close) if quote.close else 100.0

# ✅ CORRECT
close = float(quote.close)  # Trust the data
```

### 3. Online Algorithms

```python
# Use online algorithms for bounded memory
def update_ema(self, value):
    """Online EMA - O(1) memory."""
    self.ema = self.alpha * value + (1 - self.alpha) * self.ema
```

### 4. Bounded Collections

```python
from collections import deque

# Fixed-size collections
self.recent_prices = deque(maxlen=100)
self.swing_points = deque(maxlen=50)
```

### 5. Vector-to-Scalar Serialization for Multi-Parameter Indicators

**When to Use:** If you calculate similar values with different parameters (e.g., multiple EMA periods).

**The Pattern:**

```python
class MultiPeriodEMA(pcts3.sv_object):
    """Calculate EMAs with multiple periods"""

    def __init__(self):
        super().__init__()
        self.meta_name = "MultiPeriodEMA"
        self.namespace = pc.namespace_private

        # Internal vector (convenient for calculations)
        self.ema_values: List[float] = [0.0] * 5
        self.periods = [10, 20, 50, 100, 200]

        # Scalar fields for output (must match uout.json)
        self.ema_10 = 0.0
        self.ema_20 = 0.0
        self.ema_50 = 0.0
        self.ema_100 = 0.0
        self.ema_200 = 0.0

    def _calculate_emas(self, price: float):
        """Update all EMAs using internal vector"""
        for i, period in enumerate(self.periods):
            alpha = 2.0 / (period + 1)
            self.ema_values[i] = alpha * price + (1 - alpha) * self.ema_values[i]

    def copy_to_sv(self) -> pc.StructValue:
        """Convert vector to scalars for output"""
        # CRITICAL: Vector → Scalars
        self.ema_10 = self.ema_values[0]
        self.ema_20 = self.ema_values[1]
        self.ema_50 = self.ema_values[2]
        self.ema_100 = self.ema_values[3]
        self.ema_200 = self.ema_values[4]
        return super().copy_to_sv()

    def from_sv(self, sv: pc.StructValue):
        """Reconstruct vector from scalars when resuming"""
        super().from_sv(sv)
        # CRITICAL: Scalars → Vector (MUST be inverse of copy_to_sv)
        self.ema_values[0] = self.ema_10
        self.ema_values[1] = self.ema_20
        self.ema_values[2] = self.ema_50
        self.ema_values[3] = self.ema_100
        self.ema_values[4] = self.ema_200

    def _on_cycle_pass(self, time_tag):
        """Process cycle using vector calculations"""
        price = float(self.sq.close)

        # Calculate using vector (convenient)
        self._calculate_emas(price)

        # Generate signals from vector values
        if (self.ema_values[0] > self.ema_values[1] > self.ema_values[2]):
            self.signal = 1  # Uptrend across multiple periods
        elif (self.ema_values[0] < self.ema_values[1] < self.ema_values[2]):
            self.signal = -1  # Downtrend
        else:
            self.signal = 0
```

**Why This Matters:**

- **Replay Consistency**: When resuming from midpoint, `from_sv()` must correctly reconstruct your internal vector
- **State Persistence**: Every cycle, `copy_to_sv()` must save current vector values to scalar fields
- **Inverse Requirement**: `from_sv()` must exactly reverse what `copy_to_sv()` does

**Common Mistake:**

```python
# ❌ WRONG - Asymmetric conversion
def copy_to_sv(self):
    self.ema_10 = self.ema_values[0]
    self.ema_20 = self.ema_values[1]
    # Missing: ema_50, ema_100, ema_200!
    return super().copy_to_sv()

def from_sv(self, sv):
    super().from_sv(sv)
    self.ema_values[0] = self.ema_10
    self.ema_values[1] = self.ema_20
    self.ema_values[2] = self.ema_50  # Loaded but never saved!
    # Result: ema_values[2:] will be wrong on resume
```

**Testing:**

```python
# Test round-trip serialization
obj1.ema_values = [100.0, 101.0, 102.0, 103.0, 104.0]
sv = obj1.copy_to_sv()
obj2.from_sv(sv)
assert obj1.ema_values == obj2.ema_values  # MUST pass!
```

> **See Chapter 04 for detailed explanation of vector-to-scalar serialization pattern.**

## Summary

This chapter covered:

1. **Project Setup**: Files and configurations
2. **Simple Indicator**: EMA crossover example
3. **Multi-Commodity**: Manager pattern
4. **Cycle Boundaries**: Proper handling
5. **Output Serialization**: Always return list
6. **Vector-to-Scalar Serialization**: Multi-parameter indicators
7. **Best Practices**: All critical doctrines

**Key Takeaways:**

- One indicator object per commodity (DOCTRINE 1)
- Only process logical contracts (DOCTRINE 4)
- Trust dependency data (DOCTRINE 2)
- Always return list (DOCTRINE 3)
- Use online algorithms
- Bounded memory with deque
- **CRITICAL**: Override copy_to_sv()/from_sv() for vector-to-scalar conversion
- **CRITICAL**: from_sv() must be exact inverse of copy_to_sv()

**Next Steps:**

In the next chapter, we'll build a Tier 2 composite strategy that aggregates signals from multiple Tier 1 indicators.

---

**Previous:** [06 - Backtest](06-backtest.md) | **Next:** [08 - Tier 2 Composite Strategy](08-tier2-composite.md)
