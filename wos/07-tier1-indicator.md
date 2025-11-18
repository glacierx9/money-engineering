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

**Tier 1 indicators**: Transform raw market data (OHLCV) → technical signals for Tier 2 portfolio strategies.

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

# Initialization constant
MAX_BAR_REBUILD = 20  # max(sliding_window_lengths)

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

        # Initialization tracking (NOT persisted)
        self.bar_since_start = 0
        self.initialized = False

        # Reconciliation support (NOT persisted)
        self.latest_sv = None

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
        self.persistent = True

        logger.info(f"Initialized indicator for {commodity_code.decode()}")

    def initialize(self, imports, metas):
        """Initialize schemas."""
        self.load_def_from_dict(metas)
        self.set_global_imports(imports)
        self.quote.load_def_from_dict(metas)
        self.quote.set_global_imports(imports)

    def from_sv(self, sv: pc.StructValue):
        """Cache incoming StructValue for reconciliation."""
        self.latest_sv = sv

    def _from_sv(self, sv: pc.StructValue):
        """Internal deserialization helper."""
        super().from_sv(sv)

    def ready_to_serialize(self) -> bool:
        """Determine if state should be serialized."""
        return self.initialized

    def on_bar(self, bar: pc.StructValue):
        """Process incoming bar data.

        CRITICAL: Data leakage prevention pattern.
        Import data updates happen AFTER _on_cycle_pass().
        """
        ret = []  # ALWAYS return list

        # Extract metadata
        market = bar.get_market()
        code = bar.get_stock_code()
        tm = bar.get_time_tag()
        ns = bar.get_namespace()
        meta_id = bar.get_meta_id()

        # Filter for our instrument
        if not (market == self.market and
                code.startswith(self.code[:-4]) and
                code.endswith(b'<00>')): # Logical contract only
            return ret

        # Initialize timetag
        if self.timetag is None:
            self.timetag = tm

        # CRITICAL: Handle cycle boundary
        if self.timetag < tm:
            # 1. FIRST: Process with OLD data (prevent data leakage)
            self._on_cycle_pass()

            # 2. Reconcile with server data if needed
            if (not self.initialized or not overwrite) and self.latest_sv is not None:
                self._from_sv(self.latest_sv)

            # 3. Serialize state if ready
            if self.ready_to_serialize():
                if not overwrite:
                    self._reconcile()
                ret.append(self.copy_to_sv())

            # 4. Update for next cycle
            self.timetag = tm
            self.bar_since_start += 1
            self.bar_index += 1

            if self.bar_since_start >= MAX_BAR_REBUILD:
                self.initialized = True

            self.latest_sv = None

        # CRITICAL: Update imported data AFTER cycle pass (for NEXT cycle)
        if ns == self.quote.namespace and meta_id == self.quote.meta_id:
            self.quote.market = market
            self.quote.code = code
            self.quote.granularity = bar.get_granularity()
            self.quote.from_sv(bar)

        # Restore from saved state during initialization
        if self.meta_id == meta_id and self.namespace == ns and not self.initialized:
            self.from_sv(bar)

        return ret

    def _reconcile(self):
        """Reconcile calculated vs server state."""
        # Compare self.ema_fast with latest_sv.ema_fast, adjust if needed
        # Implementation depends on reconciliation strategy
        pass

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

## Bar Routing Patterns

**Best practice**: One sv_object instance per (market, code, granularity) tuple

### Pattern 1: Logic Contract (<00>) - Indicators

**Use case**: Commodity indicators outputting per logical contract

**Code pattern**: Use `<00>` for commodity, create one instance per commodity

```python
class CommodityIndicator(pcts3.sv_object):
    def __init__(self, commodity: bytes, market: bytes):
        super().__init__()
        self.market = market
        self.code = commodity + b'<00>'  # Logic contract

    def on_bar(self, bar):
        code = bar.get_stock_code()

        # Filter for this commodity's <00> contract only
        if (bar.get_market() == self.market and
            code == self.code):
            # Process this commodity's leading contract
            process(bar)
```

**Output**: Single StructValue per commodity using `<00>` code

### Pattern 2: Real Contract - Resamplers

**Use case**: Per-contract processing (e.g., ZampleQuote resampling)

**Code pattern**: Create instance for each real or logic contract

```python
class ContractResampler(pcts3.sv_object):
    def __init__(self, market: bytes, code: bytes):
        super().__init__()
        self.market = market
        self.code = code  # Exact contract (e.g., b'cu2501' or b'cu<01>')

    def on_bar(self, bar):
        # Route by exact market + code match
        if (bar.get_market() == self.market and
            bar.get_stock_code() == self.code):
            # Process this specific contract
            resample(bar)
```

**Output**: StructValue per contract with actual contract code

### Pattern 3: Placeholder/Aggregation - Indices

**Use case**: Cross-commodity aggregations (e.g., sector indices)

**Code pattern**: Use commodity code as placeholder for aggregate

```python
class SectorIndex(pcts3.sv_object):
    def __init__(self):
        super().__init__()
        # Placeholder code for index output
        self.market = b'SHFE'
        self.code = b'rb<00>'  # Not actually rebar - just placeholder

    def on_bar(self, bar):
        # Aggregate across multiple commodities
        for commodity in [b'rb', b'hc', b'i']:  # Black minerals
            if bar.get_stock_code().startswith(commodity):
                update_index(bar)
```

**Output**: Single aggregate StructValue using placeholder code

### Routing Pattern Summary

| Use Case | Instance Scope | Code Pattern | Example |
|----------|---------------|--------------|---------|
| Commodity indicators | One per commodity | `commodity + b'<00>'` | `b'cu<00>'` |
| Contract resamplers | One per contract | Exact contract code | `b'cu2501'`, `b'cu<01>'` |
| Cross-commodity aggregation | One for aggregate | Placeholder code | `b'rb<00>'` (for index) |

**Critical**: Match instance granularity to routing need to avoid state contamination.

## Cycle Boundary Handling

**CRITICAL**: Data leakage prevention pattern. Import data updates happen AFTER `_on_cycle_pass()`.

```python
def on_bar(self, bar):
    """Correct cycle boundary handling with data leakage prevention."""
    ret = []

    # Extract metadata
    tm = bar.get_time_tag()
    ns = bar.get_namespace()
    meta_id = bar.get_meta_id()

    # Initialize timetag
    if self.timetag is None:
        self.timetag = tm

    # CRITICAL: Cycle boundary detected
    if self.timetag < tm:
        # 1. FIRST: Process with OLD data (prevent data leakage)
        self._on_cycle_pass()

        # 2. Reconcile with server data if needed
        if (not self.initialized or not overwrite) and self.latest_sv is not None:
            self._from_sv(self.latest_sv)

        # 3. Generate output if ready
        if self.ready_to_serialize():
            if not overwrite:
                self._reconcile()
            ret.append(self.copy_to_sv())

        # 4. Update for new cycle
        self.timetag = tm
        self.bar_since_start += 1
        self.bar_index += 1

        if self.bar_since_start >= MAX_BAR_REBUILD:
            self.initialized = True

        self.latest_sv = None

    # CRITICAL: Update imported data AFTER cycle pass (for NEXT cycle)
    if ns == self.quote.namespace and meta_id == self.quote.meta_id:
        self.quote.from_sv(bar)

    # Restore from saved state during initialization
    if self.meta_id == meta_id and self.namespace == ns and not self.initialized:
        self.from_sv(bar)

    return ret
```

**Why This Matters:**

| Step | Purpose | Data Used |
|------|---------|-----------|
| 1. `_on_cycle_pass()` | Process cycle t | Data from cycle t |
| 2. Reconcile | Compare calc vs server | Server data from t |
| 3. Serialize | Output cycle t result | Calculated data from t |
| 4. Update counters | Advance to cycle t+1 | N/A |
| 5. Update imports | Store data for t+1 | **Data from bar (t+1)** |

If you update imports BEFORE step 1, cycle t uses data from t+1 → **data leakage bug**.

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

    def to_sv(self) -> pc.StructValue:
        """Convert vector to scalars for serialization.

        CRITICAL: from_sv() must be the exact inverse of this method.
        """
        # Vector → Scalars
        self.ema_10 = self.ema_values[0]
        self.ema_20 = self.ema_values[1]
        self.ema_50 = self.ema_values[2]
        self.ema_100 = self.ema_values[3]
        self.ema_200 = self.ema_values[4]
        return super().to_sv()

    def from_sv(self, sv: pc.StructValue):
        """Reconstruct vector from scalars when resuming.

        CRITICAL: Must be exact inverse of to_sv().
        """
        super().from_sv(sv)
        # Scalars → Vector
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

| Requirement | Purpose |
|-------------|---------|
| `from_sv() ↔ to_sv()` inverse | Replay consistency when resuming from midpoint |
| Complete mapping | All vector elements must round-trip correctly |
| No data loss | Every field in vector must have corresponding scalar |

**Note:** Since `copy_to_sv()` calls `to_sv()` internally, ensuring the `from_sv() ↔ to_sv()` inverse relationship guarantees correct behavior with `copy_to_sv()`.

**Common Mistake:**

```python
# ❌ WRONG - Asymmetric conversion
def to_sv(self):
    self.ema_10 = self.ema_values[0]
    self.ema_20 = self.ema_values[1]
    # Missing: ema_50, ema_100, ema_200!
    return super().to_sv()

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
sv = obj1.to_sv()
obj2.from_sv(sv)
assert obj1.ema_values == obj2.ema_values  # MUST pass!
```

> **See Chapter 04 for detailed explanation of `from_sv() ↔ to_sv()` inverse relationship.**

## Summary

This chapter covered:

1. **Project Setup**: Files and configurations
2. **Complete Indicator**: EMA crossover with all critical patterns
3. **Multi-Commodity**: Manager pattern (DOCTRINE 1)
4. **Data Leakage Prevention**: Import updates AFTER `_on_cycle_pass()`
5. **Initialization Tracking**: `bar_since_start` and `MAX_BAR_REBUILD`
6. **Reconciliation**: `latest_sv` caching and `_reconcile()`
7. **Cycle Boundaries**: 5-step pattern with proper timing
8. **Output Serialization**: Always return list
9. **Vector-to-Scalar**: Multi-parameter indicators with `to_sv()/from_sv()`
10. **Best Practices**: All critical doctrines

**Critical Patterns (from organic/fix001.md):**

| Pattern | Purpose | Location |
|---------|---------|----------|
| Data leakage prevention | Prevent using t+1 data for cycle t | `on_bar()` line 277-282 |
| Initialization tracking | Know when enough data accumulated | `bar_since_start >= MAX_BAR_REBUILD` |
| Reconciliation | Compare calc vs server state | `_reconcile()` in overwrite=False mode |
| `from_sv() ↔ to_sv()` | Exact inverse for state persistence | Vector-to-scalar section |

**Key Takeaways:**

- **CRITICAL**: Update imported data AFTER `_on_cycle_pass()` (data leakage prevention)
- **CRITICAL**: Track initialization with `bar_since_start` (NOT persisted)
- **CRITICAL**: Cache with `latest_sv`, reconcile with `_reconcile()`
- **CRITICAL**: `from_sv()` must be exact inverse of `to_sv()`
- One indicator object per commodity (DOCTRINE 1)
- Only process logical contracts (DOCTRINE 4)
- Trust dependency data (DOCTRINE 2)
- Always return list (DOCTRINE 3)
- Use online algorithms (bounded memory)
- Use `deque(maxlen=N)` for fixed-size collections

**Next Steps:**

In the next chapter, we'll build a Tier 2 composite strategy that aggregates signals from multiple Tier 1 indicators.

---

**Previous:** [06 - Backtest](06-backtest.md) | **Next:** [08 - Tier 2 Composite Strategy](08-tier2-composite.md)
