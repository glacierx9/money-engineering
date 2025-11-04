# Chapter 05: Stateless Design and State Management

**Learning objectives:**
- Understand stateless design principles
- Implement strategies that can resume from any point
- Use online algorithms for bounded memory
- Achieve replay consistency
- Master state persistence patterns

**Previous:** [04 - StructValue and sv_object](04-structvalue-and-sv_object.md) | **Next:** [06 - Backtest and Testing](06-backtest.md)

---

## Overview

**Stateless Design**: Strategy produces identical results regardless of stop/resume point.

**Key Property**: `Run(0→100) == Run(0→50) + Resume(50→100)`

**Chapter Coverage**:
- Bounded memory with online algorithms (EMA, Welford's variance)
- State persistence patterns (complete, minimal state in sv_object)
- Replay consistency testing (reconciliation implementation)
- Production patterns (deque constraints, deterministic computation)

## Stateless vs Stateful Comparison

| Aspect | ❌ Stateful (Bad) | ✅ Stateless (Good) |
|--------|------------------|---------------------|
| **Memory** | Unbounded growth (crashes) | O(1) bounded memory |
| **State Size** | Growing (slow persist) | Fixed size (fast persist) |
| **Resume** | Inconsistent results | Identical results |
| **Replay** | Cannot resume mid-run | Resume from any point |
| **Pattern** | `prices = []` (all history) | `ema = 0.1*p + 0.9*ema` (online) |

### The Problem: Unbounded State

```python
# ❌ BAD - Unbounded growth
class BadStrategy(pcts3.sv_object):
    def __init__(self):
        self.price_history = []  # Grows forever!

    def on_bar(self, bar):
        self.price_history.append(float(bar.close))
        avg = sum(self.price_history) / len(self.price_history)
        # After restart: price_history is empty → different results
```

### The Solution: Online Algorithm

```python
# ✅ GOOD - O(1) memory
class GoodStrategy(pcts3.sv_object):
    def __init__(self):
        self.ema = 0.0
        self.initialized = False

    def on_bar(self, bar):
        price = float(bar.close)
        if not self.initialized:
            self.ema = price
            self.initialized = True
        else:
            self.ema = 0.1 * price + 0.9 * self.ema
        # Same result regardless of restart point
```

## Stateless Design Principles

### Principle 1: Bounded Memory

**Use fixed-size data structures:**

```python
from collections import deque

class BoundedStrategy(pcts3.sv_object):
    def __init__(self):
        super().__init__()

        # ✅ GOOD - Fixed maximum size
        self.recent_prices = deque(maxlen=100)
        self.swing_points = deque(maxlen=50)
        self.volume_buffer = deque(maxlen=20)

        # ✅ GOOD - Single values
        self.ema_fast = 0.0
        self.ema_slow = 0.0
        self.atr = 0.0

        # ❌ BAD - Unbounded
        # self.all_prices = []  # WRONG!
        # self.all_volumes = []  # WRONG!
```

### Principle 2: Online Algorithms

**Update state incrementally without history:**

```python
class OnlineStrategy(pcts3.sv_object):
    """Uses online algorithms for O(1) memory."""

    def __init__(self):
        super().__init__()

        # EMA components (Welford's online algorithm)
        self.ema_mean = 0.0
        self.ema_var = 0.0
        self.alpha = 0.05

        # ADX components
        self.tr_ema = 0.0
        self.dm_plus_ema = 0.0
        self.dm_minus_ema = 0.0

        self.bar_count = 0

    def update_ema(self, value):
        """Online EMA update - no history needed."""
        self.ema_mean = self.alpha * value + (1 - self.alpha) * self.ema_mean

    def update_variance(self, value):
        """Welford's online variance algorithm."""
        delta = value - self.ema_mean
        self.ema_mean += self.alpha * delta
        self.ema_var += self.alpha * (delta * delta - self.ema_var)

    def update_atr(self, high, low, prev_close):
        """Online ATR calculation."""
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        tr = max(tr1, tr2, tr3)

        # Online EMA of TR
        self.tr_ema = self.alpha * tr + (1 - self.alpha) * self.tr_ema
```

### Principle 3: No External Dependencies

**Self-contained state:**

```python
# ❌ BAD - External state
global_cache = {}  # WRONG!

class BadStrategy(pcts3.sv_object):
    def on_bar(self, bar):
        # Depends on external global
        if 'prices' not in global_cache:
            global_cache['prices'] = []
        global_cache['prices'].append(bar.close)

# ✅ GOOD - All state internal
class GoodStrategy(pcts3.sv_object):
    def __init__(self):
        super().__init__()
        self.ema = 0.0  # All state in instance

    def on_bar(self, bar):
        price = float(bar.close)
        self.ema = 0.1 * price + 0.9 * self.ema
```

### Principle 4: Deterministic Computation

**Same input → Same output:**

```python
# ❌ BAD - Non-deterministic
import random
import time

class BadStrategy(pcts3.sv_object):
    def on_bar(self, bar):
        # Random introduces non-determinism
        noise = random.random()  # WRONG!
        signal = self.calculate_signal() + noise

        # Time-based decisions are non-deterministic
        if time.time() % 2 == 0:  # WRONG!
            return signal
        return 0

# ✅ GOOD - Deterministic
class GoodStrategy(pcts3.sv_object):
    def __init__(self):
        super().__init__()
        self.pseudo_random_state = 12345  # Seed

    def on_bar(self, bar):
        price = float(bar.close)

        # Use bar_index for deterministic "randomness"
        if self.bar_index % 2 == 0:
            return self.calculate_signal(price)
        return 0
```

## Online Algorithms

### EMA (Exponential Moving Average)

**Standard EMA:**

```python
def update_ema(self, value, alpha=0.1):
    """
    Online EMA update.

    alpha = 2 / (period + 1)
    For 20-period EMA: alpha = 2 / 21 ≈ 0.095
    """
    if not self.initialized:
        self.ema = value
        self.initialized = True
    else:
        self.ema = alpha * value + (1 - alpha) * self.ema
```

### Welford's Online Variance

**Calculate mean and variance online:**

```python
class OnlineVariance:
    """Welford's algorithm for online mean and variance."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squared differences

    def update(self, value):
        """Add new value and update statistics."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self):
        """Current variance."""
        if self.count < 2:
            return 0.0
        return self.m2 / self.count

    @property
    def std_dev(self):
        """Current standard deviation."""
        return math.sqrt(self.variance)
```

**With EMA decay:**

```python
def update_ema_variance(self, value, alpha=0.05):
    """EMA-based variance (bounded memory)."""
    delta = value - self.ema_mean
    self.ema_mean += alpha * delta
    self.ema_var += alpha * (delta * delta - self.ema_var)

    # Volatility
    volatility = math.sqrt(max(self.ema_var, 0))
```

### Online Min/Max Tracking

**Track recent min/max with deque:**

```python
from collections import deque

class OnlineMinMax:
    """Track min/max over sliding window."""

    def __init__(self, window=20):
        self.window = window
        self.values = deque(maxlen=window)

    def update(self, value):
        """Add value and get current min/max."""
        self.values.append(value)
        return min(self.values), max(self.values)
```

### ATR (Average True Range)

**Online ATR calculation:**

```python
def update_atr(self, high, low, close, prev_close, alpha=0.1):
    """Online ATR using EMA."""
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - prev_close) if prev_close > 0 else 0
    tr3 = abs(low - prev_close) if prev_close > 0 else 0
    tr = max(tr1, tr2, tr3)

    # EMA of TR
    if self.atr == 0.0:
        self.atr = tr
    else:
        self.atr = alpha * tr + (1 - alpha) * self.atr
```

## State Persistence Patterns

### Pattern 1: Minimal State

```python
class MinimalState(pcts3.sv_object):
    """Store only what's necessary."""

    def __init__(self):
        super().__init__()

        # Metadata (constants)
        self.meta_name = "MinimalState"
        self.namespace = pc.namespace_private
        self.granularity = 900
        self.market = b'SHFE'
        self.code = b'cu<00>'

        # Minimal state (automatically persisted)
        self.bar_index = 0
        self.ema = 0.0
        self.prev_close = 0.0

        # Control persistence
        self.persistent = True
```

### Pattern 2: Bounded Collections

```python
from collections import deque

class BoundedState(pcts3.sv_object):
    """Fixed-size collections for recent data."""

    def __init__(self):
        super().__init__()

        # Bounded collections (automatically persisted)
        self.recent_prices = deque(maxlen=100)
        self.swing_highs = deque(maxlen=50)
        self.swing_lows = deque(maxlen=50)

        # NOTE: deque is serialized as list
        # Maxlen is NOT preserved in serialization
        # Must restore maxlen after from_sv()
```

**Restoring deque after load:**

```python
def from_sv(self, sv):
    """Load state and restore deque constraints."""
    super().from_sv(sv)

    # Restore maxlen (not preserved in serialization)
    if not isinstance(self.recent_prices, deque):
        self.recent_prices = deque(self.recent_prices, maxlen=100)
    if not isinstance(self.swing_highs, deque):
        self.swing_highs = deque(self.swing_highs, maxlen=50)
    if not isinstance(self.swing_lows, deque):
        self.swing_lows = deque(self.swing_lows, maxlen=50)
```

### Pattern 3: Separate Persistent vs Transient

```python
class MixedState(pcts3.sv_object):
    """Separate persistent and transient state."""

    def __init__(self):
        super().__init__()

        # Persistent state (automatically saved)
        self.ema = 0.0
        self.signal = 0
        self.bar_index = 0

        # Transient state (not saved - use _ prefix)
        self._temp_buffer = []
        self._debug_info = {}
        self._cache = None

    def on_bar(self, bar):
        """Transient state rebuilt each run."""
        # Rebuild transient state
        self._temp_buffer = []
        self._cache = {}

        # Update persistent state
        self.ema = self.calculate_ema()
        self.bar_index += 1
```

## Replay Consistency

### What is Replay Consistency?

A strategy has replay consistency if:

```
Run A: Process bars 0-100 → State S100
Run B: Process bars 0-50, stop, reload, process 51-100 → State S100

State S100 from Run A == State S100 from Run B
```

**Testing replay consistency:**

```bash
python test_resuming_mode.py \
    --start 20250701203204 \
    --end 20250710203204 \
    --midpoint 20250705120000
```

This runs your strategy:
1. Continuously from start to end
2. From start to midpoint, then midpoint to end
3. Compares outputs - **must be identical**

### Achieving Replay Consistency

**Requirements:**

1. **Deterministic computation** (no random, no time-based)
2. **Complete state persistence** (all state in sv_object)
3. **Proper initialization** (from_sv restores exactly)
4. **No external dependencies** (no global state)
5. **Consistent ordering** (process bars in same order)

**Example with replay consistency:**

```python
class ReplayConsistent(pcts3.sv_object):
    """Guaranteed replay consistency."""

    def __init__(self):
        super().__init__()

        # All state persisted
        self.bar_index = 0
        self.ema = 0.0
        self.prev_close = 0.0
        self.signal = 0
        self.initialized = False

    def on_bar(self, bar):
        """Process bar deterministically."""
        close = float(bar.close)

        # Initialize
        if not self.initialized:
            self.ema = close
            self.prev_close = close
            self.initialized = True
            return

        # Deterministic update
        self.ema = 0.1 * close + 0.9 * self.ema

        # Deterministic signal
        if self.ema > self.prev_close * 1.01:
            self.signal = 1
        elif self.ema < self.prev_close * 0.99:
            self.signal = -1
        else:
            self.signal = 0

        # Update for next bar
        self.prev_close = close
        self.bar_index += 1

        # Output deterministically
        if self.bar_index % 12 == 0:  # Every 12 bars
            return [self.copy_to_sv()]
        return []
```

## Replay Consistency Testing Implementation

**Critical**: Running `test_resuming_mode.py` alone is insufficient—you must implement reconciliation in your indicator.

### Two-Run Test Workflow

| Run | Process | State |
|-----|---------|-------|
| **1. Full** | Calculate T_i at each cycle i | System saves T_i to DB |
| **2. Resume** | System pushes T'_i; Calculate T_i | Must assert: T_i == T'_i |

**Requirement**: For each cycle i after rebuilding: `assert T_i == T'_i`

### Rebuilding Phase

**Concept**: Warm-up period before meaningful output (e.g., need 50 bars for EMA calculation).

**Key Insight**: Discrepancies during rebuilding are expected and normal. Reconciliation starts only after `_rebuilding_finished()` returns True.

#### 🚨 CRITICAL: Do NOT Use `bar_index` for Rebuilding Detection

**Common Mistake**:
```python
# ❌ WRONG - Uses persisted bar_index
def _rebuilding_finished(self):
    return self.bar_index >= 165  # WRONG!
```

**Problem**: When resuming from midpoint (e.g., bar 1000), `bar_index` is already 1000, so this immediately returns True even though you haven't accumulated 165 bars since resuming. Rebuilding appears finished when it hasn't even started.

**Correct Approach**:
```python
# ✅ CORRECT - Use non-persisted counter
class Indicator(pcts3.sv_object):
    def __init__(self):
        # Persisted state
        self.bar_index = 0  # Total bars since beginning (persisted)
        self.ema = 0.0

        # Non-persisted rebuilding tracker
        self.bars_since_start = 0  # Bars in THIS run (NOT persisted)
        self.rebuilding = True

    def _rebuilding_finished(self):
        """Check if warm-up complete in THIS run."""
        WARMUP_BARS = 165
        return self.bars_since_start >= WARMUP_BARS  # ✅ Correct

    def on_bar(self, bar):
        self.bar_index += 1  # Persisted: total bars
        self.bars_since_start += 1  # NOT persisted: bars this run

        # Check rebuilding
        if self.rebuilding and self._rebuilding_finished():
            self.rebuilding = False
            logger.info(f"Rebuilding finished after {self.bars_since_start} bars")
```

**Why This Matters**:

| Scenario | `bar_index` | `bars_since_start` | Correct? |
|----------|-------------|-------------------|----------|
| **Fresh start** | 0 → 165 | 0 → 165 | Both work |
| **Resume at bar 1000** | 1000 → 1165 | 0 → 165 | Only `bars_since_start` |

**Key Difference**:
- `self.bar_index`: Persisted in sv_object, tracks total bars across all runs
- `self.bars_since_start`: NOT persisted, resets to 0 each time indicator starts

**Rule**: Use **non-persisted counter** for rebuilding detection to ensure proper warm-up on resume.

#### 🚨 CRITICAL: `from_sv()` vs `super().from_sv()` - Understanding the Difference

**The Two Methods**:

1. **`from_sv(self, sv)`**: Your overridden method with custom logic (caching, rebuilding checks)
2. **`super().from_sv(sv)`**: Parent's method that actually loads state from StructValue

**Common Mistake in `_load_from_sv`**:
```python
# ❌ WRONG - Triggers custom from_sv logic
def _load_from_sv(self, sv: pc.StructValue):
    temp = self.__class__()
    temp.from_sv(sv)  # WRONG! Calls overridden from_sv()
    return temp
```

**Problem**: This calls `temp.from_sv(sv)` which:
1. Caches sv in `temp.latest_sv`
2. Checks if rebuilding is finished
3. Maybe calls `_from_sv()` (or maybe doesn't!)
4. Does NOT directly load the state

**Result**: The temp object may not have state properly loaded, causing reconciliation failures.

**Correct Approach**:
```python
# ✅ CORRECT - Directly calls parent's from_sv
def _load_from_sv(self, sv: pc.StructValue):
    temp = self.__class__()
    # Copy metadata (required for proper deserialization)
    temp.market = self.market
    temp.code = self.code
    temp.meta_id = self.meta_id
    temp.granularity = self.granularity
    temp.namespace = self.namespace

    # Call PARENT's from_sv, bypassing overridden from_sv()
    super(self.__class__, temp).from_sv(sv)  # ✅ Correct
    return temp
```

**Why This Pattern Works**:
- `super(self.__class__, temp)`: Gets parent class proxy for temp object
- `.from_sv(sv)`: Calls parent's from_sv, not temp's overridden version
- Result: State loaded directly without custom logic interference

**Summary Table**:

| Call | What It Does | When to Use |
|------|--------------|-------------|
| `self.from_sv(sv)` | Runs your custom logic (cache, check rebuilding) | Called by framework on resume |
| `super().from_sv(sv)` | Loads state directly from StructValue | Inside `_from_sv()` during rebuilding |
| `temp.from_sv(sv)` | ❌ WRONG: Triggers custom logic on temp object | Never use this in `_load_from_sv` |
| `super(self.__class__, temp).from_sv(sv)` | ✅ Loads state into temp, bypassing custom logic | In `_load_from_sv()` for reconciliation |

### Five-Step Reconciliation Pattern

| Step | Method | Purpose | Implementation |
|------|--------|---------|----------------|
| **1** | `from_sv()` | Cache incoming state | `self.latest_sv = sv` |
| **2** | `_from_sv()` | Restore during rebuilding | `if not _rebuilding_finished(): super().from_sv(sv)` |
| **3** | `_reconcile_state()` | Compare after rebuilding | `saved = _load_from_sv(sv); assert _equal(saved)` |
| **4** | `_load_from_sv()` | Load into temp object | `super(self.__class__, temp).from_sv(sv)` ✅ |
| **5** | `_equal()` | Compare with tolerances | `math.isclose(a, b, abs_tol=1e-6, rel_tol=1e-5)` |

**Float Comparison Tolerances**:
- High precision (indicators): `1e-6, rel=1e-5`
- Medium precision (prices): `1e-3, rel=1e-4`
- Low precision (percentages): `1e-2`
- Integers (counts, regimes): Exact equality `==`

### Complete Pattern Example

```python
class ReplayConsistentIndicator(pcts3.sv_object):
    def __init__(self):
        super().__init__()
        self.meta_name = "ReplayConsistentIndicator"
        self.namespace = pc.namespace_private

        # Persisted indicator state
        self.bar_index = 0  # Total bars (persisted)
        self.ema = 0.0
        self.signal = 0
        self.timetag = 0

        # Non-persisted rebuilding management
        self.bars_since_start = 0  # Bars in THIS run (NOT persisted)
        self.rebuilding = True
        self.latest_sv = None

    def _rebuilding_finished(self):
        """Determine when indicator has enough data in THIS run."""
        WARMUP_BARS = 20
        return self.bars_since_start >= WARMUP_BARS  # ✅ Use non-persisted counter

    def from_sv(self, sv: pc.StructValue):
        """Cache incoming state for reconciliation."""
        self.latest_sv = sv

        if not self._rebuilding_finished():
            self._from_sv()

    def _from_sv(self):
        """Restore state during rebuilding phase."""
        if self.latest_sv is not None:
            logger.info(f"Restoring state at bar {self.bar_index}")
            super().from_sv(self.latest_sv)
            self.latest_sv = None

    def _load_from_sv(self, sv: pc.StructValue):
        """Load saved state into temporary object for reconciliation.

        CRITICAL: Use super(self.__class__, temp).from_sv(sv) to bypass
        the overridden from_sv() and directly load state.
        """
        temp = self.__class__()
        # Copy metadata (required for deserialization)
        temp.market = self.market
        temp.code = self.code
        temp.meta_id = self.meta_id
        temp.granularity = self.granularity
        temp.namespace = self.namespace
        # Call PARENT's from_sv, NOT temp.from_sv(sv)
        super(self.__class__, temp).from_sv(sv)  # ✅ Bypasses custom logic
        return temp

    def _equal(self, other) -> bool:
        """Compare states with proper tolerances."""
        import math

        float_equal = lambda a, b: math.isclose(a, b,
                                                abs_tol=1e-6,
                                                rel_tol=1e-5)

        if other.timetag != self.timetag:
            logger.error(f"Timetag mismatch")
            return False

        if not float_equal(other.ema, self.ema):
            logger.error(f"EMA mismatch: {other.ema} != {self.ema}")
            return False

        if other.signal != self.signal:
            logger.error(f"Signal mismatch")
            return False

        return True

    def _reconcile_state(self):
        """Compare calculated vs saved state after rebuilding."""
        if self.rebuilding or self.latest_sv is None:
            return

        saved = self._load_from_sv(self.latest_sv)
        assert self._equal(saved), \
            f"Replay consistency violation at bar {self.bar_index}"

        self.latest_sv = None

    async def on_bar(self, bar):
        """Process bar with replay consistency verification."""
        self.bar_index += 1  # Persisted: total bars
        self.bars_since_start += 1  # NOT persisted: bars this run
        self.timetag = bar.timetag

        # Calculate state
        close = float(bar.close)
        if self.bar_index == 1:
            self.ema = close
        else:
            alpha = 2.0 / 21.0
            self.ema = alpha * close + (1 - alpha) * self.ema

        self.signal = 1 if self.ema > close else -1

        # Check if rebuilding done
        if self.rebuilding and self._rebuilding_finished():
            self.rebuilding = False
            logger.info(f"Rebuilding finished after {self.bars_since_start} bars "
                       f"(total bar_index: {self.bar_index})")

        # Reconcile after rebuilding
        if not self.rebuilding:
            self._reconcile_state()

        # Output
        if not self.rebuilding:
            return [self.copy_to_sv()]
        return []
```

**Critical**: Test script provides framework; you implement reconciliation logic in indicator code.

## Common Issues and Solutions

| Issue | ❌ Anti-Pattern | ✅ Solution |
|-------|----------------|------------|
| **Growing Memory** | `self.all_prices = []` (unbounded list) | `self.prices = deque(maxlen=100)` (bounded) |
| **Non-Determinism** | `random.random()` (different each run) | `(bar_index * 9301) % 233280` (deterministic) |
| **External State** | `cached_data = {}` (global dict) | `self.ema = 0.0` (instance variable) |
| **Incomplete Persistence** | `prev = close` (local var, not saved) | `self.prev_close = close` (instance var) |
| **Wrong Rebuilding Check** | `bar_index >= 165` (persisted, wrong on resume) | `bars_since_start >= 165` (NOT persisted) |
| **Wrong _load_from_sv** | `temp.from_sv(sv)` (triggers custom logic) | `super(self.__class__, temp).from_sv(sv)` |
| **Wrong _on_cycle_pass Location** | Inside `elif bar.meta == X:` block | Outside all bar type blocks, after timetag check |
| **Missing Deque Constraints** | Deque after `from_sv()` has no maxlen | Re-create: `deque(self.prices, maxlen=100)` |

## Practical Example: Stateless Indicator

**Note**: Complete reference implementation demonstrating all principles. Use as template for new indicators.

```python
#!/usr/bin/env python3
# coding=utf-8
"""Stateless Indicator with Perfect Replay Consistency"""

import math
from collections import deque
import pycaitlyn as pc
import pycaitlynts3 as pcts3
import pycaitlynutils3 as pcu3

# Framework globals
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
    """Parse market data."""
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

class StatelessIndicator(pcts3.sv_object):
    """
    Stateless indicator with replay consistency.

    Uses online algorithms and bounded memory for:
    - EMA calculations
    - Volatility tracking
    - Swing point detection
    - Signal generation
    """

    def __init__(self, market, code):
        super().__init__()

        # Metadata (constants)
        self.meta_name = "StatelessIndicator"
        self.namespace = pc.namespace_private
        self.revision = (1 << 32) - 1
        self.granularity = 900
        self.market = market
        self.code = code

        # State (automatically persisted)
        self.bar_index = 0
        self.timetag = None

        # Online algorithm state
        self.ema_fast = 0.0
        self.ema_slow = 0.0
        self.ema_mean = 0.0
        self.ema_var = 0.0
        self.atr = 0.0

        # Bounded collections
        self.recent_highs = deque(maxlen=20)
        self.recent_lows = deque(maxlen=20)

        # Previous values for calculations
        self.prev_close = 0.0
        self.prev_high = 0.0
        self.prev_low = 0.0

        # Outputs
        self.signal = 0
        self.volatility = 0.0
        self.initialized = False

        # Data parser
        self.quote = SampleQuote()
        self.persistent = True

        # Algorithm parameters (constants)
        self.alpha_fast = 2.0 / 11.0  # 10-period EMA
        self.alpha_slow = 2.0 / 21.0  # 20-period EMA
        self.alpha_vol = 2.0 / 21.0   # Volatility decay

    def initialize(self, imports, metas):
        """Initialize metadata."""
        self.load_def_from_dict(metas)
        self.set_global_imports(imports)
        self.quote.load_def_from_dict(metas)
        self.quote.set_global_imports(imports)

    def on_bar(self, bar: pc.StructValue):
        """Process bar with guaranteed replay consistency."""
        # Filter for our instrument
        if bar.get_market() != self.market or bar.get_stock_code() != self.code:
            return []

        # Parse quote data
        if bar.get_namespace() == self.quote.namespace and \
           bar.get_meta_id() == self.quote.meta_id:
            self.quote.market = self.market
            self.quote.code = self.code
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

            results = []
            if self.bar_index > 0:
                results.append(self.copy_to_sv())

            self.timetag = tm
            self.bar_index += 1
            return results

        return []

    def _on_cycle_pass(self):
        """Process end of cycle with online algorithms."""
        # Extract OHLCV
        close = float(self.quote.close)
        high = float(self.quote.high)
        low = float(self.quote.low)
        volume = float(self.quote.volume)

        # Initialize on first bar
        if not self.initialized:
            self._initialize_state(close, high, low)
            return

        # Update EMAs (online)
        self.ema_fast = self.alpha_fast * close + (1 - self.alpha_fast) * self.ema_fast
        self.ema_slow = self.alpha_slow * close + (1 - self.alpha_slow) * self.ema_slow

        # Update volatility (Welford's algorithm with EMA)
        log_return = math.log(close / self.prev_close) if self.prev_close > 0 else 0
        delta = log_return - self.ema_mean
        self.ema_mean += self.alpha_vol * delta
        self.ema_var += self.alpha_vol * (delta * delta - self.ema_var)
        self.volatility = math.sqrt(max(self.ema_var, 0)) * math.sqrt(252)

        # Update ATR (online)
        self._update_atr(high, low, self.prev_close)

        # Update bounded collections
        self.recent_highs.append(high)
        self.recent_lows.append(low)

        # Generate signal (deterministic)
        self._generate_signal()

        # Update previous values for next bar
        self.prev_close = close
        self.prev_high = high
        self.prev_low = low

        logger.info(f"Bar {self.bar_index}: EMA_fast={self.ema_fast:.2f}, "
                   f"EMA_slow={self.ema_slow:.2f}, Signal={self.signal}")

    def _initialize_state(self, close, high, low):
        """Initialize state on first bar."""
        self.ema_fast = close
        self.ema_slow = close
        self.ema_mean = 0.0
        self.ema_var = 0.01
        self.atr = high - low
        self.prev_close = close
        self.prev_high = high
        self.prev_low = low
        self.initialized = True

    def _update_atr(self, high, low, prev_close):
        """Update ATR using online algorithm."""
        tr1 = high - low
        tr2 = abs(high - prev_close) if prev_close > 0 else 0
        tr3 = abs(low - prev_close) if prev_close > 0 else 0
        tr = max(tr1, tr2, tr3)

        # Online EMA of TR
        self.atr = self.alpha_slow * tr + (1 - self.alpha_slow) * self.atr

    def _generate_signal(self):
        """Generate signal deterministically."""
        # EMA crossover
        ema_cross = 1 if self.ema_fast > self.ema_slow else -1

        # Volatility filter
        high_vol = self.volatility > 0.3

        # Support/resistance from recent highs/lows
        if len(self.recent_highs) >= 20 and len(self.recent_lows) >= 20:
            recent_high = max(self.recent_highs)
            recent_low = min(self.recent_lows)

            # Near resistance
            if self.prev_close > recent_high * 0.995:
                self.signal = -1 if not high_vol else 0
            # Near support
            elif self.prev_close < recent_low * 1.005:
                self.signal = 1 if not high_vol else 0
            else:
                self.signal = ema_cross if not high_vol else 0
        else:
            self.signal = 0

    def from_sv(self, sv):
        """Load state and restore deque constraints."""
        super().from_sv(sv)

        # Restore deque maxlen (not preserved in serialization)
        if not isinstance(self.recent_highs, deque):
            self.recent_highs = deque(self.recent_highs, maxlen=20)
        if not isinstance(self.recent_lows, deque):
            self.recent_lows = deque(self.recent_lows, maxlen=20)

# Global indicator
indicator = StatelessIndicator(b'SHFE', b'cu<00>')

# Framework callbacks
async def on_init():
    global indicator, imports, metas, worker_no
    if worker_no != 0 and metas and imports:
        indicator.initialize(imports, metas)

async def on_ready():
    pass

async def on_bar(bar: pc.StructValue):
    global indicator, worker_no
    if worker_no != 1:
        return []
    return indicator.on_bar(bar)

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

## 🚨 CRITICAL: `_on_cycle_pass()` Calling Pattern for Multi-Source Indicators

### The Problem

When processing bars from multiple sources (e.g., SampleQuote, ZampleQuote, or multiple data feeds), `_on_cycle_pass()` must be called **once per timetag advancement**, not once per specific bar type.

**Wrong Pattern**: Calling inside specific bar processing block
```python
# ❌ WRONG - Tied to specific bar type
def on_bar(self, bar):
    if bar.meta == meta_a:
        # Process type A bars
        pass
    elif bar.meta == meta_b:
        # Process type B bars
        if self.timetag < bar.timetag:
            self._on_cycle_pass()  # WRONG! Only called for type B
            self.timetag = bar.timetag
```

**Problem**:
- `_on_cycle_pass()` only called when bar.meta == meta_b
- If type A bar arrives first with new timetag, cycle pass is skipped
- If type B bar never arrives, cycle pass never called
- Result: Missing cycle completions, incomplete calculations

### The Solution

Call `_on_cycle_pass()` **outside** all bar type processing blocks, triggered by timetag advancement, and only when data preparation is complete.

**Correct Pattern**: Timetag check independent of bar type
```python
# ✅ CORRECT - Timetag check is independent
def on_bar(self, bar):
    # Process different bar types
    if bar.meta == meta_a:
        # Process type A bars
        self.data_a_ready = True
        pass
    elif bar.meta == meta_b:
        # Process type B bars
        self.data_b_ready = True
        pass

    # Check timetag advancement (independent of bar type)
    if self.timetag < bar.timetag:
        # Only call if all required data is ready
        if self._preparation_is_ready():  # e.g., both A and B ready
            self._on_cycle_pass()
        # Always update timetag
        self.timetag = bar.timetag
```

### Implementation Requirements

**1. Timetag Check Must Be Outside Bar Type Logic**

| Aspect | ❌ Wrong | ✅ Correct |
|--------|---------|-----------|
| **Location** | Inside `elif bar.meta == X` block | Outside all bar type blocks |
| **Trigger** | Specific bar type arrival | Any bar with new timetag |
| **Reliability** | Depends on bar type order | Works regardless of order |

**2. Data Preparation Readiness Check**

```python
def _preparation_is_ready(self) -> bool:
    """Check if all required data sources are ready for this cycle.

    Returns:
        bool: True if all data needed for cycle pass is available
    """
    # Example: Need both quote data and zample data
    return (self.quote_data_ready and
            self.zample_data_ready)
```

**3. Reset Readiness Flags After Cycle Pass**

```python
def _on_cycle_pass(self):
    """Execute end-of-cycle logic when all data is ready."""
    # Your cycle pass logic here
    # ... calculations ...

    # Reset readiness flags for next cycle
    self.quote_data_ready = False
    self.zample_data_ready = False
```

### Complete Pattern Example

```python
class MultiSourceIndicator(pcts3.sv_object):
    def __init__(self):
        super().__init__()
        self.timetag = 0

        # Data readiness tracking (NOT persisted)
        self.quote_data_ready = False
        self.zample_data_ready = False

        # Persisted state
        self.ema = 0.0
        self.signal = 0

    def on_bar(self, bar):
        """Process bars from multiple sources."""
        # Route by meta type
        if bar.meta_id == self.quote.meta_id:
            # Process SampleQuote data
            self.quote.from_sv(bar)
            close = float(self.quote.close)
            self.ema = 0.1 * close + 0.9 * self.ema
            self.quote_data_ready = True

        elif bar.meta_id == self.zample.meta_id:
            # Process ZampleQuote data
            self.zample.from_sv(bar)
            volume = float(self.zample.volume)
            # ... process volume ...
            self.zample_data_ready = True

        # Timetag advancement check (OUTSIDE bar type logic)
        if self.timetag < bar.timetag:
            # Only pass cycle if all data ready
            if self._preparation_is_ready():
                self._on_cycle_pass()

            # Always update timetag
            self.timetag = bar.timetag

        return []  # Output in _on_cycle_pass()

    def _preparation_is_ready(self) -> bool:
        """Check if all required data sources have arrived."""
        return self.quote_data_ready and self.zample_data_ready

    def _on_cycle_pass(self):
        """Execute when cycle completes with all data ready."""
        # Generate signal
        self.signal = 1 if self.ema > 100.0 else -1

        # Reset for next cycle
        self.quote_data_ready = False
        self.zample_data_ready = False
```

### Why This Matters

| Scenario | Wrong Pattern | Correct Pattern |
|----------|--------------|-----------------|
| **Type A arrives first with new timetag** | Cycle pass skipped | Cycle pass triggered |
| **Type B never arrives** | Stuck waiting forever | Can check readiness, handle gracefully |
| **Bar arrival order varies** | Inconsistent behavior | Consistent behavior |
| **Missing data source** | Silent failure | Detectable via readiness check |

### Key Principle

**Timetag advancement and data readiness are orthogonal concerns:**
- **Timetag check**: "Has time moved forward?" (always check)
- **Readiness check**: "Is required data available?" (conditional cycle pass)
- **Both must be checked**, but independently

**Rule**: Never tie `_on_cycle_pass()` to specific bar types—tie it to timetag advancement + data readiness.

## Summary

This chapter covered:

1. **Stateless Design**: Strategies that produce identical results regardless of when stopped/resumed
2. **Online Algorithms**: Using bounded memory (EMA, Welford's variance, ATR calculations)
3. **State Persistence**: Minimal, complete state in sv_object
4. **Replay Consistency Testing**: Implementing reconciliation to verify T_i == T'_i
5. **The Rebuilding Phase**: Warm-up period before reconciliation starts
6. **Reconciliation Pattern**: Cache, restore, compare with proper tolerances
7. **Cycle Pass Pattern**: `_on_cycle_pass()` outside bar type logic, triggered by timetag + readiness
8. **Common Issues**: Growing memory, non-determinism, external state, wrong cycle pass location

**Key Takeaways:**

- Use online algorithms for O(1) memory
- Store only necessary state in sv_object
- Make computation deterministic (no random, no external state)
- **🚨 CRITICAL #1**: Use `bars_since_start` (NOT persisted) for rebuilding detection, NOT `bar_index`
- **🚨 CRITICAL #2**: In `_load_from_sv()`, use `super(self.__class__, temp).from_sv(sv)`, NOT `temp.from_sv(sv)`
- **🚨 CRITICAL #3**: Place `_on_cycle_pass()` outside bar type logic, check timetag advancement independently
- Understand difference: `from_sv()` = custom logic, `super().from_sv()` = actual state loading
- Multi-source pattern: Check readiness before calling `_on_cycle_pass()`, reset flags after
- Implement reconciliation methods: `_load_from_sv()`, `_equal()`, `_reconcile_state()`
- Cache incoming state in `from_sv()`, restore during rebuilding in `_from_sv()`
- Only assert equality after rebuilding finishes
- Use `math.isclose()` with appropriate tolerances for float comparison
- Test with test_resuming_mode.py (but implement assertions in your code!)
- Use deque with maxlen for bounded collections
- Restore deque constraints after from_sv()
- **Critical**: `test_resuming_mode.py` only provides the framework—you must implement reconciliation

**Next Steps:**

In the next chapter, we'll explore backtesting - running your stateless strategies through historical data to validate performance and tune parameters.

---

**Previous:** [04 - StructValue and sv_object](04-structvalue-and-sv_object.md) | **Next:** [06 - Backtest and Testing](06-backtest.md)
