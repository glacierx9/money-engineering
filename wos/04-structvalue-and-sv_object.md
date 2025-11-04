# Chapter 4: StructValue and sv_object

**Learning objectives:**
- Understand StructValue as the framework's data container
- Master sv_object pattern for automatic serialization
- Learn metadata management and routing
- Use from_sv() and copy_to_sv() effectively

**Previous:** [03 - Programming Basics and CLI](03-programming-basics-and-cli.md) | **Next:** [05 - Stateless Design](05-stateless-design.md)

---

## Overview

StructValue is the fundamental data container in the Wolverine framework. Every piece of market data, indicator output, and strategy signal flows through the system as a StructValue. The `sv_object` class provides automatic serialization, allowing your strategy's state to be saved and restored transparently.

This chapter explains how these concepts work together to enable stateless,resumable strategies.

## StructValue: The Universal Data Container

### What is a StructValue?

A StructValue is a structured data container that wraps:
- **Metadata**: Market, instrument code, timestamp, granularity
- **Data Type**: Namespace and meta_id identifying the data structure
- **Field Values**: The actual data (OHLCV, indicators, signals, etc.)

Think of it as a self-describing data packet that flows through the system.

```python
import pycaitlyn as pc

# StructValue contains:
# 1. Metadata (who, what, when, where)
# 2. Type information (namespace, meta_id)
# 3. Field values (the actual data)

def inspect_struct_value(sv: pc.StructValue):
    """Inspect a StructValue's metadata."""

    # WHO: Which market and instrument
    market = sv.get_market()           # e.g., b'SHFE'
    code = sv.get_stock_code()         # e.g., b'cu<00>'

    # WHEN: Timestamp
    time_tag = sv.get_time_tag()       # Milliseconds since epoch
    time_string = ts_parse(time_tag)   # Human-readable

    # WHAT: Data type
    namespace = sv.get_namespace()     # Global or private
    meta_id = sv.get_meta_id()         # Type identifier

    # WHERE: Timeframe
    granularity = sv.get_granularity() # Seconds (900 = 15min)

    print(f"Market: {market.decode()}")
    print(f"Code: {code.decode()}")
    print(f"Time: {time_string}")
    print(f"Namespace: {'global' if namespace == pc.namespace_global else 'private'}")
    print(f"Meta ID: {meta_id}")
    print(f"Granularity: {granularity}s")
```

### Metadata Fields

**Core Metadata:**

| Method | Returns | Description |
|--------|---------|-------------|
| `get_market()` | `bytes` | Market identifier (e.g., `b'SHFE'`, `b'DCE'`) |
| `get_stock_code()` | `bytes` | Instrument code (e.g., `b'cu<00>'`, `b'i2501'`) |
| `get_time_tag()` | `int` | Timestamp in milliseconds |
| `get_granularity()` | `int` | Bar size in seconds |
| `get_namespace()` | `int` | `pc.namespace_global` or `pc.namespace_private` |
| `get_meta_id()` | `int` | Type identifier for routing |

**Example - Routing Based on Metadata:**

```python
def route_struct_value(sv: pc.StructValue):
    """Route StructValue to appropriate handler."""

    namespace = sv.get_namespace()
    meta_id = sv.get_meta_id()
    market = sv.get_market()
    code = sv.get_stock_code()

    # Route to price data handler
    if namespace == pc.namespace_global:
        if self.quote.meta_id == meta_id:
            return self.handle_price_data(sv)

    # Route to indicator data handler
    elif namespace == pc.namespace_private:
        if self.indicator.meta_id == meta_id:
            return self.handle_indicator_data(sv)

    # Unknown type
    return None
```

### Namespaces

Two namespaces separate different data domains:

**1. Global Namespace (pc.namespace_global)**

Public, shared data available to all strategies:
- Raw market data (SampleQuote, ZampleQuote)
- Exchange reference data
- Tick data
- Order book data

```python
class SampleQuote(pcts3.sv_object):
    """Global namespace for market data."""
    def __init__(self):
        super().__init__()
        self.namespace = pc.namespace_global  # Public data
        self.meta_name = "SampleQuote"
```

**2. Private Namespace (pc.namespace_private)**

Strategy-specific data, isolated per strategy:
- Custom indicators
- Trading signals
- Strategy state
- Portfolio metrics

```python
class MyIndicator(pcts3.sv_object):
    """Private namespace for custom data."""
    def __init__(self):
        super().__init__()
        self.namespace = pc.namespace_private  # Private to this strategy
        self.meta_name = "MyIndicator"
```

## sv_object: Automatic Serialization

The `sv_object` class provides automatic state persistence. All instance variables are automatically saved to and loaded from StructValue.

### Basic sv_object Pattern

```python
import pycaitlynts3 as pcts3
import pycaitlyn as pc

class MyIndicator(pcts3.sv_object):
    """
    sv_object automatically serializes all instance variables.
    No manual save/load code needed!
    """

    def __init__(self):
        super().__init__()

        # METADATA (set once, never change)
        self.meta_name = "MyIndicator"
        self.namespace = pc.namespace_private
        self.revision = (1 << 32) - 1  # Latest version
        self.granularity = 900
        self.market = b'SHFE'
        self.code = b'cu<00>'

        # STATE (automatically persisted)
        self.bar_index = 0
        self.ema_value = 0.0
        self.total_volume = 0
        self.signal_strength = 0.0

        # Control serialization
        self.persistent = True  # Enable auto-persistence

    def on_bar(self, bar: pc.StructValue):
        """Process bar and return serialized state."""

        # Update state
        self.bar_index += 1
        self.ema_value = self.calculate_ema()

        # Serialize to StructValue (automatic!)
        return [self.copy_to_sv()]
```

**What gets serialized:**
- All instance variables (primitives, lists, dicts)
- Arrays (numpy arrays, lists)
- Nested objects (if also sv_objects)

**What doesn't get serialized:**
- Variables starting with `_` (private)
- Objects without serialization support
- Set `persistent = False` to disable

### from_sv() - Loading State

Load state from a StructValue:

```python
class MyIndicator(pcts3.sv_object):
    def load_historical_state(self, sv: pc.StructValue):
        """Load state from StructValue."""

        # Must set metadata BEFORE from_sv()
        self.market = sv.get_market()
        self.code = sv.get_stock_code()
        self.granularity = sv.get_granularity()

        # Parse StructValue into instance variables
        self.from_sv(sv)

        # Now all fields are populated
        print(f"Loaded bar_index: {self.bar_index}")
        print(f"Loaded ema_value: {self.ema_value}")
```

**Critical Pattern - Set Metadata First:**

```python
# ✅ CORRECT - Set metadata before from_sv()
parser = SampleQuote()
parser.market = bar.get_market()
parser.code = bar.get_stock_code()
parser.granularity = bar.get_granularity()
parser.from_sv(bar)  # Now it knows how to parse

# ❌ WRONG - Missing metadata
parser = SampleQuote()
parser.from_sv(bar)  # ERROR! No market/code/granularity set
```

### copy_to_sv() - Saving State

Serialize current state to StructValue:

```python
class MyIndicator(pcts3.sv_object):
    def serialize_output(self):
        """Serialize current state to StructValue."""

        # Update state
        self.bar_index += 1
        self.signal_strength = self.calculate_signal()

        # Create StructValue with current state
        sv = self.copy_to_sv()

        return sv
```

**The sv contains:**
- All metadata (market, code, timestamp, etc.)
- All instance variables serialized as fields
- Ready to be sent to other strategies or persisted

## Metadata Management

### Loading Metadata Definitions

Every sv_object needs metadata definitions to know:
- Which fields exist
- Field types (int, double, string, etc.)
- Field order
- Serialization format

```python
async def on_init():
    """Initialize with metadata definitions."""
    global strategy, imports, metas

    if metas and imports:
        # Load metadata definitions
        strategy.load_def_from_dict(metas)

        # Set up imports for dependencies
        strategy.set_global_imports(imports)

        # Initialize any nested sv_objects
        strategy.price_data.load_def_from_dict(metas)
        strategy.price_data.set_global_imports(imports)
```

### The metas Dictionary

The `metas` dictionary maps metadata keys to definitions:

```python
# metas structure
metas = {
    (namespace, meta_name): [
        schema,          # Field definitions
        namespace,       # Namespace
        meta_name,       # Name
        granularity,     # Granularity
        market,          # Market
        meta_id,         # Type ID
        code,            # Instrument
        revision         # Version
    ],
    # More definitions...
}

# Example lookup
key = (pc.namespace_global, "SampleQuote")
if key in metas:
    meta_def = metas[key]
    meta_id = meta_def[5]  # Extract meta_id
```

### The imports Dictionary

The `imports` dictionary defines data dependencies:

```python
# imports structure (from uin.json)
imports = {
    "global": {
        "SampleQuote": {
            "fields": ["open", "high", "low", "close", "volume", "turnover"],
            "granularities": [300, 900],
            "markets": ["SHFE", "DCE"],
            "securities": [["cu", "al"], ["i", "j"]]
        }
    },
    "private": {
        "MyIndicator": {
            "fields": ["signal_strength", "confidence"],
            # ...
        }
    }
}
```

## Multiple sv_objects Pattern

Common pattern: Create separate sv_objects for parsing different data types.

```python
class SampleQuote(pcts3.sv_object):
    """Parser for global market data."""
    def __init__(self):
        super().__init__()
        self.meta_name = "SampleQuote"
        self.namespace = pc.namespace_global
        self.revision = (1 << 32) - 1
        # OHLCV fields
        self.open = None
        self.high = None
        self.low = None
        self.close = None
        self.volume = None
        self.turnover = None

class ZampleQuote(pcts3.sv_object):
    """Parser for private sampled data."""
    def __init__(self):
        super().__init__()
        self.meta_name = "ZampleQuote"
        self.namespace = pc.namespace_private
        self.revision = (1 << 32) - 1
        # Same fields as SampleQuote
        self.open = None
        self.high = None
        self.low = None
        self.close = None
        self.volume = None
        self.turnover = None

class MyIndicator(pcts3.sv_object):
    """Main indicator uses parsers."""
    def __init__(self):
        super().__init__()
        self.meta_name = "MyIndicator"
        self.namespace = pc.namespace_private

        # Create parsers for different data sources
        self.sq = SampleQuote()
        self.zq = ZampleQuote()

        # State
        self.ema = 0.0

    def initialize(self, imports, metas):
        """Initialize all sv_objects."""
        self.load_def_from_dict(metas)
        self.set_global_imports(imports)

        # Initialize parsers
        self.sq.load_def_from_dict(metas)
        self.sq.set_global_imports(imports)
        self.zq.load_def_from_dict(metas)
        self.zq.set_global_imports(imports)

    def on_bar(self, bar: pc.StructValue):
        """Route to appropriate parser."""
        ns = bar.get_namespace()
        meta_id = bar.get_meta_id()

        # Parse with SampleQuote
        if ns == self.sq.namespace and meta_id == self.sq.meta_id:
            self.sq.market = bar.get_market()
            self.sq.code = bar.get_stock_code()
            self.sq.granularity = bar.get_granularity()
            self.sq.from_sv(bar)
            self.process_price_data(self.sq)

        # Parse with ZampleQuote
        elif ns == self.zq.namespace and meta_id == self.zq.meta_id:
            self.zq.market = bar.get_market()
            self.zq.code = bar.get_stock_code()
            self.zq.granularity = bar.get_granularity()
            self.zq.from_sv(bar)
            self.process_sampled_data(self.zq)

        return []

    def process_price_data(self, quote):
        """Process parsed market data."""
        close = float(quote.close)
        self.ema = 0.1 * close + 0.9 * self.ema
```

## Data Access Patterns

### Accessing Field Values

After parsing with `from_sv()`, access fields directly:

```python
class SampleQuote(pcts3.sv_object):
    def __init__(self):
        super().__init__()
        self.open = None
        self.high = None
        self.low = None
        self.close = None
        self.volume = None
        self.turnover = None

# Usage
quote = SampleQuote()
quote.from_sv(bar)

# Direct field access - NO FALLBACK VALUES
current_price = float(quote.close)  # Trust the data
high_price = float(quote.high)
volume = float(quote.volume)

# Calculate with confidence
atr = high_price - float(quote.low)
```

**CRITICAL: No Fallback Logic (DOCTRINE 2)**

Never use fallback values for dependency data:

```python
# ❌ WRONG - Fallback masks data issues
current_price = float(quote.close) if quote.close is not None else 100.0

# ✅ CORRECT - Trust dependency data completely
current_price = float(quote.close)  # Will raise if None - GOOD!
```

### Working with Arrays

sv_object supports arrays and lists:

```python
class PortfolioState(pcts3.sv_object):
    def __init__(self):
        super().__init__()
        self.meta_name = "PortfolioState"
        self.namespace = pc.namespace_private

        # Arrays automatically serialized
        self.signals = []        # List of signals
        self.leverages = []      # List of leverages
        self.markets = []        # List of markets
        self.codes = []          # List of codes
        self.prices = []         # List of prices

    def add_position(self, signal, leverage, market, code, price):
        """Add position to arrays."""
        self.signals.append(signal)
        self.leverages.append(leverage)
        self.markets.append(market)
        self.codes.append(code)
        self.prices.append(price)

    def serialize(self):
        """Arrays automatically included in StructValue."""
        return self.copy_to_sv()
```

## Practical Examples

### Example 1: Simple Price Parser

```python
class PriceParser(pcts3.sv_object):
    """Parse and store recent prices."""

    def __init__(self):
        super().__init__()
        self.meta_name = "SampleQuote"
        self.namespace = pc.namespace_global
        self.revision = (1 << 32) - 1

        # Fields matching SampleQuote
        self.open = None
        self.high = None
        self.low = None
        self.close = None
        self.volume = None
        self.turnover = None

    def parse_bar(self, bar: pc.StructValue) -> dict:
        """Parse bar and return OHLCV dict."""

        # Set metadata
        self.market = bar.get_market()
        self.code = bar.get_stock_code()
        self.granularity = bar.get_granularity()

        # Parse
        self.from_sv(bar)

        # Extract values
        return {
            'open': float(self.open),
            'high': float(self.high),
            'low': float(self.low),
            'close': float(self.close),
            'volume': float(self.volume),
            'turnover': float(self.turnover)
        }
```

### Example 2: Multi-Timeframe Indicator

```python
class MultiTimeframeIndicator(pcts3.sv_object):
    """Indicator with multiple data sources."""

    def __init__(self, market, code):
        super().__init__()

        # Metadata
        self.meta_name = "MTFIndicator"
        self.namespace = pc.namespace_private
        self.revision = (1 << 32) - 1
        self.granularity = 900
        self.market = market
        self.code = code

        # Parsers for different timeframes
        self.quote_5min = SampleQuote()
        self.quote_15min = SampleQuote()
        self.quote_1h = ZampleQuote()

        # State
        self.ema_5min = 0.0
        self.ema_15min = 0.0
        self.ema_1h = 0.0
        self.signal = 0

    def initialize(self, imports, metas):
        """Initialize all parsers."""
        self.load_def_from_dict(metas)
        self.set_global_imports(imports)

        # Initialize parsers
        for parser in [self.quote_5min, self.quote_15min, self.quote_1h]:
            parser.load_def_from_dict(metas)
            parser.set_global_imports(imports)

    def on_bar(self, bar: pc.StructValue):
        """Process bars from multiple timeframes."""
        ns = bar.get_namespace()
        meta_id = bar.get_meta_id()
        granularity = bar.get_granularity()

        # Route to correct parser
        if granularity == 300:  # 5min
            parser = self.quote_5min
        elif granularity == 900:  # 15min
            parser = self.quote_15min
        elif granularity == 3600:  # 1h
            parser = self.quote_1h
        else:
            return []

        # Parse if correct type
        if ns == parser.namespace and meta_id == parser.meta_id:
            parser.market = self.market
            parser.code = self.code
            parser.granularity = granularity
            parser.from_sv(bar)

            # Update corresponding EMA
            close = float(parser.close)
            if granularity == 300:
                self.ema_5min = 0.1 * close + 0.9 * self.ema_5min
            elif granularity == 900:
                self.ema_15min = 0.05 * close + 0.95 * self.ema_15min
            elif granularity == 3600:
                self.ema_1h = 0.02 * close + 0.98 * self.ema_1h

            # Generate signal from multi-timeframe alignment
            if self.ema_5min > self.ema_15min > self.ema_1h:
                self.signal = 1
            elif self.ema_5min < self.ema_15min < self.ema_1h:
                self.signal = -1
            else:
                self.signal = 0

            # Output on main timeframe (15min)
            if granularity == 900:
                return [self.copy_to_sv()]

        return []
```

### Example 3: Composite Signal Parser

```python
class TierOneSignal(pcts3.sv_object):
    """Parser for Tier 1 indicator signals."""

    def __init__(self):
        super().__init__()
        self.meta_name = "MyIndicator"
        self.namespace = pc.namespace_private
        self.revision = (1 << 32) - 1

        # Signal fields
        self.signal_strength = 0.0
        self.confidence = 0.0
        self.regime = 0
        self.trend_strength = 0.0

class TierTwoComposite(pcts3.sv_object):
    """Composite strategy aggregating Tier 1 signals."""

    def __init__(self):
        super().__init__()
        self.meta_name = "CompositeStrategy"
        self.namespace = pc.namespace_private
        self.revision = (1 << 32) - 1

        # Signal parsers for each instrument
        self.signal_parsers = {}

        # Create parsers for each commodity
        for market, codes in [('SHFE', ['cu', 'al']), ('DCE', ['i', 'j'])]:
            for code in codes:
                key = (market, code)
                parser = TierOneSignal()
                parser.market = market.encode()
                parser.code = (code + '<00>').encode()
                self.signal_parsers[key] = parser

        # Aggregated state
        self.aggregate_signal = 0.0
        self.active_positions = 0

    def initialize(self, imports, metas):
        """Initialize all parsers."""
        self.load_def_from_dict(metas)
        self.set_global_imports(imports)

        for parser in self.signal_parsers.values():
            parser.load_def_from_dict(metas)
            parser.set_global_imports(imports)

    def on_bar(self, bar: pc.StructValue):
        """Process Tier 1 signals."""
        market = bar.get_market()
        code = bar.get_stock_code()

        # Find matching parser
        for (m, c), parser in self.signal_parsers.items():
            if market == m.encode() and code.startswith(c.encode()):
                if bar.get_meta_id() == parser.meta_id:
                    # Parse signal
                    parser.from_sv(bar)

                    # Process signal
                    if abs(parser.signal_strength) > 0.5 and parser.confidence > 0.6:
                        self.aggregate_signal += parser.signal_strength
                        self.active_positions += 1

                    break

        return []
```

## Common Patterns and Best Practices

### ✅ Good Patterns

**1. Clean Parser Separation:**

```python
# Separate parsers for different data types
class MyStrategy(pcts3.sv_object):
    def __init__(self):
        super().__init__()
        self.price_parser = SampleQuote()
        self.signal_parser = OtherStrategySignal()

    def initialize(self, imports, metas):
        self.load_def_from_dict(metas)
        self.price_parser.load_def_from_dict(metas)
        self.signal_parser.load_def_from_dict(metas)
```

**2. Metadata Before Parsing:**

```python
# Always set metadata before from_sv()
parser.market = bar.get_market()
parser.code = bar.get_stock_code()
parser.granularity = bar.get_granularity()
parser.from_sv(bar)  # Now safe
```

**3. Trust Dependency Data:**

```python
# No fallback logic - trust the source
price = float(quote.close)
volume = float(quote.volume)
high = float(quote.high)
```

### ❌ Anti-Patterns

**1. Missing Metadata:**

```python
# ❌ WRONG - No metadata set
parser = SampleQuote()
parser.from_sv(bar)  # ERROR!
```

**2. Fallback Values:**

```python
# ❌ WRONG - Masks data issues
price = float(quote.close) if quote.close else 100.0

# ✅ CORRECT - Fail fast
price = float(quote.close)  # Raises if None
```

**3. Modifying Metadata:**

```python
# ❌ WRONG - Never change metadata
def on_bar(self, bar):
    self.market = bar.get_market()  # WRONG!
    self.code = bar.get_stock_code()  # WRONG!
```

### 🔄 Advanced Pattern: Vector-to-Scalar Serialization

**Critical Concept:** When using lists/vectors internally but storing as separate scalar fields in uout.json.

**The Challenge:**

You might calculate similar values with different parameters using a vector:
```python
# Internal representation - convenient for calculations
self.ema_values: List[float] = [0.0] * 5  # 5 different EMA periods
```

But uout.json defines them as separate fields:
```json
{
  "fields": ["ema_1", "ema_2", "ema_3", "ema_4", "ema_5"],
  "defs": [
    {"field_type": "double", "precision": 6},
    {"field_type": "double", "precision": 6},
    {"field_type": "double", "precision": 6},
    {"field_type": "double", "precision": 6},
    {"field_type": "double", "precision": 6}
  ]
}
```

**The Solution: Custom Serialization**

You **MUST** override `copy_to_sv()` and `from_sv()` to handle the conversion:

```python
class MultiPeriodIndicator(pcts3.sv_object):
    """Indicator with multiple periods calculated via vector"""

    def __init__(self):
        super().__init__()
        self.meta_name = "MultiPeriodIndicator"
        self.namespace = pc.namespace_private
        self.revision = (1 << 32) - 1

        # Internal vector representation (for calculation)
        self.ema_values: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.periods = [10, 20, 50, 100, 200]

        # Scalar fields for serialization (match uout.json)
        self.ema_1 = 0.0
        self.ema_2 = 0.0
        self.ema_3 = 0.0
        self.ema_4 = 0.0
        self.ema_5 = 0.0

    def _calculate_emas(self, price: float):
        """Update all EMA values using vector"""
        for i, period in enumerate(self.periods):
            alpha = 2.0 / (period + 1)
            self.ema_values[i] = alpha * price + (1 - alpha) * self.ema_values[i]

    def copy_to_sv(self) -> pc.StructValue:
        """
        Convert vector to scalar fields for serialization

        CRITICAL: This is called when saving state
        """
        # Vector → Scalars (for output)
        self.ema_1 = self.ema_values[0]
        self.ema_2 = self.ema_values[1]
        self.ema_3 = self.ema_values[2]
        self.ema_4 = self.ema_values[3]
        self.ema_5 = self.ema_values[4]

        # Call parent to serialize scalar fields
        return super().copy_to_sv()

    def from_sv(self, sv: pc.StructValue):
        """
        Reconstruct vector from scalar fields when resuming

        CRITICAL: Must be inverse of copy_to_sv()
        """
        # Call parent to deserialize scalar fields
        super().from_sv(sv)

        # Scalars → Vector (reconstruct internal state)
        self.ema_values[0] = self.ema_1
        self.ema_values[1] = self.ema_2
        self.ema_values[2] = self.ema_3
        self.ema_values[3] = self.ema_4
        self.ema_values[4] = self.ema_5
```

**Why This Matters:**

1. **Replay Consistency**: When framework resumes from midpoint:
   - It deserializes using `from_sv()`
   - Your internal vector MUST be correctly reconstructed
   - Otherwise calculations will be wrong

2. **State Persistence**: Every cycle:
   - Framework calls `copy_to_sv()` to save state
   - Your scalars must match current vector values
   - Otherwise state is lost

3. **Inverse Requirement**:
   ```python
   # This MUST be true:
   obj1.copy_to_sv()  # Vector → Scalars
   sv = obj1.copy_to_sv()
   obj2.from_sv(sv)   # Scalars → Vector
   # obj2.ema_values == obj1.ema_values  # MUST be equal!
   ```

**Complete Example:**

```python
def on_bar(self, bar: pc.StructValue) -> List[pc.StructValue]:
    ret = []

    market = bar.get_market()
    code = bar.get_stock_code()
    ns = bar.get_namespace()
    meta_id = bar.get_meta_id()

    if self.sq.namespace == ns and self.sq.meta_id == meta_id:
        if code.endswith(b'<00>'):
            self.sq.market = market
            self.sq.code = code
            self.sq.granularity = bar.get_granularity()
            self.sq.from_sv(bar)

            # Calculate using internal vector
            price = float(self.sq.close)
            self._calculate_emas(price)  # Updates self.ema_values

            # Generate signal based on vector values
            if self.ema_values[0] > self.ema_values[1] > self.ema_values[2]:
                self.signal = 1

            # Serialize - copy_to_sv() converts vector to scalars
            if self.ready_to_serialize():
                ret.append(self.copy_to_sv())  # Internally: vector → scalars

    return ret
```

**Testing Your Implementation:**

```python
# Test serialization round-trip
def test_serialization():
    obj1 = MultiPeriodIndicator()
    obj1.initialize(imports, metas)

    # Set some test values
    obj1.ema_values = [100.5, 101.2, 102.3, 103.1, 104.5]

    # Serialize
    sv = obj1.copy_to_sv()

    # Deserialize into new object
    obj2 = MultiPeriodIndicator()
    obj2.initialize(imports, metas)
    obj2.from_sv(sv)

    # Verify vectors match
    assert obj1.ema_values == obj2.ema_values  # MUST pass!
    print("✅ Serialization round-trip successful")
```

**Common Mistakes:**

❌ **Forgetting to override copy_to_sv():**
```python
# Vector values calculated but never saved!
def _calculate_emas(self, price):
    self.ema_values[i] = alpha * price + (1 - alpha) * self.ema_values[i]
# Missing: Vector → Scalars in copy_to_sv()
# Result: All EMA fields output as 0.0
```

❌ **Forgetting to override from_sv():**
```python
# Scalars loaded but vector not reconstructed!
# Result: Calculations use stale vector values on resume
```

❌ **Asymmetric conversion:**
```python
def copy_to_sv(self):
    self.ema_1 = self.ema_values[0]
    self.ema_2 = self.ema_values[1]
    # Missing ema_3, ema_4, ema_5!
    return super().copy_to_sv()

def from_sv(self, sv):
    super().from_sv(sv)
    self.ema_values[0] = self.ema_1
    self.ema_values[1] = self.ema_2
    self.ema_values[2] = self.ema_3  # Loaded but never saved!
    self.ema_values[3] = self.ema_4  # Loaded but never saved!
    self.ema_values[4] = self.ema_5  # Loaded but never saved!
# Result: Replay inconsistency - different values on resume!
```

**Key Principle:**

> **from_sv() must be the exact inverse of copy_to_sv()**
>
> Any transformation applied in copy_to_sv() must be reversed in from_sv()

## Summary

This chapter covered:

1. **StructValue**: Universal data container with metadata
2. **sv_object**: Automatic serialization class
3. **Metadata Management**: Loading and routing with metas/imports
4. **from_sv()**: Loading state from StructValue
5. **copy_to_sv()**: Saving state to StructValue
6. **Multiple sv_objects**: Parser pattern for different data types
7. **Vector-to-Scalar Serialization**: Custom copy_to_sv()/from_sv() for internal vectors
8. **Best Practices**: Trust data, set metadata, separate parsers

**Key Takeaways:**

- StructValue wraps metadata + type info + field values
- sv_object provides automatic state persistence
- Always set metadata before from_sv()
- Trust dependency data - no fallback logic
- Use separate sv_objects for different data types
- Initialize all sv_objects with load_def_from_dict()
- **CRITICAL**: Override copy_to_sv() and from_sv() when using vectors internally
- **CRITICAL**: from_sv() must be the exact inverse of copy_to_sv()

**Next Steps:**

In the next chapter, we'll explore the Singularity principle - designing stateless strategies that can be stopped and resumed at any point without losing consistency.

---

**Previous:** [03 - Programming Basics and CLI](03-programming-basics-and-cli.md) | **Next:** [05 - Stateless Design](05-stateless-design.md)
