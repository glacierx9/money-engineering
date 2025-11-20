# Chapter 3: Programming Basics and CLI

**Learning objectives:**
- Understand Wolverine's base classes and framework architecture
- Master calculator3_test.py CLI for running strategies
- Learn required module structure and callbacks
- Use framework globals effectively

**Previous:** [02 - uin.json and uout.json](02-uin-and-uout.md) | **Next:** [04 - StructValue and sv_object](04-structvalue-and-sv_object.md)

---

## Overview

The Wolverine framework provides base classes and a testing CLI that together form the foundation for building trading strategies. This chapter covers the programming patterns, CLI usage, and required framework integration points that every strategy must implement.

## Base Classes Architecture

### The Foundation: pcts3.sv_object

Every indicator in Wolverine extends `pcts3.sv_object`, which provides:

```python
import pycaitlynts3 as pcts3
import pycaitlyn as pc

class MyIndicator(pcts3.sv_object):
    """Base class for all indicators provides automatic serialization."""

    def __init__(self):
        super().__init__()

        # Metadata (REQUIRED - set these as constants)
        self.meta_name = "MyIndicator"      # Must match uout.json export name
        self.namespace = pc.namespace_private  # Or pc.namespace_global
        self.granularity = 900              # Bar size in seconds (15 minutes)
        self.market = b'SHFE'               # Market identifier
        self.code = b'cu<00>'               # Instrument code
        self.revision = (1 << 32) - 1       # Version number

        # State variables (automatically persisted)
        self.bar_index = 0
        self.ema_value = 0.0
        self.signal_strength = 0.0

        # Control persistence
        self.persistent = True               # Enable automatic state persistence
```

**Key concepts:**

1. **Automatic Serialization**: All instance variables are automatically saved/loaded
2. **Metadata Constants**: Meta fields should be set once and never changed
3. **State Persistence**: Framework handles save/load via `copy_to_sv()` and `from_sv()`

### Strategy Base Classes

For Tier 2 (composite) and Tier 3 (execution) strategies:

```python
import strategyc3 as sc3
import composite_strategyc3 as csc3

# Tier 2: Composite Strategy (portfolio management)
class MyComposite(csc3.composite_strategy):
    def __init__(self, initial_money=10000000.0, basket_count=10):
        super().__init__(initial_money, basket_count)

        # Portfolio state arrays
        self.strategies = []      # Individual basket strategies
        self.keys = []           # Basket identifiers
        self.signals = []        # Current signals
        self.leverages = []      # Leverage per basket
        self.capitals = []       # Capital per basket

# Tier 3: Execution Strategy (order management)
class MyExecutor(sc3.strategy):
    def __init__(self, initial_money=10000000.0):
        super().__init__(initial_money)

        # Trading state
        self.signal = 0          # Current signal: -1, 0, 1
        self.leverage = 1.0      # Position leverage
        self.price = 0.0         # Current price
        self.volume = 0.0        # Position size
```

## Module Structure

Every strategy module must follow this structure:

```python
#!/usr/bin/env python3
# coding=utf-8
"""
Strategy Name - Brief Description
"""

# ============================================================================
# IMPORTS
# ============================================================================

import pycaitlyn as pc
import pycaitlynts3 as pcts3
import pycaitlynutils3 as pcu3

# ============================================================================
# FRAMEWORK GLOBALS (REQUIRED)
# ============================================================================

use_raw = True                   # Use raw data mode
overwrite = False                # Overwrite existing data (False for production)
granularity = 900                # Primary granularity in seconds
schema: pc.IndexSchema = None    # Schema (set by framework)
max_workers = 1                  # Number of workers
worker_no = None                 # Current worker ID (set by framework)
exports = {}                     # Export configuration (set by framework)
imports = {}                     # Import configuration (set by framework)
metas = {}                       # Metadata definitions (set by framework)
logger = pcu3.vanilla_logger()   # Logger instance

# ============================================================================
# STRATEGY IMPLEMENTATION
# ============================================================================

class MyStrategy(pcts3.sv_object):
    """Your strategy implementation"""
    pass

# Global strategy instance
strategy = MyStrategy()

# ============================================================================
# FRAMEWORK CALLBACKS (REQUIRED)
# ============================================================================

async def on_init():
    """Initialize strategy - called once at startup."""
    global strategy, imports, metas, worker_no
    if worker_no != 0 and metas and imports:
        strategy.load_def_from_dict(metas)
        strategy.set_global_imports(imports)

async def on_ready():
    """Called when strategy is ready to process data."""
    pass

async def on_bar(bar: pc.StructValue):
    """Process incoming bar data - main processing loop."""
    global strategy
    return strategy.on_bar(bar)

async def on_market_open(market, tradeday, time_tag, time_string):
    """Called when market opens."""
    pass

async def on_market_close(market, tradeday, timetag, timestring):
    """Called when market closes."""
    pass

async def on_tradeday_begin(market, tradeday, time_tag, time_string):
    """Called at beginning of trading day."""
    pass

async def on_tradeday_end(market, tradeday, timetag, timestring):
    """Called at end of trading day."""
    pass

async def on_reference(market, tradeday, data, timetag, timestring):
    """Process reference data (tick sizes, contract specs, etc.)."""
    pass

async def on_historical(params, records):
    """Process historical data - typically not used."""
    pass
```

### Framework Global Variables

**Critical globals that must be present:**

| Variable | Type | Purpose | Typical Value |
|----------|------|---------|---------------|
| `use_raw` | bool | Use raw data format | `True` |
| `overwrite` | bool | Overwrite on replay | `False` (production) |
| `granularity` | int | Primary bar size (seconds) | `900` (15min) |
| `schema` | IndexSchema | Schema definition | `None` (set by framework) |
| `max_workers` | int | Number of workers | `1` |
| `worker_no` | int | Current worker ID | Set by framework |
| `exports` | dict | Export configuration | Set by framework |
| `imports` | dict | Import configuration | Set by framework |
| `metas` | dict | Metadata definitions | Set by framework |
| `logger` | Logger | Logging instance | `pcu3.vanilla_logger()` |

**Example usage:**

```python
# Check worker ID before processing
async def on_init():
    global worker_no, strategy
    if worker_no != 0:  # Only initialize for processing worker
        strategy.initialize(imports, metas)
    logger.info(f"Worker {worker_no} initialized")

# Use overwrite flag for testing vs production
class MyStrategy(pcts3.sv_object):
    def __init__(self):
        super().__init__()
        self.overwrite = overwrite  # Use global setting
```

## Package Imports

### Standard Framework Imports

**Always use direct imports for WallE packages:**

```python
# ✅ CORRECT - Direct imports (current standard)
from WallE.sqsm.extractor import Extractor, FeatureLoader
from WallE.sqsm.inference import InferenceClient
from WallE.ml.feature_engineering import FeatureTransformer
```

**Never use try/except fallback pattern:**

```python
# ❌ WRONG - Legacy pattern (outdated)
try:
    from WallE.sqsm.extractor import Extractor
except ImportError:
    from sqsm.extractor import Extractor  # WRONG! Causes confusion
```

**Rationale**: All WallE packages are now properly distributed and installed. The try/except pattern was used for local debugging but is no longer needed and causes import confusion.

**Reference**: See `export_walle.py` for complete list of available WallE packages.

### Core Framework Imports

**Required imports for all strategies:**

```python
import pycaitlyn as pc              # Core framework
import pycaitlynts3 as pcts3        # sv_object base class
import pycaitlynutils3 as pcu3      # Utilities (logging, time parsing)
```

**Optional imports based on strategy type:**

```python
import strategyc3 as sc3                    # Tier 3 (execution)
import composite_strategyc3 as csc3         # Tier 2 (portfolio)
from collections import deque               # Bounded collections
import math                                 # Math operations
```

## Time Format Handling

### Standard Time Format

**Framework standard: Milliseconds since epoch**

All `timetag` values from StructValue instances use milliseconds since Unix epoch:

```python
# Extract timetag from bar
tm = bar.get_time_tag()  # Returns: milliseconds since epoch (int)

# Example: 2025-01-15 09:30:00 → 1705304400000
```

### Time Conversion Functions

**pycaitlynutils3 provides rich time APIs:**

| Function | Purpose | Example |
|----------|---------|---------|
| `pcu3.ts_parse(timetag)` | ms → human-readable string | `"2025-01-15 09:30:00"` |
| `pcu3.ts_to_datetime(timetag)` | ms → datetime object | `datetime(2025, 1, 15, 9, 30)` |
| `pcu3.datetime_to_ts(dt)` | datetime → ms | `1705304400000` |

**Example usage:**

```python
def on_bar(self, bar):
    tm = bar.get_time_tag()  # ms since epoch

    # Convert for logging
    timestamp_str = pcu3.ts_parse(tm)
    logger.info(f"Processing bar at {timestamp_str}")

    # Convert for calculations
    dt = pcu3.ts_to_datetime(tm)
    hour = dt.hour
    if 9 <= hour < 11:  # Morning session
        # ... process
        pass
```

### Converting to/from Other Formats

**YYYYMMDDhhmmss format (for inference servers):**

```python
# CLI typically uses YYYYmmddHHMMSS
current_time_str = "20250115093000"  # From inference server

# Convert to ms since epoch for framework
from datetime import datetime
dt = datetime.strptime(current_time_str, "%Y%m%d%H%M%S")
timetag_ms = int(dt.timestamp() * 1000)

# Convert back when needed
dt = pcu3.ts_to_datetime(timetag_ms)
current_time_str = dt.strftime("%Y%m%d%H%M%S")
```

**Key Principle**: Always store and manipulate time as milliseconds since epoch internally. Convert only at boundaries (CLI input, external API calls, logging).

**Reference**: See `pycaitlynutils3.py` for complete time API documentation.

## Multi-Worker Architecture

### worker_no Pattern

**Framework design**: Multiple parallel processes for backtesting performance

| worker_no | Role | Responsibilities |
|-----------|------|------------------|
| **0** | Coordinator (reduce) | No calculation; optional `on_reduce()` to aggregate results from workers 1-N |
| **1...N** | Workers (map) | Actual bar processing and calculations |

**Framework behavior**:
- Sets `worker_no` at runtime for each process
- Routes bars to workers by key (market, commodity, granularity)
- Worker 0 aggregates outputs or forwards to server

**Critical patterns**:

```python
# Initialization: Only workers load metadata
async def on_init():
    global worker_no, strategy, imports, metas
    if worker_no != 0 and metas and imports:  # Workers only
        strategy.load_def_from_dict(metas)
        strategy.set_global_imports(imports)

# Processing: Only worker 1 processes bars (single-worker mode)
async def on_bar(bar: pc.StructValue):
    global worker_no, strategy
    if worker_no != 1:  # Skip in coordinator
        return []
    return strategy.on_bar(bar)

# Optional: Coordinator aggregates results
async def on_reduce(results: List[pc.StructValue]):
    """Called on worker 0 to aggregate outputs from workers 1-N"""
    global strategy
    # Aggregate logic here
    return aggregated_results
```

**Reference**: See `/home/wolverine/bin/running/` for framework implementation:
- `calculator3.py`: Main entry point
- `calculatorp3.py`: Live environment launcher (spawns workers via SSH)
- `calculator3_test.py`: Dev/test backtest runner

## Logic Contracts

**Purpose**: Abstract expiring contracts (futures/options) into stable identifiers

**Pattern**: Contract codes ending with `<NN>` where NN is 00-99

| Logic Contract | Meaning | Selection Criteria |
|----------------|---------|-------------------|
| `<00>` | Leading contract | Highest volume or open interest |
| `<01>` | Nearest to expire | 1st contract by expiration date |
| `<N>` | N-th nearest | N-th contract by expiration date |

**Contract mapping**:
- Maintained daily by framework (not indicator)
- Updated in Singularity/Reference data
- Pushed via `on_reference()` callback ~10 min before trading day

**Example**:
```python
# Code patterns
b'cu<00>'  # Copper leading contract (logical)
b'cu2501'  # Copper Jan 2025 contract (real)

# Filtering in on_bar
code = bar.get_stock_code()
if code.endswith(b'<00>'):  # Logical contract only
    process_logical(bar)
elif re.match(rb'cu\d{4}', code):  # Real contract
    process_real(bar)
```

**Reference**: Spot markets (stocks) don't use logic contracts (no expiry).

## Required Callbacks

### Initialization Flow

**1. on_init() - Strategy Initialization**

Called once at startup. Initialize your strategy with metadata:

```python
async def on_init():
    """Initialize strategy with metadata schemas."""
    global strategy, imports, metas, worker_no

    # Only initialize in processing worker (worker 1)
    if worker_no != 0 and metas and imports:
        # Load metadata definitions
        strategy.load_def_from_dict(metas)

        # Set up imports for data dependencies
        strategy.set_global_imports(imports)

        # Initialize any custom sv_objects
        strategy.price_data.load_def_from_dict(metas)
        strategy.price_data.set_global_imports(imports)

        logger.info("Strategy initialized successfully")
```

**2. on_ready() - Ready Signal**

Called when framework is ready to send data:

```python
async def on_ready():
    """Strategy is ready to receive data."""
    global strategy
    logger.info(f"{strategy.meta_name} ready to process bars")
```

### Data Processing Flow

**3. on_bar() - Main Processing Loop**

The heart of your strategy - processes every bar:

```python
async def on_bar(bar: pc.StructValue):
    """
    Process incoming bar data.

    Args:
        bar: StructValue containing OHLCV data or signals

    Returns:
        List of StructValue outputs (empty list if no output)
    """
    global strategy, worker_no

    # Only process in worker 1
    if worker_no != 1:
        return []

    # Process bar and return results
    results = strategy.on_bar(bar)

    # ALWAYS return a list
    return results if isinstance(results, list) else []
```

**Critical pattern - ALWAYS return list:**

```python
# ✅ CORRECT - Always return list
async def on_bar(bar):
    results = []
    if should_output:
        results.append(strategy.copy_to_sv())
    return results  # Empty or with outputs

# ❌ WRONG - Never return None or single object
async def on_bar(bar):
    if should_output:
        return strategy.copy_to_sv()  # WRONG!
    return None  # WRONG!
```

### Market Event Callbacks

**4. Market Lifecycle Events**

```python
async def on_market_open(market, tradeday, time_tag, time_string):
    """Called when market opens for trading."""
    logger.info(f"Market {market} opened on {tradeday}")

async def on_market_close(market, tradeday, timetag, timestring):
    """Called when market closes."""
    logger.info(f"Market {market} closed on {tradeday}")

async def on_tradeday_begin(market, tradeday, time_tag, time_string):
    """
    Called ~10 min before trading day starts.

    CRITICAL for strategies:
    - target_instrument selection (which contract to trade)
    - Contract rolling (switch from old to new contract)
    - Daily initialization
    """
    global strategy
    strategy.on_tradeday_begin(market, tradeday)

async def on_tradeday_end(market, tradeday, timetag, timestring):
    """
    Called at end of trading day.

    Use for:
    - End-of-day calculations
    - Daily reporting
    - State cleanup
    """
    global strategy
    strategy.on_tradeday_end(market, tradeday)
    logger.info(f"End of day: PV={strategy.pv:.2f}, NV={strategy.nv:.4f}")
```

**5. on_reference() - Singularity/Reference Data**

**CRITICAL**: Required for ALL strategies (Tier 2/3). Composite strategies MUST forward to sub-strategies.

**Purpose**: Receive contract mappings, tick sizes, and market metadata

**Timing**: Pushed ~10 min before trading day begins

**Contains**:
- Logic contract → Real contract mappings (<00> → cu2501, etc.)
- Available contracts, stocks, indices, options
- Tick sizes (`min_variation_unit`)
- Contract multipliers
- Trading hours per market

```python
async def on_reference(market, tradeday, data, timetag, timestring):
    """
    CRITICAL: Process Singularity/Reference data.

    Args:
        market: Market identifier (b'SHFE', b'DCE', etc.)
        tradeday: Trading day (YYYYMMDD format)
        data: Dict containing contract mappings and metadata
        timetag: Timestamp (ms since epoch)
        timestring: Human-readable time
    """
    global strategy

    # Extract commodity reference data
    if 'commodity' in data:
        commodities = data['commodity']

        # Tick sizes for precise pricing
        tick_sizes = commodities.get('min_variation_unit', [])
        codes = commodities.get('code', [])

        # Contract mappings for logic contracts
        logic_mappings = commodities.get('logic_contract_map', {})

        # Store for strategy use
        for i, code in enumerate(codes):
            if i < len(tick_sizes):
                strategy.tick_sizes[f"{market}/{code}"] = tick_sizes[i]

        # Update logic contract mappings
        strategy.update_contract_mappings(logic_mappings)

        logger.info(f"Reference data updated: {len(codes)} commodities")

# For composite strategies: MUST forward to sub-strategies
async def on_reference(market, tradeday, data, timetag, timestring):
    """Forward reference data to all sub-strategies"""
    global composite, worker_no
    if worker_no == 1:  # Only in worker
        composite.on_reference(market, tradeday, data)
```

**Reference**: `/home/wolverine/bin/running/calculator3.py` for data structure

## calculator3_test.py CLI

The main testing CLI for running strategies.

### Basic Command Structure

```bash
python /home/wolverine/bin/running/calculator3_test.py \
    --testcase /path/to/strategy/ \
    --algoname MyStrategy \
    --sourcefile MyStrategy.py \
    --start 20240101000000 \
    --end 20240131000000 \
    --granularity 900 \
    --tm wss://10.99.100.116:4433/tm \
    --tm-master 10.99.100.116:6102 \
    --rails https://10.99.100.116:4433/private-api/ \
    --token YOUR_TOKEN_HERE \
    --category 1 \
    --is-managed 1 \
    --restore-length 2592000000 \
    --multiproc 1
```

### CLI Arguments Reference

**Required Arguments:**

| Argument | Description | Example |
|----------|-------------|---------|
| `--testcase` | Path to strategy directory | `/workspaces/Margarita/` |
| `--algoname` | Strategy name (must match meta_name) | `Margarita` |
| `--sourcefile` | Python file name | `Margarita.py` |
| `--start` | Start time (YYYYMMDDhhmmss) | `20240101000000` |
| `--end` | End time (YYYYMMDDhhmmss) | `20240131235959` |
| `--granularity` | Bar size in seconds | `900` (15 min) |

**Server Configuration:**

| Argument | Description | Example |
|----------|-------------|---------|
| `--tm` | TimeMachine server URL | `wss://10.99.100.116:4433/tm` |
| `--tm-master` | TimeMachine master address | `10.99.100.116:6102` |
| `--rails` | API endpoint URL | `https://10.99.100.116:4433/private-api/` |
| `--token` | Authentication token | `YOUR_TOKEN_HERE` |

**Strategy Configuration:**

| Argument | Description | Values |
|----------|-------------|--------|
| `--category` | Strategy category | `1` (indicators), `2` (strategies) |
| `--is-managed` | Managed mode flag | `0` (no), `1` (yes) |
| `--restore-length` | Historical data to restore (ms) | `2592000000` (30 days) |
| `--multiproc` | Enable multiprocessing | `0` (no), `1` (yes) |

### Common Usage Patterns

**Quick Test (7 days):**

```bash
python calculator3_test.py \
    --testcase ${PWD}/ \
    --algoname MyIndicator \
    --sourcefile MyIndicator.py \
    --start 20250703203204 \
    --end 20250710203204 \
    --granularity 300 \
    --tm wss://10.99.100.116:4433/tm \
    --tm-master 10.99.100.116:6102 \
    --rails https://10.99.100.116:4433/private-api/ \
    --token YOUR_TOKEN \
    --category 1 \
    --is-managed 1 \
    --restore-length 864000000 \
    --multiproc 1
```

**Full Backtest (2+ years):**

```bash
python calculator3_test.py \
    --testcase ${PWD}/ \
    --algoname MyIndicator \
    --sourcefile MyIndicator.py \
    --start 20230101000000 \
    --end 20250925000000 \
    --granularity 300 \
    --tm wss://10.99.100.116:4433/tm \
    --tm-master 10.99.100.116:6102 \
    --rails https://10.99.100.116:4433/private-api/ \
    --token YOUR_TOKEN \
    --category 1 \
    --is-managed 1 \
    --restore-length 864000000 \
    --multiproc 1
```

**Replay Consistency Test:**

```bash
python test_resuming_mode.py \
    --start 20250701203204 \
    --end 20250710203204 \
    --midpoint 20250705120000
```

### VSCode Debug Configuration

Create `.vscode/launch.json` for easy debugging:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "MyStrategy - Quick Test",
            "type": "debugpy",
            "request": "launch",
            "stopOnEntry": false,
            "python": "python",
            "program": "/home/wolverine/bin/running/calculator3_test.py",
            "args": [
                "--testcase", "${workspaceFolder}/",
                "--algoname", "MyStrategy",
                "--sourcefile", "MyStrategy.py",
                "--start", "20250703203204",
                "--end", "20250710203204",
                "--granularity", "300",
                "--tm", "wss://10.99.100.116:4433/tm",
                "--tm-master", "10.99.100.116:6102",
                "--rails", "https://10.99.100.116:4433/private-api/",
                "--token", "YOUR_TOKEN_HERE",
                "--category", "1",
                "--is-managed", "1",
                "--restore-length", "864000000",
                "--multiproc", "1"
            ],
            "cwd": "${workspaceFolder}",
            "envFile": "~/.env",
            "env": {
                "PYTHONPATH": "/home/wolverine/bin/running:${env:PYTHONPATH}"
            },
            "console": "integratedTerminal"
        },
        {
            "name": "MyStrategy - Full Backtest",
            "type": "debugpy",
            "request": "launch",
            "stopOnEntry": false,
            "python": "python",
            "program": "/home/wolverine/bin/running/calculator3_test.py",
            "args": [
                "--testcase", "${workspaceFolder}/",
                "--algoname", "MyStrategy",
                "--sourcefile", "MyStrategy.py",
                "--start", "20230101000000",
                "--end", "20250925000000",
                "--granularity", "300",
                "--tm", "wss://10.99.100.116:4433/tm",
                "--tm-master", "10.99.100.116:6102",
                "--rails", "https://10.99.100.116:4433/private-api/",
                "--token", "YOUR_TOKEN_HERE",
                "--category", "1",
                "--is-managed", "1",
                "--restore-length", "864000000",
                "--multiproc", "1"
            ],
            "cwd": "${workspaceFolder}",
            "envFile": "~/.env",
            "env": {
                "PYTHONPATH": "/home/wolverine/bin/running:${env:PYTHONPATH}"
            },
            "console": "integratedTerminal"
        },
        {
            "name": "Test Resuming Mode",
            "type": "debugpy",
            "request": "launch",
            "stopOnEntry": false,
            "python": "python",
            "program": "${workspaceFolder}/test_resuming_mode.py",
            "args": [],
            "cwd": "${workspaceFolder}",
            "envFile": "~/.env",
            "env": {
                "PYTHONPATH": "/home/wolverine/bin/running:${env:PYTHONPATH}"
            },
            "console": "integratedTerminal"
        }
    ]
}
```

**Debug in VSCode:**
1. Open your strategy project
2. Set breakpoints by clicking left of line numbers
3. Press F5 or select configuration from Run menu
4. Use Debug Console to inspect variables
5. Step through code with F10 (step over) / F11 (step into)

## Logging Best Practices

```python
import pycaitlynutils3 as pcu3

# Create logger
logger = pcu3.vanilla_logger()

# Timestamp formatting
def log_with_timestamp(timetag, message):
    """Log with formatted timestamp."""
    timestamp = pcu3.ts_parse(timetag)
    logger.info(f"[{timestamp}] {message}")

# Example usage in strategy
class MyStrategy(pcts3.sv_object):
    def on_bar(self, bar):
        timetag = bar.get_time_tag()
        log_with_timestamp(timetag, f"Processing bar {self.bar_index}")

        # Detailed logging in development
        logger.debug(f"Price: {self.current_price:.2f}, "
                    f"Volume: {self.current_volume:.0f}")

        # Info level for key events
        logger.info(f"Signal generated: {self.signal_strength:.3f}")

        # Warning for unusual conditions
        if self.volume_spike > 2.0:
            logger.warning(f"Volume spike detected: {self.volume_spike:.2f}x")

        # Error for critical issues
        try:
            self.calculate_indicator()
        except Exception as e:
            logger.error(f"Indicator calculation failed: {e}")
```

## Common Patterns and Anti-Patterns

### ✅ Good Patterns

**1. Proper Initialization:**

```python
class MyIndicator(pcts3.sv_object):
    def __init__(self):
        super().__init__()

        # Set metadata as constants
        self.meta_name = "MyIndicator"
        self.namespace = pc.namespace_private
        self.granularity = 900
        self.market = b'SHFE'
        self.code = b'cu<00>'
        self.revision = (1 << 32) - 1

        # Initialize state
        self.bar_index = 0
        self.initialized = False
        self.persistent = True

    def initialize(self, imports, metas):
        """Separate initialization method."""
        self.load_def_from_dict(metas)
        self.set_global_imports(imports)
```

**2. Proper Bar Processing:**

```python
async def on_bar(bar: pc.StructValue):
    """Always return list, handle worker check."""
    global strategy, worker_no

    if worker_no != 1:
        return []

    results = strategy.on_bar(bar)
    return results if isinstance(results, list) else []
```

**3. Cycle Boundary Handling:**

```python
def on_bar(self, bar):
    tm = bar.get_time_tag()

    # Initialize timetag
    if self.timetag is None:
        self.timetag = tm

    # Handle cycle boundaries
    if self.timetag < tm:
        self._on_cycle_pass(tm)

        # Output if ready
        results = []
        if self.ready_to_serialize():
            results.append(self.copy_to_sv())

        # Update for next cycle
        self.timetag = tm
        self.bar_index += 1

        return results

    # Process current cycle data
    self._process_bar_data(bar)
    return []
```

### ❌ Anti-Patterns

**1. Modifying Metadata:**

```python
# ❌ WRONG - Never modify metadata after init
class BadIndicator(pcts3.sv_object):
    def on_bar(self, bar):
        self.market = bar.get_market()  # WRONG!
        self.code = bar.get_stock_code()  # WRONG!
```

**2. Returning Wrong Type:**

```python
# ❌ WRONG - Never return None or single object
async def on_bar(bar):
    if should_output:
        return strategy.copy_to_sv()  # WRONG!
    return None  # WRONG!

# ✅ CORRECT - Always return list
async def on_bar(bar):
    results = []
    if should_output:
        results.append(strategy.copy_to_sv())
    return results
```

**3. Not Checking Worker ID:**

```python
# ❌ WRONG - Will process in all workers
async def on_bar(bar):
    return strategy.on_bar(bar)

# ✅ CORRECT - Only process in worker 1
async def on_bar(bar):
    global worker_no
    if worker_no != 1:
        return []
    return strategy.on_bar(bar)
```

**4. Missing Framework Globals:**

```python
# ❌ WRONG - Missing required globals
import pycaitlyn as pc

# Strategy code...

# ✅ CORRECT - All required globals present
import pycaitlyn as pc
import pycaitlynutils3 as pcu3

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
```

## Working Example: Simple EMA Indicator

Complete working example showing all concepts:

```python
#!/usr/bin/env python3
# coding=utf-8
"""Simple EMA Crossover Indicator"""

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
    """Parse SampleQuote data."""
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

class EMAIndicator(pcts3.sv_object):
    """Simple EMA crossover indicator."""

    def __init__(self, market, code):
        super().__init__()

        # Metadata (constants)
        self.meta_name = "EMAIndicator"
        self.namespace = pc.namespace_private
        self.granularity = 900
        self.market = market
        self.code = code
        self.revision = (1 << 32) - 1

        # State variables
        self.bar_index = 0
        self.timetag = None
        self.ema_fast = 0.0
        self.ema_slow = 0.0
        self.signal = 0  # -1, 0, 1
        self.initialized = False
        self.persistent = True

        # Data parser
        self.quote = SampleQuote()

        # EMA parameters
        self.alpha_fast = 2.0 / 11.0  # 10-period EMA
        self.alpha_slow = 2.0 / 21.0  # 20-period EMA

    def initialize(self, imports, metas):
        """Initialize with metadata."""
        self.load_def_from_dict(metas)
        self.set_global_imports(imports)
        self.quote.load_def_from_dict(metas)
        self.quote.set_global_imports(imports)

    def on_bar(self, bar: pc.StructValue):
        """Process bar data."""
        # Check if this bar is for us
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
        """Process end of cycle."""
        close = float(self.quote.close)

        # Initialize EMAs
        if not self.initialized:
            self.ema_fast = close
            self.ema_slow = close
            self.initialized = True
            return

        # Update EMAs
        self.ema_fast = self.alpha_fast * close + (1 - self.alpha_fast) * self.ema_fast
        self.ema_slow = self.alpha_slow * close + (1 - self.alpha_slow) * self.ema_slow

        # Generate signal
        if self.ema_fast > self.ema_slow:
            self.signal = 1  # Bullish
        elif self.ema_fast < self.ema_slow:
            self.signal = -1  # Bearish
        else:
            self.signal = 0  # Neutral

        logger.info(f"Bar {self.bar_index}: EMA_fast={self.ema_fast:.2f}, "
                   f"EMA_slow={self.ema_slow:.2f}, Signal={self.signal}")

# Global indicator
indicator = EMAIndicator(b'SHFE', b'cu<00>')

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

## Summary

This chapter covered:

1. **Base Classes**: `pcts3.sv_object`, `sc3.strategy`, `csc3.composite_strategy`
2. **Module Structure**: Required imports, globals, callbacks
3. **Framework Globals**: Purpose and usage of each global variable
4. **Required Callbacks**: Initialization, processing, market events
5. **CLI Usage**: calculator3_test.py arguments and patterns
6. **VSCode Integration**: Debug configurations for easy testing
7. **Best Practices**: Patterns to follow and anti-patterns to avoid

**Key Takeaways:**

- Every strategy must extend base classes and implement required callbacks
- Framework globals must be present at module level
- ALWAYS return list from `on_bar()` callback
- Use `worker_no` to control which worker processes data
- Metadata should be constants, never modified after initialization
- VSCode debug configurations make testing much easier

**Next Steps:**

In the next chapter, we'll dive deep into StructValue and sv_object patterns, understanding how data flows through the framework and how automatic serialization works.

---

**Previous:** [02 - uin.json and uout.json](02-uin-and-uout.md) | **Next:** [04 - StructValue and sv_object](04-structvalue-and-sv_object.md)
