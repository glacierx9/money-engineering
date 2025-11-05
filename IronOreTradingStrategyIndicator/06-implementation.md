# Chapter 6: Implementation

## Code Structure

Complete implementation structure for IronOreIndicator.py following WOS patterns.

---

## File Organization

```
IronOreIndicator/
├── IronOreIndicator.py    # Main indicator implementation
├── uin.json               # Input configuration (OHLCV schema)
├── uout.json              # Output configuration (signals + indicators)
└── analysis.ipynb         # Jupyter notebook for P&L visualization
```

---

## Class Structure

### SampleQuote Class

**Purpose**: Parse OHLCV data from global namespace

```python
class SampleQuote(pcts3.sv_object):
    """Parse SampleQuote (OHLCV) data from global namespace"""

    def __init__(self):
        super().__init__()

        # Metadata - CONSTANTS
        self.meta_name = "SampleQuote"
        self.namespace = pc.namespace_global
        self.revision = (1 << 32) - 1
        self.granularity = 900

        # OHLCV fields (automatically populated by from_sv)
        self.open = None
        self.high = None
        self.low = None
        self.close = None
        self.volume = None
        self.turnover = None
```

### IronOreIndicator Class

**Purpose**: Multi-indicator confirmation system with regime detection

**Class Organization**:

```python
class IronOreIndicator(pcts3.sv_object):
    """
    Iron Ore Multi-Indicator Confirmation System

    Layers:
    1. Trend: Triple EMA + MACD
    2. Mean Reversion: RSI + Bollinger Bands
    3. Volatility: ATR + BB Width
    4. Liquidity: Volume EMA

    Generates regime-aware buy/sell signals with position scaling
    """

    def __init__(self):
        # Metadata
        # State variables
        # Indicator states
        # Risk management states
        # Dependencies

    def initialize(self, imports, metas):
        # Initialize metadata schemas

    def on_bar(self, bar):
        # Process incoming bars

    def _on_cycle_pass(self, time_tag):
        # Process end of cycle

    # Indicator update methods
    def _update_emas(self, close):
    def _update_macd(self):
    def _update_rsi(self, close):
    def _update_bollinger_bands(self, close):
    def _update_atr(self, high, low, close):
    def _update_volume_ema(self, volume):

    # Regime detection
    def _detect_regime(self):
    def _is_strong_uptrend(self):
    def _is_strong_downtrend(self):
    def _is_high_volatility_chaos(self):

    # Signal generation
    def _generate_signal(self):
    def _check_uptrend_buy(self):
    def _check_downtrend_sell(self):
    def _check_ranging_buy(self):
    def _check_ranging_sell(self):

    # Risk management
    def _calculate_position_size(self):
    def _calculate_stop_loss(self, entry_price, direction):
    def _check_profit_targets(self, entry_price, current_price, direction):

    # Utilities
    def ready_to_serialize(self):
```

---

## State Variables Declaration

### Metadata (CONSTANTS)

```python
# Metadata - CONSTANTS (never change after initialization)
self.meta_name = "IronOreIndicator"
self.namespace = pc.namespace_private
self.granularity = 900
self.market = b'DCE'
self.code = b'i<00>'
self.revision = (1 << 32) - 1
```

### Tracking Variables

```python
# State variables (automatically persisted by framework)
self.bar_index = 0
self.timetag = None
self.initialized = False
```

### Triple EMA States

```python
# EMA states
self.ema_12 = 0.0
self.ema_26 = 0.0
self.ema_50 = 0.0
```

### MACD States

```python
# MACD states
self.macd = 0.0
self.macd_signal = 0.0
self.macd_histogram = 0.0
```

### RSI States

```python
# RSI states (online algorithm)
self.rsi = 50.0
self.gain_ema = 0.0
self.loss_ema = 0.0
self.prev_close = 0.0
```

### Bollinger Band States

```python
# Bollinger Band states (Welford's online variance)
self.bb_n = 0
self.bb_mean = 0.0
self.bb_m2 = 0.0  # Sum of squared differences from mean
self.bb_variance = 0.0
self.bb_std_dev = 0.0
self.bb_upper = 0.0
self.bb_middle = 0.0
self.bb_lower = 0.0
self.bb_width = 0.0
self.bb_width_pct = 0.0
```

### ATR States

```python
# ATR states
self.atr = 0.0
self.mean_atr = 0.0
self.atr_count = 0
```

### Volume States

```python
# Volume EMA
self.volume_ema = 0.0
self.volume = 0.0
```

### Current Bar Data

```python
# Current bar OHLCV
self.open = 0.0
self.high = 0.0
self.low = 0.0
self.close = 0.0
```

### Regime States

```python
# Regime tracking
self.regime = 3  # Default to ranging
self.regime_prev = 3
self.regime_confirmation_count = 0
self.regime_1_bars = 0
self.regime_2_bars = 0
self.regime_3_bars = 0
self.regime_4_bars = 0
```

### Signal States

```python
# Signal outputs
self.signal = 0          # -1 (sell), 0 (neutral), 1 (buy)
self.confidence = 0.0    # Signal confidence [0.0, 1.0]
self.position_state = 0  # 0 = flat, 1 = in position
```

### Risk Management States

```python
# Position management
self.current_position_size = 0.0
self.entry_price = 0.0
self.stop_loss_price = 0.0
self.target_1_hit = False
self.target_2_hit = False

# Daily tracking
self.daily_pnl = 0.0
self.daily_trades = 0
self.current_tradeday = None

# Performance tracking
self.peak_equity = 1000000.0
self.current_equity = 1000000.0
self.total_trades = 0
self.winning_trades = 0
self.losing_trades = 0
```

### Alpha Parameters

```python
# EMA parameters (alpha = 2 / (period + 1))
self.alpha_12 = 2.0 / 13.0   # 0.1538
self.alpha_26 = 2.0 / 27.0   # 0.0741
self.alpha_50 = 2.0 / 51.0   # 0.0392
self.alpha_9 = 2.0 / 10.0    # 0.2000
self.alpha_14 = 2.0 / 15.0   # 0.1333
self.alpha_20 = 2.0 / 21.0   # 0.0952
```

### Dependencies

```python
# Dependency sv_objects
self.sq = SampleQuote()

# Control persistence
self.persistent = True
```

---

## Initialization Method

```python
def initialize(self, imports, metas):
    """Initialize metadata schemas for all sv_objects"""
    self.load_def_from_dict(metas)
    self.set_global_imports(imports)

    # Initialize dependencies
    self.sq.load_def_from_dict(metas)
    self.sq.set_global_imports(imports)
```

---

## Bar Processing Pipeline

### on_bar Method

```python
def on_bar(self, bar: pc.StructValue) -> List[pc.StructValue]:
    """
    Process incoming market data bars

    Args:
        bar: StructValue containing market data

    Returns:
        List of StructValue outputs (empty list if no output this cycle)
    """
    ret = []  # ALWAYS return list

    # Extract metadata
    market = bar.get_market()
    code = bar.get_stock_code()
    tm = bar.get_time_tag()
    ns = bar.get_namespace()
    meta_id = bar.get_meta_id()

    # Filter for our market/instrument
    if market != self.market:
        return ret

    # Route to appropriate sv_object
    if self.sq.namespace == ns and self.sq.meta_id == meta_id:
        # Filter for logical contracts only (ending in <00>)
        if code.endswith(b'<00>'):
            # Set metadata before from_sv
            self.sq.market = market
            self.sq.code = code
            self.sq.granularity = bar.get_granularity()

            # Parse data into sv_object
            self.sq.from_sv(bar)

            # Handle cycle boundaries
            if self.timetag is None:
                self.timetag = tm

            if self.timetag < tm:
                # New cycle - process previous cycle's data
                self._on_cycle_pass(tm)

                # Serialize state if ready
                if self.ready_to_serialize():
                    ret.append(self.copy_to_sv())

                # Update for next cycle
                self.timetag = tm
                self.bar_index += 1

    return ret  # ALWAYS return list
```

### _on_cycle_pass Method

```python
def _on_cycle_pass(self, time_tag):
    """
    Process end of cycle - calculate indicators and generate signals

    Pipeline:
    1. Extract OHLCV data
    2. Initialize on first bar
    3. Update all indicators (online algorithms)
    4. Detect market regime
    5. Generate trading signal
    6. Update risk management
    7. Log signal if generated
    """

    # Extract OHLCV data
    self.open = float(self.sq.open)
    self.high = float(self.sq.high)
    self.low = float(self.sq.low)
    self.close = float(self.sq.close)
    self.volume = float(self.sq.volume)

    # Initialize on first bar
    if not self.initialized:
        self._initialize_state()
        return

    # Update indicators (order matters for dependencies)
    self._update_emas(self.close)
    self._update_macd()
    self._update_rsi(self.close)
    self._update_bollinger_bands(self.close)
    self._update_atr(self.high, self.low, self.close)
    self._update_volume_ema(self.volume)

    # Detect regime
    self._detect_regime()

    # Generate signal
    self._generate_signal()

    # Update risk management
    if self.signal != 0:
        self._update_position_state()

    # Log signal generation
    if self.signal != 0:
        logger.info(
            f"Bar {self.bar_index}: Signal={self.signal}, "
            f"Confidence={self.confidence:.3f}, "
            f"Regime={self.regime}, "
            f"RSI={self.rsi:.2f}, "
            f"MACD={self.macd:.4f}, "
            f"EMA12={self.ema_12:.2f}, "
            f"EMA26={self.ema_26:.2f}, "
            f"Close={self.close:.2f}"
        )
```

---

## Indicator Update Methods

### EMA Updates

```python
def _update_emas(self, close):
    """Update triple EMA (online algorithm)"""
    self.ema_12 = self.alpha_12 * close + (1 - self.alpha_12) * self.ema_12
    self.ema_26 = self.alpha_26 * close + (1 - self.alpha_26) * self.ema_26
    self.ema_50 = self.alpha_50 * close + (1 - self.alpha_50) * self.ema_50
```

### MACD Update

```python
def _update_macd(self):
    """Update MACD and signal line"""
    # MACD line (already have EMA 12 and 26)
    self.macd = self.ema_12 - self.ema_26

    # Signal line (9-period EMA of MACD)
    self.macd_signal = (self.alpha_9 * self.macd +
                        (1 - self.alpha_9) * self.macd_signal)

    # Histogram
    self.macd_histogram = self.macd - self.macd_signal
```

### RSI Update

```python
def _update_rsi(self, close):
    """Update RSI (online algorithm via gain/loss EMAs)"""

    # Price change
    change = close - self.prev_close
    gain = max(change, 0.0)
    loss = max(-change, 0.0)

    # Update gain/loss EMAs
    self.gain_ema = self.alpha_14 * gain + (1 - self.alpha_14) * self.gain_ema
    self.loss_ema = self.alpha_14 * loss + (1 - self.alpha_14) * self.loss_ema

    # Calculate RSI
    if self.loss_ema > 0:
        rs = self.gain_ema / self.loss_ema
        self.rsi = 100.0 - (100.0 / (1.0 + rs))
    else:
        self.rsi = 100.0  # No losses = max RSI

    # Update previous close for next iteration
    self.prev_close = close
```

### Bollinger Bands Update (Welford's Algorithm)

```python
def _update_bollinger_bands(self, close):
    """Update Bollinger Bands using Welford's online variance algorithm"""

    self.bb_n += 1
    n = min(self.bb_n, 20)  # Cap at period

    # Welford's algorithm for mean and variance
    delta = close - self.bb_mean
    self.bb_mean += delta / n
    delta2 = close - self.bb_mean
    self.bb_m2 += delta * delta2

    # Calculate variance and std dev
    if n > 1:
        self.bb_variance = self.bb_m2 / (n - 1)
        self.bb_std_dev = math.sqrt(self.bb_variance)
    else:
        self.bb_variance = 0.0
        self.bb_std_dev = 0.0

    # Calculate bands
    self.bb_middle = self.bb_mean
    self.bb_upper = self.bb_mean + (2.0 * self.bb_std_dev)
    self.bb_lower = self.bb_mean - (2.0 * self.bb_std_dev)
    self.bb_width = self.bb_upper - self.bb_lower

    # Width percentage
    if self.bb_middle > 0:
        self.bb_width_pct = (self.bb_width / self.bb_middle) * 100.0
    else:
        self.bb_width_pct = 0.0
```

### ATR Update

```python
def _update_atr(self, high, low, close):
    """Update ATR (online algorithm)"""

    # True Range
    tr1 = high - low
    tr2 = abs(high - self.prev_close) if self.prev_close > 0 else 0
    tr3 = abs(low - self.prev_close) if self.prev_close > 0 else 0
    tr = max(tr1, tr2, tr3)

    # ATR (EMA of True Range)
    self.atr = self.alpha_14 * tr + (1 - self.alpha_14) * self.atr

    # Update mean ATR (online mean)
    self.atr_count += 1
    delta = self.atr - self.mean_atr
    self.mean_atr += delta / min(self.atr_count, 100)
```

### Volume EMA Update

```python
def _update_volume_ema(self, volume):
    """Update volume EMA"""
    self.volume_ema = (self.alpha_20 * volume +
                       (1 - self.alpha_20) * self.volume_ema)
```

---

## State Initialization

```python
def _initialize_state(self):
    """Initialize indicator state on first bar"""

    # Initialize EMAs
    self.ema_12 = self.close
    self.ema_26 = self.close
    self.ema_50 = self.close

    # Initialize MACD
    self.macd = 0.0
    self.macd_signal = 0.0
    self.macd_histogram = 0.0

    # Initialize RSI
    self.rsi = 50.0
    self.gain_ema = 0.0
    self.loss_ema = 0.0
    self.prev_close = self.close

    # Initialize Bollinger Bands
    self.bb_n = 1
    self.bb_mean = self.close
    self.bb_m2 = 0.0
    self.bb_variance = 0.0
    self.bb_std_dev = 0.0
    self.bb_upper = self.close
    self.bb_middle = self.close
    self.bb_lower = self.close
    self.bb_width = 0.0
    self.bb_width_pct = 0.0

    # Initialize ATR
    self.atr = self.high - self.low
    self.mean_atr = self.atr
    self.atr_count = 1

    # Initialize Volume
    self.volume_ema = self.volume

    # Initialize signals
    self.signal = 0
    self.confidence = 0.0
    self.regime = 3

    self.initialized = True

    logger.info(f"Initialized: close={self.close:.2f}, volume={self.volume:.2f}")
```

---

## Serialization Control

```python
def ready_to_serialize(self) -> bool:
    """
    Determine if state should be serialized

    Returns:
        True if bar_index > 0 and initialized
    """
    return self.bar_index > 0 and self.initialized
```

---

## Framework Callbacks

```python
# Global instance
indicator = IronOreIndicator()

async def on_init():
    """Initialize indicator with metadata schemas"""
    global indicator, imports, metas, worker_no
    if worker_no != 0 and metas and imports:
        indicator.initialize(imports, metas)
        logger.info("IronOreIndicator initialized")

async def on_bar(bar: pc.StructValue):
    """Process incoming bars"""
    global indicator, worker_no
    if worker_no != 1:
        return []
    return indicator.on_bar(bar)

async def on_ready():
    """Called when framework is ready"""
    logger.info("IronOreIndicator ready")

# Additional callbacks (empty implementations)
async def on_market_open(market, tradeday, time_tag, time_string):
    pass

async def on_market_close(market, tradeday, timetag, timestring):
    pass

async def on_reference(market, tradeday, data, timetag, timestring):
    pass

async def on_tradeday_begin(market, tradeday, time_tag, time_string):
    pass

async def on_tradeday_end(market, tradeday, timetag, timestring):
    pass

async def on_historical(params, records):
    pass
```

---

## Next Steps

Proceed to Chapter 7 for uin.json and uout.json configuration.
