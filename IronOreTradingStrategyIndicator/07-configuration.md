# Chapter 7: Configuration

## Configuration Files

Two JSON configuration files define input and output schemas.

---

## uin.json (Input Configuration)

**Purpose**: Define input data schema for OHLCV bars from global namespace.

**File**: `IronOreIndicator/uin.json`

```json
{
  "SampleQuote": {
    "granularity": 900,
    "namespace": "global",
    "revision": 4294967295,
    "fields": [
      {
        "name": "open",
        "type": "double",
        "desc": "Opening price"
      },
      {
        "name": "high",
        "type": "double",
        "desc": "Highest price"
      },
      {
        "name": "low",
        "type": "double",
        "desc": "Lowest price"
      },
      {
        "name": "close",
        "type": "double",
        "desc": "Closing price"
      },
      {
        "name": "volume",
        "type": "double",
        "desc": "Trading volume"
      },
      {
        "name": "turnover",
        "type": "double",
        "desc": "Trading turnover"
      }
    ]
  }
}
```

**Field Specifications**:

| Field    | Type   | Description                   |
|----------|--------|-------------------------------|
| open     | double | Bar opening price             |
| high     | double | Bar highest price             |
| low      | double | Bar lowest price              |
| close    | double | Bar closing price             |
| volume   | double | Bar trading volume            |
| turnover | double | Bar trading turnover (value)  |

**Metadata**:
- `granularity`: 900 seconds (15 minutes)
- `namespace`: "global" (framework-provided data)
- `revision`: 4294967295 (max uint32, latest version)

---

## uout.json (Output Configuration)

**Purpose**: Define output data schema for indicator values and signals.

**File**: `IronOreIndicator/uout.json`

```json
{
  "IronOreIndicator": {
    "granularity": 900,
    "namespace": "private",
    "revision": 4294967295,
    "fields": [
      {
        "name": "ema_12",
        "type": "double",
        "desc": "12-period EMA"
      },
      {
        "name": "ema_26",
        "type": "double",
        "desc": "26-period EMA"
      },
      {
        "name": "ema_50",
        "type": "double",
        "desc": "50-period EMA"
      },
      {
        "name": "macd",
        "type": "double",
        "desc": "MACD line"
      },
      {
        "name": "macd_signal",
        "type": "double",
        "desc": "MACD signal line"
      },
      {
        "name": "macd_histogram",
        "type": "double",
        "desc": "MACD histogram"
      },
      {
        "name": "rsi",
        "type": "double",
        "desc": "RSI value (0-100)"
      },
      {
        "name": "bb_upper",
        "type": "double",
        "desc": "Bollinger Band upper"
      },
      {
        "name": "bb_middle",
        "type": "double",
        "desc": "Bollinger Band middle (SMA)"
      },
      {
        "name": "bb_lower",
        "type": "double",
        "desc": "Bollinger Band lower"
      },
      {
        "name": "bb_width",
        "type": "double",
        "desc": "Bollinger Band width (absolute)"
      },
      {
        "name": "bb_width_pct",
        "type": "double",
        "desc": "Bollinger Band width percentage"
      },
      {
        "name": "atr",
        "type": "double",
        "desc": "Average True Range"
      },
      {
        "name": "volume_ema",
        "type": "double",
        "desc": "20-period volume EMA"
      },
      {
        "name": "regime",
        "type": "int32",
        "desc": "Market regime (1=uptrend, 2=downtrend, 3=ranging, 4=chaos)"
      },
      {
        "name": "signal",
        "type": "int32",
        "desc": "Trading signal (-1=sell, 0=neutral, 1=buy)"
      },
      {
        "name": "confidence",
        "type": "double",
        "desc": "Signal confidence (0.0-1.0)"
      },
      {
        "name": "position_state",
        "type": "int32",
        "desc": "Position state (0=flat, 1=in position)"
      },
      {
        "name": "bar_index",
        "type": "int64",
        "desc": "Bar counter"
      }
    ]
  }
}
```

**Field Specifications**:

### Trend Indicators

| Field          | Type   | Description              |
|----------------|--------|--------------------------|
| ema_12         | double | 12-period EMA            |
| ema_26         | double | 26-period EMA            |
| ema_50         | double | 50-period EMA            |
| macd           | double | MACD line (ema_12 - ema_26) |
| macd_signal    | double | 9-period EMA of MACD     |
| macd_histogram | double | MACD - MACD Signal       |

### Mean Reversion Indicators

| Field     | Type   | Description                    |
|-----------|--------|--------------------------------|
| rsi       | double | RSI value (0-100)              |
| bb_upper  | double | Bollinger Band upper (mean + 2σ) |
| bb_middle | double | Bollinger Band middle (SMA)    |
| bb_lower  | double | Bollinger Band lower (mean - 2σ) |
| bb_width  | double | Band width (upper - lower)     |
| bb_width_pct | double | Band width as % of middle   |

### Volatility Indicators

| Field | Type   | Description          |
|-------|--------|----------------------|
| atr   | double | Average True Range   |

### Liquidity Indicators

| Field      | Type   | Description          |
|------------|--------|----------------------|
| volume_ema | double | 20-period volume EMA |

### Regime & Signals

| Field          | Type   | Description                                  |
|----------------|--------|----------------------------------------------|
| regime         | int32  | Market regime (1/2/3/4)                      |
| signal         | int32  | Trading signal (-1=sell, 0=neutral, 1=buy)   |
| confidence     | double | Signal confidence (0.0 to 1.0)               |
| position_state | int32  | Position state (0=flat, 1=in position)       |
| bar_index      | int64  | Bar counter (for debugging)                  |

**Metadata**:
- `granularity`: 900 seconds (15 minutes)
- `namespace`: "private" (user-defined indicator)
- `revision`: 4294967295 (max uint32, latest version)

---

## Data Flow Diagram

```
Input (uin.json):
┌─────────────────┐
│  SampleQuote    │
│  (global)       │
├─────────────────┤
│ • open          │
│ • high          │
│ • low           │
│ • close         │
│ • volume        │
│ • turnover      │
└─────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   IronOreIndicator.py       │
│   Processing Pipeline       │
├─────────────────────────────┤
│ 1. Update EMAs              │
│ 2. Update MACD              │
│ 3. Update RSI               │
│ 4. Update Bollinger Bands   │
│ 5. Update ATR               │
│ 6. Update Volume EMA        │
│ 7. Detect Regime            │
│ 8. Generate Signal          │
└─────────────────────────────┘
        │
        ▼
Output (uout.json):
┌─────────────────┐
│ IronOreIndicator│
│  (private)      │
├─────────────────┤
│ • ema_12/26/50  │
│ • macd family   │
│ • rsi           │
│ • bb family     │
│ • atr           │
│ • volume_ema    │
│ • regime        │
│ • signal        │
│ • confidence    │
│ • position_state│
└─────────────────┘
```

---

## Configuration Validation

### uin.json Validation

```bash
# Check JSON syntax
python3 -m json.tool uin.json

# Expected output: Valid JSON (no errors)
```

### uout.json Validation

```bash
# Check JSON syntax
python3 -m json.tool uout.json

# Expected output: Valid JSON (no errors)
```

---

## Field Type Reference

### Supported Types

| Type   | Description           | Example Values        |
|--------|----------------------|-----------------------|
| double | 64-bit floating point | 123.45, -67.89        |
| int32  | 32-bit signed integer | -1, 0, 1, 2, 3, 4     |
| int64  | 64-bit signed integer | 1000, 2000, 999999    |

---

## Namespace Reference

| Namespace | Usage                                      |
|-----------|--------------------------------------------|
| global    | Framework-provided data (OHLCV, reference) |
| private   | User-defined indicators and signals        |

---

## Revision Management

**Revision Field**: Controls schema versioning

- `(1 << 32) - 1` = 4294967295 = Maximum uint32 value
- Indicates latest/current revision
- Framework uses revision for schema compatibility checks

---

## Granularity Reference

| Granularity | Seconds | Description      |
|-------------|---------|------------------|
| 60          | 1 min   | 1-minute bars    |
| 300         | 5 min   | 5-minute bars    |
| 900         | 15 min  | 15-minute bars   |
| 1800        | 30 min  | 30-minute bars   |
| 3600        | 1 hour  | 1-hour bars      |

**This Strategy**: 900 seconds (15-minute bars)

---

## Configuration Best Practices

1. **Match Granularity**: Ensure uin.json and uout.json have matching granularity (900)
2. **Use Descriptive Names**: Field names should be self-documenting
3. **Include Descriptions**: Add "desc" field for all outputs
4. **Type Consistency**: Use double for prices/indicators, int32 for signals/regimes
5. **Namespace Separation**: Keep input (global) and output (private) separate

---

## Loading Configuration

Framework automatically loads configuration files:

```python
# In indicator code
def initialize(self, imports, metas):
    """Initialize metadata schemas"""
    # Framework provides metas dict from uin.json and uout.json
    self.load_def_from_dict(metas)
    self.set_global_imports(imports)

    # Initialize dependencies
    self.sq.load_def_from_dict(metas)
    self.sq.set_global_imports(imports)
```

---

## Next Steps

Proceed to Chapter 8 for backtesting procedures and data fetching.
