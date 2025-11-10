# Chapter 02: uin.json and uout.json - Input and Output Configuration

## Overview

The `uin.json` (input universe) and `uout.json` (output universe) files are the configuration backbone of any Wolverine indicator or strategy. They define:

- **uin.json**: What data your indicator consumes (inputs)
- **uout.json**: What data your indicator produces (outputs)

These JSON files tell the framework exactly what markets, securities, granularities, and fields to process, ensuring type safety and proper data routing.

## uin.json - Input Universe Configuration

### Purpose

`uin.json` specifies:
1. Which data sources to import (SampleQuote, ZampleQuote, other indicators)
2. What fields from those sources you need
3. Which granularities (timeframes) to subscribe to
4. Which markets and securities to monitor

### Basic Structure

```json
{
  "global": {
    "imports": {
      "DataSourceName": {
        "fields": [...],
        "granularities": [...],
        "revision": 4294967295,
        "markets": [...],
        "security_categories": [...],
        "securities": [...]
      }
    }
  },
  "private": {
    "imports": {
      "PrivateDataSource": {
        ...
      }
    }
  }
}
```

### Complete Example

```json
{
  "global": {
    "imports": {
      "SampleQuote": {
        "fields": [
          "open",
          "high",
          "low",
          "close",
          "volume",
          "turnover"
        ],
        "granularities": [300, 900, 3600],
        "revision": 4294967295,
        "markets": ["DCE", "SHFE", "CZCE"],
        "security_categories": [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        "securities": [
          ["i", "j"],
          ["cu", "al", "rb", "sc", "fu"],
          ["TA", "MA"]
        ]
      }
    }
  },
  "private": {
    "imports": {
      "ZampleQuote": {
        "fields": [
          "open",
          "high",
          "low",
          "close",
          "volume",
          "turnover"
        ],
        "granularities": [14400],
        "revision": 4294967295,
        "markets": ["DCE", "SHFE", "CZCE"],
        "security_categories": [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        "securities": [
          ["i", "j"],
          ["cu", "al", "rb", "sc", "fu"],
          ["TA", "MA"]
        ]
      }
    }
  }
}
```

### Field Descriptions

**fields**: Array of field names to import from the data source
- For SampleQuote/ZampleQuote: `["open", "high", "low", "close", "volume", "turnover"]`
- For other indicators: Check their uout.json for available fields
- **IMPORTANT**: Use `"turnover"` not `"amount"` for volume-weighted price data

**granularities**: Array of time granularities in seconds
- Available for SampleQuote: `[60, 300, 900, 1800, 3600, 86400]` (1min, 5min, 15min, 30min, 1H, 1D)
- Plus weekly and monthly
- Use ZampleQuote for missing granularities (e.g., `[14400]` for 4H)

**revision**: Schema version
- Always use `4294967295` (latest version constant: `(1 << 32) - 1`)

**markets**: Array of market identifiers
- `["DCE", "SHFE", "CZCE", "CFFEX"]`

**security_categories**: Array of arrays, one per market
- Usually `[[1, 2, 3], [1, 2, 3], [1, 2, 3]]` for safety
- Each sub-array corresponds to a market

**securities**: Array of arrays, securities grouped by market
- DCE: `["i", "j", "m", "y", ...]` (lowercase)
- SHFE: `["cu", "al", "rb", "sc", "fu", ...]` (lowercase)
- CZCE: `["TA", "MA", "FG", "SR", ...]` (UPPERCASE - special case!)

### 🚨 Critical Array Alignment Rule

**markets**, **security_categories**, and **securities** arrays MUST have identical length and alignment:

```json
{
  "markets": ["DCE", "SHFE", "CZCE"],          // Length: 3
  "security_categories": [
    [1, 2, 3],   // For DCE
    [1, 2, 3],   // For SHFE
    [1, 2, 3]    // For CZCE
  ],                                          // Length: 3
  "securities": [
    ["i", "j"],                // DCE securities
    ["cu", "al", "rb"],        // SHFE securities
    ["TA", "MA"]               // CZCE securities
  ]                                          // Length: 3
}
```

**Each index corresponds across all three arrays:**
- `markets[0]` ↔ `security_categories[0]` ↔ `securities[0]`
- `markets[1]` ↔ `security_categories[1]` ↔ `securities[1]`
- `markets[2]` ↔ `security_categories[2]` ↔ `securities[2]`

### Namespace Organization

**global namespace**: Public data available to all
- SampleQuote: Standard market data
- Shared indicators
- Reference data

**private namespace**: Private calculations
- ZampleQuote: Custom sampled data
- Your indicator imports from other private indicators
- Internal dependencies

### Importing Other Indicators

To use another indicator's output as input:

```json
{
  "private": {
    "imports": {
      "MyOtherIndicator": {
        "fields": [
          "signal",
          "confidence",
          "trend_strength"
        ],
        "granularities": [900],
        "revision": 4294967295,
        "markets": ["DCE"],
        "security_categories": [[1, 2, 3]],
        "securities": [["i", "j"]]
      }
    }
  }
}
```

### Granularity Selection Strategy

**Principle**: Only import granularities you actually use

```json
// ❌ BAD: Importing unused granularities wastes resources
"granularities": [60, 300, 900, 1800, 3600, 86400]  // Using only 900

// ✅ GOOD: Only what you need
"granularities": [300, 900, 3600]  // Multi-timeframe analysis
```

**Why This Matters:**
- Reduces memory consumption
- Improves data pipeline performance
- Faster backtest execution
- Lower resource usage

## uout.json - Output Universe Configuration

### Purpose

`uout.json` specifies:
1. What fields your indicator outputs
2. Field types and precision
3. Display names for visualization
4. Which markets and securities your indicator covers
5. Output granularities

### Basic Structure

```json
{
  "private": {
    "markets": [...],
    "security_categories": [...],
    "securities": [...],
    "sample_granularities": {
      "type": "min",
      "cycles": [...]
    },
    "export": {
      "XXX": {
        "fields": [...],
        "defs": [...],
        "revision": -1
      }
    }
  }
}
```

### Complete Example

```json
{
  "private": {
    "markets": ["DCE", "SHFE", "CZCE"],
    "security_categories": [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
    "securities": [
      ["i", "j"],
      ["cu", "al", "rb", "sc", "fu"],
      ["TA", "MA"]
    ],
    "sample_granularities": {
      "type": "min",
      "cycles": [300],
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
          {
            "field_type": "int64",
            "display_name": "Preserved Field",
            "precision": 0,
            "sample_type": 0,
            "multiple": 1
          },
          {
            "field_type": "integer",
            "display_name": "Bar Index",
            "precision": 0,
            "sample_type": 0,
            "multiple": 1
          },
          {
            "field_type": "double",
            "display_name": "EMA Fast",
            "precision": 6,
            "sample_type": 0,
            "multiple": 1
          },
          {
            "field_type": "double",
            "display_name": "EMA Slow",
            "precision": 6,
            "sample_type": 0,
            "multiple": 1
          },
          {
            "field_type": "integer",
            "display_name": "Signal",
            "precision": 0,
            "sample_type": 0,
            "multiple": 1
          },
          {
            "field_type": "double",
            "display_name": "Confidence",
            "precision": 6,
            "sample_type": 0,
            "multiple": 1
          }
        ],
        "revision": -1
      }
    }
  }
}
```

### Field Definitions

**export name**: MUST be "XXX"
```json
"export": {
  "XXX": {  // CRITICAL: Always use "XXX", framework replaces this
    ...
  }
}
```

**fields**: Array of field names (must match Python class attributes)
```json
"fields": [
  "_preserved_field",  // MUST be first
  "your_field_1",
  "your_field_2"
]
```

**defs**: Array of field definitions (one per field, same order)
```json
"defs": [
  {
    "field_type": "int64",      // Field data type
    "display_name": "...",      // Human-readable name
    "precision": 0,             // Decimal places
    "sample_type": 0,           // Usually 0
    "multiple": 1               // Usually 1
  }
]
```

### 🚨 Critical _preserved_field Requirement

**MANDATORY**: `"_preserved_field"` MUST be:
1. The FIRST field in the fields array
2. Type `"int64"`
3. Precision `0`

```json
"fields": [
  "_preserved_field",  // MUST BE FIRST
  "other_field_1",
  "other_field_2"
],
"defs": [
  {
    "field_type": "int64",   // MUST BE int64
    "precision": 0,          // MUST BE 0
    "display_name": "Preserved Field",
    "sample_type": 0,
    "multiple": 1
  },
  // Other field definitions...
]
```

**Why This Exists:**
- Automatically populated by calculator3.py with time_tag
- Framework requirement for state persistence
- DO NOT manually set this field in your code

### Legal Field Types

The framework supports exactly **8 data types**:

**Scalar Types:**
1. `"integer"` - 32-bit integer
2. `"int64"` - 64-bit integer
3. `"double"` - Double precision float
4. `"string"` - String data

**Vector Types:**
5. `"vInteger"` - Vector of 32-bit integers
6. `"vInt64"` - Vector of 64-bit integers
7. `"vDouble"` - Vector of double precision floats
8. `"vString"` - Vector of strings

### Field Precision Guidelines

**precision** determines decimal places for display and comparison:

```json
// Integer fields: precision 0
{"field_type": "integer", "precision": 0}    // Exact equality
{"field_type": "int64", "precision": 0}      // Exact equality

// Double fields: precision 2-8
{"field_type": "double", "precision": 2}     // 2 decimal places (e.g., time)
{"field_type": "double", "precision": 4}     // 4 decimal places (e.g., prices)
{"field_type": "double", "precision": 6}     // 6 decimal places (most indicators)

// String fields: precision 0
{"field_type": "string", "precision": 0}
```

**Choosing Precision:**
- **Prices**: 4-6 (depends on instrument tick size)
- **Percentages/Ratios**: 6
- **Counts/Indices**: 0 (integers)
- **Time durations**: 2
- **Regime classifications**: 0 (integers)

### Sample Granularities

Specifies output granularity:

```json
"sample_granularities": {
  "type": "min",                    // Type: "min", "enum", etc.
  "cycles": [300],                  // 300 seconds = 5 minutes
  "cycle_lengths": [0]              // Usually [0]
}
```

**Common Granularities:**
- 60: 1 minute
- 300: 5 minutes
- 900: 15 minutes
- 1800: 30 minutes
- 3600: 1 hour
- 14400: 4 hours
- 86400: 1 day

**Output Granularity Rule:**
- Should match your minimum input granularity
- If you import [300, 900, 3600], output at 300 (minimum dependency)

### Vector Fields for Multi-Commodity Indicators

When outputting arrays of values:

```json
{
  "fields": [
    "_preserved_field",
    "forecast_horizons",        // Vector field
    "regime_strengths"          // Vector field
  ],
  "defs": [
    {
      "field_type": "int64",
      "precision": 0,
      "display_name": "Preserved Field",
      "sample_type": 0,
      "multiple": 1
    },
    {
      "field_type": "vDouble",   // Vector of doubles
      "precision": 6,
      "display_name": "Forecast Horizons",
      "sample_type": 0,
      "multiple": 1
    },
    {
      "field_type": "vInteger",  // Vector of integers
      "precision": 0,
      "display_name": "Regime Strengths",
      "sample_type": 0,
      "multiple": 1
    }
  ]
}
```

**In Python:**
```python
class MyIndicator(pcts3.sv_object):
    def __init__(self):
        super().__init__()
        self.forecast_horizons = [0.0] * 5  // Automatically handled as vDouble
        self.regime_strengths = [0] * 3     // Automatically handled as vInteger
```

## Configuration Best Practices

### 1. Array Alignment

**Always verify alignment:**

```python
# Quick check in Python
markets = ["DCE", "SHFE", "CZCE"]
sec_cats = [[1,2,3], [1,2,3], [1,2,3]]
securities = [["i","j"], ["cu","al"], ["TA","MA"]]

assert len(markets) == len(sec_cats) == len(securities), "Array length mismatch!"
```

### 2. Minimal Granularity Import

```json
// ❌ BAD: Importing everything
"granularities": [60, 300, 900, 1800, 3600, 86400]  // Only using 900

// ✅ GOOD: Only what's needed
"granularities": [900]  // Primary timeframe only

// ✅ BETTER: Multi-timeframe but minimal
"granularities": [300, 900, 3600]  // 5min, 15min, 1H analysis
```

### 3. Field Naming Convention

Use descriptive, consistent names:

```json
// ✅ GOOD: Clear, descriptive names
"fields": [
  "ema_fast_10",
  "ema_slow_20",
  "rsi_14",
  "trend_regime",
  "signal_strength"
]

// ❌ BAD: Ambiguous names
"fields": [
  "val1",
  "val2",
  "x",
  "flag"
]
```

### 4. Precision Consistency

Match precision to data characteristics:

```json
// Price-related: 4-6 decimal places
{"field_type": "double", "precision": 6, "display_name": "EMA Price"}

// Percentages/Ratios: 6 decimal places
{"field_type": "double", "precision": 6, "display_name": "Return Pct"}

// Counts/Indices: 0 decimal places
{"field_type": "integer", "precision": 0, "display_name": "Bar Count"}
```

### 5. CZCE Special Case

**Remember: CZCE uses UPPERCASE:**

```json
{
  "markets": ["DCE", "SHFE", "CZCE"],
  "securities": [
    ["i", "j"],              // DCE: lowercase
    ["cu", "al"],            // SHFE: lowercase
    ["TA", "MA", "FG"]       // CZCE: UPPERCASE!
  ]
}
```

## Common Mistakes and Solutions

### Mistake 1: Array Length Mismatch

```json
// ❌ WRONG
{
  "markets": ["DCE", "SHFE"],                    // Length: 2
  "security_categories": [[1,2,3], [1,2,3], [1,2,3]],  // Length: 3
  "securities": [["i"], ["cu"]]                  // Length: 2
}

// ✅ CORRECT
{
  "markets": ["DCE", "SHFE"],                    // Length: 2
  "security_categories": [[1,2,3], [1,2,3]],     // Length: 2
  "securities": [["i","j"], ["cu","al"]]         // Length: 2
}
```

### Mistake 2: Missing _preserved_field

```json
// ❌ WRONG
"fields": [
  "bar_index",        // Missing _preserved_field!
  "indicator_value"
]

// ✅ CORRECT
"fields": [
  "_preserved_field",  // MUST BE FIRST
  "bar_index",
  "indicator_value"
]
```

### Mistake 3: Wrong Export Name

```json
// ❌ WRONG
"export": {
  "MyIndicator": {  // Should be "XXX"
    ...
  }
}

// ✅ CORRECT
"export": {
  "XXX": {  // Always "XXX"
    ...
  }
}
```

### Mistake 4: Using "amount" Instead of "turnover"

```json
// ❌ WRONG
"fields": ["open", "high", "low", "close", "volume", "amount"]

// ✅ CORRECT
"fields": ["open", "high", "low", "close", "volume", "turnover"]
```

### Mistake 5: CZCE Lowercase

```json
// ❌ WRONG
{"markets": ["CZCE"], "securities": [["ta", "ma"]]}  // Lowercase!

// ✅ CORRECT
{"markets": ["CZCE"], "securities": [["TA", "MA"]]}  // UPPERCASE for CZCE
```

## Validation Checklist

Before running your indicator, verify:

**uin.json:**
- [ ] Array lengths match (markets, security_categories, securities)
- [ ] Granularities are minimal and necessary
- [ ] Field names are correct (especially "turnover" not "amount")
- [ ] CZCE securities are UPPERCASE
- [ ] Revision is 4294967295

**uout.json:**
- [ ] Export name is "XXX"
- [ ] "_preserved_field" is first field
- [ ] "_preserved_field" is int64, precision 0
- [ ] Array lengths match (same as uin.json)
- [ ] Field types are legal (one of 8 types)
- [ ] Precision values are appropriate
- [ ] Field names match Python class attributes

---

## Output StructValue Patterns and Visualization

### Understanding Output Multiplicity

**StructValue Index**: (meta_id, granularity, market, stock_code, time_tag)

| Component | Tier-1 Indicator | Tier-2 Composite |
|-----------|------------------|------------------|
| **meta_id** | Fixed (one per indicator) | Fixed (one per strategy) |
| **granularity** | Fixed (from uout.json) | Fixed (from uout.json) |
| **market** | Multiple markets | Single market (placeholder) |
| **stock_code** | Multiple securities | Single placeholder (`<00>`, `<01>`) |
| **Output Count** | N × M (markets × securities) | 1 (single synthetic) |

### Tier-1 Output Pattern

**Configuration**:
```json
{
  "markets": ["DCE", "SHFE"],
  "securities": [["i", "j"], ["cu", "al", "rb"]]
}
```

**Outputs Generated**:
- `DCE/i<00>` (iron ore on DCE)
- `DCE/j<00>` (coking coal on DCE)
- `SHFE/cu<00>` (copper on SHFE)
- `SHFE/al<00>` (aluminum on SHFE)
- `SHFE/rb<00>` (rebar on SHFE)

Total: 5 separate StructValue streams

**Visualization**: Pick ONE commodity as example
```python
# Query one commodity
market = "SHFE"
code = "cu<00>"  # Copper
```

### Tier-2 Output Pattern

**Configuration**:
```json
{
  "markets": ["DCE"],
  "securities": [["COMPOSITE"]]
}
```

**Outputs Generated**:
- `DCE/COMPOSITE<00>` (single synthetic instrument)

Total: 1 StructValue stream

**Real Exposures**: Stored in StructValue fields (not in market/code)
- Fields like `markets`, `codes`, `positions` define actual basket holdings

**Visualization**: Use the single placeholder
```python
# Query the placeholder
market = "DCE"  # Or whatever market in uout.json
code = "COMPOSITE<00>"  # Or whatever securities field
```

### securities Field Semantics

**Tier-1 Indicators**:
- `securities` = List of actual commodities to calculate
- Each commodity gets its own StructValue output
- Framework generates (market, commodity) pairs automatically

**Tier-2 Composites**:
- `securities` = Placeholder name (e.g., "COMPOSITE", "PORTFOLIO")
- Single output with placeholder stock_code
- Actual positions stored in StructValue fields

### Querying Calculated Data

Use svr3 module to fetch StructValues from server:

```python
import svr3

# Tier-1: Query one commodity
client = svr3.sv_reader(
    start, end,
    "MyIndicator",  # Indicator name
    900,            # Granularity
    "private",
    "symbol",
    ["SHFE"],       # Market from uout.json
    ["cu<00>"],     # ONE commodity to visualize
    # ... other params
)

# Tier-2: Query the placeholder
client = svr3.sv_reader(
    start, end,
    "MyComposite",  # Strategy name
    300,
    "private",
    "symbol",
    ["DCE"],        # Market from uout.json
    ["COMPOSITE<00>"],  # Placeholder from uout.json
    # ... other params
)
```

**Reference**: See Chapter 10 - Visualization for complete svr3 usage patterns.

---

## Summary

Configuration files are critical for indicator operation:

**uin.json** defines:
- Input data sources and fields
- Granularities to subscribe to
- Markets and securities to monitor
- Dependencies on other indicators

**uout.json** defines:
- Output fields and types
- Field precision and display names
- Output granularity
- Export configuration

**Critical Requirements:**
- Array alignment across markets/securities
- "_preserved_field" as first field in uout.json
- "XXX" as export algorithm name
- CZCE uses UPPERCASE codes
- Minimal granularity imports

**Next Chapter**: [Chapter 03 - Programming Basics and CLI](./03-programming-basics-and-cli.md) - Learn the base classes, CLI tools, and programming patterns.
