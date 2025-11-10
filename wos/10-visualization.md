# Chapter 10: Visualization and Analysis

**Learning objectives:**
- Fetch calculated indicator data using svr3
- Create visualization scripts for analysis
- Understand Tier-1 vs Tier-2 data patterns
- Generate diagnostic plots
- Optimize indicator parameters

**Previous:** [09 - Tier 3 Strategy](09-tier3-strategy.md) | **Next:** [11 - Fine-tune and Iterate](11-fine-tune-and-iterate.md)

---

## Overview

Visualization fetches calculated StructValues from server to analyze indicator behavior, identify issues, and optimize parameters. The svr3 module provides async API for flexible data retrieval.

## The svr3 Module

**Purpose**: Async module for fetching calculated indicator data from server.

**Use Case**: Query any indicator or composite strategy's performance metrics for tuning.

### Data Format

**save_by_symbol() returns**: `List[Dict]` where each dict contains:

| Field Category | Fields | Description |
|----------------|--------|-------------|
| **Header Fields** | `time_tag`, `granularity`, `market`, `code`, `namespace` | Automatically included for all rows |
| **Custom Fields** | All fields from uout.json | Your indicator's output fields |

**time_tag Format**: Unix timestamp in milliseconds (e.g., `1672761600000`)

**Usage Note**: Use `time_tag` directly for time-series x-axis - automatically skips weekends/holidays.

### StructValue Indexing

**Index Components**: (meta_id, granularity, market, stock_code, time_tag)

| Tier | meta_id | granularity | market/stock_code Pattern | Output Count |
|------|---------|-------------|---------------------------|--------------|
| **Tier-1** | Fixed | Fixed | Multiple (m, s) pairs from uout.json | Many commodities |
| **Tier-2** | Fixed | Fixed | Single placeholder (m, `<00>` or `<01>`) | One synthetic |

**Tier-1 Pattern**:
- Outputs for ALL (market, security) where `commodity(security)` in `securities[market]` from uout.json
- Example: uout.json has `securities: [["i", "j"], ["cu", "al"]]`
- Outputs: `DCE/i<00>`, `DCE/j<00>`, `SHFE/cu<00>`, `SHFE/al<00>`
- **Visualization**: Pick ONE commodity as example (e.g., `cu<00>`)

**Tier-2 Pattern**:
- Single output with placeholder stock_code (e.g., `COMPOSITE<00>`)
- Real exposures defined in StructValue fields (markets, codes arrays)
- **Visualization**: Use single (market, placeholder_code) from uout.json

### svr3.sv_reader Signature

```python
client = svr3.sv_reader(
    start_date,              # int: YYYYMMDDHHmmss format
    end_date,                # int: YYYYMMDDHHmmss format
    meta_name,               # str: Indicator name (e.g., "MyIndicator")
    granularity,             # int: Seconds (e.g., 300 for 5min)
    namespace,               # str: "global" or "private"
    mode,                    # str: "symbol" (standard mode)
    markets,                 # List[str]: ["DCE", "SHFE", ...]
    codes,                   # List[str]: ["cu<00>", "i<00>", ...]
    persistent,              # bool: False (no auto-save to file)
    rails_url,               # str: Backend API URL
    ws_url,                  # str: WebSocket URL for data fetch
    username,                # str: "" (leave blank)
    password,                # str: "" (leave blank, use token)
    tm_master_endpoint,      # Tuple[str, int]: Time machine server
)
client.token = "YOUR_TOKEN"  # Set token after creation
```

### Pattern 1: Single Connection, Single Fetch

```python
import asyncio
import svr3
from typing import List, Dict

class IndicatorFetcher:
    def __init__(self, token: str):
        self.token = token
        self.start_date = 20230103203204
        self.end_date = 20230510203204

    async def connect_and_fetch(self, market: str, instrument: str) -> List[Dict]:
        """Connect and fetch data for one market/instrument pair"""

        print(f"🔄 Fetching {market}/{instrument}...")

        client = svr3.sv_reader(
            self.start_date,
            self.end_date,
            "MyIndicator",                              # Your indicator name
            300,                                        # 5-minute granularity
            "private",                                  # Namespace
            "symbol",                                   # Mode
            [market],                                   # Markets
            [instrument],                               # Instruments
            False,                                      # No persistent file
            "https://10.99.100.116:4433/private-api/", # Rails URL
            "wss://10.99.100.116:4433/tm",             # WebSocket URL
            "",                                         # Username (blank)
            "",                                         # Password (blank)
            ("10.99.100.116", 6102),                   # Time machine master
        )
        client.token = self.token

        await client.login()
        await client.connect()
        client.ws_task = asyncio.create_task(client.ws_loop())
        await client.shakehand()

        ret = await client.save_by_symbol()
        data = ret[1][1]  # Extract List[Dict] from result tuple

        # Cleanup
        client.stop()
        await client.join()

        return data  # List[Dict] with header fields + uout.json fields

# Usage (Interactive Mode - VS Code/Cursor notebooks)
if __name__ == '__main__':
    fetcher = IndicatorFetcher("YOUR_TOKEN")
    data = await fetcher.connect_and_fetch("SHFE", "cu<00>")
    print(f"Fetched {len(data)} records")

# Usage (Regular Mode - Python interpreter)
if __name__ == '__main__':
    async def main():
        fetcher = IndicatorFetcher("YOUR_TOKEN")
        data = await fetcher.connect_and_fetch("SHFE", "cu<00>")
        print(f"Fetched {len(data)} records")

    asyncio.run(main())
```

### Pattern 2: Single Connection, Multiple Fetches (Recommended)

**Use Case**: Fetch multiple market/instrument pairs efficiently by reusing connection.

```python
class MultiFetcher:
    def __init__(self, token: str):
        self.token = token
        self.start_date = 20230103203204
        self.end_date = 20230510203204
        self.client = None

    async def connect(self):
        """Establish connection (call once)"""

        self.client = svr3.sv_reader(
            self.start_date,
            self.end_date,
            "MyIndicator",
            300,
            "private",
            "symbol",
            ["DCE"],                                    # Initial market
            ["i<00>"],                                  # Initial instrument
            False,
            "https://10.99.100.116:4433/private-api/",
            "wss://10.99.100.116:4433/tm",
            "",
            "",
            ("10.99.100.116", 6102),
        )
        self.client.token = self.token

        await self.client.login()
        await self.client.connect()
        self.client.ws_task = asyncio.create_task(self.client.ws_loop())
        await self.client.shakehand()

    async def fetch(self, market: str, instrument: str) -> List[Dict]:
        """Fetch data by updating markets/codes and reusing connection"""

        self.client.markets = [market]
        self.client.codes = [instrument]

        ret = await self.client.save_by_symbol()
        return ret[1][1]  # List[Dict] with header + custom fields

    async def close(self):
        """Clean up connection"""

        self.client.stop()
        await self.client.join()

# Usage
async def main():
    fetcher = MultiFetcher("YOUR_TOKEN")

    await fetcher.connect()

    # Fetch multiple instruments
    cu_data = await fetcher.fetch("SHFE", "cu<00>")
    al_data = await fetcher.fetch("SHFE", "al<00>")
    i_data = await fetcher.fetch("DCE", "i<00>")

    await fetcher.close()

    print(f"Fetched cu: {len(cu_data)}, al: {len(al_data)}, i: {len(i_data)}")

if __name__ == '__main__':
    asyncio.run(main())
```

### Async/Await Usage Patterns

| Mode | Entry Pattern | Use Case |
|------|---------------|----------|
| **Interactive** | `await main_entry()` | VS Code/Cursor notebooks, interactive analysis |
| **Regular** | `asyncio.run(main_entry())` | Python interpreter, production scripts |

**Compatible Pattern** (works in both modes):

```python
async def main_entry():
    # ... async code ...
    pass

if __name__ == '__main__':
    try:
        # Interactive mode (notebooks)
        await main_entry()
    except SyntaxError:
        # Regular mode (interpreter)
        import asyncio
        asyncio.run(main_entry())
```

---

## Complete Visualization Example

```python
#!/usr/bin/env python3
"""Tier-1 Indicator Visualization"""

import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import asyncio
import svr3

class IndicatorVisualizer:
    def __init__(self, token: str, start: int, end: int):
        self.token = token
        self.start_date = start
        self.end_date = end
        self.client = None

    async def connect(self, indicator_name: str, granularity: int):
        """Connect to server"""

        self.client = svr3.sv_reader(
            self.start_date,
            self.end_date,
            indicator_name,
            granularity,
            "private",
            "symbol",
            ["SHFE"],
            ["cu<00>"],
            False,
            "https://10.99.100.116:4433/private-api/",
            "wss://10.99.100.116:4433/tm",
            "",
            "",
            ("10.99.100.116", 6102),
        )
        self.client.token = self.token

        await self.client.login()
        await self.client.connect()
        self.client.ws_task = asyncio.create_task(self.client.ws_loop())
        await self.client.shakehand()

    async def fetch(self, market: str, code: str) -> pd.DataFrame:
        """Fetch data and convert to DataFrame"""

        self.client.markets = [market]
        self.client.codes = [code]

        ret = await self.client.save_by_symbol()
        data = ret[1][1]  # List[Dict] with header + custom fields

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Convert time_tag (unix ms) to datetime for plotting
        df['datetime'] = pd.to_datetime(df['time_tag'], unit='ms')

        return df

    async def close(self):
        """Cleanup"""

        self.client.stop()
        await self.client.join()

def plot_indicator_signals(df):
    """Plot indicator signals over time (x-axis naturally skips weekends/holidays)."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    # Plot 1: Indicator values
    ax1 = axes[0]
    ax1.plot(df['datetime'], df['ema_fast'], label='EMA Fast', color='blue')
    ax1.plot(df['datetime'], df['ema_slow'], label='EMA Slow', color='red')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.set_title('Indicator Values')
    ax1.grid(True)

    # Plot 2: Signals
    ax2 = axes[1]
    ax2.scatter(df['datetime'], df['signal'], c=df['signal'],
                cmap='RdYlGn', alpha=0.6)
    ax2.set_ylabel('Signal')
    ax2.set_ylim(-1.5, 1.5)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_title('Trading Signals')
    ax2.grid(True)

    # Plot 3: Bar index
    ax3 = axes[2]
    ax3.fill_between(df['datetime'], 0, df['confidence'],
                     alpha=0.3, color='green')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Bar Index')
    ax3.set_title('Bar Counter')
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig('indicator_analysis.png', dpi=150)
    print("Saved: indicator_analysis.png")

def main():
    """Main analysis workflow."""
    print(f"Fetching data for {STRATEGY_NAME}...")

    # Run async fetch
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    df = loop.run_until_complete(fetch_indicator_data())
    loop.close()

    print(f"Loaded {len(df)} bars")

    # Generate plots
    if len(df) > 0:
        plot_indicator_signals(df)
        print("\nAnalysis complete!")
    else:
        print("No data to visualize")

if __name__ == "__main__":
    main()
```

## Key Visualizations

### 1. Indicator Time Series

```python
def plot_time_series(df):
    """Plot all indicator values over time (x-axis skips weekends/holidays)."""
    fields = ['ema_fast', 'ema_slow', 'tsi', 'vai', 'mdi']

    fig, axes = plt.subplots(len(fields), 1, figsize=(15, 3*len(fields)))

    for i, field in enumerate(fields):
        axes[i].plot(df['datetime'], df[field])
        axes[i].set_title(f'{field.upper()}')
        axes[i].grid(True)

    plt.tight_layout()
    plt.savefig('time_series.png', dpi=150)
```

### 2. Correlation Analysis

```python
def analyze_correlations(df):
    """Analyze correlations between indicators."""
    corr_matrix = df[['ema_fast', 'ema_slow', 'signal', 'confidence']].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Indicator Correlations')
    plt.tight_layout()
    plt.savefig('correlations.png', dpi=150)
```

### 3. Signal Performance

```python
def analyze_signal_performance(df):
    """Analyze signal performance (x-axis skips weekends/holidays)."""
    # Calculate returns
    df['returns'] = df['close'].pct_change()

    # Signal-based returns
    df['signal_returns'] = df['signal'].shift(1) * df['returns']

    # Cumulative returns
    cumulative = (1 + df['signal_returns']).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], cumulative)
    plt.title('Cumulative Strategy Returns')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.savefig('performance.png', dpi=150)
```

## Parameter Analysis

### Finding Optimal Parameters

```python
def parameter_sweep(start, end):
    """Test different parameter combinations."""
    results = []

    # Test different EMA periods
    for fast in [5, 10, 15, 20]:
        for slow in [20, 30, 40, 50]:
            if slow <= fast:
                continue

            # Fetch data with these parameters
            df = run_backtest_with_params(fast, slow)

            # Calculate performance metrics
            sharpe = calculate_sharpe_ratio(df)
            max_dd = calculate_max_drawdown(df)
            win_rate = calculate_win_rate(df)

            results.append({
                'fast': fast,
                'slow': slow,
                'sharpe': sharpe,
                'max_dd': max_dd,
                'win_rate': win_rate
            })

    return pd.DataFrame(results)

def plot_parameter_heatmap(results_df):
    """Visualize parameter sweep results."""
    pivot = results_df.pivot(index='fast', columns='slow', values='sharpe')

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn')
    plt.title('Sharpe Ratio by EMA Parameters')
    plt.xlabel('Slow EMA Period')
    plt.ylabel('Fast EMA Period')
    plt.savefig('parameter_sweep.png', dpi=150)
```

---

## Market/Instrument Selection Guide

| Indicator Type | uout.json Pattern | Visualization Query | Example |
|----------------|-------------------|---------------------|---------|
| **Tier-1** | `securities: [["i", "j"], ["cu", "al"]]` | Pick ONE commodity | `("SHFE", "cu<00>")` |
| **Tier-2** | `securities: [["COMPOSITE"]]` | Use placeholder | `("DCE", "COMPOSITE<00>")` |

**Tier-1 Selection Strategy**:
```python
# Check uout.json securities field
# Example: securities: [["i", "j"], ["cu", "al", "rb"]]

# Pick one from each market for analysis
tier1_samples = [
    ("DCE", "i<00>"),      # Iron ore
    ("SHFE", "cu<00>"),    # Copper
]
```

**Tier-2 Selection Strategy**:
```python
# Tier-2 has single synthetic instrument
# Query uses market + placeholder from uout.json

tier2_query = ("DCE", "COMPOSITE<00>")  # Or whatever placeholder in uout.json
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Empty data returned | Wrong market/code combination | Check uout.json securities field, use correct (market, code) |
| Connection timeout | Wrong rails_url or ws_url | Verify server endpoints, check network |
| Token error | Invalid or expired token | Get fresh token from server admin |
| Missing indicator | Backtest not run or failed | Run backtest first, check logs |
| Wrong granularity | Indicator outputs different granularity | Match granularity with uout.json sample_granularities |

**Debugging Steps**:
1. Verify indicator ran successfully (check server logs)
2. Confirm market/code exists in uout.json
3. Check date range has data (align with backtest period)
4. Validate token with simple query
5. Test connection with known working indicator first

---

## Data Format and Time Handling

### Returned Data Structure

**save_by_symbol() returns**: `List[Dict]` where each dictionary contains:

```python
{
    # Header fields (automatically included)
    'time_tag': 1672761600000,      # Unix timestamp in milliseconds
    'granularity': 900,              # Granularity in seconds
    'market': 'SHFE',                # Market identifier
    'code': 'cu<00>',                # Instrument code
    'namespace': 'private',          # Namespace

    # Custom fields from uout.json
    'ema_fast': 123.45,
    'ema_slow': 120.30,
    'signal': 1.0,
    # ... all other fields defined in uout.json
}
```

### Time Axis Best Practice

**Critical**: Use `time_tag` for x-axis to naturally skip weekends/holidays.

```python
# Convert to DataFrame
df = pd.DataFrame(data)

# Convert time_tag (unix ms) to datetime
df['datetime'] = pd.to_datetime(df['time_tag'], unit='ms')

# Plot time series (automatically skips weekends/holidays)
plt.plot(df['datetime'], df['my_field'])
plt.xlabel('Time')
plt.show()
```

**Why this works**: The returned data only contains bars where market was open. Using `time_tag` directly creates properly spaced x-axis without manual weekend/holiday filtering.

**Anti-pattern** ❌:
```python
# DON'T create artificial continuous time axis
date_range = pd.date_range(start, end, freq='15min')  # Includes weekends!
```

**Best practice** ✅:
```python
# DO use actual time_tag values from data
df['datetime'] = pd.to_datetime(df['time_tag'], unit='ms')  # Only market hours
```

---

## Summary

**Core Concepts**:
- svr3 async module fetches calculated data from server as `List[Dict]`
- Each dict contains header fields (`time_tag`, `granularity`, `market`, `code`, `namespace`) + custom fields from uout.json
- Tier-1: Multiple (market, commodity) outputs → pick ONE for visualization
- Tier-2: Single placeholder output → use market + placeholder from uout.json
- Connection lifecycle: connect() → fetch() → close()
- Two patterns: single fetch or reuse connection for multiple fetches

**Critical Requirements**:
1. Use `svr3.sv_reader()` with 12 parameters (not `svr3.Client()`)
2. Set `client.token` after creation
3. Call `login() → connect() → ws_loop() → shakehand()` before fetching
4. Use `client.save_by_symbol()` to fetch data, result in `ret[1][1]` (List[Dict])
5. Convert `time_tag` (unix ms) to datetime: `pd.to_datetime(df['time_tag'], unit='ms')`
6. Use `time_tag`-derived datetime for x-axis (auto-skips weekends/holidays)
7. Call `client.stop()` and `await client.join()` for cleanup
8. Match market/code with uout.json securities field
9. Use async/await patterns (interactive vs regular mode)

**Key Tools**:
- svr3: Server data fetching
- pandas: Data manipulation
- matplotlib/seaborn: Plotting
- asyncio: Async execution

**Reference**: See MargaritaComposite/margarita_composite_viz.py for working Tier-2 example

**Next**: Iterative optimization workflow.

---

**Previous:** [09 - Tier 3 Strategy](09-tier3-strategy.md) | **Next:** [11 - Fine-tune and Iterate](11-fine-tune-and-iterate.md)
