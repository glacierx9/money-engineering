# Chapter 10: Visualization and Analysis

**Learning objectives:**
- Create visualization scripts for indicator analysis
- Connect to svr3 server and authenticate
- Fetch data using svr3 API properly
- Analyze parameter performance
- Generate diagnostic plots
- Optimize indicator parameters

**Previous:** [09 - Tier 3 Strategy](09-tier3-strategy.md) | **Next:** [11 - Fine-tune and Iterate](11-fine-tune-and-iterate.md)

---

## Overview

Visualization helps understand indicator behavior, identify issues, and optimize parameters. This chapter shows how to create analysis scripts using the svr3 API with proper connection handling and authentication.

## SVR3 Server Configuration

### Environment Setup

**IMPORTANT**: Store sensitive credentials in environment variables:

```bash
# Add to your .bashrc or .zshrc
export SVR_HOST="10.99.100.116"
export SVR_TOKEN="your_token_here"
```

Load in your script:

```python
import os

SVR_HOST = os.getenv("SVR_HOST", "10.99.100.116")
TOKEN = os.getenv("SVR_TOKEN")
if not TOKEN:
    raise ValueError("SVR_TOKEN environment variable not set")
```

### Connection Parameters

The svr3 server requires several connection parameters:

```python
# Server configuration
SVR_HOST = os.getenv("SVR_HOST", "10.99.100.116")
RAILS_URL = f"https://{SVR_HOST}:4433/private-api/"
WS_URL = f"wss://{SVR_HOST}:4433/tm"  # Note: /tm endpoint (not /ws)
TM_MASTER = (SVR_HOST, 6102)  # Tuple format: (host, port)
TOKEN = os.getenv("SVR_TOKEN")

# Data configuration
GRANULARITY = 900  # 15 minutes
NAMESPACE = "global"  # "global" for SampleQuote (raw market data)
                      # "private" for your custom indicators
```

### Key Points

1. **Namespace**:
   - `"global"`: For raw market data (SampleQuote, OHLCV)
   - `"private"`: For your custom indicators/strategies

2. **Time Format**:
   - Must be integers in YYYYMMDDHHMMSS format
   - Example: `20251016000000` (not strings!)

3. **Endpoints**:
   - Rails: `https://{host}:4433/private-api/`
   - WebSocket: `wss://{host}:4433/tm` (not `/ws`)
   - TM Master: Tuple `(host, 6102)`

## Connection and Data Fetching

### Complete Connection Sequence

The connection to svr3 requires a specific sequence of steps:

```python
import asyncio
import svr3
import pandas as pd
from typing import Optional, Dict, List

async def fetch_data_from_server(
    market: str,
    code: str,
    start_date: str,  # "YYYYMMDD"
    end_date: str,    # "YYYYMMDD"
    algoname: str = "SampleQuote",
    namespace: str = "global"
) -> Optional[pd.DataFrame]:
    """
    Fetch data from svr3 server with proper connection sequence.

    Args:
        market: Market name (e.g., "DCE", "SHFE", "CZCE")
        code: Instrument code (e.g., "i<00>", "cu<00>", "SR<00>")
        start_date: Start date as YYYYMMDD string
        end_date: End date as YYYYMMDD string
        algoname: Algorithm/data source name
        namespace: "global" for raw data, "private" for custom indicators

    Returns:
        DataFrame with fetched data, or None if fetch fails
    """
    try:
        # Convert dates to integer timestamps
        start_time = int(start_date + "000000")  # e.g., 20251016000000
        end_time = int(end_date + "235959")      # e.g., 20251030235959

        # Load connection parameters from environment
        svr_host = os.getenv("SVR_HOST")
        token = os.getenv("SVR_TOKEN")

        rails_url = f"https://{svr_host}:4433/private-api/"
        ws_url = f"wss://{svr_host}:4433/tm"
        tm_master = (svr_host, 6102)

        # Create sv_reader instance
        # IMPORTANT: Arguments are positional, order matters!
        reader = svr3.sv_reader(
            start_time,              # start_date (int)
            end_time,                # end_date (int)
            algoname,                # algorithm name
            GRANULARITY,             # granularity (900s = 15min)
            namespace,               # namespace ("global" or "private")
            "symbol",                # work_mode (always "symbol")
            [market],                # markets (list)
            [code],                  # codes (list of logical contracts)
            False,                   # persistent (False for one-time fetch)
            rails_url,               # rails_url
            ws_url,                  # ws_url
            "",                      # user (empty string)
            "",                      # hashed_password (empty string)
            tm_master,               # tm_master (tuple)
        )

        # Set token after initialization (not in constructor)
        reader.token = token

        # Connection sequence - EXACT ORDER REQUIRED:
        # 1. Login to authenticate
        await reader.login()

        # 2. Connect to the server
        await reader.connect()

        # 3. Start WebSocket loop in background
        reader.ws_task = asyncio.create_task(reader.ws_loop())

        # 4. Perform handshake
        await reader.shakehand()

        # Fetch data using save_by_symbol (not fetch_by_code)
        print(f"   📥 Fetching data for {market}/{code}...")
        ret = await reader.save_by_symbol()

        # Extract data from result
        # Result structure: ret[1][1] contains the data
        data = ret[1][1]

        if not data or len(data) == 0:
            print(f"   ⚠️  No data returned")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Parse timestamps if needed
        if 'time_tag' in df.columns:
            try:
                import pycaitlynutils3 as pcu3
                df['timestamp'] = pd.to_datetime(df['time_tag'].apply(pcu3.ts_parse))
            except:
                # Fallback if pcu3 not available
                if df['time_tag'].dtype in ['int64', 'float64']:
                    df['timestamp'] = pd.to_datetime(df['time_tag'], unit='ms')
                else:
                    df['timestamp'] = pd.to_datetime(df['time_tag'])

        print(f"   ✓ Fetched {len(df)} bars")
        return df

    except Exception as e:
        import traceback
        print(f"   ❌ Error fetching data: {e}")
        print(traceback.format_exc())
        return None
```

### Running Async Functions

Handle both Jupyter and standalone environments:

```python
# For standalone scripts
def fetch_data_sync(*args, **kwargs):
    """Synchronous wrapper for async fetch function."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # In Jupyter with nest_asyncio
            return loop.run_until_complete(fetch_data_from_server(*args, **kwargs))
        else:
            return loop.run_until_complete(fetch_data_from_server(*args, **kwargs))
    except RuntimeError:
        # No event loop, create new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(fetch_data_from_server(*args, **kwargs))
        loop.close()
        return result

# Usage
df = fetch_data_sync(
    market="DCE",
    code="i<00>",
    start_date="20251016",
    end_date="20251030",
    algoname="SampleQuote",
    namespace="global"
)
```

### Jupyter Notebook Support

For interactive notebooks, install nest_asyncio:

```python
try:
    import nest_asyncio
    nest_asyncio.apply()
    print("✓ nest_asyncio applied (Jupyter compatibility)")
except ImportError:
    print("ℹ️  Install with: pip install nest-asyncio")
```

## Visualization Script Template

```python
#!/usr/bin/env python3
"""Indicator Visualization Script"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import asyncio
import svr3

# Configuration - Load from environment
SVR_HOST = os.getenv("SVR_HOST", "10.99.100.116")
TOKEN = os.getenv("SVR_TOKEN")
if not TOKEN:
    raise ValueError("SVR_TOKEN environment variable not set!")

RAILS_URL = f"https://{SVR_HOST}:4433/private-api/"
WS_URL = f"wss://{SVR_HOST}:4433/tm"
TM_MASTER = (SVR_HOST, 6102)

# Data configuration
STRATEGY_NAME = "IronOreIndicator"  # Your indicator name
NAMESPACE = "private"  # "private" for custom indicators
MARKET = "DCE"
CODE = "i<00>"
GRANULARITY = 900

# Time range (YYYYMMDD format)
START_DATE = "20251016"
END_DATE = "20251030"

async def fetch_indicator_data():
    """Fetch indicator data using svr3."""
    start_time = int(START_DATE + "000000")
    end_time = int(END_DATE + "235959")

    # Create reader
    reader = svr3.sv_reader(
        start_time, end_time,
        STRATEGY_NAME,
        GRANULARITY,
        NAMESPACE,
        "symbol",
        [MARKET],
        [CODE],
        False,
        RAILS_URL,
        WS_URL,
        "", "",
        TM_MASTER,
    )
    reader.token = TOKEN

    # Connection sequence
    await reader.login()
    await reader.connect()
    reader.ws_task = asyncio.create_task(reader.ws_loop())
    await reader.shakehand()

    # Fetch data
    ret = await reader.save_by_symbol()
    data = ret[1][1]

    return pd.DataFrame(data)

def plot_indicator_signals(df):
    """Plot indicator signals over time."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    # Plot 1: Indicator values
    ax1 = axes[0]
    ax1.plot(df['timestamp'], df['indicator_value'], label='Indicator', color='blue')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.set_title('Indicator Values')
    ax1.grid(True)

    # Plot 2: Signals
    ax2 = axes[1]
    ax2.scatter(df['timestamp'], df['signal'], c=df['signal'],
                cmap='RdYlGn', alpha=0.6)
    ax2.set_ylabel('Signal')
    ax2.set_ylim(-1.5, 1.5)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_title('Trading Signals')
    ax2.grid(True)

    # Plot 3: Bar index
    ax3 = axes[2]
    ax3.plot(df['timestamp'], df['bar_index'], color='green')
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
    """Plot all indicator values over time."""
    fields = ['ema_fast', 'ema_slow', 'tsi', 'vai', 'mdi']

    fig, axes = plt.subplots(len(fields), 1, figsize=(15, 3*len(fields)))

    for i, field in enumerate(fields):
        axes[i].plot(df['timestamp'], df[field])
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
    """Analyze signal performance."""
    # Calculate returns
    df['returns'] = df['close'].pct_change()

    # Signal-based returns
    df['signal_returns'] = df['signal'].shift(1) * df['returns']

    # Cumulative returns
    cumulative = (1 + df['signal_returns']).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], cumulative)
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

## Summary

Visualization helps:
- Understand indicator behavior
- Identify parameter issues
- Optimize performance
- Debug problems
- Communicate results

Key tools:
- svr3 API for data fetching
- pandas for data manipulation
- matplotlib/seaborn for plotting

**Next**: Iterative optimization workflow.

---

**Previous:** [09 - Tier 3 Strategy](09-tier3-strategy.md) | **Next:** [11 - Fine-tune and Iterate](11-fine-tune-and-iterate.md)
