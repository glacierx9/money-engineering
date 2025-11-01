#!/usr/bin/env python3
# coding=utf-8
# %%
"""
# Multi-Instrument Technical Indicator Visualization (N×M Grid)

This script visualizes multiple instruments across multiple indicators in an N×M grid layout:
- N rows: One row per instrument
- M columns: One column per indicator type

**How to use:**
- In VSCode, open this file and it will be recognized as an interactive notebook
- Click "Run Cell" or use Shift+Enter to execute cells
- Plots will display inline in the interactive window
"""

# %% [markdown]
"""
## 📊 Setup and Configuration
"""

# %%
# Import required libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import svr3 for real data fetching
try:
    import svr3
    SVR3_AVAILABLE = True
    print("✓ svr3 module available")
except ImportError:
    SVR3_AVAILABLE = False
    print("⚠️  svr3 module not available - will use sample data")

# Try to import nest_asyncio for Jupyter compatibility
try:
    import nest_asyncio
    nest_asyncio.apply()
    print("✓ nest_asyncio applied (Jupyter compatibility)")
except ImportError:
    print("ℹ️  nest_asyncio not available (install with: pip install nest-asyncio)")

# Configure inline plotting for interactive mode
try:
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
except:
    pass

# Set style for better-looking plots
sns.set_style("darkgrid")
plt.rcParams['font.size'] = 9

print("✓ Libraries imported successfully")

# %%
# Configuration parameters
GRANULARITY = 900  # 15 minutes
NAMESPACE = "global"  # SampleQuote is in GLOBAL namespace, not private!

# Time range: 2 WEEKS focused analysis (past 2 weeks till yesterday)
# Today is 2025-10-31, so yesterday is 2025-10-30
END_DATE = "20251030"    # October 30, 2025 (yesterday)
START_DATE = "20251016"  # October 16, 2025 (2 weeks before yesterday)

# Output directory setup - timestamped subfolder (created only when saving images)
from datetime import datetime as dt
OUTPUT_DIR = f"output_images/{dt.now().strftime('%Y-%m-%d_%H%M%S')}_run"
_output_dir_created = False

def save_figure(filename, **kwargs):
    """Save figure to OUTPUT_DIR, creating directory only when needed."""
    global _output_dir_created
    if not _output_dir_created:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"📁 Output directory created: {OUTPUT_DIR}")
        _output_dir_created = True
    full_path = f"{OUTPUT_DIR}/{filename}"
    plt.savefig(full_path, **kwargs)
    print(f"💾 Saved: {full_path}")
    return full_path

# SVR3 Server Configuration - Load from environment variables
SVR_HOST = os.getenv("SVR_HOST", "10.99.100.116")
TOKEN = os.getenv("SVR_TOKEN")

if not TOKEN:
    raise ValueError(
        "SVR_TOKEN environment variable not set!\n"
        "Please set it with: export SVR_TOKEN='your_token_here'\n"
        "Or add it to your .bashrc/.zshrc file"
    )

# Derive connection URLs from host
RAILS_URL = f"https://{SVR_HOST}:4433/private-api/"
WS_URL = f"wss://{SVR_HOST}:4433/tm"  # Note: /tm endpoint (not /ws)
TM_MASTER = (SVR_HOST, 6102)  # Tuple format

print(f"📡 Server configuration:")
print(f"   Host: {SVR_HOST}")
print(f"   Rails: {RAILS_URL}")
print(f"   WS: {WS_URL}")
print(f"   TM Master: {TM_MASTER}")
print(f"   Token: {'✓ Loaded from environment' if TOKEN else '✗ Not set'}")

# Data source preference
USE_REAL_DATA = True  # Set to True to fetch from server

# Define instruments from different markets
# SampleQuote (global namespace) has data for all commodities
INSTRUMENTS = [
    {"market": "DCE", "code": "i<00>", "name": "Iron Ore"},
    {"market": "DCE", "code": "m<00>", "name": "Soybean Meal"},
    {"market": "SHFE", "code": "rb<00>", "name": "Rebar"},
    {"market": "SHFE", "code": "cu<00>", "name": "Copper"},
    {"market": "CZCE", "code": "SR<00>", "name": "Sugar"},
    {"market": "CZCE", "code": "TA<00>", "name": "PTA"},
]

print(f"📅 Analysis period: {START_DATE} to {END_DATE} (2 weeks)")
print(f"📊 Instruments: {len(INSTRUMENTS)}")
for inst in INSTRUMENTS:
    print(f"   • {inst['market']}/{inst['code']}: {inst['name']}")
print(f"📡 Data source: {'Real (svr3)' if USE_REAL_DATA and SVR3_AVAILABLE else 'Sample (generated)'}")

# %% [markdown]
"""
## 📥 Data Loading Functions
"""

# %%
def fetch_real_data_for_instrument(
    instrument: Dict,
    start_date: str,
    end_date: str
) -> Optional[pd.DataFrame]:
    """
    Fetch real market data from svr3 server.

    Args:
        instrument: Dictionary with market, code, name
        start_date: Start date as YYYYMMDD string
        end_date: End date as YYYYMMDD string

    Returns:
        DataFrame with real market data, or None if fetch fails
    """
    if not SVR3_AVAILABLE:
        return None

    try:
        import asyncio

        # Format time as integers (YYYYMMDDHHMMSS) - svr3 requires int, not string!
        start_time = int(start_date + "000000")  # e.g., 20241217000000
        end_time = int(end_date + "235959")      # e.g., 20241231235959

        print(f"   🔌 Connecting to svr3 reader...")

        async def fetch_data():
            """Async function to fetch data from svr3 - following margarita_viz.py pattern"""
            # Create sv_reader instance (positional arguments like margarita_viz.py)
            # Fetching from SampleQuote (global namespace - raw OHLCV market data)
            reader = svr3.sv_reader(
                start_time,              # start_date
                end_time,                # end_date
                "SampleQuote",           # algoname - Raw OHLCV market data (global namespace)
                GRANULARITY,             # granularity (900s = 15min)
                NAMESPACE,               # ns ("global" for SampleQuote)
                "symbol",                # work_mode
                [instrument['market']],  # markets
                [instrument['code']],    # codes (logical contract like "i<00>")
                False,                   # persistent
                RAILS_URL,               # rails_url
                WS_URL,                  # ws_url
                "",                      # user (empty string)
                "",                      # hashed_password (empty string)
                TM_MASTER,               # tm_master (tuple)
            )
            # Set token after initialization
            reader.token = TOKEN

            # Connection sequence (exact order from margarita_viz.py)
            await reader.login()
            await reader.connect()
            reader.ws_task = asyncio.create_task(reader.ws_loop())
            await reader.shakehand()

            # Fetch data using save_by_symbol (not fetch_by_code)
            print(f"   📥 Fetching data...")
            ret = await reader.save_by_symbol()

            # Debug: Check what was returned
            print(f"   📊 Result structure: type={type(ret)}, len={len(ret) if hasattr(ret, '__len__') else 'N/A'}")

            data = ret[1][1]  # Extract data from result
            print(f"   📊 Data extracted: {len(data)} records")

            return data

        # Run async function - handle both Jupyter and standalone environments
        try:
            # Try to get existing event loop (Jupyter/IPython)
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # In Jupyter, nest_asyncio should allow this
                raw_data = loop.run_until_complete(fetch_data())
            else:
                # Loop exists but not running
                raw_data = loop.run_until_complete(fetch_data())
        except RuntimeError:
            # No event loop exists, create a new one (standalone script)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            raw_data = loop.run_until_complete(fetch_data())
            loop.close()

        if not raw_data or len(raw_data) == 0:
            print(f"   ⚠️  No data returned from server")
            print(f"   💡 Possible reasons:")
            print(f"      1. No data available for this time period ({start_date} - {end_date})")
            print(f"      2. Market was closed during this period")
            print(f"      3. Instrument {instrument['code']} not available on {instrument['market']}")
            print(f"   💡 Tip: Try a different date range or check if market was trading")
            return None

        # Convert raw data (list of dicts) to DataFrame
        df = pd.DataFrame(raw_data)

        # Add instrument metadata
        df['market'] = instrument['market']
        df['code'] = instrument['code']
        df['name'] = instrument['name']

        # Convert time_tag to datetime (following margarita pattern)
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
        elif 'timestamp' in df.columns:
            if df['timestamp'].dtype in ['int64', 'float64']:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        print(f"   ✓ Fetched {len(df)} bars from server")
        return df

    except Exception as e:
        import traceback
        print(f"   ❌ Error fetching from svr3: {e}")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Debug traceback:")
        print(traceback.format_exc())
        print(f"   📋 Debug info:")
        print(f"      Market: {instrument['market']}")
        print(f"      Code: {instrument['code']}")
        print(f"      Start: {start_date} → {int(start_date + '000000')}")
        print(f"      End: {end_date} → {int(end_date + '235959')}")
        return None


def generate_sample_data_for_instrument(
    instrument: Dict,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Generate sample OHLCV data for a specific instrument over 2 weeks.

    Args:
        instrument: Dictionary with market, code, name
        start_date: Start date as YYYYMMDD string
        end_date: End date as YYYYMMDD string
    """
    # Parse dates
    start = pd.to_datetime(start_date, format='%Y%m%d')
    end = pd.to_datetime(end_date, format='%Y%m%d')

    # Generate 15-minute bars for 2 weeks (approximately 8 bars/hour * 8 hours/day * 14 days)
    n_bars = int((end - start).days * 64)  # Approximate trading bars per day
    dates = pd.date_range(start=start, periods=n_bars, freq='15T')

    # Use different seed per instrument for variety
    seed = hash(instrument['code']) % 10000
    np.random.seed(seed)

    n = len(dates)

    # Base price varies by commodity
    base_prices = {
        'i<00>': 800,    # Iron Ore
        'm<00>': 3500,   # Soybean Meal
        'rb<00>': 3800,  # Rebar
        'cu<00>': 68000, # Copper
        'SR<00>': 6000,  # Sugar
        'TA<00>': 5800,  # PTA
    }
    base_price = base_prices.get(instrument['code'], 1000)

    # Generate realistic price movement
    trend = np.linspace(0, base_price * 0.05, n)  # 5% trend over period
    seasonal = (base_price * 0.02) * np.sin(2 * np.pi * np.arange(n) / 96)  # Daily cycle
    random_walk = np.cumsum(np.random.randn(n) * base_price * 0.003)  # Random volatility

    # Combine components
    price = base_price + trend + seasonal + random_walk

    # Generate OHLCV data
    df = pd.DataFrame({
        'timestamp': dates,
        'market': instrument['market'],
        'code': instrument['code'],
        'name': instrument['name'],
        'open': price + np.random.randn(n) * base_price * 0.002,
        'high': price + np.abs(np.random.randn(n)) * base_price * 0.005,
        'low': price - np.abs(np.random.randn(n)) * base_price * 0.005,
        'close': price,
        'volume': np.abs(np.random.randn(n)) * 10000 + 50000,
    })

    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    df['turnover'] = df['close'] * df['volume']

    return df


def load_all_instruments(instruments: List[Dict], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Load data for all instruments from svr3 server.
    Raises error if data cannot be fetched.

    Returns:
        Dictionary mapping instrument key to DataFrame
    """
    if not USE_REAL_DATA:
        raise ValueError("USE_REAL_DATA is set to False. Set it to True to fetch real data.")

    if not SVR3_AVAILABLE:
        raise ImportError("svr3 module not available. Cannot fetch real data.")

    data = {}

    for inst in instruments:
        key = f"{inst['market']}_{inst['code']}"
        print(f"📊 Loading {inst['market']}/{inst['code']} ({inst['name']})...")

        # Fetch real data from server
        print(f"   🌐 Fetching real data from server...")
        df = fetch_real_data_for_instrument(inst, start_date, end_date)

        if df is None:
            raise RuntimeError(
                f"Failed to fetch data for {inst['market']}/{inst['code']}. "
                f"Check server connection and credentials."
            )

        data[key] = df
        print(f"   ✓ Loaded {len(df)} bars\n")

    return data

print("✓ Data loading functions defined")

# %% [markdown]
"""
## 📊 Load Data for All Instruments
"""

# %%
print("🔄 Loading data for all instruments...")
instrument_data = load_all_instruments(INSTRUMENTS, START_DATE, END_DATE)

print(f"\n✅ Data loaded successfully!")
print(f"   • Total instruments: {len(instrument_data)}")
for key, df in instrument_data.items():
    print(f"   • {key}: {len(df):,} bars")

# %% [markdown]
"""
## 🧮 Technical Indicator Calculations
"""

# %%
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators for a dataframe.

    Indicators calculated:
    - SMA (20, 50)
    - EMA (20)
    - Smoothed indicators: HMA, KAMA, TEMA, SuperSmoother
    - RSI (14) with smoothed version
    - MACD with smoothed histogram
    - Bollinger Bands
    - ATR
    - Volume indicators
    - Combined buy/sell signals
    """
    df = df.copy()

    # Moving Averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # Hull Moving Average (HMA) - Very smooth, low lag
    period = 20
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    wma_half = df['close'].rolling(window=half_period).apply(
        lambda x: np.dot(x, np.arange(1, half_period + 1)) / np.arange(1, half_period + 1).sum(), raw=True
    )
    wma_full = df['close'].rolling(window=period).apply(
        lambda x: np.dot(x, np.arange(1, period + 1)) / np.arange(1, period + 1).sum(), raw=True
    )
    raw_hma = 2 * wma_half - wma_full
    df['hma_20'] = raw_hma.rolling(window=sqrt_period).apply(
        lambda x: np.dot(x, np.arange(1, sqrt_period + 1)) / np.arange(1, sqrt_period + 1).sum(), raw=True
    )

    # Kaufman Adaptive Moving Average (KAMA) - Adapts to volatility
    def kama(series, period=20, fast=2, slow=30):
        change = abs(series - series.shift(period))
        volatility = series.diff().abs().rolling(period).sum()
        er = change / volatility  # Efficiency Ratio
        sc = (er * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2  # Smoothing Constant

        kama_values = np.zeros(len(series))
        kama_values[period - 1] = series.iloc[period - 1]

        for i in range(period, len(series)):
            kama_values[i] = kama_values[i - 1] + sc.iloc[i] * (series.iloc[i] - kama_values[i - 1])

        return pd.Series(kama_values, index=series.index)

    df['kama_20'] = kama(df['close'])

    # Triple Exponential Moving Average (TEMA) - Extra smooth
    ema1 = df['close'].ewm(span=20, adjust=False).mean()
    ema2 = ema1.ewm(span=20, adjust=False).mean()
    ema3 = ema2.ewm(span=20, adjust=False).mean()
    df['tema_20'] = 3 * ema1 - 3 * ema2 + ema3

    # SuperSmoother Filter - Removes high-frequency noise
    def supersmoother(series, period=10):
        a1 = np.exp(-np.sqrt(2) * np.pi / period)
        b1 = 2 * a1 * np.cos(np.sqrt(2) * np.pi / period)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3

        ss = np.zeros(len(series))
        for i in range(2, len(series)):
            ss[i] = c1 * (series.iloc[i] + series.iloc[i-1]) / 2 + c2 * ss[i-1] + c3 * ss[i-2]

        return pd.Series(ss, index=series.index)

    df['supersmoother'] = supersmoother(df['close'])

    # RSI (original and smoothed)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_smooth'] = df['rsi'].ewm(span=5, adjust=False).mean()  # Smoothed RSI

    # MACD with smoothed histogram
    df['macd_line'] = df['close'].ewm(span=12, adjust=False).mean() - \
                      df['close'].ewm(span=26, adjust=False).mean()
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd_line'] - df['macd_signal']
    df['macd_hist_smooth'] = df['macd_histogram'].ewm(span=3, adjust=False).mean()

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (rolling_std * 2)
    df['bb_lower'] = df['bb_middle'] - (rolling_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']  # Normalized width

    # ATR
    df['h_l'] = df['high'] - df['low']
    df['h_pc'] = abs(df['high'] - df['close'].shift(1))
    df['l_pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    df['atr_percent'] = (df['atr'] / df['close']) * 100  # ATR as percentage of price
    df.drop(['h_l', 'h_pc', 'l_pc', 'tr'], axis=1, inplace=True)

    # Volume indicators
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']

    # COMBINED BUY/SELL SIGNALS
    # Initialize signal as neutral
    df['signal'] = 0
    df['signal_strength'] = 0.0

    # Trend signals (smoothed MA crossovers)
    df['trend_signal'] = 0
    df.loc[df['hma_20'] > df['tema_20'], 'trend_signal'] = 1  # Bullish
    df.loc[df['hma_20'] < df['tema_20'], 'trend_signal'] = -1  # Bearish

    # Momentum signals (smoothed RSI)
    df['momentum_signal'] = 0
    df.loc[df['rsi_smooth'] < 30, 'momentum_signal'] = 1  # Oversold - buy
    df.loc[df['rsi_smooth'] > 70, 'momentum_signal'] = -1  # Overbought - sell
    df.loc[(df['rsi_smooth'] > 40) & (df['rsi_smooth'] < 60), 'momentum_signal'] = 0  # Neutral zone

    # MACD signals (smoothed histogram)
    df['macd_signal_flag'] = 0
    df.loc[df['macd_hist_smooth'] > 0, 'macd_signal_flag'] = 1  # Bullish
    df.loc[df['macd_hist_smooth'] < 0, 'macd_signal_flag'] = -1  # Bearish

    # Bollinger Band signals
    df['bb_signal'] = 0
    df.loc[df['close'] < df['bb_lower'], 'bb_signal'] = 1  # Below lower band - buy
    df.loc[df['close'] > df['bb_upper'], 'bb_signal'] = -1  # Above upper band - sell

    # Volume confirmation
    df['volume_confirm'] = df['volume_ratio'] > 1.2  # High volume

    # COMBINED SIGNAL: Consensus of multiple indicators
    # Buy signal: Need at least 2 bullish signals
    buy_score = (
        (df['trend_signal'] == 1).astype(int) +
        (df['momentum_signal'] == 1).astype(int) +
        (df['macd_signal_flag'] == 1).astype(int) +
        (df['bb_signal'] == 1).astype(int)
    )

    # Sell signal: Need at least 2 bearish signals
    sell_score = (
        (df['trend_signal'] == -1).astype(int) +
        (df['momentum_signal'] == -1).astype(int) +
        (df['macd_signal_flag'] == -1).astype(int) +
        (df['bb_signal'] == -1).astype(int)
    )

    # Generate final signal
    df.loc[buy_score >= 2, 'signal'] = 1  # BUY
    df.loc[sell_score >= 2, 'signal'] = -1  # SELL

    # Signal strength (0-4)
    df['signal_strength'] = buy_score - sell_score

    # Volume-confirmed signals (stronger signals)
    df['confirmed_signal'] = 0
    df.loc[(df['signal'] == 1) & df['volume_confirm'], 'confirmed_signal'] = 1
    df.loc[(df['signal'] == -1) & df['volume_confirm'], 'confirmed_signal'] = -1

    return df


print("🧮 Calculating indicators for all instruments...")
for key in instrument_data.keys():
    instrument_data[key] = calculate_indicators(instrument_data[key])
    print(f"   ✓ {key} indicators calculated")

print("✅ All indicators calculated!")

# %% [markdown]
"""
## 📊 N×M Grid Visualization

Create a comprehensive N×M grid where:
- **N rows**: One per instrument (6 instruments)
- **M columns**: One per indicator type (5 indicators)

Indicators displayed:
1. Price with Moving Averages (SMA 20, 50, EMA 20)
2. RSI (Relative Strength Index)
3. MACD (Moving Average Convergence Divergence)
4. Bollinger Bands
5. Volume Analysis
"""

# %%
def create_nm_grid_visualization(
    data: Dict[str, pd.DataFrame],
    instruments: List[Dict]
):
    """
    Create N×M grid: N instruments × M indicators
    """
    N = len(instruments)  # Number of instruments (rows)
    M = 5  # Number of indicators (columns)

    # Create figure with subplots
    fig, axes = plt.subplots(N, M, figsize=(24, 4*N))

    # Ensure axes is 2D
    if N == 1:
        axes = axes.reshape(1, -1)

    # Column titles (indicator types)
    indicator_titles = [
        'Price & Moving Averages',
        'RSI',
        'MACD',
        'Bollinger Bands',
        'Volume Analysis'
    ]

    # Plot each instrument (row)
    for i, inst in enumerate(instruments):
        key = f"{inst['market']}_{inst['code']}"
        df = data[key]

        # Row label (instrument name)
        row_label = f"{inst['market']}/{inst['code']}\n{inst['name']}"

        # Column 0: Price with MAs
        ax = axes[i, 0]
        ax.plot(df['timestamp'], df['close'], label='Close',
                color='black', linewidth=1.2, alpha=0.9)
        ax.plot(df['timestamp'], df['sma_20'], label='SMA 20',
                color='blue', linewidth=1, alpha=0.7)
        ax.plot(df['timestamp'], df['sma_50'], label='SMA 50',
                color='red', linewidth=1, alpha=0.7)
        ax.plot(df['timestamp'], df['ema_20'], label='EMA 20',
                color='green', linewidth=1, alpha=0.7, linestyle='--')
        ax.set_ylabel(row_label, fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        if i == 0:
            ax.set_title(indicator_titles[0], fontsize=11, fontweight='bold')

        # Column 1: RSI
        ax = axes[i, 1]
        ax.plot(df['timestamp'], df['rsi'], color='purple', linewidth=1.2)
        ax.axhline(y=70, color='r', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=30, color='g', linestyle='--', alpha=0.5, linewidth=1)
        ax.fill_between(df['timestamp'], 30, 70, alpha=0.1, color='gray')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        if i == 0:
            ax.set_title(indicator_titles[1], fontsize=11, fontweight='bold')

        # Column 2: MACD
        ax = axes[i, 2]
        ax.plot(df['timestamp'], df['macd_line'], label='MACD',
                color='blue', linewidth=1)
        ax.plot(df['timestamp'], df['macd_signal'], label='Signal',
                color='red', linewidth=1)
        colors = ['green' if x >= 0 else 'red' for x in df['macd_histogram']]
        ax.bar(df['timestamp'], df['macd_histogram'], color=colors, alpha=0.3, width=0.5)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        if i == 0:
            ax.set_title(indicator_titles[2], fontsize=11, fontweight='bold')

        # Column 3: Bollinger Bands
        ax = axes[i, 3]
        ax.plot(df['timestamp'], df['close'], label='Close',
                color='black', linewidth=1.2)
        ax.plot(df['timestamp'], df['bb_middle'], label='BB Mid',
                color='blue', linewidth=1)
        ax.plot(df['timestamp'], df['bb_upper'], label='BB Upper',
                color='red', linewidth=0.8, linestyle='--')
        ax.plot(df['timestamp'], df['bb_lower'], label='BB Lower',
                color='green', linewidth=0.8, linestyle='--')
        ax.fill_between(df['timestamp'], df['bb_upper'], df['bb_lower'],
                        alpha=0.1, color='gray')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        if i == 0:
            ax.set_title(indicator_titles[3], fontsize=11, fontweight='bold')

        # Column 4: Volume
        ax = axes[i, 4]
        ax.bar(df['timestamp'], df['volume'], color='steelblue', alpha=0.6)
        ax.plot(df['timestamp'], df['volume_sma_20'], color='red',
                linewidth=1.5, label='Vol SMA 20')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        if i == 0:
            ax.set_title(indicator_titles[4], fontsize=11, fontweight='bold')

    # Main title
    plt.suptitle(
        f'Multi-Instrument Technical Analysis Grid (N×M) - 2 Week Analysis\n'
        f'{len(instruments)} Instruments × {M} Indicators | Period: {START_DATE} - {END_DATE}',
        fontsize=14, fontweight='bold', y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    save_figure('multi_instrument_nm_grid.png', dpi=150, bbox_inches='tight')
    plt.show()


# Execute the visualization
print("\n📊 Creating N×M Grid Visualization...")
create_nm_grid_visualization(instrument_data, INSTRUMENTS)
print("✅ Visualization complete!")

# %% [markdown]
"""
## 📊 Smoothed Indicators & Signal Dashboard

Noise-reduced indicators with buy/sell signal interpretation for cleaner trading signals.
"""

# %%
def plot_smoothed_indicators_dashboard(df: pd.DataFrame, instrument: Dict):
    """
    Create dashboard with SMOOTHED indicators and clear buy/sell signals.

    This removes noise and shows:
    - Smoothed price trends (HMA, KAMA, TEMA, SuperSmoother)
    - Smoothed RSI
    - Smoothed MACD histogram
    - Combined buy/sell signals with interpretation
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(5, 2, height_ratios=[2.5, 1, 1, 1, 1.5], hspace=0.3, wspace=0.3)

    # Panel 1: Smoothed Price Indicators (left, spans 2 rows)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['timestamp'], df['close'], label='Close Price',
             color='black', linewidth=1.5, alpha=0.6)
    ax1.plot(df['timestamp'], df['hma_20'], label='HMA (Hull MA)',
             color='#2E86DE', linewidth=2.5, alpha=0.9)
    ax1.plot(df['timestamp'], df['kama_20'], label='KAMA (Adaptive)',
             color='#10AC84', linewidth=2.5, alpha=0.9)
    ax1.plot(df['timestamp'], df['tema_20'], label='TEMA (Triple EMA)',
             color='#EE5A24', linewidth=2.5, alpha=0.9)
    ax1.plot(df['timestamp'], df['supersmoother'], label='SuperSmoother',
             color='#9B59B6', linewidth=2, alpha=0.8, linestyle='--')

    # Mark buy/sell signals on price chart
    buy_signals = df[df['confirmed_signal'] == 1]
    sell_signals = df[df['confirmed_signal'] == -1]
    ax1.scatter(buy_signals['timestamp'], buy_signals['close'],
                color='green', marker='^', s=200, alpha=0.9, label='BUY', zorder=5)
    ax1.scatter(sell_signals['timestamp'], sell_signals['close'],
                color='red', marker='v', s=200, alpha=0.9, label='SELL', zorder=5)

    ax1.set_title(f"{instrument['market']}/{instrument['code']} - {instrument['name']} | Smoothed Price Trends",
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Panel 2: Smoothed RSI (left)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df['timestamp'], df['rsi'], label='RSI (raw)',
             color='gray', linewidth=1, alpha=0.4)
    ax2.plot(df['timestamp'], df['rsi_smooth'], label='RSI (smoothed)',
             color='purple', linewidth=2.5, alpha=0.9)
    ax2.axhline(y=70, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=30, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.fill_between(df['timestamp'], 30, 70, alpha=0.1, color='gray')
    ax2.set_ylabel('RSI', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.set_title('Smoothed RSI (5-period EMA)', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Smoothed MACD Histogram (right)
    ax3 = fig.add_subplot(gs[1, 1])
    colors_raw = ['lightgreen' if x >= 0 else 'lightcoral' for x in df['macd_histogram']]
    ax3.bar(df['timestamp'], df['macd_histogram'], color=colors_raw, alpha=0.3, label='MACD Hist (raw)')
    ax3.plot(df['timestamp'], df['macd_hist_smooth'], label='MACD Hist (smoothed)',
             color='darkblue', linewidth=2.5, alpha=0.9)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_ylabel('MACD Histogram', fontsize=11, fontweight='bold')
    ax3.set_title('Smoothed MACD Histogram (3-period EMA)', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Signal Strength Over Time (left)
    ax4 = fig.add_subplot(gs[2, 0])
    colors_signal = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in df['signal_strength']]
    ax4.bar(df['timestamp'], df['signal_strength'], color=colors_signal, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_ylabel('Signal Strength', fontsize=11, fontweight='bold')
    ax4.set_title('Combined Signal Strength (-4 to +4)', fontsize=11, fontweight='bold')
    ax4.set_ylim(-4.5, 4.5)
    ax4.grid(True, alpha=0.3)

    # Panel 5: Volume Confirmation (right)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.bar(df['timestamp'], df['volume_ratio'], color='steelblue', alpha=0.6)
    ax5.axhline(y=1.2, color='red', linestyle='--', linewidth=1.5, label='High Vol Threshold')
    ax5.axhline(y=1.0, color='black', linestyle='-', linewidth=1)
    ax5.set_ylabel('Volume Ratio', fontsize=11, fontweight='bold')
    ax5.set_title('Volume Confirmation (Vol / SMA20)', fontsize=11, fontweight='bold')
    ax5.legend(loc='upper left', fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Panel 6: Bollinger Band Width (left)
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.plot(df['timestamp'], df['bb_width'], color='orange', linewidth=2)
    ax6.fill_between(df['timestamp'], 0, df['bb_width'], alpha=0.3, color='orange')
    ax6.set_ylabel('BB Width', fontsize=11, fontweight='bold')
    ax6.set_title('Bollinger Band Width (Volatility Measure)', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Panel 7: ATR Percentage (right)
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.plot(df['timestamp'], df['atr_percent'], color='red', linewidth=2)
    ax7.fill_between(df['timestamp'], 0, df['atr_percent'], alpha=0.3, color='red')
    ax7.set_ylabel('ATR %', fontsize=11, fontweight='bold')
    ax7.set_title('ATR as Percentage of Price', fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # Panel 8: Signal Interpretation Table (bottom, spans full width)
    ax8 = fig.add_subplot(gs[4, :])
    ax8.axis('off')

    # Calculate signal statistics
    latest_signal = df['confirmed_signal'].iloc[-1]
    latest_strength = df['signal_strength'].iloc[-1]
    total_buy_signals = (df['confirmed_signal'] == 1).sum()
    total_sell_signals = (df['confirmed_signal'] == -1).sum()
    avg_signal_strength = df['signal_strength'].mean()

    # Current indicator states
    latest_idx = len(df) - 1
    trend_state = "🟢 BULLISH" if df['trend_signal'].iloc[latest_idx] == 1 else "🔴 BEARISH" if df['trend_signal'].iloc[latest_idx] == -1 else "⚪ NEUTRAL"
    momentum_state = "🟢 OVERSOLD (BUY)" if df['momentum_signal'].iloc[latest_idx] == 1 else "🔴 OVERBOUGHT (SELL)" if df['momentum_signal'].iloc[latest_idx] == -1 else "⚪ NEUTRAL"
    macd_state = "🟢 BULLISH" if df['macd_signal_flag'].iloc[latest_idx] == 1 else "🔴 BEARISH" if df['macd_signal_flag'].iloc[latest_idx] == -1 else "⚪ NEUTRAL"
    bb_state = "🟢 BELOW LOWER (BUY)" if df['bb_signal'].iloc[latest_idx] == 1 else "🔴 ABOVE UPPER (SELL)" if df['bb_signal'].iloc[latest_idx] == -1 else "⚪ WITHIN BANDS"
    vol_state = "✅ HIGH VOLUME" if df['volume_confirm'].iloc[latest_idx] else "⚠️  LOW VOLUME"

    signal_text = f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║  📊 SIGNAL INTERPRETATION & CURRENT STATE                                                                     ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                                ║
    ║  🎯 LATEST SIGNAL:  {'🟢 BUY' if latest_signal == 1 else '🔴 SELL' if latest_signal == -1 else '⚪ NEUTRAL'}  (Strength: {latest_strength:.1f}/4)                                              ║
    ║                                                                                                                ║
    ║  📈 TREND (HMA vs TEMA):       {trend_state}                                                          ║
    ║  ⚡ MOMENTUM (Smoothed RSI):   {momentum_state}                                                       ║
    ║  📊 MACD (Smoothed Hist):      {macd_state}                                                           ║
    ║  📉 BOLLINGER BANDS:           {bb_state}                                                             ║
    ║  📦 VOLUME CONFIRMATION:       {vol_state}                                                            ║
    ║                                                                                                                ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║  📊 PERIOD STATISTICS (Past 2 Weeks)                                                                          ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                                ║
    ║  ✅ Total BUY Signals:         {total_buy_signals}                                                                         ║
    ║  ❌ Total SELL Signals:        {total_sell_signals}                                                                         ║
    ║  📊 Average Signal Strength:   {avg_signal_strength:.2f}                                                                   ║
    ║                                                                                                                ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║  💡 HOW SIGNALS ARE GENERATED                                                                                 ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                                ║
    ║  BUY SIGNAL:  Requires at least 2 of the following bullish conditions:                                       ║
    ║    • Trend: HMA > TEMA (Hull MA above Triple EMA)                                                            ║
    ║    • Momentum: Smoothed RSI < 30 (oversold)                                                                  ║
    ║    • MACD: Smoothed histogram > 0 (bullish momentum)                                                         ║
    ║    • Bollinger: Price < lower band (potential bounce)                                                        ║
    ║                                                                                                                ║
    ║  SELL SIGNAL: Requires at least 2 of the following bearish conditions:                                       ║
    ║    • Trend: HMA < TEMA (Hull MA below Triple EMA)                                                            ║
    ║    • Momentum: Smoothed RSI > 70 (overbought)                                                                ║
    ║    • MACD: Smoothed histogram < 0 (bearish momentum)                                                         ║
    ║    • Bollinger: Price > upper band (potential reversal)                                                      ║
    ║                                                                                                                ║
    ║  CONFIRMED SIGNAL: Buy/Sell + Volume > 1.2x average (high volume confirmation)                               ║
    ║                                                                                                                ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """

    ax8.text(0.02, 0.95, signal_text, transform=ax8.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(
        f'Smoothed Indicators Dashboard - {instrument["market"]}/{instrument["code"]} ({instrument["name"]})\n'
        f'Period: {START_DATE} to {END_DATE} | Noise-Reduced Signals for Clearer Trading',
        fontsize=15, fontweight='bold', y=0.995
    )

    save_figure(f"smoothed_dashboard_{instrument['market']}_{instrument['code']}.png",
                dpi=150, bbox_inches='tight')
    plt.show()


# Create smoothed dashboard for each instrument
print("\n📊 Creating Smoothed Indicators Dashboards...")
for inst in INSTRUMENTS:
    key = f"{inst['market']}_{inst['code']}"
    print(f"\n   Generating dashboard for {inst['market']}/{inst['code']} ({inst['name']})...")
    plot_smoothed_indicators_dashboard(instrument_data[key], inst)
print("✅ All smoothed dashboards complete!")

# %% [markdown]
"""
## 📊 Individual Instrument Deep Dive

For detailed analysis of a specific instrument, select one and visualize all indicators.
"""

# %%
def plot_single_instrument_dashboard(df: pd.DataFrame, instrument: Dict):
    """
    Create a comprehensive dashboard for a single instrument.
    """
    fig, axes = plt.subplots(5, 1, figsize=(16, 14),
                             gridspec_kw={'height_ratios': [2, 1, 1, 1, 1]})

    # 1. Price with MAs and Bollinger Bands
    ax = axes[0]
    ax.plot(df['timestamp'], df['close'], label='Close', color='black', linewidth=2)
    ax.plot(df['timestamp'], df['sma_20'], label='SMA 20', color='blue', linewidth=1.5)
    ax.plot(df['timestamp'], df['sma_50'], label='SMA 50', color='red', linewidth=1.5)
    ax.plot(df['timestamp'], df['bb_upper'], color='red', linewidth=1, linestyle='--', alpha=0.5)
    ax.plot(df['timestamp'], df['bb_lower'], color='green', linewidth=1, linestyle='--', alpha=0.5)
    ax.fill_between(df['timestamp'], df['bb_upper'], df['bb_lower'], alpha=0.1, color='gray')
    ax.set_ylabel('Price', fontsize=11)
    ax.set_title(f"{instrument['market']}/{instrument['code']} - {instrument['name']}",
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Volume
    ax = axes[1]
    ax.bar(df['timestamp'], df['volume'], color='steelblue', alpha=0.6)
    ax.plot(df['timestamp'], df['volume_sma_20'], color='red', linewidth=2)
    ax.set_ylabel('Volume', fontsize=11)
    ax.grid(True, alpha=0.3)

    # 3. RSI
    ax = axes[2]
    ax.plot(df['timestamp'], df['rsi'], color='purple', linewidth=1.5)
    ax.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    ax.fill_between(df['timestamp'], 30, 70, alpha=0.1, color='gray')
    ax.set_ylabel('RSI', fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # 4. MACD
    ax = axes[3]
    ax.plot(df['timestamp'], df['macd_line'], label='MACD', color='blue', linewidth=1.5)
    ax.plot(df['timestamp'], df['macd_signal'], label='Signal', color='red', linewidth=1.5)
    colors = ['green' if x >= 0 else 'red' for x in df['macd_histogram']]
    ax.bar(df['timestamp'], df['macd_histogram'], color=colors, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_ylabel('MACD', fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 5. ATR
    ax = axes[4]
    ax.plot(df['timestamp'], df['atr'], color='red', linewidth=1.5)
    ax.fill_between(df['timestamp'], 0, df['atr'], alpha=0.3, color='red')
    ax.set_ylabel('ATR', fontsize=11)
    ax.set_xlabel('Time', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(f"instrument_dashboard_{instrument['market']}_{instrument['code']}.png",
                dpi=150, bbox_inches='tight')
    plt.show()


# Example: Plot dashboard for first instrument
print("\n📊 Creating detailed dashboard for first instrument...")
first_inst = INSTRUMENTS[0]
first_key = f"{first_inst['market']}_{first_inst['code']}"
plot_single_instrument_dashboard(instrument_data[first_key], first_inst)
print("✅ Dashboard complete!")

# %% [markdown]
"""
## 📊 Statistical Summary for All Instruments
"""

# %%
print("="*80)
print("📊 MULTI-INSTRUMENT STATISTICAL SUMMARY")
print("="*80)

for inst in INSTRUMENTS:
    key = f"{inst['market']}_{inst['code']}"
    df = instrument_data[key]

    print(f"\n{'='*80}")
    print(f"📈 {inst['market']}/{inst['code']} - {inst['name']}")
    print(f"{'='*80}")

    print(f"\n📅 Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"📊 Total Bars: {len(df):,}")

    print(f"\n💰 Price Statistics")
    print(f"   Current: {df['close'].iloc[-1]:.2f}")
    print(f"   Mean: {df['close'].mean():.2f}")
    print(f"   Min: {df['close'].min():.2f}")
    print(f"   Max: {df['close'].max():.2f}")
    print(f"   Range: {df['close'].max() - df['close'].min():.2f}")

    df['returns'] = df['close'].pct_change()
    print(f"\n📈 Returns")
    print(f"   Mean: {df['returns'].mean() * 100:.4f}%")
    print(f"   Std Dev: {df['returns'].std() * 100:.4f}%")

    print(f"\n📊 RSI")
    print(f"   Current: {df['rsi'].iloc[-1]:.2f}")
    print(f"   Mean: {df['rsi'].mean():.2f}")
    overbought = (df['rsi'] > 70).sum()
    oversold = (df['rsi'] < 30).sum()
    print(f"   Overbought (>70): {overbought:,} bars ({overbought/len(df)*100:.1f}%)")
    print(f"   Oversold (<30): {oversold:,} bars ({oversold/len(df)*100:.1f}%)")

    print(f"\n📊 ATR")
    print(f"   Current: {df['atr'].iloc[-1]:.2f}")
    print(f"   Mean: {df['atr'].mean():.2f}")

print("\n" + "="*80)
print("✅ Analysis complete!")
print("="*80)

# %% [markdown]
"""
## 🎯 Summary

This visualization provides:

1. **N×M Grid Layout**: 6 instruments × 5 indicators = 30 subplots
   - Each row shows one instrument across all indicators
   - Each column shows one indicator across all instruments
   - Easy comparison between instruments and indicators

2. **2-Week Focused Timeframe**: Short-term data for relevant analysis

3. **Multiple Markets Covered**:
   - DCE: Iron Ore (i), Soybean Meal (m)
   - SHFE: Rebar (rb), Copper (cu)
   - CZCE: Sugar (SR), PTA (TA)

4. **Key Technical Indicators**:
   - Trend: Moving Averages (SMA, EMA)
   - Momentum: RSI, MACD
   - Volatility: Bollinger Bands, ATR
   - Volume: Volume analysis with moving average

5. **Real Data from Server**:
   - Fetches real market data from r3 server
   - Requires proper server configuration and credentials
   - No fallback - fails if data cannot be fetched

**Configuration:**
- Update connection parameters:
  - `SVR_HOST`: Server IP address
  - `USER`: Your username
  - `HASHED_PASSWORD`: Your hashed password
  - `RAILS_URL`, `WS_URL`, `TM_MASTER`: Auto-configured from host
- Set `USE_REAL_DATA = True` (default)
- Customize instrument selection in INSTRUMENTS list
- Adjust time range by modifying START_DATE and END_DATE
- Add more indicators by expanding the M dimension
"""

# %%
