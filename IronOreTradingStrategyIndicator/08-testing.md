# Chapter 8: Testing

## Backtest Overview

Complete backtesting procedure using historical data from svr3 server.

---

## Test Periods

**Full Backtest Period**: 2024-01-01 to 2024-12-31 (1 year)

**Evaluation Period**: 2024-10-01 to 2024-12-31 (Q4 2024, 3 months)

**Purpose**:
- Full period: Generate all signals, validate strategy logic
- Evaluation period: Calculate P&L metrics, assess performance

---

## Data Fetching from svr3

### Connection Setup

```python
import pycaitlynts3 as pcts3

# Create svr3 reader
reader = pcts3.SvrDataReader()

# Configure connection
params = {
    'market': b'DCE',
    'code': b'i<00>',
    'granularity': 900,
    'start_date': '2024-01-01',
    'end_date': '2024-12-31'
}
```

### Fetch Historical Data

```python
async def fetch_historical_data(market, code, start_date, end_date, granularity=900):
    """
    Fetch historical OHLCV data from svr3

    Args:
        market: Market code (b'DCE')
        code: Instrument code (b'i<00>')
        start_date: Start date string 'YYYY-MM-DD'
        end_date: End date string 'YYYY-MM-DD'
        granularity: Bar granularity in seconds (900)

    Returns:
        DataFrame with OHLCV data
    """

    reader = pcts3.SvrDataReader()

    # Convert dates to timestamps
    start_ts = pcts3.date_to_timetag(start_date)
    end_ts = pcts3.date_to_timetag(end_date)

    # Fetch data
    ret = await reader.save_by_symbol(
        market=market,
        code=code,
        granularity=granularity,
        start_time=start_ts,
        end_time=end_ts
    )

    if ret is None:
        raise ValueError("Failed to fetch data from svr3")

    # Parse to DataFrame
    df = parse_sv_to_dataframe(ret)

    return df
```

### Parse StructValue to DataFrame

```python
import pandas as pd

def parse_sv_to_dataframe(sv_records):
    """
    Parse StructValue records to pandas DataFrame

    Args:
        sv_records: List of StructValue objects

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """

    data = []

    for sv in sv_records:
        row = {
            'timestamp': sv.get_time_tag(),
            'open': float(sv.get_field('open')),
            'high': float(sv.get_field('high')),
            'low': float(sv.get_field('low')),
            'close': float(sv.get_field('close')),
            'volume': float(sv.get_field('volume')),
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

    return df
```

---

## Running Framework Backtest

### Framework Execution

```bash
# Navigate to indicator directory
cd /home/user/money-engineering/IronOreIndicator

# Run framework (assumes framework is configured)
# Framework will:
# 1. Load uin.json and uout.json
# 2. Initialize IronOreIndicator.py
# 3. Fetch historical data
# 4. Process bars through on_bar()
# 5. Output indicator values to database/file
```

**Note**: Framework execution depends on pycaitlyn installation and configuration.

---

## Simulation-Based Backtest (Jupyter Notebook)

### Load Indicator Outputs

```python
# Load indicator outputs (after framework run)
import pandas as pd

# Option 1: Load from CSV (if framework outputs to CSV)
indicator_df = pd.read_csv('indicator_outputs.csv')

# Option 2: Load from database (if framework outputs to DB)
# Use pcts3.SvrDataReader to fetch indicator outputs

# Expected columns:
# timestamp, ema_12, ema_26, ema_50, macd, macd_signal, macd_histogram,
# rsi, bb_upper, bb_middle, bb_lower, bb_width, bb_width_pct,
# atr, volume_ema, regime, signal, confidence, position_state
```

### Simulate Trading

```python
import numpy as np

def simulate_trading(indicator_df, ohlcv_df, capital=1000000):
    """
    Simulate trading based on indicator signals

    Args:
        indicator_df: DataFrame with indicator outputs
        ohlcv_df: DataFrame with OHLCV data
        capital: Initial capital

    Returns:
        DataFrame with trade log and P&L
    """

    # Merge indicator and price data
    df = pd.merge(
        ohlcv_df[['timestamp', 'open', 'high', 'low', 'close']],
        indicator_df,
        on='timestamp',
        how='inner'
    )

    # Initialize simulation state
    position = 0  # 0 = flat, 1 = long, -1 = short
    position_size = 0  # Number of contracts
    entry_price = 0.0
    stop_loss = 0.0
    cash = capital
    equity = capital

    trades = []
    equity_curve = []

    for idx, row in df.iterrows():
        timestamp = row['timestamp']
        open_price = row['open']
        high_price = row['high']
        low_price = row['low']
        close_price = row['close']
        signal = row['signal']
        confidence = row['confidence']
        regime = row['regime']

        # Check stop loss (intrabar)
        if position != 0:
            if position == 1 and low_price <= stop_loss:
                # Stop loss hit on long
                exit_price = stop_loss
                pnl = (exit_price - entry_price) * position_size
                cash += pnl
                equity = cash

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': timestamp,
                    'direction': 'LONG',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'size': position_size,
                    'pnl': pnl,
                    'exit_reason': 'STOP_LOSS'
                })

                position = 0
                position_size = 0

            elif position == -1 and high_price >= stop_loss:
                # Stop loss hit on short
                exit_price = stop_loss
                pnl = (entry_price - exit_price) * position_size
                cash += pnl
                equity = cash

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': timestamp,
                    'direction': 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'size': position_size,
                    'pnl': pnl,
                    'exit_reason': 'STOP_LOSS'
                })

                position = 0
                position_size = 0

        # Process signal (execute at next bar open)
        if signal == 1 and position == 0:
            # BUY signal
            position = 1
            entry_price = open_price  # Execute at open (next bar)
            stop_loss = entry_price * 0.97  # 3% stop

            # Calculate position size (20% of capital initially)
            position_value = cash * 0.20 * confidence
            position_size = int(position_value / entry_price)

            entry_time = timestamp

        elif signal == -1 and position != 0:
            # SELL signal (exit)
            exit_price = open_price

            if position == 1:
                pnl = (exit_price - entry_price) * position_size
                direction = 'LONG'
            else:
                pnl = (entry_price - exit_price) * position_size
                direction = 'SHORT'

            cash += pnl
            equity = cash

            trades.append({
                'entry_time': entry_time,
                'exit_time': timestamp,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': position_size,
                'pnl': pnl,
                'exit_reason': 'SIGNAL'
            })

            position = 0
            position_size = 0

        # Record equity
        if position != 0:
            # Mark-to-market equity
            if position == 1:
                unrealized_pnl = (close_price - entry_price) * position_size
            else:
                unrealized_pnl = (entry_price - close_price) * position_size
            equity = cash + unrealized_pnl
        else:
            equity = cash

        equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'cash': cash,
            'position': position
        })

    return pd.DataFrame(trades), pd.DataFrame(equity_curve)
```

---

## Performance Metrics

### Calculate Metrics

```python
def calculate_metrics(trades_df, equity_df, capital=1000000):
    """
    Calculate backtest performance metrics

    Returns:
        Dictionary with all metrics
    """

    # Total P&L
    total_pnl = trades_df['pnl'].sum()

    # Number of trades
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])

    # Win rate
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # Average win/loss
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0

    # Profit factor
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Max win/loss
    max_win = trades_df['pnl'].max() if total_trades > 0 else 0
    max_loss = trades_df['pnl'].min() if total_trades > 0 else 0

    # Drawdown
    equity_df['peak'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = equity_df['equity'] - equity_df['peak']
    equity_df['drawdown_pct'] = equity_df['drawdown'] / equity_df['peak']
    max_drawdown = equity_df['drawdown'].min()
    max_drawdown_pct = equity_df['drawdown_pct'].min()

    # Sharpe ratio (assuming 252 trading days, 0% risk-free rate)
    returns = equity_df['equity'].pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 0 else 0

    # Return
    total_return = (equity_df['equity'].iloc[-1] - capital) / capital

    metrics = {
        'total_pnl': total_pnl,
        'total_return_pct': total_return * 100,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_win': max_win,
        'max_loss': max_loss,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct * 100,
        'sharpe_ratio': sharpe,
        'final_equity': equity_df['equity'].iloc[-1]
    }

    return metrics
```

### Print Metrics

```python
def print_metrics(metrics):
    """Print formatted metrics"""

    print("=" * 60)
    print("BACKTEST PERFORMANCE METRICS")
    print("=" * 60)
    print(f"Total P&L:           ${metrics['total_pnl']:,.2f}")
    print(f"Total Return:        {metrics['total_return_pct']:.2f}%")
    print(f"Final Equity:        ${metrics['final_equity']:,.2f}")
    print()
    print(f"Total Trades:        {metrics['total_trades']}")
    print(f"Winning Trades:      {metrics['winning_trades']}")
    print(f"Losing Trades:       {metrics['losing_trades']}")
    print(f"Win Rate:            {metrics['win_rate']*100:.2f}%")
    print()
    print(f"Average Win:         ${metrics['avg_win']:,.2f}")
    print(f"Average Loss:        ${metrics['avg_loss']:,.2f}")
    print(f"Profit Factor:       {metrics['profit_factor']:.2f}")
    print()
    print(f"Max Win:             ${metrics['max_win']:,.2f}")
    print(f"Max Loss:            ${metrics['max_loss']:,.2f}")
    print()
    print(f"Max Drawdown:        ${metrics['max_drawdown']:,.2f}")
    print(f"Max Drawdown %:      {metrics['max_drawdown_pct']:.2f}%")
    print()
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
    print("=" * 60)
```

---

## Validation Checks

### Data Quality Checks

```python
def validate_data(df):
    """Validate OHLCV data quality"""

    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print("Warning: Missing values detected:")
        print(missing[missing > 0])

    # Check for zero/negative prices
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if (df[col] <= 0).any():
            print(f"Warning: Zero or negative values in {col}")

    # Check OHLC relationship
    invalid_ohlc = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    )

    if invalid_ohlc.any():
        print(f"Warning: Invalid OHLC relationships in {invalid_ohlc.sum()} bars")

    # Check for gaps (missing timestamps)
    df['time_diff'] = df['timestamp'].diff()
    expected_diff = 900  # 15 minutes
    gaps = df[df['time_diff'] > expected_diff * 1.5]

    if len(gaps) > 0:
        print(f"Warning: {len(gaps)} gaps detected in data")

    print("Data validation complete.")
```

---

## Test Execution Checklist

1. **Fetch Historical Data**
   - [ ] Connect to svr3 server
   - [ ] Fetch DCE i<00> data (2024-01-01 to 2024-12-31)
   - [ ] Validate data quality
   - [ ] Save to CSV/database

2. **Run Indicator**
   - [ ] Load uin.json and uout.json
   - [ ] Initialize IronOreIndicator
   - [ ] Process all bars through on_bar()
   - [ ] Save indicator outputs

3. **Simulate Trading**
   - [ ] Load indicator signals
   - [ ] Simulate position entries/exits
   - [ ] Track P&L and equity curve
   - [ ] Save trade log

4. **Calculate Metrics**
   - [ ] Compute all performance metrics
   - [ ] Validate against success criteria
   - [ ] Generate metrics report

5. **Visualize Results**
   - [ ] Plot equity curve
   - [ ] Plot drawdown curve
   - [ ] Plot regime distribution
   - [ ] Generate trade analysis charts

---

## Expected Test Duration

- **Data Fetch**: 2-5 minutes (depends on network)
- **Indicator Processing**: 10-30 minutes (1 year of 15-min bars)
- **Simulation**: 1-2 minutes
- **Visualization**: < 1 minute

**Total**: ~15-40 minutes for complete backtest

---

## Next Steps

Proceed to Chapter 9 for Jupyter notebook P&L visualization setup.
