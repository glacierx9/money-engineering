# Chapter 9: Visualization

## Jupyter Notebook Setup

Complete P&L visualization setup for backtest analysis.

---

## Notebook Structure

**File**: `IronOreIndicator/analysis.ipynb`

**Sections**:
1. Imports and Configuration
2. Data Loading
3. Simulation
4. Metrics Calculation
5. Equity Curve Visualization
6. Drawdown Analysis
7. Regime Distribution
8. Trade Analysis
9. Indicator Correlation
10. Export Results

---

## Cell 1: Imports and Configuration

```python
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Jupyter display settings
%matplotlib inline
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Figure size defaults
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10

# Configuration
CAPITAL = 1_000_000
START_DATE = '2024-01-01'
END_DATE = '2024-12-31'
EVAL_START = '2024-10-01'  # Evaluation period start
```

---

## Cell 2: Data Loading

```python
# Load OHLCV data (from CSV or fetch from svr3)
ohlcv_df = pd.read_csv('ohlcv_data.csv')
ohlcv_df['datetime'] = pd.to_datetime(ohlcv_df['timestamp'], unit='s')

# Load indicator outputs (from framework run)
indicator_df = pd.read_csv('indicator_outputs.csv')
indicator_df['datetime'] = pd.to_datetime(indicator_df['timestamp'], unit='s')

# Merge datasets
df = pd.merge(
    ohlcv_df[['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']],
    indicator_df,
    on='timestamp',
    how='inner'
)

print(f"Loaded {len(df)} bars from {df['datetime'].min()} to {df['datetime'].max()}")
print(f"Data columns: {list(df.columns)}")
```

---

## Cell 3: Trading Simulation

```python
def simulate_trading(df, capital=CAPITAL):
    """Simulate trading with indicator signals"""

    position = 0  # 0 = flat, 1 = long
    position_size = 0
    entry_price = 0.0
    entry_time = None
    stop_loss = 0.0

    cash = capital
    trades = []
    equity_curve = []

    for idx, row in df.iterrows():
        timestamp = row['timestamp']
        dt = row['datetime']
        open_price = row['open']
        high_price = row['high']
        low_price = row['low']
        close_price = row['close']
        signal = row['signal']
        confidence = row['confidence']
        regime = row['regime']

        # Check stop loss
        if position == 1 and low_price <= stop_loss:
            exit_price = stop_loss
            pnl = (exit_price - entry_price) * position_size
            cash += pnl

            trades.append({
                'entry_time': entry_time,
                'exit_time': dt,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': position_size,
                'pnl': pnl,
                'return_pct': (exit_price - entry_price) / entry_price * 100,
                'regime': regime,
                'exit_reason': 'STOP_LOSS'
            })

            position = 0
            position_size = 0

        # Process signals
        if signal == 1 and position == 0:
            # BUY
            position = 1
            entry_price = open_price
            entry_time = dt
            stop_loss = entry_price * 0.97  # 3% stop

            # Position size: 20% of capital, adjusted by confidence
            position_value = cash * 0.20 * confidence
            position_size = int(position_value / entry_price)

        elif signal == -1 and position == 1:
            # SELL (exit)
            exit_price = open_price
            pnl = (exit_price - entry_price) * position_size
            cash += pnl

            trades.append({
                'entry_time': entry_time,
                'exit_time': dt,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': position_size,
                'pnl': pnl,
                'return_pct': (exit_price - entry_price) / entry_price * 100,
                'regime': regime,
                'exit_reason': 'SIGNAL'
            })

            position = 0
            position_size = 0

        # Calculate equity
        if position == 1:
            unrealized_pnl = (close_price - entry_price) * position_size
            equity = cash + unrealized_pnl
        else:
            equity = cash

        equity_curve.append({
            'timestamp': timestamp,
            'datetime': dt,
            'equity': equity,
            'cash': cash,
            'position': position
        })

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)

    return trades_df, equity_df

# Run simulation
trades_df, equity_df = simulate_trading(df)

print(f"\nSimulation complete:")
print(f"Total trades: {len(trades_df)}")
print(f"Final equity: ${equity_df['equity'].iloc[-1]:,.2f}")
```

---

## Cell 4: Calculate Metrics

```python
def calculate_metrics(trades_df, equity_df):
    """Calculate performance metrics"""

    # Total P&L
    total_pnl = trades_df['pnl'].sum()

    # Trades
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # Win/Loss averages
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0

    # Profit factor
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Drawdown
    equity_df['peak'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = equity_df['equity'] - equity_df['peak']
    equity_df['drawdown_pct'] = equity_df['drawdown'] / equity_df['peak'] * 100
    max_drawdown = equity_df['drawdown'].min()
    max_drawdown_pct = equity_df['drawdown_pct'].min()

    # Returns
    total_return = (equity_df['equity'].iloc[-1] - CAPITAL) / CAPITAL * 100

    # Sharpe ratio
    returns = equity_df['equity'].pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 0 else 0

    metrics = {
        'Total P&L': f"${total_pnl:,.2f}",
        'Total Return': f"{total_return:.2f}%",
        'Total Trades': total_trades,
        'Winning Trades': winning_trades,
        'Losing Trades': losing_trades,
        'Win Rate': f"{win_rate*100:.2f}%",
        'Average Win': f"${avg_win:,.2f}",
        'Average Loss': f"${avg_loss:,.2f}",
        'Profit Factor': f"{profit_factor:.2f}",
        'Max Drawdown': f"${max_drawdown:,.2f}",
        'Max Drawdown %': f"{max_drawdown_pct:.2f}%",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Final Equity': f"${equity_df['equity'].iloc[-1]:,.2f}"
    }

    return metrics, equity_df

metrics, equity_df = calculate_metrics(trades_df, equity_df)

# Print metrics
print("\n" + "="*60)
print("BACKTEST PERFORMANCE METRICS")
print("="*60)
for key, value in metrics.items():
    print(f"{key:<20} {value}")
print("="*60)
```

---

## Cell 5: Equity Curve Visualization

```python
# Plot equity curve
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Equity curve
axes[0].plot(equity_df['datetime'], equity_df['equity'], linewidth=2, label='Equity')
axes[0].axhline(y=CAPITAL, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
axes[0].fill_between(equity_df['datetime'], CAPITAL, equity_df['equity'],
                      where=(equity_df['equity'] >= CAPITAL), color='green', alpha=0.2)
axes[0].fill_between(equity_df['datetime'], CAPITAL, equity_df['equity'],
                      where=(equity_df['equity'] < CAPITAL), color='red', alpha=0.2)
axes[0].set_ylabel('Equity ($)', fontsize=12)
axes[0].set_title('Equity Curve', fontsize=14, fontweight='bold')
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)

# Format y-axis
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.2f}M'))

# Drawdown
axes[1].fill_between(equity_df['datetime'], 0, equity_df['drawdown_pct'],
                      color='red', alpha=0.3)
axes[1].plot(equity_df['datetime'], equity_df['drawdown_pct'], color='darkred', linewidth=1.5)
axes[1].set_ylabel('Drawdown (%)', fontsize=12)
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('pnl_curve.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nEquity curve saved as pnl_curve.png")
```

---

## Cell 6: Regime Distribution Analysis

```python
# Regime distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar count by regime
regime_counts = df['regime'].value_counts().sort_index()
regime_labels = {1: 'Strong Uptrend', 2: 'Strong Downtrend', 3: 'Sideways', 4: 'Chaos'}
regime_counts.index = regime_counts.index.map(regime_labels)

axes[0].bar(regime_counts.index, regime_counts.values, color=['green', 'red', 'blue', 'orange'])
axes[0].set_ylabel('Number of Bars', fontsize=12)
axes[0].set_title('Market Regime Distribution', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Trade count by regime
if len(trades_df) > 0:
    trade_regime_counts = trades_df['regime'].value_counts().sort_index()
    trade_regime_counts.index = trade_regime_counts.index.map(regime_labels)

    axes[1].bar(trade_regime_counts.index, trade_regime_counts.values,
                color=['green', 'red', 'blue', 'orange'])
    axes[1].set_ylabel('Number of Trades', fontsize=12)
    axes[1].set_title('Trades by Regime', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('regime_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Cell 7: Trade Analysis

```python
if len(trades_df) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # P&L distribution
    axes[0, 0].hist(trades_df['pnl'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('P&L ($)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('P&L Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Return distribution
    axes[0, 1].hist(trades_df['return_pct'], bins=30, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Return (%)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Return Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Cumulative P&L
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    axes[1, 0].plot(range(len(trades_df)), trades_df['cumulative_pnl'],
                    linewidth=2, color='green')
    axes[1, 0].fill_between(range(len(trades_df)), 0, trades_df['cumulative_pnl'],
                             alpha=0.3, color='green')
    axes[1, 0].set_xlabel('Trade Number', fontsize=12)
    axes[1, 0].set_ylabel('Cumulative P&L ($)', fontsize=12)
    axes[1, 0].set_title('Cumulative P&L', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Win/Loss by regime
    regime_pnl = trades_df.groupby('regime')['pnl'].sum()
    regime_pnl.index = regime_pnl.index.map(regime_labels)
    colors = ['green' if x > 0 else 'red' for x in regime_pnl.values]
    axes[1, 1].bar(regime_pnl.index, regime_pnl.values, color=colors)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 1].set_ylabel('Total P&L ($)', fontsize=12)
    axes[1, 1].set_title('P&L by Regime', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('trade_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
```

---

## Cell 8: Indicator Correlation Heatmap

```python
# Select indicator columns
indicator_cols = ['ema_12', 'ema_26', 'ema_50', 'macd', 'rsi',
                  'bb_width_pct', 'atr', 'volume_ema', 'close']

# Calculate correlation matrix
corr_matrix = df[indicator_cols].corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Indicator Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Cell 9: Evaluation Period Analysis

```python
# Filter for evaluation period (Q4 2024)
eval_df = equity_df[equity_df['datetime'] >= EVAL_START].copy()
eval_trades_df = trades_df[trades_df['exit_time'] >= EVAL_START].copy()

# Calculate eval metrics
eval_metrics, _ = calculate_metrics(eval_trades_df, eval_df)

print("\n" + "="*60)
print(f"EVALUATION PERIOD METRICS ({EVAL_START} to {END_DATE})")
print("="*60)
for key, value in eval_metrics.items():
    print(f"{key:<20} {value}")
print("="*60)
```

---

## Cell 10: Export Results

```python
# Save results
trades_df.to_csv('trade_log.csv', index=False)
equity_df.to_csv('equity_curve.csv', index=False)

# Save metrics to JSON
import json

all_results = {
    'backtest_metrics': metrics,
    'evaluation_metrics': eval_metrics,
    'configuration': {
        'capital': CAPITAL,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'eval_start': EVAL_START
    }
}

with open('backtest_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\nResults saved:")
print("- trade_log.csv")
print("- equity_curve.csv")
print("- backtest_results.json")
print("- pnl_curve.png")
print("- regime_distribution.png")
print("- trade_analysis.png")
print("- correlation_heatmap.png")
```

---

## Expected Outputs

**Charts**:
1. `pnl_curve.png` - Equity curve and drawdown
2. `regime_distribution.png` - Bar/trade counts by regime
3. `trade_analysis.png` - P&L distribution and cumulative P&L
4. `correlation_heatmap.png` - Indicator correlation matrix

**Data Files**:
1. `trade_log.csv` - Complete trade history
2. `equity_curve.csv` - Daily equity values
3. `backtest_results.json` - All metrics in JSON format

---

## Next Steps

Proceed to Chapter 10 for complete reproduction workflow.
