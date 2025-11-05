# Chapter 10: Reproduction Steps

## Complete Workflow

Step-by-step instructions to reproduce the Iron Ore trading strategy from scratch.

---

## Prerequisites

**Required Software**:
- Python 3.8+
- pycaitlyn framework
- pycaitlynts3
- pycaitlynutils3
- pandas, numpy, matplotlib, seaborn
- Jupyter notebook
- Access to svr3 data server

**Verify Installation**:
```bash
python3 -c "import pycaitlyn; import pycaitlynts3; import pycaitlynutils3; print('Framework OK')"
python3 -c "import pandas; import numpy; import matplotlib; print('Libraries OK')"
jupyter --version
```

---

## Step 1: Create Directory Structure

```bash
# Navigate to project root
cd /home/user/money-engineering

# Create indicator directory
mkdir -p IronOreIndicator
cd IronOreIndicator
```

---

## Step 2: Create Configuration Files

### Create uin.json

```bash
cat > uin.json << 'EOF'
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
EOF
```

### Create uout.json

```bash
cat > uout.json << 'EOF'
{
  "IronOreIndicator": {
    "granularity": 900,
    "namespace": "private",
    "revision": 4294967295,
    "fields": [
      {"name": "ema_12", "type": "double", "desc": "12-period EMA"},
      {"name": "ema_26", "type": "double", "desc": "26-period EMA"},
      {"name": "ema_50", "type": "double", "desc": "50-period EMA"},
      {"name": "macd", "type": "double", "desc": "MACD line"},
      {"name": "macd_signal", "type": "double", "desc": "MACD signal line"},
      {"name": "macd_histogram", "type": "double", "desc": "MACD histogram"},
      {"name": "rsi", "type": "double", "desc": "RSI value (0-100)"},
      {"name": "bb_upper", "type": "double", "desc": "Bollinger Band upper"},
      {"name": "bb_middle", "type": "double", "desc": "Bollinger Band middle"},
      {"name": "bb_lower", "type": "double", "desc": "Bollinger Band lower"},
      {"name": "bb_width", "type": "double", "desc": "Bollinger Band width"},
      {"name": "bb_width_pct", "type": "double", "desc": "BB width percentage"},
      {"name": "atr", "type": "double", "desc": "Average True Range"},
      {"name": "volume_ema", "type": "double", "desc": "Volume EMA"},
      {"name": "regime", "type": "int32", "desc": "Market regime (1-4)"},
      {"name": "signal", "type": "int32", "desc": "Trading signal (-1/0/1)"},
      {"name": "confidence", "type": "double", "desc": "Signal confidence"},
      {"name": "position_state", "type": "int32", "desc": "Position state (0/1)"},
      {"name": "bar_index", "type": "int64", "desc": "Bar counter"}
    ]
  }
}
EOF
```

### Validate JSON

```bash
python3 -m json.tool uin.json > /dev/null && echo "uin.json: Valid"
python3 -m json.tool uout.json > /dev/null && echo "uout.json: Valid"
```

---

## Step 3: Implement IronOreIndicator.py

**Reference**: See Chapter 6 for complete implementation

**Create file**:
```bash
touch IronOreIndicator.py
```

**Implementation checklist**:
- [ ] Import statements (pycaitlyn, pycaitlynts3, pycaitlynutils3)
- [ ] Framework globals (use_raw, overwrite, granularity, etc.)
- [ ] SampleQuote class
- [ ] IronOreIndicator class
  - [ ] __init__ (all state variables)
  - [ ] initialize method
  - [ ] on_bar method
  - [ ] _on_cycle_pass method
  - [ ] Indicator update methods (EMAs, MACD, RSI, BB, ATR, Volume)
  - [ ] Regime detection methods
  - [ ] Signal generation methods
  - [ ] ready_to_serialize method
- [ ] Global indicator instance
- [ ] Framework callbacks (on_init, on_bar, on_ready, etc.)

**Key sections to implement**:

1. **State Variables** (from Chapter 6)
2. **Indicator Updates** (from Chapter 2)
3. **Regime Detection** (from Chapter 3)
4. **Signal Logic** (from Chapter 4)
5. **Risk Management** (from Chapter 5)

---

## Step 4: Fetch Historical Data

### Create data_fetch.py

```python
#!/usr/bin/env python3
"""Fetch historical data from svr3 server"""

import asyncio
import pandas as pd
import pycaitlynts3 as pcts3

async def fetch_data():
    """Fetch DCE Iron Ore data for 2024"""

    reader = pcts3.SvrDataReader()

    # Convert dates to timestamps
    start_ts = pcts3.date_to_timetag('2024-01-01')
    end_ts = pcts3.date_to_timetag('2024-12-31')

    print("Fetching data from svr3...")
    print(f"Market: DCE")
    print(f"Instrument: i<00>")
    print(f"Period: 2024-01-01 to 2024-12-31")
    print(f"Granularity: 900s (15 minutes)")

    # Fetch data
    ret = await reader.save_by_symbol(
        market=b'DCE',
        code=b'i<00>',
        granularity=900,
        start_time=start_ts,
        end_time=end_ts
    )

    if ret is None:
        print("Error: Failed to fetch data")
        return

    # Parse to DataFrame
    data = []
    for sv in ret:
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
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

    # Save to CSV
    df.to_csv('ohlcv_data.csv', index=False)

    print(f"\nData saved: {len(df)} bars")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"File: ohlcv_data.csv")

if __name__ == '__main__':
    asyncio.run(fetch_data())
```

### Run data fetch

```bash
python3 data_fetch.py
```

**Expected output**:
```
Fetching data from svr3...
Market: DCE
Instrument: i<00>
Period: 2024-01-01 to 2024-12-31
Granularity: 900s (15 minutes)

Data saved: XXXXX bars
Date range: 2024-01-01 to 2024-12-31
File: ohlcv_data.csv
```

---

## Step 5: Run Framework (Generate Indicator Outputs)

**Note**: Framework execution depends on pycaitlyn installation. This step may vary based on your framework setup.

```bash
# Option 1: Framework command (if available)
# pycaitlyn run IronOreIndicator.py

# Option 2: Custom framework runner
# python3 run_framework.py --indicator IronOreIndicator.py --start 2024-01-01 --end 2024-12-31

# Expected outputs:
# - Logs showing bar processing
# - indicator_outputs.csv (or database entries)
```

**Verify indicator outputs**:
```bash
# Check that indicator_outputs.csv exists and has expected columns
head indicator_outputs.csv
```

---

## Step 6: Create Jupyter Notebook

### Create analysis.ipynb

**Reference**: See Chapter 9 for complete notebook structure

**Cell structure**:
1. Imports and Configuration
2. Data Loading
3. Trading Simulation
4. Metrics Calculation
5. Equity Curve Visualization
6. Regime Distribution
7. Trade Analysis
8. Indicator Correlation
9. Evaluation Period Analysis
10. Export Results

### Launch Jupyter

```bash
jupyter notebook analysis.ipynb
```

### Run all cells

**Keyboard**: Shift+Enter on each cell, or Cell → Run All

**Expected outputs**:
- Metrics printed to console
- Charts displayed inline
- Files saved:
  - `pnl_curve.png`
  - `regime_distribution.png`
  - `trade_analysis.png`
  - `correlation_heatmap.png`
  - `trade_log.csv`
  - `equity_curve.csv`
  - `backtest_results.json`

---

## Step 7: Verify Results

### Check Success Criteria

From Chapter 1, verify:
- [ ] Positive cumulative P&L over evaluation period
- [ ] Sharpe ratio > 1.0
- [ ] Maximum drawdown < 15%
- [ ] Win rate > 45%
- [ ] Profit factor > 1.3

### Review Charts

1. **pnl_curve.png**: Equity should show upward trend
2. **regime_distribution.png**: Balanced regime distribution
3. **trade_analysis.png**: More winning trades than losing
4. **correlation_heatmap.png**: Indicators not perfectly correlated

### Review Metrics

```bash
cat backtest_results.json
```

Look for:
- `total_return_pct` > 0
- `sharpe_ratio` > 1.0
- `max_drawdown_pct` < 15
- `win_rate` > 0.45
- `profit_factor` > 1.3

---

## Step 8: Iterate and Optimize (Optional)

If results don't meet criteria:

### Parameter Tuning

Adjust in IronOreIndicator.py:
- EMA periods (12/26/50)
- RSI thresholds (30/70)
- BB standard deviations (2.0)
- Volume multiplier (1.5)
- ATR multiplier for regime detection (1.5)

### Signal Filtering

Enhance signal logic:
- Add time-of-day filters
- Require multiple confirmation bars
- Adjust position sizing based on volatility

### Risk Management

Tighten controls:
- Reduce position size (10% instead of 20%)
- Tighter stop loss (2% instead of 3%)
- More conservative profit targets

**After changes**:
1. Re-run framework (Step 5)
2. Re-run notebook (Step 6)
3. Compare results

---

## Step 9: Documentation

### Create README.md

```bash
cat > README.md << 'EOF'
# Iron Ore Trading Strategy Indicator

Multi-indicator confirmation system for DCE Iron Ore futures.

## Quick Start

1. Fetch data: `python3 data_fetch.py`
2. Run indicator: Framework execution
3. Analyze: `jupyter notebook analysis.ipynb`

## Strategy

- 7 technical indicators
- 4 market regimes
- Multi-confirmation signals
- Position scaling: 20% → 40% → 60%

## Results

See `backtest_results.json` for performance metrics.

## Documentation

Complete documentation in `../IronOreTradingStrategyIndicator/`
EOF
```

---

## Complete Directory Structure

After all steps:

```
IronOreIndicator/
├── IronOreIndicator.py          # Main indicator implementation
├── uin.json                      # Input configuration
├── uout.json                     # Output configuration
├── data_fetch.py                 # Data fetching script
├── ohlcv_data.csv               # Historical OHLCV data
├── indicator_outputs.csv        # Indicator outputs from framework
├── analysis.ipynb               # Jupyter notebook
├── pnl_curve.png                # Equity curve chart
├── regime_distribution.png      # Regime analysis chart
├── trade_analysis.png           # Trade statistics chart
├── correlation_heatmap.png      # Indicator correlation
├── trade_log.csv                # Complete trade history
├── equity_curve.csv             # Equity time series
├── backtest_results.json        # All metrics
└── README.md                    # Quick reference
```

---

## Troubleshooting

### Data Fetch Fails

**Error**: "cannot unpack non-iterable NoneType object"

**Solution**: Check svr3 connection, verify market/code/dates are correct

### Framework Errors

**Error**: "meta_id not found"

**Solution**: Verify uin.json and uout.json are in same directory as IronOreIndicator.py

### Indicator Errors

**Error**: Division by zero in RSI

**Solution**: Add safety checks: `if self.loss_ema > 0`

### No Signals Generated

**Solution**: Lower confirmation thresholds, check regime detection logic

---

## Validation Checklist

- [ ] All 10 documentation chapters read
- [ ] Directory created: IronOreIndicator/
- [ ] uin.json created and validated
- [ ] uout.json created and validated
- [ ] IronOreIndicator.py implemented with all methods
- [ ] data_fetch.py created and executed
- [ ] ohlcv_data.csv contains 2024 data
- [ ] Framework run completed
- [ ] indicator_outputs.csv generated
- [ ] analysis.ipynb created with all 10 cells
- [ ] Jupyter notebook executed successfully
- [ ] All charts generated
- [ ] backtest_results.json created
- [ ] Metrics meet success criteria
- [ ] README.md created

---

## Timeline Estimate

| Step | Task | Duration |
|------|------|----------|
| 1 | Create directory | 1 min |
| 2 | Create JSON configs | 5 min |
| 3 | Implement IronOreIndicator.py | 2-4 hours |
| 4 | Fetch historical data | 5 min |
| 5 | Run framework | 10-30 min |
| 6 | Create Jupyter notebook | 30-60 min |
| 7 | Verify results | 15 min |
| 8 | Iterate (if needed) | 1-2 hours |
| 9 | Documentation | 10 min |

**Total**: 4-8 hours (depending on experience)

---

## Final Notes

**Reproduction Goal**: Any developer with framework access should be able to:
1. Follow these steps sequentially
2. Generate identical indicator logic
3. Produce similar backtest results (within variance of data updates)

**Success**: Positive P&L curve over Q4 2024 evaluation period with acceptable risk metrics.

**Next Steps After Reproduction**:
- Deploy to paper trading
- Monitor live performance
- Compare live vs backtest results
- Iterate on strategy improvements

---

## Support

For framework-specific questions, refer to WOS documentation in `../wos/`

For strategy questions, review Chapters 1-9 in this documentation.

---

**End of Documentation**
