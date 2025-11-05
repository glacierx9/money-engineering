# Iron Ore Trading Strategy Indicator

Multi-indicator confirmation system for DCE Iron Ore futures trading.

## Overview

**Type**: Tier-1 Indicator (WOS Framework)
**Market**: DCE (Dalian Commodity Exchange)
**Instrument**: i<00> (Iron Ore logical contract)
**Granularity**: 900 seconds (15 minutes)
**Strategy**: 7-indicator multi-confirmation with market regime detection

## Features

- **Triple EMA** (12/26/50) for trend identification
- **MACD** for momentum confirmation
- **RSI** for mean reversion signals
- **Bollinger Bands** for price extremes
- **ATR** for volatility measurement
- **Volume EMA** for liquidity confirmation
- **Market Regime Detection** (4 regimes: Uptrend, Downtrend, Ranging, Chaos)
- **Progressive Position Sizing** (20% → 40% → 60%)
- **Risk Management** (3% stops, 5%/8% profit targets)

## Quick Start

```bash
# 1. Verify framework installation
python3 -c "import pycaitlyn; import pycaitlynts3; print('OK')"

# 2. Run indicator (framework execution - refer to WOS docs)
# Framework will process OHLCV data and generate signals

# 3. Analyze results
jupyter notebook analysis.ipynb
```

## Complete Documentation

**Full implementation guide and reproduction steps:**

📚 **[IronOreTradingStrategyIndicator Documentation](../IronOreTradingStrategyIndicator/INDEX.md)**

The documentation provides:
- Strategy overview and objectives
- Technical indicator specifications
- Market regime classification rules
- Signal generation logic
- Risk management procedures
- Complete implementation guide
- Backtesting procedures
- Jupyter notebook visualization setup
- Step-by-step reproduction workflow

## Project Structure

```
IronOreIndicator/
├── IronOreIndicator.py       # Main indicator implementation
├── uin.json                   # Input configuration (OHLCV schema)
├── uout.json                  # Output configuration (signals + indicators)
├── analysis.ipynb             # P&L visualization notebook
├── test_resuming_mode.py      # Replay consistency test
├── README.md                  # This file
└── wos/ → ../wos             # Framework documentation (symlink)
```

## Strategy Performance

**Backtest Period**: 2024-01-01 to 2024-12-31
**Evaluation Period**: 2024-10-01 to 2024-12-31

**Target Metrics**:
- Positive cumulative P&L
- Sharpe ratio > 1.0
- Maximum drawdown < 15%
- Win rate > 45%
- Profit factor > 1.3

## Development

### Framework Testing

```bash
# Quick test (7 days)
python /home/wolverine/bin/running/calculator3_test.py \
    --testcase . \
    --algoname IronOreIndicator \
    --sourcefile IronOreIndicator.py \
    --start 20250703203204 \
    --end 20250710203204 \
    --granularity 900 \
    --tm wss://10.99.100.116:4433/tm \
    --tm-master 10.99.100.116:6102 \
    --rails https://10.99.100.116:4433/private-api/ \
    --token YOUR_TOKEN \
    --category 1 \
    --is-managed 1 \
    --restore-length 864000000 \
    --multiproc 1
```

### Replay Consistency Test

```bash
python test_resuming_mode.py
```

## Resources

- **Strategy Documentation**: [../IronOreTradingStrategyIndicator/](../IronOreTradingStrategyIndicator/INDEX.md)
- **WOS Framework**: [./wos/INDEX.md](./wos/INDEX.md) - Complete framework guide
- **Tier-1 Patterns**: [./wos/07-tier1-indicator.md](./wos/07-tier1-indicator.md)
- **Backtest Guide**: [./wos/06-backtest.md](./wos/06-backtest.md)

## Implementation Notes

This indicator follows WOS framework patterns:
- Stateless design with online algorithms (O(1) memory)
- Cycle boundary handling for bar processing
- sv_object serialization for state persistence
- Replay consistency guaranteed

All technical indicators use online computation algorithms ensuring bounded memory usage and deterministic replay.

---

For complete implementation details and reproduction steps, see the [full documentation](../IronOreTradingStrategyIndicator/INDEX.md).
