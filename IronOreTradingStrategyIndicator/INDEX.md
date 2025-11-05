# Iron Ore Trading Strategy Indicator - Documentation

**Comprehensive Multi-Indicator Confirmation System for DCE Iron Ore Futures**

## Table of Contents

1. [Overview](01-overview.md) - Strategy summary and objectives
2. [Technical Indicators](02-technical-indicators.md) - Seven indicator specifications
3. [Market Regimes](03-market-regimes.md) - Four regime classifications
4. [Signal Logic](04-signal-logic.md) - Buy/sell signal generation rules
5. [Risk Management](05-risk-management.md) - Position sizing and risk controls
6. [Implementation](06-implementation.md) - Code structure and algorithms
7. [Configuration](07-configuration.md) - uin.json and uout.json setup
8. [Testing](08-testing.md) - Backtesting procedures
9. [Visualization](09-visualization.md) - Jupyter notebook P&L analysis
10. [Reproduction Steps](10-reproduction-steps.md) - Complete workflow

## Quick Start

```bash
# 1. Navigate to indicator directory
cd /home/user/money-engineering/IronOreIndicator

# 2. Review configuration files
cat uin.json uout.json

# 3. Run indicator (framework required)
# Framework will execute IronOreIndicator.py

# 4. Launch Jupyter for P&L visualization
jupyter notebook analysis.ipynb
```

## System Requirements

- Python 3.x
- pycaitlyn framework
- pycaitlynts3
- pycaitlynutils3
- pandas, numpy, matplotlib
- Jupyter notebook
- Access to svr3 data server

## Strategy Summary

- **Market**: DCE (Dalian Commodity Exchange)
- **Instrument**: i<00> (Iron Ore logical contract)
- **Granularity**: 900 seconds (15 minutes)
- **Capital**: $1,000,000
- **Position Sizing**: 20% → 40% → 60% progressive scaling
- **Backtest Period**: 2024-01-01 to 2024-12-31
- **Evaluation Period**: 2024-10-01 to 2024-12-31

## Documentation Purpose

This documentation enables complete reproduction of the Iron Ore trading strategy indicator system. Follow chapters sequentially for implementation.
