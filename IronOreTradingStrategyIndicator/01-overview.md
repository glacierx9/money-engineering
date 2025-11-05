# Chapter 1: Overview

## Strategy Objective

Generate profitable buy/sell signals for DCE Iron Ore futures using multi-indicator confirmation and market regime detection.

## Core Approach

**Multi-Layer Confirmation System**

Layer 1: **Trend Identification**
- Triple EMA (12/26/50 periods)
- MACD (12/26/9 configuration)

Layer 2: **Mean Reversion**
- RSI (14-period)
- Bollinger Bands (20-period, 2 std dev)

Layer 3: **Volatility & Risk**
- ATR (14-period)
- Bollinger Band Width

Layer 4: **Liquidity Confirmation**
- Volume EMA (20-period)

## Market Regime Detection

Four distinct regimes drive strategy behavior:

1. **Strong Uptrend**: Momentum-based long entries
2. **Strong Downtrend**: Momentum-based short entries
3. **Sideways/Ranging**: Mean reversion trades
4. **High Volatility Chaos**: No trading (capital preservation)

## Signal Generation Philosophy

**Multi-Confirmation Requirement**: ALL indicators must agree before generating signal.

**Position State Machine**: Alternating buy/sell signals (cannot buy twice consecutively).

**Progressive Position Sizing**: Scale into positions (20% → 40% → 60% max).

## Target Market

- **Exchange**: DCE (Dalian Commodity Exchange)
- **Instrument**: i<00> (Iron Ore logical contract)
- **Granularity**: 900 seconds (15-minute bars)
- **Session**: Day trading and swing positions
- **Data Source**: svr3 production server

## Capital Allocation

- **Total Capital**: $1,000,000
- **Initial Position**: 20% ($200,000)
- **Scale 1**: 40% ($400,000)
- **Scale 2**: 60% ($600,000)
- **Cash Reserve**: 40% minimum ($400,000)

## Risk Parameters

- **Stop Loss**: 3% per position
- **Profit Target 1**: 5% (partial exit)
- **Profit Target 2**: 8% (full exit)
- **Maximum Position**: 60% of capital
- **Regime Filter**: No trades in high volatility chaos

## Backtest Specifications

- **Full Backtest**: 2024-01-01 to 2024-12-31 (1 year)
- **Evaluation Period**: 2024-10-01 to 2024-12-31 (Q4 2024)
- **Signal Frequency**: Every 15-minute bar
- **Execution**: Next bar open (no look-ahead bias)

## Expected Outputs

1. **IronOreIndicator.py**: Complete indicator implementation
2. **uin.json**: Input configuration (OHLCV data schema)
3. **uout.json**: Output configuration (all indicators + signals)
4. **analysis.ipynb**: Jupyter notebook with P&L visualization
5. **Backtest Results**: Trade log, metrics, P&L curve

## Success Criteria

- Positive cumulative P&L over evaluation period
- Sharpe ratio > 1.0
- Maximum drawdown < 15%
- Win rate > 45%
- Profit factor > 1.3

## Implementation Framework

All code follows WOS (Wolverine Operating System) patterns:
- Stateless design with online algorithms
- O(1) memory complexity
- Cycle boundary handling
- sv_object serialization
- Replay consistency guaranteed

## Next Steps

Proceed to Chapter 2 for detailed technical indicator specifications.
