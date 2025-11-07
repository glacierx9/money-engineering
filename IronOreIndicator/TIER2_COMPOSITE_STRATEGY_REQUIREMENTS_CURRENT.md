# Tier-2 Composite Trading Strategy - CURRENT IMPLEMENTATION STATUS

**Document Version**: 2.0 - AS-IMPLEMENTED
**Date**: 2025-11-07
**Status**: DEBUGGING - Basket PV Not Updating Issue
**Framework**: WOS (Wolverine Operating System)
**Strategy Type**: Multi-Instrument Portfolio Management (Tier-2)

---

## CRITICAL BUG STATUS

**Current Problem**: Basket trading mechanism not working properly
- ✅ Trades show as executing in logs
- ❌ `basket.price` stays 0.00 (should reflect market price)
- ❌ `basket.pv` stays at ¥250,000 (should change with positions)
- ❌ Composite `self.pv` stays frozen at ¥1,000,000
- ❌ `analysis.ipynb` shows "Empty data" (no output written)

**Symptoms**:
```
Market data routed: SHFE/cu2411 -> Basket 1, price=76570.00, basket.price=0.00 ❌
LONG basket 2: DCE/m, contracts=84, price=2958.00
  -> Basket 2 after entry: cash=¥250,000, pv=¥250,000, signal=1 ❌
```

**See**: `QUESTIONS_FOR_WOS_DEVELOPER.md` for comprehensive debugging questions

---

## Executive Summary

This document describes the **CURRENT IMPLEMENTATION** of a Tier-2 composite strategy managing 3 futures instruments. The strategy is **BUILT BUT NOT FUNCTIONAL** due to a critical bug where basket PV values don't update after trades execute.

**Primary Objective**: Debug and fix basket trading mechanism to achieve positive P&L through diversified, risk-managed trading.

---

## 1. Current Implementation Status

### 1.1 What's BUILT ✅

**Tier-1 Indicators** (All 3):
- ✅ IronOreIndicatorRelaxed (DCE/i<00>) - WORKING
- ✅ CopperIndicator (SHFE/cu<00>) - WORKING
- ✅ SoybeanIndicator (DCE/m<00>) - WORKING
- ✅ All export "close" field correctly in position 2
- ✅ All generate signals with confidence, regime, strength
- ✅ All use 7-indicator system (EMA, MACD, RSI, BB, ATR, Volume)

**Tier-2 CompositeStrategy**:
- ✅ Imports signals from all 3 Tier-1 indicators via `uin.json`
- ✅ Manages 3 baskets using `composite_strategyc3` base class
- ✅ Allocates capital via `_allocate(basket_idx, market, code+b'<00>', capital, 1.0)`
- ✅ Processes Tier-1 signals via signal parsers (IronOreSignalParser, etc.)
- ✅ Implements RiskManager with all limits
- ✅ Implements Rebalancer with drift detection
- ✅ Entry/exit logic with confidence thresholds (50%/40% - relaxed)
- ✅ Stop-loss checking (3% per position)
- ✅ Circuit breakers (10% max DD, 3% daily loss)
- ✅ Logging and portfolio metrics tracking

### 1.2 What's BROKEN ❌

**Critical Issues**:
1. **Basket Price Updates** - `basket.price` stays 0.00, never receives market data
2. **Basket PV Calculation** - `basket.pv` doesn't update after trades
3. **Market Data Routing** - `super().on_bar(bar)` doesn't route data to baskets
4. **Data Output** - No data written to server (analysis.ipynb fails)

**Root Cause Hypotheses**:
- Framework's `composite_strategy.on_bar()` doesn't auto-route market data?
- Need to call `basket.on_price_updated()` manually?
- Baskets allocated with logical contracts (`i<00>`) but market data has specific contracts (`i2501`)?
- Missing initialization step for baskets?

---

## 2. Current Architecture (AS-IMPLEMENTED)

```
┌────────────────────────────────────────────────────┐
│  TIER 1: Three Indicators (WORKING ✅)             │
│  ─────────────────────────────────────────────────│
│  1. IronOreIndicatorRelaxed (DCE/i<00>)           │
│  2. CopperIndicator (SHFE/cu<00>)                  │
│  3. SoybeanIndicator (DCE/m<00>)                   │
│                                                    │
│  Each exports: close, signal, confidence, regime   │
└──────────────────┬─────────────────────────────────┘
                   │
                   │ Tier-1 Signals (working ✅)
                   ▼
┌────────────────────────────────────────────────────┐
│  TIER 2: COMPOSITE STRATEGY (BROKEN ❌)            │
│  ─────────────────────────────────────────────────│
│  • Signal aggregation (works ✅)                   │
│  • Basket management (broken ❌)                   │
│  • Market data routing (broken ❌)                 │
│  • Position tracking (broken ❌)                   │
│  • PV calculation (broken ❌)                      │
└──────────────────┬─────────────────────────────────┘
                   │
                   │ Output (empty ❌)
                   ▼
┌────────────────────────────────────────────────────┐
│  OUTPUT: Portfolio Performance (NO DATA ❌)        │
│  • analysis.ipynb shows "Empty data"               │
│  • Likely due to basket.pv not updating            │
└────────────────────────────────────────────────────┘
```

---

## 3. Current File Structure

```
IronOreIndicator/
├── IronOreIndicatorRelaxed.py          # Tier-1 (working ✅)
├── uin.json                             # Import config (working ✅)
├── uout.json                            # Export config (working ✅)
│
├── CopperIndicator/
│   ├── CopperIndicator.py              # Tier-1 (working ✅)
│   ├── uin.json                         # (working ✅)
│   └── uout.json                        # (working ✅)
│
├── SoybeanIndicator/
│   ├── SoybeanIndicator.py             # Tier-1 (working ✅)
│   ├── uin.json                         # (working ✅)
│   └── uout.json                        # (working ✅)
│
├── CompositeStrategy/
│   ├── CompositeStrategy.py            # Tier-2 (broken ❌)
│   ├── uin.json                         # (correct ✅)
│   ├── uout.json                        # (correct ✅)
│   ├── analysis.ipynb                   # (fails - no data ❌)
│   ├── QUESTIONS_FOR_WOS_DEVELOPER.md  # (debugging questions)
│   └── test_output.log                  # (test logs)
│
├── TIER2_COMPOSITE_STRATEGY_REQUIREMENTS.md        # Original spec
└── TIER2_COMPOSITE_STRATEGY_REQUIREMENTS_CURRENT.md # This file
```

---

## 4. Current CompositeStrategy Implementation

### 4.1 Basket Configuration (CORRECT ✅)

```python
BASKET_COUNT = 3
INITIAL_CAPITAL = 1000000.0

BASKET_MAPPING = {
    0: (b'DCE', b'i'),   # Iron Ore
    1: (b'SHFE', b'cu'), # Copper
    2: (b'DCE', b'm'),   # Soybean Meal
}
```

### 4.2 Basket Initialization (CURRENT IMPLEMENTATION)

```python
def _initialize_baskets(self, initial_cash: float):
    """Allocate baskets to instruments"""
    base_capital = initial_cash * 0.25  # 25% each

    for basket_idx in range(BASKET_COUNT):
        market, code = self.basket_to_instrument[basket_idx]
        instrument_code = code + b'<00>'  # e.g., b'i<00>'

        # Allocate basket
        self._allocate(basket_idx, market, instrument_code, base_capital, 1.0)
```

**Status**: Allocates correctly (cash transfers), but baskets don't receive market data

### 4.3 Market Data Routing (BROKEN ❌)

**Current Implementation**:
```python
def on_bar(self, bar: pc.StructValue) -> List[pc.StructValue]:
    market = bar.get_market()
    code = bar.get_stock_code()
    tm = bar.get_time_tag()
    ns = bar.get_namespace()

    # ... cycle boundary handling ...

    # Route bars based on namespace
    if ns == pc.namespace_global:
        # Market data (SampleQuote) - route to baskets via framework
        super().on_bar(bar)  # ❌ DOESN'T WORK - basket.price stays 0.00

        # Log for debugging
        commodity = self.calc_commodity(code)
        key = (market, commodity)
        if key in self.instrument_to_basket:
            basket_idx = self.instrument_to_basket[key]
            basket = self.strategies[basket_idx]
            close_price = bar.get_double(3)
            logger.info(f"Market data routed: {market.decode()}/{code.decode()} -> Basket {basket_idx}, "
                       f"price={close_price:.2f}, basket.price={basket.price:.2f} ❌")

    elif ns == pc.namespace_private:
        # Tier-1 indicator signals
        self._process_tier1_signal(bar)  # ✅ WORKS

    return []
```

**Problem**: `super().on_bar(bar)` is called but `basket.price` remains 0.00

**Hypothesis**: Need to manually call `basket.on_price_updated()` or parse SampleQuote ourselves?

### 4.4 Signal Processing (WORKING ✅)

```python
def _process_tier1_signal(self, bar: pc.StructValue):
    """Parse incoming Tier-1 indicator signals"""
    for basket_idx, parser in self.parsers.items():
        if ns == parser.namespace and meta_id == parser.meta_id:
            parser.from_sv(bar)  # Parse signal

            self.tier1_signals[basket_idx] = {
                'close': parser.close,  # Price from Tier-1
                'signal': parser.signal,
                'confidence': parser.confidence,
                'regime': parser.regime,
                'signal_strength': parser.signal_strength,
                # ... more fields ...
            }
```

**Status**: ✅ Works correctly, signals parsed and stored

### 4.5 Entry Execution (PARTIALLY WORKING ⚠️)

```python
def _execute_entry(self, basket_idx: int, signal_data: Dict):
    """Execute entry for basket"""
    basket = self.strategies[basket_idx]
    market, code = self.basket_to_instrument[basket_idx]

    # Calculate position size using price from Tier-1 signal
    allocated_capital = self.pv * self.base_allocation_pct
    current_price = signal_data['close']  # From Tier-1, not basket
    contracts = int(allocated_capital / current_price)

    # Risk check...

    # Execute via framework
    basket._signal(current_price, basket.timetag, signal)

    # Log basket state
    logger.info(f"  -> Basket {basket_idx} after entry: cash=¥{basket.cash:,.0f}, "
               f"pv=¥{basket.pv:,.0f}, signal={basket.signal}")
```

**Status**:
- ✅ `basket._signal()` is called correctly
- ✅ `basket.signal` changes to 1 (position open)
- ❌ `basket.pv` stays at ¥250,000 (doesn't reflect position value)
- ❌ `basket.cash` doesn't change

**Problem**: Either `basket._signal()` isn't working, or basket needs market price context to calculate PV

### 4.6 Entry Conditions (RELAXED - 50%/40%)

```python
def _should_enter(self, signal_data: Dict) -> bool:
    """Entry condition checks (relaxed thresholds)"""
    if signal_data['signal'] == 0:
        return False

    # RELAXED from 0.60 to 0.50
    if signal_data['confidence'] < 0.50:
        return False

    if signal_data['regime'] == 4:  # Avoid chaos
        return False

    # RELAXED from 0.50 to 0.40
    if signal_data['signal_strength'] < 0.40:
        return False

    return True
```

**Status**: ✅ Working (trades execute), but too strict (only 3 trades in 7 days)

### 4.7 Exit Conditions (CURRENT IMPLEMENTATION)

```python
def _should_exit(self, basket_idx: int, signal_data: Dict, basket) -> bool:
    """Exit condition checks"""
    # 1. Stop-loss check (3%)
    entry_price = self.entry_prices[basket_idx]
    current_price = signal_data['close']
    if self._check_stop_loss(entry_price, current_price, basket.signal):
        return True

    # 2. Signal reversal
    if signal_data['signal'] * basket.signal < 0:
        return True

    # 3. Confidence dropped below 30%
    if signal_data['confidence'] < 0.30:
        return True

    # 4. Chaos regime
    if signal_data['regime'] == 4:
        return True

    return False
```

**Status**: ✅ Logic correct, but can't verify effectiveness due to PV bug

---

## 5. Current Configuration Files

### 5.1 CompositeStrategy/uin.json (AS-IMPLEMENTED)

```json
{
  "global": {
    "imports": {
      "SampleQuote": {
        "fields": ["open", "high", "low", "close", "volume", "turnover"],
        "granularities": [900],
        "revision": 4294967295,
        "markets": ["DCE", "SHFE"],
        "security_categories": [[1, 2, 3]],
        "securities": [["i", "m"], ["cu"]]
      }
    }
  },
  "private": {
    "imports": {
      "IronOreIndicatorRelaxed": {
        "fields": [
          "_preserved_field",
          "bar_index",
          "close",  // ADDED - price data from Tier-1
          "ema_12",
          "ema_26",
          "ema_50",
          "macd",
          "macd_signal",
          "macd_histogram",
          "rsi",
          "bb_upper",
          "bb_middle",
          "bb_lower",
          "bb_width",
          "bb_width_pct",
          "atr",
          "volume_ema",
          "regime",
          "signal",
          "confidence",
          "signal_strength",
          "desired_contracts",
          "contracts_to_add",
          "position_state",
          "cash",
          "contracts_held",
          "portfolio_value"
        ],
        "granularities": [900],
        "markets": ["DCE"],
        "security_categories": [[1, 2, 3]],
        "securities": [["i"]]
      },
      "CopperIndicator": {
        // Same structure, market="SHFE", securities=[["cu"]]
      },
      "SoybeanIndicator": {
        // Same structure, market="DCE", securities=[["m"]]
      }
    }
  }
}
```

**Status**: ✅ Correct - imports both market data and Tier-1 signals

### 5.2 CompositeStrategy/uout.json (AS-IMPLEMENTED)

```json
{
  "private": {
    "markets": ["COMPOSITE"],
    "security_categories": [[1]],
    "securities": [["PORTFOLIO"]],
    "sample_granularities": {
      "type": "min",
      "cycles": [900],
      "cycle_lengths": [0]
    },
    "export": {
      "XXX": {
        "fields": [
          "_preserved_field",
          "bar_index",
          "active_positions",
          "total_signals_processed",
          "portfolio_exposure_pct",
          "cash_reserve_pct"
        ],
        "defs": [
          {"field_type": "int64", "display_name": "Preserved Field", "precision": 0, "sample_type": 0, "multiple": 1},
          {"field_type": "integer", "display_name": "Bar Index", "precision": 0, "sample_type": 0, "multiple": 1},
          {"field_type": "integer", "display_name": "Active Positions", "precision": 0, "sample_type": 0, "multiple": 1},
          {"field_type": "integer", "display_name": "Total Signals Processed", "precision": 0, "sample_type": 0, "multiple": 1},
          {"field_type": "double", "display_name": "Portfolio Exposure Pct", "precision": 6, "sample_type": 0, "multiple": 1},
          {"field_type": "double", "display_name": "Cash Reserve Pct", "precision": 6, "sample_type": 0, "multiple": 1}
        ],
        "revision": -1
      }
    }
  }
}
```

**Status**: ✅ Correct format (XXX placeholder, full field defs)

**Note**: Uses "XXX" because framework requires this for composite strategies

---

## 6. Test Results (CURRENT STATE)

### 6.1 Quick Test (7 days: 2024-10-25 to 2024-11-01)

**Command**:
```bash
cd CompositeStrategy
python3 calculator3_test.py --testcase . --algoname CompositeStrategy \
  --sourcefile CompositeStrategy.py --start 20241025000000 --end 20241101000000 \
  --granularity 900 --multiproc 1 --category 1
```

**Results**:
- ✅ Test completes without errors
- ✅ All 3 Tier-1 signals received
- ✅ Market data arrives (60+ bars logged)
- ✅ 3 trades executed (LONG basket 2)
- ❌ `basket.price` stays 0.00 throughout
- ❌ `basket.pv` stays ¥250,000 after trades
- ❌ `self.pv` stays ¥1,000,000 (frozen)
- ❌ `NV` stays 1.0000 (correct given frozen PV)

**Sample Logs**:
```
[2025-11-07 03:31:45] Market data routed: DCE/i2411 -> Basket 0, price=759.00, basket.price=0.00 ❌
[2025-11-07 03:31:48] LONG basket 2: DCE/m, contracts=84, price=2958.00, conf=0.60
[2025-11-07 03:31:48]   -> Basket 2 after entry: cash=¥250,000, pv=¥250,000, signal=1 ❌
[2025-11-07 03:31:48] [Bar 20] PV=¥1,000,000, NV=1.0000, Active=1/3 ❌
[2025-11-07 03:31:48] CLOSE basket 2: DCE/m, entry=2958.00, exit=2958.00, P&L=+0.00%
[2025-11-07 03:31:48]   -> Basket 2 after exit: cash=¥250,000, pv=¥250,000, signal=0 ❌
```

### 6.2 analysis.ipynb Results

**Cell 2**: Fetch Composite Strategy data
```python
reader = svr3.sv_reader(
    meta_name="CompositeStrategy",
    markets=["COMPOSITE"],
    codes=["PORTFOLIO"],
    namespace=pc.namespace_private,
    ...
)
```

**Result**: ❌ "Empty data" - No data written to server

**Hypothesis**: Either `ready_to_serialize()` returns False, or data isn't being written due to bug

---

## 7. Debugging Attempts Made

### 7.1 Attempt 1: Add "close" field to Tier-1 exports ✅ DONE
- Modified all 3 Tier-1 uout.json files
- Added "close" field at position 2
- Modified signal parsers to extract close price
- **Result**: Trades execute using Tier-1 prices, but basket.pv still doesn't update

### 7.2 Attempt 2: Manual market data routing ❌ FAILED
- Extracted commodity from specific contracts (`i2501` → `i`)
- Called `_update_basket_price()` to manually set `basket.price`
- **Result**: Didn't work - framework state corrupted, NV exploded

### 7.3 Attempt 3: Call super().on_bar(bar) for market data ❌ FAILED
- Removed manual routing
- Called `super().on_bar(bar)` expecting framework to route automatically
- **Result**: `basket.price` stays 0.00 - framework doesn't route automatically

### 7.4 Attempt 4: Don't manually set basket.pv ✅ CORRECT APPROACH
- Removed all manual `basket.pv = ...` assignments
- Removed manual `self.pv = ...` assignments
- Let framework calculate via `_save()` and `_sync()`
- **Result**: No improvement - basket.pv still frozen because baskets aren't receiving market data

### 7.5 Attempt 5: Add initialization flag ✅ DONE (but doesn't fix PV bug)
- Added `self.initialized = False` flag
- Set to True after first cycle
- Changed `ready_to_serialize()` to check `self.initialized and self.bar_index > 0`
- **Result**: Proper initialization, but doesn't fix basket PV bug

---

## 8. Current Theories on Root Cause

### 8.1 Theory 1: Missing `basket.on_price_updated()` call
**Hypothesis**: Baskets need explicit price updates via `basket.on_price_updated(tm, market, code, open, close, high, low, volume)`

**Evidence**:
- `super().on_bar(bar)` doesn't update `basket.price`
- WOS docs may show manual price update pattern

**Next Step**: Ask WOS developer if this method exists and how to use it

### 8.2 Theory 2: Logical vs Specific Contract Mismatch
**Hypothesis**: Baskets allocated with `i<00>` but market data arrives with `i2501`, framework can't match

**Evidence**:
- Allocation uses `code + b'<00>'` (e.g., `b'i<00>'`)
- Market data shows `i2411`, `i2501`, `cu2411`, `m2501`
- No matching mechanism visible

**Next Step**: Ask WOS developer how contract matching works

### 8.3 Theory 3: Need SampleQuote Parser in CompositeStrategy
**Hypothesis**: Must manually parse SampleQuote bars and extract OHLCV

**Evidence**:
- No `SampleQuote` sv_object instantiated in `__init__()`
- May need to manually parse market data

**Next Step**: Ask WOS developer if manual parsing is required

### 8.4 Theory 4: basket._signal() Needs Price Context
**Hypothesis**: `basket._signal()` can't calculate PV without `basket.price` being set first

**Evidence**:
- `basket.signal` changes (position opens)
- But `basket.pv` doesn't change (no position value calculated)
- `basket.price` is 0.00 when `_signal()` is called

**Next Step**: Ask WOS developer about `basket._signal()` requirements

---

## 9. Questions for WOS Developer

**See**: `QUESTIONS_FOR_WOS_DEVELOPER.md` for complete list (50+ questions)

**Top Priority Questions**:
1. How should market data reach baskets? `super().on_bar()` or `basket.on_price_updated()`?
2. Does `_allocate()` with `b'i<00>'` automatically match market data for `i2501`?
3. Why does `basket.price` stay 0.00 after `super().on_bar(bar)`?
4. Why doesn't `basket.pv` update after `basket._signal()` is called?
5. How should PV be calculated - by framework or manually?

---

## 10. Next Steps

### 10.1 Immediate Actions
1. ✅ Send questions file to WOS developer
2. ✅ Send this current implementation document for context
3. ⏳ Wait for answers from WOS developer
4. ⏳ Implement correct market data routing pattern
5. ⏳ Fix basket PV calculation
6. ⏳ Verify analysis.ipynb works

### 10.2 Once Bug Fixed
1. Run full 7-day test and verify:
   - ✅ `basket.price` updates with market data
   - ✅ `basket.pv` changes after trades
   - ✅ `self.pv` reflects portfolio value correctly
   - ✅ `analysis.ipynb` loads data successfully

2. Optimize entry thresholds (currently too strict)
3. Run 1-month backtest
4. Analyze P&L and adjust parameters
5. Run full 3-month backtest

---

## 11. Implementation Differences from Original Spec

### 11.1 Changes Made During Implementation

| Original Spec | Current Implementation | Reason |
|--------------|----------------------|--------|
| Confidence threshold: 60% | **50%** | Relaxed to get more trades |
| Signal strength: 50% | **40%** | Relaxed to get more trades |
| Position sizing from allocated capital | From **signal_data['close']** (Tier-1 price) | Workaround - basket.price is 0.00 |
| Automatic basket management | Manual **entry_prices** tracking | Workaround for PV bug |
| uout.json uses strategy name | Uses **"XXX"** placeholder | Framework requirement |
| Expect ~300-1500 trades | Only **3 trades** in 7 days | Entry thresholds too strict |

### 11.2 Additional Features Added

1. **Initialization flag**: `self.initialized` to control data output
2. **Extensive logging**: Debug logs for market data routing, basket state
3. **Price extraction from Tier-1**: Use `signal_data['close']` for trades
4. **Commodity extraction**: `calc_commodity()` to match specific→logical contracts

---

## 12. Performance Metrics (CANNOT MEASURE DUE TO BUG)

**Target vs Current**:

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Total Return | >0% | **Cannot measure** | ❌ PV frozen |
| Max Drawdown | <10% | **0%** (frozen) | ❌ No variation |
| Portfolio PV | Changes with trades | **¥1,000,000** (frozen) | ❌ Bug |
| Basket PV | Changes with positions | **¥250,000** (frozen) | ❌ Bug |
| NV | >1.0 preferred | **1.0000** (frozen) | ❌ PV frozen |
| Trades Executed | 300-1500 (3 months) | **3** (7 days) | ⚠️ Too few |
| Active Positions | 0-3 | **0-1** | ⚠️ Rarely enter |

---

## 13. Code Quality Status

- ✅ WOS framework patterns followed
- ✅ Configuration files valid
- ✅ No syntax errors
- ✅ Online algorithms (EMA, bounded memory)
- ✅ Proper cycle boundary handling
- ✅ sv_object pattern for signal parsers
- ✅ Comprehensive logging
- ❌ **NOT FUNCTIONAL** - Critical bug prevents actual trading

---

## 14. Summary for WOS Developer

### 14.1 What We Need Help With

**Primary Issue**: Basket trading mechanism not working
- Baskets are allocated correctly (cash transfers)
- Tier-1 signals are received correctly
- `basket._signal()` is called correctly
- BUT: `basket.price` stays 0.00 and `basket.pv` doesn't update

**Specific Questions**:
1. How should market data reach baskets?
2. How does `_allocate()` relate to market data subscriptions?
3. Why doesn't `super().on_bar(bar)` update `basket.price`?
4. What's the correct way to update basket prices and PV?

**What Works**:
- ✅ All 3 Tier-1 indicators generate signals
- ✅ CompositeStrategy receives all signals
- ✅ Signal parsing works perfectly
- ✅ Risk management logic implemented
- ✅ Entry/exit logic implemented

**What Doesn't Work**:
- ❌ Market data not reaching baskets
- ❌ Basket prices stay 0.00
- ❌ Basket PV doesn't reflect positions
- ❌ Portfolio PV frozen
- ❌ No data output for analysis

### 14.2 Files to Review

1. **CompositeStrategy.py** - Main implementation (lines 476-505 for market data routing)
2. **uin.json** - Configuration (imports both global and private)
3. **uout.json** - Export configuration
4. **QUESTIONS_FOR_WOS_DEVELOPER.md** - Comprehensive debugging questions
5. **This document** - Complete current state

### 14.3 Expected Behavior After Fix

```
BEFORE:
  Market data arrives → super().on_bar(bar) → basket.price stays 0.00 ❌

AFTER (expected):
  Market data arrives → [correct routing] → basket.price = 2958.00 ✅

BEFORE:
  basket._signal(price, tm, 1) → basket.pv stays ¥250,000 ❌

AFTER (expected):
  basket._signal(price, tm, 1) → basket.pv = ¥250,000 + position_value ✅
```

---

## 15. Appendix: Complete Test Logs

**See**: `test_output.log` in CompositeStrategy/ directory

**Key Log Sections**:
- Basket initialization (lines 1-10)
- Market data arrival (lines 12-15)
- Signal processing (lines 16-25)
- Trade execution (lines 27-40)
- Portfolio state (lines 41-60)

---

**Document End**

This document represents the **ACTUAL CURRENT STATE** of the Tier-2 Composite Strategy implementation as of 2025-11-07. Send this along with `QUESTIONS_FOR_WOS_DEVELOPER.md` to get help debugging the basket trading mechanism.

**Status**: AWAITING WOS DEVELOPER GUIDANCE 🚧
