# Chapter 12: Complete Example Project

**Learning objectives:**
- Build a complete trading system from scratch
- Apply all learned concepts
- Follow best practices throughout
- Deploy to production
- Monitor and maintain

**Previous:** [11 - Fine-tune and Iterate](11-fine-tune-and-iterate.md)

---

## Overview

This chapter walks through building a complete 3-tier trading system from idea to production deployment.

## Project: Mean Reversion System

### Strategy Concept

Trade mean reversion in commodity futures:
- **Tier 1**: RSI-based mean reversion indicator
- **Tier 2**: Portfolio manager for multiple commodities
- **Tier 3**: Execution with risk management

### Phase 1: Tier 1 Indicator

**File**: `MeanReversionIndicator.py`

```python
#!/usr/bin/env python3
"""RSI Mean Reversion Indicator"""

import math
from collections import deque
import pycaitlyn as pc
import pycaitlynts3 as pcts3
import pycaitlynutils3 as pcu3

# Globals
use_raw = True
overwrite = False
granularity = 900
schema = None
max_workers = 1
worker_no = None
exports = {}
imports = {}
metas = {}
logger = pcu3.vanilla_logger()

class SampleQuote(pcts3.sv_object):
    def __init__(self):
        super().__init__()
        self.meta_name = "SampleQuote"
        self.namespace = pc.namespace_global
        self.revision = (1 << 32) - 1
        self.open = self.high = self.low = self.close = None
        self.volume = self.turnover = None

class MeanReversionIndicator(pcts3.sv_object):
    """RSI-based mean reversion indicator."""

    def __init__(self, commodity, market):
        super().__init__()
        self.meta_name = "MeanReversion"
        self.namespace = pc.namespace_private
        self.revision = (1 << 32) - 1
        self.granularity = 900
        self.market = market
        self.code = commodity + b'<00>'

        # RSI calculation
        self.rsi = 50.0
        self.gain_ema = 0.0
        self.loss_ema = 0.0
        self.alpha = 2.0 / 15.0  # 14-period RSI

        # State
        self.bar_index = 0
        self.timetag = None
        self.prev_close = 0.0
        self.signal = 0
        self.confidence = 0.0
        self.initialized = False
        self.bars_since_start = 0
        self.latest_sv = None
        self.WARMUP_PERIOD = 80 # As per documentation example

        # Data parser
        self.quote = SampleQuote()
        self.persistent = True

    def _rebuild_finished(self):
        return self.bars_since_start >= self.WARMUP_PERIOD

    def _from_sv(self, sv):
        # This will set all the fields that are marked as persistent
        super().from_sv(sv)

    def _load_from_sv(self, sv):
        temp = self.__class__(self.code[:-4].decode(), self.market)
        # Copy metadata
        temp.market = self.market
        temp.code = self.code
        temp.meta_id = self.meta_id
        temp.granularity = self.granularity
        temp.namespace = self.namespace
        # Call PARENT's from_sv directly
        super(self.__class__, temp).from_sv(sv)
        return temp

    def _equal(self, other, tolerance=1e-9):
        # Compare relevant fields for reconciliation
        if not isinstance(other, MeanReversionIndicator):
            return False
        
        # Compare numerical fields with tolerance
        if not (math.isclose(self.rsi, other.rsi, rel_tol=tolerance) and
                math.isclose(self.gain_ema, other.gain_ema, rel_tol=tolerance) and
                math.isclose(self.loss_ema, other.loss_ema, rel_tol=tolerance) and
                math.isclose(self.alpha, other.alpha, rel_tol=tolerance) and
                math.isclose(self.prev_close, other.prev_close, rel_tol=tolerance) and
                math.isclose(self.confidence, other.confidence, rel_tol=tolerance)):
            return False

        # Compare other fields directly
        if not (self.bar_index == other.bar_index and
                self.timetag == other.timetag and
                self.signal == other.signal and
                self.initialized == other.initialized):
            return False
            
        return True

    def _reconcile_state(self):
        if self.latest_sv is None:
            # This should not happen if _reconcile_state is called correctly
            logger.error(f"Reconciliation failed for {self.market}-{self.code}: latest_sv is None")
            return

        saved_state = self._load_from_sv(self.latest_sv)
        if not self._equal(saved_state):
            logger.error(f"Reconciliation failed for {self.market}-{self.code} at timetag {self.timetag}: "
                         f"Calculated: rsi={self.rsi}, gain_ema={self.gain_ema}, loss_ema={self.loss_ema}, "
                         f"prev_close={self.prev_close}, signal={self.signal}, confidence={self.confidence}. "
                         f"Saved: rsi={saved_state.rsi}, gain_ema={saved_state.gain_ema}, loss_ema={saved_state.loss_ema}, "
                         f"prev_close={saved_state.prev_close}, signal={saved_state.signal}, confidence={saved_state.confidence}.")
            raise AssertionError("State reconciliation failed.")

    def calculate_state(self):
        close = float(self.quote.close)

        if not self.initialized:
            self.prev_close = close
            self.initialized = True
            return

        # Calculate RSI
        change = close - self.prev_close
        gain = max(change, 0)
        loss = max(-change, 0)

        self.gain_ema = self.alpha * gain + (1 - self.alpha) * self.gain_ema
        self.loss_ema = self.alpha * loss + (1 - self.alpha) * self.loss_ema

        if self.loss_ema > 0:
            rs = self.gain_ema / self.loss_ema
            self.rsi = 100 - (100 / (1 + rs))
        else:
            self.rsi = 100

        # Generate mean reversion signal
        if self.rsi < 30:
            self.signal = 1  # Oversold - buy
            self.confidence = (30 - self.rsi) / 30
        elif self.rsi > 70:
            self.signal = -1  # Overbought - sell
            self.confidence = (self.rsi - 70) / 30
        else:
            self.signal = 0
            self.confidence = 0.0

        self.prev_close = close

    def initialize(self, imports, metas):
        self.load_def_from_dict(metas)
        self.set_global_imports(imports)
        self.quote.load_def_from_dict(metas)
        self.quote.set_global_imports(imports)

    def on_bar(self, bar: pc.StructValue):
        # 1. Extract bar info
        market = bar.get_market()
        code = bar.get_stock_code()
        granularity = bar.get_granularity()
        tm = bar.get_time_tag()
        

        ret = [] # Initialize results list

        
        # 2. Initialize timetag if None and granularity matches
        if self.granularity == granularity and self.timetag is None:
            self.timetag = tm

        # 3. Check timetag advancement (CRITICAL: granularity match)
        if self.granularity == granularity and self.timetag is not None and self.timetag < tm:
            # 3a. Calculate frame with OLD data
            self.calculate_state()

            # 3b. Reconcile (after rebuilding finished)
            if self.latest_sv is not None and self._rebuild_finished():
                self._reconcile_state()

            # 3c. Restore state (during rebuilding)
            # The 'overwrite' global is typically used in the runner to force state overwrite
            # This part assumes 'overwrite' is a global or passed in a similar fashion.
            # For this example, we'll assume it's a global variable as seen in the original file.
            if not self._rebuild_finished() and self.latest_sv is not None:
                self._from_sv(self.latest_sv)

            # 3d. Output if ready
            if self._rebuild_finished(): # Only output after warm-up
                ret.append(self.copy_to_sv())

            # 3e. Update state (AFTER all processing for the current cycle)
            self.latest_sv = None # Clear cached latest_sv after use
            self.timetag = tm
            self.bar_index += 1  # Persisted counter
            self.bars_since_start += 1  # Non-persisted counter
            self.initialized = self._rebuild_finished() # Update initialized flag based on warm-up

        # 4. Import data (AFTER cycle pass and state updates)
        # This caches the incoming bar data into self.quote for the NEXT calculation cycle.
        if bar.get_namespace() == self.quote.namespace and \
           bar.get_meta_id() == self.quote.meta_id and \
           bar.get_market() == self.market and \
           bar.get_stock_code().startswith(self.code[:-4]) and \
           bar.get_stock_code().endswith(b'<00>') and \
           bar.get_granularity() == self.granularity:
            self.quote.from_sv(bar)
        
        # If it's a saved state SV, cache it
        if bar.get_namespace() == self.namespace and \
           bar.get_meta_id() == self.meta_id and \
            bar.get_stock_code() == self.code and \
            bar.get_granularity() == self.granularity and \
            bar.get_market() == self.market:
            self.latest_sv = bar
        return ret

manager = IndicatorManager()

async def on_init():
    global manager, imports, metas, worker_no
    if worker_no != 0 and metas and imports:
        manager.initialize(imports, metas)

async def on_ready():
    pass

async def on_bar(bar):
    global manager, worker_no
    if worker_no != 1:
        return []
    return manager.on_bar(bar)

# Other callbacks...
async def on_market_open(market, tradeday, time_tag, time_string):
    pass
async def on_market_close(market, tradeday, timetag, timestring):
    pass
async def on_tradeday_begin(market, tradeday, time_tag, time_string):
    pass
async def on_tradeday_end(market, tradeday, timetag, timestring):
    pass
async def on_reference(market, tradeday, data, timetag, timestring):
    pass
async def on_historical(params, records):
    pass
```

### Phase 2: Test and Validate

```bash
# Quick test
python calculator3_test.py --testcase ./ --algoname MeanReversion \
    --sourcefile MeanReversionIndicator.py \
    --start 20250703000000 --end 20250710000000 \
    --granularity 900 --category 1

# Replay consistency
python test_resuming_mode.py

# Full backtest
python calculator3_test.py --testcase ./ --algoname MeanReversion \
    --sourcefile MeanReversionIndicator.py \
    --start 20230101000000 --end 20250925000000 \
    --granularity 900 --category 1
```

### Phase 3: Visualize and Optimize

```python
# Create visualization script
import svr3
import pandas as pd
import matplotlib.pyplot as plt

def analyze_indicator():
    client = svr3.Client("10.99.100.116", 8080, "TOKEN")
    data = client.fetch(
        namespace="private",
        strategy="MeanReversion",
        market="SHFE",
        code="cu<00>",
        granularity=900,
        start="20250701000000",
        end="20250710000000"
    )

    df = pd.DataFrame(data)

    # Plot RSI and signals
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    ax1.plot(df['timestamp'], df['rsi'])
    ax1.axhline(y=30, color='g', linestyle='--')
    ax1.axhline(y=70, color='r', linestyle='--')
    ax1.set_title('RSI')

    ax2.scatter(df['timestamp'], df['signal'], c=df['signal'], cmap='RdYlGn')
    ax2.set_title('Signals')

    plt.savefig('analysis.png')

analyze_indicator()
```

### Phase 4: Tier 2 Composite

Build portfolio manager (see Chapter 8 for template).

### Phase 5: Production Deployment

```bash
# Final checks
1. Replay consistency test passes
2. Full backtest performance acceptable  
3. Code review completed
4. Risk parameters validated
5. Monitoring set up

# Deploy
Deploy to production server
Enable monitoring and alerts
Start with small capital
Monitor closely for first week
Gradually increase allocation
```

## Summary

Complete workflow:
1. Design Tier 1 indicator
2. Implement with all best practices
3. Test (quick, replay, full backtest)
4. Visualize and optimize
5. Build Tier 2 composite
6. Build Tier 3 executor (if live trading)
7. Validate thoroughly
8. Deploy to production
9. Monitor and maintain

**Congratulations!** You've learned the complete Wolverine framework for building production trading systems.

---

**Previous:** [11 - Fine-tune and Iterate](11-fine-tune-and-iterate.md) | **Return to:** [01 - Overview](01-overview.md)