# Iron Ore Indicator System - Requirements

**Document**: REQUIREMENTS.md
**Version**: 1.0
**Last Updated**: 2025-11-04
**Purpose**: Complete specification for building iron ore tier-1 indicator with backtesting and P&L visualization
**Target**: AI agents and developers
**Container**: IronOreIndicator/
**Framework**: WOS (Wolverine Operating System)

---

## 1. Project Overview

### 1.1 Objective

Build a production-ready tier-1 indicator for iron ore futures that:
1. Generates buy/sell signals using technical analysis
2. Backtests against 1 year of historical data (2024-01-01 to 2024-12-31)
3. Evaluates P&L performance on recent 3 months (2024-10-01 to 2024-12-31)
4. Visualizes results in interactive Jupyter notebook with P&L curves

### 1.2 Deliverables

**Location**: `IronOreIndicator/`

| File | Type | Purpose |
|------|------|---------|
| `IronOreIndicator.py` | Python | Main indicator implementation |
| `uin.json` | JSON | Input configuration |
| `uout.json` | JSON | Output configuration |
| `test_resuming_mode.py` | Python | Replay consistency test |
| `analysis.ipynb` | Jupyter | P&L visualization and metrics |
| `.env` | Environment | Server credentials (SVR_HOST, SVR_TOKEN) |

### 1.3 Success Criteria

- [ ] Replay consistency test passes
- [ ] 1-year backtest completes successfully (~24,000 bars)
- [ ] Memory usage bounded (O(1) growth)
- [ ] P&L curve generated from real svr3 data
- [ ] Performance metrics: Sharpe > 1.0, Win Rate > 50%, Max DD < 20%

---

## 2. Data Specifications

### 2.1 Input Data

**Source**: SampleQuote (global namespace)
**Market**: DCE (Dalian Commodity Exchange)
**Instrument**: i<00> (iron ore logical contract)
**Granularity**: 900 seconds (15 minutes)
**Fields**: open, high, low, close, volume, turnover

**Time Ranges**:
- **Backtest**: 2024-01-01 00:00:00 to 2024-12-31 23:59:59 (1 year)
- **P&L Evaluation**: 2024-10-01 00:00:00 to 2024-12-31 23:59:59 (3 months)

### 2.2 Output Data

**Namespace**: private
**Fields**:

| Field | Type | Precision | Description |
|-------|------|-----------|-------------|
| `_preserved_field` | int64 | 0 | Framework-managed timestamp |
| `bar_index` | integer | 0 | Bar counter |
| `ema_fast` | double | 6 | Fast EMA value |
| `ema_slow` | double | 6 | Slow EMA value |
| `rsi` | double | 6 | RSI indicator (0-100) |
| `volume_ema` | double | 6 | Volume moving average |
| `signal` | integer | 0 | Trade signal: -1 (sell), 0 (neutral), 1 (buy) |
| `confidence` | double | 6 | Signal confidence [0.0, 1.0] |

---

## 3. Indicator Algorithm

### 3.1 Technical Analysis Components

**Strategy**: Multi-indicator confirmation system

**Components**:
1. **EMA Crossover**: Trend detection
   - Fast EMA: 10-period
   - Slow EMA: 20-period

2. **RSI**: Mean reversion detection
   - Period: 14
   - Oversold: < 30
   - Overbought: > 70

3. **Volume Confirmation**: Liquidity filter
   - Volume EMA: 20-period
   - Threshold: 1.2× average

### 3.2 Signal Generation Logic

**Pseudocode**:
```
For each bar:
  1. Update EMAs (fast, slow, volume) using online algorithm
  2. Update RSI using online gain/loss EMAs

  3. Generate signal:
     BUY if:
       - EMA_fast > EMA_slow (uptrend)
       - RSI < 30 (oversold)
       - volume > 1.2 × volume_EMA (high liquidity)
       - confidence = (30 - RSI) / 30

     SELL if:
       - EMA_fast < EMA_slow (downtrend)
       - RSI > 70 (overbought)
       - volume > 1.2 × volume_EMA (high liquidity)
       - confidence = (RSI - 70) / 30

     NEUTRAL otherwise:
       - signal = 0, confidence = 0.0
```

### 3.3 Implementation Requirements

**Stateless Design** (WOS Doctrine):
- Use online algorithms (EMA updates: `new = α × value + (1 - α) × old`)
- No unbounded memory (no growing lists)
- All state in instance variables (automatically persisted)
- Deterministic (no random, time-based, or external state)

**Reference**: wos/05-stateless-design.md for online algorithm patterns

**Online Algorithm Formulas**:
```python
# EMA update
ema = alpha × current_price + (1 - alpha) × ema_previous
where alpha = 2 / (period + 1)

# RSI calculation (online)
change = close - prev_close
gain = max(change, 0)
loss = max(-change, 0)
gain_ema = alpha × gain + (1 - alpha) × gain_ema_previous
loss_ema = alpha × loss + (1 - alpha) × loss_ema_previous
rs = gain_ema / loss_ema
rsi = 100 - (100 / (1 + rs))
```

---

## 4. Implementation Patterns

### 4.1 Module Structure

**Required Components** (from WOS framework):

1. **Framework Globals**:
```python
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
```

2. **SampleQuote Parser** (sv_object):
```python
class SampleQuote(pcts3.sv_object):
    def __init__(self):
        self.meta_name = "SampleQuote"
        self.namespace = pc.namespace_global
        self.revision = (1 << 32) - 1
        # Fields: open, high, low, close, volume, turnover
```

3. **IronOreIndicator Class** (sv_object):
```python
class IronOreIndicator(pcts3.sv_object):
    def __init__(self):
        # Metadata (constants)
        self.meta_name = "IronOreIndicator"
        self.namespace = pc.namespace_private
        self.granularity = 900
        self.market = b'DCE'
        self.code = b'i<00>'

        # State variables (auto-persisted)
        # ... (see Section 3.3)
```

4. **Framework Callbacks**:
```python
async def on_init():
    # Initialize with metas and imports
async def on_bar(bar):
    # Process bars, ALWAYS return list
# ... (other callbacks)
```

**Reference**: wos/03-programming-basics-and-cli.md (lines 91-176) for complete module structure

### 4.2 Cycle Boundary Pattern

**Critical Pattern**:
```python
def on_bar(self, bar):
    # 1. Filter for DCE/i<00>
    if not (bar.get_market() == self.market and
            bar.get_stock_code().endswith(b'<00>')):
        return []

    # 2. Parse SampleQuote data
    # ...

    tm = bar.get_time_tag()

    # 3. Initialize timetag
    if self.timetag is None:
        self.timetag = tm

    # 4. Handle cycle boundary
    if self.timetag < tm:
        self._on_cycle_pass()  # Process end of cycle

        results = []
        if self.bar_index > 0 and self.initialized:
            results.append(self.copy_to_sv())

        self.timetag = tm
        self.bar_index += 1
        return results  # ALWAYS list

    return []
```

**Reference**: wos/07-tier1-indicator.md (lines 419-476) for cycle boundary handling

---

## 5. Testing Requirements

### 5.1 Replay Consistency Test (MANDATORY)

**Purpose**: Verify stateless design

**Test File**: `test_resuming_mode.py`

**Execution**:
```bash
python test_resuming_mode.py \
    --start 20241001000000 \
    --end 20241031000000 \
    --midpoint 20241015120000
```

**Pass Criteria**: Outputs from continuous run MUST match outputs from stop-resume run (bit-for-bit identical)

**Reference**: wos/06-backtest.md (lines 74-100); templates/test_resuming_mode.py.template

### 5.2 Full Backtest (1 Year)

**Command**:
```bash
python /home/wolverine/bin/running/calculator3_test.py \
    --testcase ${PWD}/ \
    --algoname IronOreIndicator \
    --sourcefile IronOreIndicator.py \
    --start 20240101000000 \
    --end 20241231235959 \
    --granularity 900 \
    --tm wss://${SVR_HOST}:4433/tm \
    --tm-master ${SVR_HOST}:6102 \
    --rails https://${SVR_HOST}:4433/private-api/ \
    --token ${SVR_TOKEN} \
    --category 1 \
    --is-managed 1 \
    --restore-length 864000000 \
    --multiproc 1
```

**Expected Results**:
- ~24,000 bars processed (96 bars/day × 250 trading days)
- Signals generated: 500-2000 (varies by market conditions)
- Execution time: 5-15 minutes
- Memory: Constant (no growth)

---

## 6. P&L Calculation and Visualization

### 6.1 Data Fetching from SVR3

**Purpose**: Retrieve indicator outputs for P&L analysis

**Server Configuration**:
```python
SVR_HOST = os.getenv("SVR_HOST")  # e.g., "10.99.100.116"
SVR_TOKEN = os.getenv("SVR_TOKEN")
RAILS_URL = f"https://{SVR_HOST}:4433/private-api/"
WS_URL = f"wss://{SVR_HOST}:4433/tm"
TM_MASTER = (SVR_HOST, 6102)
```

**Connection Sequence** (CRITICAL ORDER):
```python
import svr3
import asyncio

async def fetch_data():
    # 1. Create reader
    reader = svr3.sv_reader(
        20241001000000,  # start (int)
        20241231235959,  # end (int)
        "IronOreIndicator",  # algoname
        900,  # granularity
        "private",  # namespace
        "symbol",  # work_mode
        ["DCE"],  # markets
        ["i<00>"],  # codes
        False,  # persistent
        RAILS_URL, WS_URL, "", "", TM_MASTER
    )

    # 2. Set token
    reader.token = SVR_TOKEN

    # 3. Connection sequence (DO NOT CHANGE ORDER)
    await reader.login()
    await reader.connect()
    reader.ws_task = asyncio.create_task(reader.ws_loop())
    await reader.shakehand()

    # 4. Fetch data
    ret = await reader.save_by_symbol()
    data = ret[1][1]

    return pd.DataFrame(data)
```

**Reference**: wos/10-visualization.md (lines 76-192)

### 6.2 P&L Calculation

**Backtesting Logic**:

```
Input: DataFrame with columns [timestamp, close, signal, confidence]
Output: DataFrame with columns [entry_time, exit_time, direction, pnl, ...]

Algorithm:
1. Initialize: position = None, trades = []

2. For each bar:
   a. If position is None AND signal ≠ 0 AND confidence > 0.6:
      - Open position: entry_price = close, direction = signal

   b. If position is not None:
      - If signal reverses OR confidence < 0.3:
        - Close position: exit_price = close
        - Calculate PnL: (exit_price - entry_price) × direction
        - Append to trades list
        - Set position = None

3. Calculate cumulative P&L: cumsum(trades['pnl'])

4. Calculate metrics:
   - Total PnL = sum(trades['pnl'])
   - Win Rate = wins / total_trades
   - Sharpe = mean(returns) / std(returns) × sqrt(252)
   - Max Drawdown = max(peak - trough) / peak
```

**Position Sizing**: 1 contract per signal (can be scaled by confidence)

### 6.3 Jupyter Notebook Visualization

**File**: `analysis.ipynb`

**Notebook Cells**:

**Cell 1: Imports and Setup**
```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import asyncio
import svr3
from datetime import datetime

%matplotlib inline
sns.set_style("darkgrid")

# Load environment
from dotenv import load_dotenv
load_dotenv()

SVR_HOST = os.getenv("SVR_HOST")
SVR_TOKEN = os.getenv("SVR_TOKEN")
```

**Cell 2: Data Fetching**
```python
# Fetch indicator data from svr3 server
# (Use connection sequence from Section 6.1)

df = fetch_data()
print(f"Loaded {len(df)} bars")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
df.head()
```

**Cell 3: P&L Calculation**
```python
# Calculate trades and P&L
# (Use algorithm from Section 6.2)

trades = calculate_trades(df)
df['cumulative_pnl'] = calculate_cumulative_pnl(trades)
df['drawdown'] = calculate_drawdown(df['cumulative_pnl'])

print(f"Total trades: {len(trades)}")
print(f"Total P&L: ¥{trades['pnl'].sum():,.2f}")
```

**Cell 4: Performance Metrics**
```python
# Calculate and display metrics

metrics = {
    'total_trades': len(trades),
    'win_rate': (trades['pnl'] > 0).sum() / len(trades),
    'total_pnl': trades['pnl'].sum(),
    'avg_win': trades[trades['pnl'] > 0]['pnl'].mean(),
    'avg_loss': trades[trades['pnl'] < 0]['pnl'].mean(),
    'profit_factor': trades[trades['pnl'] > 0]['pnl'].sum() / abs(trades[trades['pnl'] < 0]['pnl'].sum()),
    'sharpe_ratio': calculate_sharpe(trades),
    'max_drawdown': df['drawdown'].min()
}

# Display as formatted table
pd.DataFrame([metrics]).T.rename(columns={0: 'Value'})
```

**Cell 5: P&L Curve Visualization**
```python
# Create 4-panel visualization

fig, axes = plt.subplots(4, 1, figsize=(15, 12))

# Panel 1: Cumulative P&L
axes[0].plot(df['timestamp'], df['cumulative_pnl'], linewidth=2)
axes[0].set_title('Cumulative P&L', fontsize=14, fontweight='bold')
axes[0].set_ylabel('P&L (CNY)')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Panel 2: Price and Signals
buy_signals = df[df['signal'] == 1]
sell_signals = df[df['signal'] == -1]
axes[1].plot(df['timestamp'], df['close'], alpha=0.6, color='gray', label='Price')
axes[1].scatter(buy_signals['timestamp'], buy_signals['close'],
               color='green', marker='^', s=100, label='Buy', alpha=0.8)
axes[1].scatter(sell_signals['timestamp'], sell_signals['close'],
               color='red', marker='v', s=100, label='Sell', alpha=0.8)
axes[1].set_title('Price and Trading Signals', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Price (CNY)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Panel 3: Drawdown
axes[2].fill_between(df['timestamp'], 0, df['drawdown']*100,
                     color='red', alpha=0.3)
axes[2].set_title('Drawdown', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Drawdown (%)')
axes[2].grid(True, alpha=0.3)

# Panel 4: Trade P&L Distribution
axes[3].hist(trades['pnl'], bins=50, edgecolor='black', alpha=0.7)
axes[3].axvline(x=0, color='black', linestyle='--', linewidth=2)
axes[3].set_title('Individual Trade P&L Distribution', fontsize=14, fontweight='bold')
axes[3].set_xlabel('P&L (CNY)')
axes[3].set_ylabel('Frequency')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pnl_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Cell 6: Signal Analysis**
```python
# Analyze signal characteristics

signal_analysis = df[df['signal'] != 0].groupby('signal').agg({
    'confidence': ['mean', 'std', 'min', 'max'],
    'rsi': 'mean',
    'ema_fast': 'mean',
    'signal': 'count'
}).round(4)

signal_analysis
```

**Cell 7: Monthly Performance**
```python
# Break down P&L by month

df['month'] = pd.to_datetime(df['timestamp']).dt.to_period('M')
monthly_pnl = df.groupby('month')['cumulative_pnl'].last().diff()

fig, ax = plt.subplots(figsize=(12, 6))
monthly_pnl.plot(kind='bar', ax=ax, color=['green' if x > 0 else 'red' for x in monthly_pnl])
ax.set_title('Monthly P&L', fontsize=14, fontweight='bold')
ax.set_ylabel('P&L (CNY)')
ax.set_xlabel('Month')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

**Required Outputs**:
1. Interactive plots displayed in notebook
2. `pnl_analysis.png` saved to disk
3. Metrics table showing all performance statistics
4. Monthly breakdown of returns

---

## 7. Acceptance Criteria

### 7.1 Functional Requirements

- [ ] **Indicator runs successfully**: Processes 1 year of data (2024-01-01 to 2024-12-31)
- [ ] **Signals generated**: At least 100 signals over the year (both buy and sell)
- [ ] **Replay consistency**: Test passes with identical outputs
- [ ] **Memory bounded**: No memory growth during backtest (O(1) complexity)
- [ ] **Data fetching**: Successfully retrieves data from svr3 server

### 7.2 Performance Requirements

- [ ] **Sharpe Ratio**: > 1.0 (risk-adjusted returns)
- [ ] **Win Rate**: > 50% (more winning trades than losing)
- [ ] **Profit Factor**: > 1.5 (total wins / total losses)
- [ ] **Max Drawdown**: < 20% (maximum peak-to-trough decline)
- [ ] **Total P&L**: Positive over 3-month evaluation period

### 7.3 Code Quality Requirements

- [ ] **WOS Compliance**: Follows all framework patterns (stateless, sv_object, cycle boundaries)
- [ ] **Doctrines Applied**: All 4 WOS doctrines followed (see wos/01-overview.md, lines 443-465)
- [ ] **No Code Smells**: No unbounded memory, no global state, no non-determinism
- [ ] **Configuration Valid**: uin.json and uout.json pass framework validation
- [ ] **Documentation**: README.md explains setup, CLAUDE.md has development notes

### 7.4 Visualization Requirements

- [ ] **Jupyter Notebook**: analysis.ipynb runs without errors
- [ ] **P&L Curve**: 4-panel visualization generated
- [ ] **Metrics Table**: All performance metrics calculated and displayed
- [ ] **Real Data**: Uses actual svr3 server data (not mock/simulated)
- [ ] **Interactive**: Notebook allows parameter adjustment and re-run

---

## 8. Development Workflow

### 8.1 Setup Phase (30 minutes)

```bash
# 1. Navigate to container
cd IronOreIndicator/

# 2. Verify files exist
ls -la  # Should see: uin.json, uout.json, IronOreIndicator.py, etc.

# 3. Create .env file
cat > .env << EOF
SVR_HOST=10.99.100.116
SVR_TOKEN=your_token_here
EOF

# 4. Verify WOS documentation access
ls -la wos/  # Should see symlink to ../wos/
```

### 8.2 Implementation Phase (2-3 hours)

1. **Review existing code**: IronOreIndicator/IronOreIndicator.py
2. **Verify configuration**: uin.json, uout.json
3. **Implement indicator logic**: Follow Section 3 (Algorithm) and Section 4 (Patterns)
4. **Add logging**: Use `logger.info()` for signal generation

### 8.3 Testing Phase (1 hour)

```bash
# 1. Quick test (7 days)
python /home/wolverine/bin/running/calculator3_test.py \
    --testcase ${PWD}/ --algoname IronOreIndicator \
    --sourcefile IronOreIndicator.py \
    --start 20241025000000 --end 20241101000000 \
    --granularity 900 --category 1 \
    --tm wss://${SVR_HOST}:4433/tm \
    --tm-master ${SVR_HOST}:6102 \
    --rails https://${SVR_HOST}:4433/private-api/ \
    --token ${SVR_TOKEN} \
    --is-managed 1 --restore-length 864000000 --multiproc 1

# 2. Replay consistency (MANDATORY)
python test_resuming_mode.py

# 3. Full backtest (1 year)
# (Use command from Section 5.2, adjust dates to 20240101-20241231)
```

### 8.4 Visualization Phase (1 hour)

```bash
# 1. Create Jupyter notebook
jupyter notebook analysis.ipynb

# 2. Execute all cells sequentially
# 3. Verify P&L curve generated
# 4. Export notebook to HTML for sharing
jupyter nbconvert --to html analysis.ipynb
```

### 8.5 Validation Phase (30 minutes)

```bash
# 1. Check all acceptance criteria (Section 7)
# 2. Review performance metrics
# 3. Verify replay consistency test passed
# 4. Confirm P&L curve shows reasonable results
# 5. Document any issues in CLAUDE.md
```

**Total Time**: ~4-6 hours

---

## 9. Reference Materials

### 9.1 WOS Framework Documentation

**Location**: `wos/` directory (symlinked from IronOreIndicator/)

**Key Chapters**:
- `01-overview.md` - Framework architecture and doctrines
- `02-uin-and-uout.md` - Configuration file specifications
- `03-programming-basics-and-cli.md` - Module structure and callbacks
- `04-structvalue-and-sv_object.md` - Data serialization patterns
- `05-stateless-design.md` - Online algorithms and replay consistency
- `06-backtest.md` - Testing procedures
- `07-tier1-indicator.md` - Tier-1 indicator patterns
- `10-visualization.md` - Data fetching and plotting

### 9.2 Templates

**Location**: `../templates/`

- `indicator.py.template` - Base indicator structure
- `indicator_viz.py.template` - Visualization script template
- `test_resuming_mode.py.template` - Replay test template
- `launch.json.template` - VS Code debug configuration

### 9.3 Existing Implementation

**Location**: `IronOreIndicator/`

- `IronOreIndicator.py` - Working indicator (may need updates)
- `ironoreindicator_viz.py` - Python visualization script (convert to notebook)
- `uin.json`, `uout.json` - Configuration files
- `CLAUDE.md` - Development notes

### 9.4 External Documentation

- **svr3 API**: Connection patterns in wos/10-visualization.md
- **Jupyter Notebooks**: https://jupyter.org/documentation
- **pandas**: https://pandas.pydata.org/docs/
- **matplotlib**: https://matplotlib.org/stable/contents.html

---

## 10. Troubleshooting

### 10.1 Common Issues

**Issue**: Replay consistency test fails

**Solution**: Check for non-deterministic code (random, time-based), verify all state variables persisted, ensure from_sv()/copy_to_sv() symmetry

**Issue**: No signals generated

**Solution**: Check threshold values too strict (RSI < 30, RSI > 70), verify confidence calculation, add debug logging

**Issue**: Memory grows during backtest

**Solution**: Replace unbounded lists with online algorithms (EMA) or bounded deques (maxlen parameter)

**Issue**: svr3 connection fails

**Solution**: Verify SVR_TOKEN in .env, check connection sequence order (Section 6.1), confirm server accessible

**Issue**: Jupyter notebook won't load data

**Solution**: Ensure nest_asyncio installed (`pip install nest-asyncio`), verify async fetch function runs correctly

### 10.2 Debugging Commands

```bash
# Check configuration validity
python -c "import json; print(json.load(open('uin.json')))"

# Verify environment variables
echo $SVR_HOST
echo $SVR_TOKEN

# Test svr3 connectivity
python -c "import svr3; print(svr3.__version__)"

# Check indicator imports
python -c "import IronOreIndicator; print('OK')"
```

---

## Summary

**This document specifies**:
1. Complete tier-1 indicator for iron ore (DCE/i<00>)
2. 1-year backtest (2024-01-01 to 2024-12-31)
3. 3-month P&L evaluation (2024-10-01 to 2024-12-31)
4. Jupyter notebook visualization with real svr3 data
5. All deliverables in IronOreIndicator/ container

**Expected Timeline**: 4-6 hours total

**Framework**: Follows WOS patterns (referenced, not duplicated)

**Target Users**: AI agents and developers

**Success Criteria**: Section 7 (all checkboxes must pass)

---

**Next Steps for AI Agent**:
1. Read this requirements document
2. Review referenced WOS chapters for implementation patterns
3. Implement/update IronOreIndicator.py following Section 3-4
4. Run tests following Section 5
5. Create analysis.ipynb following Section 6.3
6. Validate against Section 7 acceptance criteria
