# Chapter 05: Stateless Design and Replay Consistency

**Learning objectives:**
- Understand why replay consistency is critical
- Apply stateless design principles
- Implement reconciliation pattern
- Prevent common consistency violations

**Previous:** [04 - StructValue and sv_object](04-structvalue-and-sv_object.md) | **Next:** [06 - Backtest and Testing](06-backtest.md)

---

## The Problem: Why Stateless Design Matters

### Production Reality

**System behavior**: Strategies run continuously but can restart at any time:
- Server crashes
- Code updates
- Resource management
- Market session boundaries

**Replay consistency requirement**:
```
Run(0→100) must equal Run(0→50) + Resume(50→100)
```

**Why it matters**:
- **Backtesting validity**: Historical results must match live behavior
- **Recovery correctness**: Restart at bar 1000 must continue as if never stopped
- **Performance attribution**: Track why strategy made specific decisions
- **Regulatory compliance**: Reproducible decision trail

### The Root Cause: Unbounded State

```python
# ❌ Violates replay consistency
class BadStrategy:
    def __init__(self):
        self.price_history = []  # Grows unbounded

    def on_bar(self, bar):
        self.price_history.append(bar.close)
        avg = sum(self.price_history) / len(self.price_history)
```

**Three failures**:

| Issue | Consequence |
|-------|-------------|
| **Memory growth** | Crashes after weeks of runtime |
| **State size** | Serialization takes minutes, blocking execution |
| **Resume inconsistency** | After restart: empty history → different calculations |

**Production failure scenario**:
1. Run from bar 0→1000, history has 1000 prices, avg = mean(1000 prices)
2. System restarts at bar 1000
3. Resume from bar 1000→1100, history starts empty
4. Bar 1001: avg = mean(1 price) ≠ mean(1001 prices)
5. **Different signals, different trades, different P&L**

---

## Stateless Design Principles

### Principle 1: Bounded Memory

**Doctrine**: Every data structure must have fixed maximum size.

**Implementation**: Use online algorithms or bounded collections (deque with maxlen).

**Trade-off**: O(1) memory vs complete history.

**Verification**: `sizeof(state) = constant` regardless of runtime duration.

### Principle 2: Deterministic Computation

**Doctrine**: Same input sequence → Same output sequence, always.

**Violations**:

| Source | Example | Fix |
|--------|---------|-----|
| Random | `random.random()` | Use `(bar_index * seed) % modulus` |
| Time | `datetime.now()` | Use `bar.time_tag` |
| External state | Global variables | Instance variables only |
| Floating order | `dict` iteration (pre-3.7) | Ordered structures |

### Principle 3: Complete State Persistence

**Doctrine**: All state needed for next calculation must be in `sv_object` fields.

**Rule**: If variable affects future output, it must be instance variable (persisted).

**Anti-pattern**: Local variables carrying state across bars.

### Principle 4: Data Leakage Prevention

**Doctrine**: Cycle t calculations must NOT use data arriving in cycle t+1.

**Pattern**:
```python
def on_bar(self, bar):
    # Timetag advancement = cycle boundary
    if self.timetag < bar.timetag:
        # 1. Calculate with OLD data (from previous bars)
        self._on_cycle_pass()

        # 2. Update timetag
        self.timetag = bar.timetag

    # 3. Import NEW data (for NEXT cycle)
    if bar.meta_id == self.quote.meta_id:
        self.quote.from_sv(bar)  # Available NEXT cycle
```

**Why**: Live trading receives bar for time t+1 AFTER decisions for time t must be made. Backtest must match this constraint.

---

## Replay Consistency

### Definition

**Replay consistency**: Strategy produces identical state and output regardless of stop/resume points.

**Formal requirement**:
```
For any bars B[0...n] and split point k:
  process(B[0...n]) == resume(process(B[0...k]), B[k+1...n])
```

**Test**: `test_resuming_mode.py` runs strategy twice:
1. Continuously: bars 0→100
2. Split: bars 0→50, save state, resume 50→100

**Requirement**: Output must be identical (within floating-point tolerance).

### Why Replay Fails

**Common causes**:

| Cause | Why It Breaks | Detection |
|-------|---------------|-----------|
| **Unbounded state** | History lost on resume | Different calculations |
| **Non-determinism** | Random/time-based logic | Outputs diverge |
| **Incomplete persistence** | Missing fields in sv_object | State not restored |
| **Data leakage** | Using t+1 data in cycle t | Backtest ≠ live |
| **Wrong rebuilding** | Using persisted counter | Rebuilding skipped on resume |

---

## The Reconciliation Pattern

### Purpose

**Goal**: Verify that recalculated state matches previously saved state.

**When**: After warm-up period (rebuilding phase) completes.

**Pattern**:
```
For each bar after rebuilding:
  1. Load saved state from database: S_saved
  2. Calculate state from scratch: S_calculated
  3. Assert: S_calculated == S_saved (within tolerance)
```

### The Rebuilding Phase

**Problem**: Indicators need warm-up period (e.g., 50 bars for EMA calculation).

**Rebuilding**: On resume, restore persisted state, then verify after warm-up.

**Critical distinction**:

| Counter | Behavior | Use Case |
|---------|----------|----------|
| `self.bar_index` | **Persisted**, carries across restarts | Total bars processed |
| `self.bars_since_start` | **NOT persisted**, resets to 0 each run | Rebuilding detection |

**Why it matters**:
```python
# ❌ WRONG - Uses persisted counter
def _rebuilding_finished(self):
    return self.bar_index >= 50  # Immediately true if resuming at bar 1000

# ✅ CORRECT - Uses non-persisted counter
def _rebuilding_finished(self):
    return self.bars_since_start >= 50  # Counts from 0 each run
```

### Five-Step Reconciliation Pattern

| Step | Method | Purpose |
|------|--------|---------|
| **1. Cache** | `from_sv(sv)` | Store incoming saved state: `self.latest_sv = sv` |
| **2. Restore** | `_from_sv()` | During rebuilding: `super().from_sv(self.latest_sv)` |
| **3. Compare** | `_reconcile_state()` | After rebuilding: `assert self._equal(saved_state)` |
| **4. Load** | `_load_from_sv(sv)` | Create temp object with saved state for comparison |
| **5. Verify** | `_equal(other)` | Compare fields with floating-point tolerance |

**Critical detail in Step 4**:
```python
# ❌ WRONG - Triggers custom from_sv logic
def _load_from_sv(self, sv):
    temp = self.__class__()
    temp.from_sv(sv)  # Calls overridden from_sv!
    return temp

# ✅ CORRECT - Bypasses custom logic
def _load_from_sv(self, sv):
    temp = self.__class__()
    # Copy metadata for deserialization
    temp.market = self.market
    temp.code = self.code
    temp.meta_id = self.meta_id
    temp.granularity = self.granularity
    temp.namespace = self.namespace
    # Call PARENT's from_sv directly
    super(self.__class__, temp).from_sv(sv)
    return temp
```

**Why**: `temp.from_sv(sv)` calls your overridden method with caching/rebuilding logic, NOT state loading. Must call parent directly.

---

## Critical Rules

### 🚨 Rule 1: Data Leakage Prevention

**Doctrine**: Update imports AFTER `_on_cycle_pass()`, never before.

```python
# ✅ CORRECT
if self.timetag < bar.timetag:
    self._on_cycle_pass()  # Uses OLD data
    self.timetag = bar.timetag

# Now import NEW data (for next cycle)
self.quote.from_sv(bar)
```

**Verification**: In backtest, bar for time t+1 arrives AFTER decisions for time t. Live trading has same constraint.

### 🚨 Rule 2: Rebuilding Detection

**Doctrine**: Use non-persisted counter for warm-up tracking.

```python
def __init__(self):
    self.bar_index = 0  # Persisted
    self.bars_since_start = 0  # NOT persisted

def _rebuilding_finished(self):
    return self.bars_since_start >= WARMUP  # Resets to 0 on each run

def on_bar(self, bar):
    self.bar_index += 1  # Total bars
    self.bars_since_start += 1  # Bars this run
```

### 🚨 Rule 3: Cycle Pass Location

**Doctrine**: `_on_cycle_pass()` triggered by timetag advancement, independent of bar types.

```python
# ❌ WRONG - Inside bar type logic
if bar.meta_id == quote_meta:
    if self.timetag < bar.timetag:
        self._on_cycle_pass()  # Only called for quote bars!

# ✅ CORRECT - Outside bar type logic
if self.timetag < bar.timetag:
    if self._all_data_ready():  # Check readiness
        self._on_cycle_pass()  # Called regardless of bar type
    self.timetag = bar.timetag

# Then route bars
if bar.meta_id == quote_meta:
    self.quote.from_sv(bar)
elif bar.meta_id == zample_meta:
    self.zample.from_sv(bar)
```

**Why**: Multi-source indicators need all data before cycle pass. Tying to specific bar type causes missed cycles.

### 🚨 Rule 4: Bounded Collections

**Doctrine**: Use `deque(maxlen=N)` for fixed-size collections.

**Critical**: `maxlen` is NOT preserved in serialization.

```python
def __init__(self):
    self.prices = deque(maxlen=100)

def from_sv(self, sv):
    super().from_sv(sv)
    # Restore maxlen (lost during serialization)
    if not isinstance(self.prices, deque):
        self.prices = deque(self.prices, maxlen=100)
```

### 🚨 Rule 5: Float Comparison Tolerance

**Doctrine**: Use `math.isclose()` with appropriate tolerances.

```python
def _equal(self, other):
    # High precision for indicators
    if not math.isclose(self.ema, other.ema, abs_tol=1e-6, rel_tol=1e-5):
        return False

    # Exact for integers
    if self.signal != other.signal:
        return False

    return True
```

| Data Type | Tolerance | Rationale |
|-----------|-----------|-----------|
| Indicators (EMA, volatility) | `abs=1e-6, rel=1e-5` | Accumulation of floating-point errors |
| Prices | `abs=1e-3, rel=1e-4` | Tick size precision |
| Integers (signals, regimes) | Exact `==` | No tolerance |

---

## Common Violations

| Violation | Symptom | Fix |
|-----------|---------|-----|
| **Unbounded growth** | Memory crash after weeks | Use `deque(maxlen=N)` or online algorithms |
| **Data leakage** | Backtest better than live | Update imports AFTER cycle pass |
| **Wrong rebuilding check** | Reconciliation fails on resume | Use `bars_since_start`, NOT `bar_index` |
| **Wrong `_load_from_sv`** | Assertion failures | `super(self.__class__, temp).from_sv(sv)` |
| **Cycle pass in bar block** | Missed cycles, incomplete data | Check timetag outside bar routing |
| **Missing `maxlen` restore** | Unbounded after resume | Re-create deque in `from_sv()` |
| **Non-determinism** | Different outputs same inputs | No random/time-based logic |
| **External state** | Global variables | All state in instance |

---

## Summary

**Core problem**: Strategies must produce identical results regardless of when stopped/resumed.

**Why it's hard**:
- Unbounded state (memory crashes, incomplete restoration)
- Non-deterministic computation (time, random)
- Data leakage (using future data)
- Incomplete persistence (missing state)

**Solution principles**:
1. **Bounded memory**: Fixed-size structures, online algorithms
2. **Deterministic**: No random, no external state, no time-based logic
3. **Complete persistence**: All state in `sv_object` fields
4. **Data leakage prevention**: Import AFTER cycle pass

**Reconciliation**:
- Verify recalculated state matches saved state
- Requires warm-up period (rebuilding phase)
- Use non-persisted counter for rebuilding detection
- Compare with floating-point tolerance

**Critical rules**:
- 🚨 Update imports AFTER `_on_cycle_pass()`
- 🚨 Use `bars_since_start` (NOT persisted) for rebuilding
- 🚨 In `_load_from_sv()`: `super(self.__class__, temp).from_sv(sv)`
- 🚨 Place `_on_cycle_pass()` outside bar routing logic
- 🚨 Restore `deque` maxlen in `from_sv()`

**Verification**: `test_resuming_mode.py` runs split-resume test; you implement reconciliation assertions.

**Reference**: See templates for complete reconciliation implementation.

---

**Previous:** [04 - StructValue and sv_object](04-structvalue-and-sv_object.md) | **Next:** [06 - Backtest and Testing](06-backtest.md)
