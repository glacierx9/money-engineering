# Chapter 05: Stateless Design and Replay Consistency

**Learning objectives:**

- Understand frame-based calculation model
- Recognize timetag as the driver of computation
- Apply stateless design principles
- Implement rebuilding and reconciliation
- Prevent common consistency violations

**Previous:** [04 - StructValue and sv_object](04-structvalue-and-sv_object.md) | **Next:** [06 - Backtest and Testing](06-backtest.md)

---

## Fundamentals: Frame-Based Calculation

### What is a Frame?

**Frame**: A single StructValue representing state at specific time T for granularity G.

**Time progression**: As time moves forward, frames become historical and new frames are generated.

**Key insight**: `on_bar()` receives data, but **timetag change** triggers calculation.

### The Driver: Timetag

**Critical distinction**:

| Event | Role | Action |
|-------|------|--------|
| `on_bar()` arrival | **Data collection** | Import and cache data for NEXT frame |
| `timetag` change | **Calculation trigger** | Calculate and output frame for PREVIOUS period |

**Root cause of errors**: Mistakenly calculating frame on every `on_bar()` instead of on timetag advancement.

**Correct pattern**:

```python
def on_bar(self, bar):
    # Timetag advancement = time moves forward
    if self.timetag < bar.timetag:
        # Calculate frame for PREVIOUS period (using OLD data)
        self._on_cycle_pass()

        # Update to new time
        self.timetag = bar.timetag

    # Import data for NEXT calculation
    self.quote.from_sv(bar)
```

### Timetag Definition

**Bar time range**: `[x, y)` where x = start time, y = end time

**Timetag assignment**: `self.timetag = y` (end time)

**Critical requirement**: Only update `self.timetag` from bars matching `self.granularity`.

**Why**: Different granularities have different time boundaries. Using mismatched granularity corrupts time tracking.

```python
# ✅ CORRECT - Granularity check
if bar.granularity == self.granularity and self.timetag < bar.timetag:
    self._on_cycle_pass()
    self.timetag = bar.timetag

# ❌ WRONG - No granularity check
if self.timetag < bar.timetag:  # May use wrong granularity!
    self._on_cycle_pass()
```

---

## The Problem: Why Stateless Design Matters

### Production Reality

**System behavior**: Strategies run continuously but restart frequently:

- Server crashes and restarts
- Code updates and redeployments
- Resource management (memory cleanup)
- Market session boundaries

**Replay consistency requirement**:

```
Run(0→100) must equal Run(0→50) + Resume(50→100)
```

**Why it matters**:

- **Backtesting validity**: Historical results must match live behavior
- **Recovery correctness**: Restart at bar 1000 must continue as if never stopped
- **Performance attribution**: Reproduce decisions for analysis
- **Regulatory compliance**: Auditable decision trail

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

**Pattern**: Import data AFTER `_on_cycle_pass()`, never before.

**Why**: Live trading receives bar for time t+1 AFTER decisions for time t. Backtest must match this constraint.

---

## The Rebuilding System

### System Design

**Intrinsic behavior**: Every backtest run rewinds `rebuild_length` seconds of historical data to replay.

**Purpose**: Warm-up period to reconstruct state before meaningful calculations.

**When rebuilding occurs**: Every time indicator starts (fresh start or resume).

**Rebuilding period**: System pushes historical data covering `rebuild_length` seconds before target start time.

### Two Tasks During Rebuilding

**Task 1: Restore from previous output**

- System pushes historical output SVs from database
- Indicator must restore persisted state
- **Critical**: Restored state must match what was saved (production requirement)

**Task 2: Reconstruct non-persisted state**

- Some state is NOT saved (transient, computed fields)
- Must rebuild from historical input data
- Requires sufficient historical data (hence `rebuild_length`)

**Theoretical guarantee**: If incoming data unchanged + algorithm unchanged → reconstructed state = original state.

### Rebuilding vs Production

**Rebuilding phase** (warm-up):

- Load saved states during rebuilding
- Accumulate data for calculations
- **Do NOT assert consistency** (still warming up)
- Track with `bars_since_start` (non-persisted counter)

**Production phase** (after rebuilding):

- Calculate states from scratch
- **Assert**: calculated state == saved state (reconciliation)
- Output meaningful results
- Normal operation

---

## The Reconciliation Pattern

### Purpose

**Goal**: Verify that recalculated state matches previously saved state.

**When**: After rebuilding phase completes.

**Pattern**:

```
For each bar after rebuilding:
  1. Calculate state from current data: S_calculated
  2. Load saved state from database: S_saved
  3. Assert: S_calculated == S_saved (within tolerance)
  4. If match: continue; if mismatch: ERROR
```

### Rebuilding Detection

**Critical distinction**:

| Counter | Behavior | Use Case |
|---------|----------|----------|
| `self.bar_index` | **Persisted**, carries across restarts | Total bars processed |
| `self.bars_since_start` | **NOT persisted**, resets to 0 each run | Rebuilding detection |

**Why it matters**:

```python
# ❌ WRONG - Uses persisted counter
def _rebuild_finished(self):
    return self.bar_index >= 80  # Immediately true if resuming at bar 1000

# ✅ CORRECT - Uses non-persisted counter
def _rebuild_finished(self):
    return self.bars_since_start >= 80  # Counts from 0 each run
```

### Five-Step Pattern

| Step | Method | Purpose |
|------|--------|---------|
| **1. Cache** | `from_sv(sv)` | Store incoming saved state: `self.latest_sv = sv` |
| **2. Restore** | `_from_sv()` | During rebuilding: `super().from_sv(self.latest_sv)` |
| **3. Compare** | `_reconcile_state()` | After rebuilding: `assert self._equal(saved)` |
| **4. Load** | `_load_from_sv(sv)` | Create temp object with saved state for comparison |
| **5. Verify** | `_equal(other)` | Compare fields with floating-point tolerance |

### Operational Sequence in on_bar

**Critical ordering** (from fix004.md example):

```python
def on_bar(self, bar):
    # 1. Extract bar info
    tm = bar.get_time_tag()
    granularity = bar.get_granularity()

    # 2. Initialize timetag
    if granularity == self.granularity and self.timetag is None:
        self.timetag = tm

    # 3. Check timetag advancement (CRITICAL: granularity match)
    if self.granularity == granularity and self.timetag < tm:
        # 3a. Calculate frame with OLD data
        self._on_cycle_pass(tm)

        # 3b. Reconcile (after rebuilding finished)
        if self.latest_sv is not None and self.initialized:
            self._reconcile_state()

        # 3c. Restore state (during rebuilding)
        if (not self.initialized or not overwrite) and self.latest_sv is not None:
            self._from_sv()

        # 3d. Output if ready
        if self.ready_to_serialize():
            ret.append(self.copy_to_sv())

        # 3e. Update state (AFTER all processing)
        self.latest_sv = None
        self.timetag = tm
        self.bar_index += 1  # Persisted
        self.bars_since_start += 1  # NOT persisted
        self.initialized = self._rebuild_finished()

    # 4. Process extractor and parent
    self.extractor.on_bar(bar)
    super().on_bar(bar)

    # 5. Import data (AFTER cycle pass)
    if bar.market == self.market and bar.code == self.code and \
       bar.granularity == self.granularity and bar.meta_id == self.meta_id:
        self.from_sv(bar)  # Cache for reconciliation

    return ret
```

**Key sequence points**:

1. Timetag advancement check includes granularity match
2. `_on_cycle_pass()` uses OLD data (before timetag update)
3. Reconciliation happens AFTER calculation (if initialized)
4. Restoration happens DURING rebuilding (if not initialized)
5. State updates (timetag, counters) happen AFTER all processing
6. Data import happens LAST (for NEXT cycle)

---

## Critical Rules

### 🚨 Rule 1: Granularity-Matched Timetag Update

**Doctrine**: Only update `self.timetag` from bars matching `self.granularity`.

```python
# ✅ CORRECT
if bar.granularity == self.granularity and self.timetag < bar.timetag:
    self._on_cycle_pass()
    self.timetag = bar.timetag

# ❌ WRONG - Missing granularity check
if self.timetag < bar.timetag:
    self._on_cycle_pass()
```

**Why**: Different granularities have different time boundaries. Mismatched updates corrupt time tracking.

### 🚨 Rule 2: Data Leakage Prevention

**Doctrine**: Import data AFTER `_on_cycle_pass()`, never before.

```python
# ✅ CORRECT
if self.timetag < bar.timetag:
    self._on_cycle_pass()  # Uses OLD data
    self.timetag = bar.timetag

# Import NEW data (for next cycle)
self.quote.from_sv(bar)
```

**Verification**: Live trading receives data for time t+1 AFTER decisions for time t.

### 🚨 Rule 3: Rebuilding Detection

**Doctrine**: Use non-persisted counter for warm-up tracking.

```python
def __init__(self):
    self.bar_index = 0  # Persisted
    self.bars_since_start = 0  # NOT persisted

def _rebuild_finished(self):
    return self.bars_since_start >= WARMUP  # Resets to 0 on each run

def on_bar(self, bar):
    self.bar_index += 1  # Total bars
    self.bars_since_start += 1  # Bars this run
```

### 🚨 Rule 4: State Update Timing

**Doctrine**: Update timetag, counters, and flags AFTER all processing.

**Order**:

1. Calculate: `_on_cycle_pass()`
2. Reconcile: `_reconcile_state()` (if initialized)
3. Restore: `_from_sv()` (if rebuilding)
4. Output: `copy_to_sv()`
5. **Then update**: `self.timetag = tm`, `self.bar_index += 1`, etc.

**Why**: Calculations need current state values. Premature updates corrupt calculations.

### 🚨 Rule 5: Load Saved State Correctly

**Doctrine**: In `_load_from_sv()`, bypass custom logic with `super(self.__class__, temp).from_sv(sv)`.

```python
# ❌ WRONG - Triggers custom from_sv logic
def _load_from_sv(self, sv):
    temp = self.__class__()
    temp.from_sv(sv)  # Calls overridden from_sv!
    return temp

# ✅ CORRECT - Bypasses custom logic
def _load_from_sv(self, sv):
    temp = self.__class__()
    # Copy metadata
    temp.market = self.market
    temp.code = self.code
    temp.meta_id = self.meta_id
    temp.granularity = self.granularity
    temp.namespace = self.namespace
    # Call PARENT's from_sv directly
    super(self.__class__, temp).from_sv(sv)
    return temp
```

**Why**: `temp.from_sv(sv)` calls overridden method with caching logic, NOT state loading.

---

## Common Violations

| Violation | Symptom | Fix |
|-----------|---------|-----|
| **Missing granularity check** | Timetag corruption, wrong boundaries | Check `bar.granularity == self.granularity` |
| **Data leakage** | Backtest better than live | Import AFTER `_on_cycle_pass()` |
| **Wrong rebuilding check** | Reconciliation fails on resume | Use `bars_since_start`, NOT `bar_index` |
| **Premature state update** | Calculations use wrong values | Update timetag/counters AFTER processing |
| **Wrong `_load_from_sv`** | Assertion failures | `super(self.__class__, temp).from_sv(sv)` |
| **Unbounded growth** | Memory crash after weeks | Use `deque(maxlen=N)` or online algorithms |
| **Non-determinism** | Different outputs same inputs | No random/time-based logic |
| **External state** | Resume inconsistency | All state in instance variables |

---

## Summary

**Core concepts**:

- **Frame**: StructValue at time T for granularity G
- **Driver**: Timetag change triggers calculation (not on_bar arrival)
- **Timetag definition**: End time y of bar range [x, y), must match granularity
- **Rebuilding**: System always rewinds `rebuild_length` for warm-up
- **Two tasks**: (1) Restore saved state, (2) Reconstruct computed state

**Why stateless matters**:

- Production systems restart frequently
- Results must be identical after restart
- Unbounded state causes memory crashes and inconsistency

**Four principles**:

1. **Bounded memory**: Fixed-size structures, online algorithms
2. **Deterministic**: No random, no external state, no time-based logic
3. **Complete persistence**: All state in `sv_object` fields
4. **Data leakage prevention**: Import AFTER cycle pass

**Rebuilding system**:

- Rewinds historical data automatically
- Two phases: rebuilding (warm-up) vs production (reconciliation)
- Track with non-persisted `bars_since_start` counter

**Reconciliation**:

- After rebuilding: verify calculated == saved
- Five-step pattern: Cache, Restore, Compare, Load, Verify
- Use floating-point tolerance for comparison

**Five critical rules**:

1. 🚨 **Granularity-matched timetag**: Only update from matching granularity
2. 🚨 **Data leakage prevention**: Import AFTER `_on_cycle_pass()`
3. 🚨 **Rebuilding detection**: Use `bars_since_start` (NOT persisted)
4. 🚨 **State update timing**: Update timetag/counters AFTER processing
5. 🚨 **Load saved state**: `super(self.__class__, temp).from_sv(sv)`

**Operational sequence**:

1. Check timetag advancement (with granularity match)
2. Calculate with OLD data
3. Reconcile (if initialized) or restore (if rebuilding)
4. Output frame
5. Update state (timetag, counters, flags)
6. Import NEW data (for next cycle)

**Verification**: `test_resuming_mode.py` runs split-resume test; you implement reconciliation assertions.

**Reference**: See templates for complete implementation.

---

**Previous:** [04 - StructValue and sv_object](04-structvalue-and-sv_object.md) | **Next:** [06 - Backtest and Testing](06-backtest.md)
