# Questions for WOS Framework Developer

**Context**: Implementing a Tier-2 Composite Strategy that manages 3 baskets (Iron Ore, Copper, Soybean). The strategy compiles and runs without errors, but:
- `basket.price` stays 0.00 even after market data arrives
- `basket.pv` stays at ¥250,000 (initial allocation) even after trades execute
- Composite `self.pv` stays frozen at ¥1,000,000
- `basket.signal` changes correctly (0 → 1 → 0), indicating `basket._signal()` is being called
- `analysis.ipynb` shows "Empty data" when fetching from private namespace

**Current Implementation**:
- Allocates baskets with `_allocate(basket_idx, market, code+b'<00>', capital, 1.0)` (using logical contracts)
- Calls `super().on_bar(bar)` for market data (namespace_global)
- Manually processes Tier-1 signals (namespace_private)
- Calls `basket._signal(price, timetag, signal)` to execute trades

---

## SECTION 1: Market Data Routing to Baskets

### Q1.1: How should market data reach baskets?
When a `SampleQuote` bar arrives in `namespace_global`, which of the following is correct?

**Option A**: Call `super().on_bar(bar)` and the framework automatically routes to baskets based on market/code matching
```python
if ns == pc.namespace_global:
    super().on_bar(bar)  # Framework handles everything?
```

**Option B**: Manually parse `SampleQuote` and call `basket.on_price_updated()`
```python
if ns == pc.namespace_global:
    sample_quote.from_sv(bar)
    basket.on_price_updated(tm, market, code, open, close, high, low, volume)
```

**Option C**: Manually parse and set `basket.price` directly
```python
if ns == pc.namespace_global:
    sample_quote.from_sv(bar)
    basket.price = float(sample_quote.close)
```

**Option D**: Something else entirely?

**Current behavior**: We call `super().on_bar(bar)` but `basket.price` stays 0.00, suggesting the framework isn't routing automatically.

---

### Q1.2: Does `composite_strategy.on_bar()` route market data?
Does the base class `composite_strategyc3.composite_strategy.on_bar(bar)` method:
- **Parse** SampleQuote bars?
- **Match** them to baskets based on market/code?
- **Update** basket.price automatically?
- Or does it do **nothing** and expect subclasses to handle routing?

---

### Q1.3: What is `basket.on_price_updated()` and when should it be called?
- Does this method exist in the basket (strategyc3) base class?
- What are its exact parameters?
- Does it update `basket.price` and recalculate `basket.pv`?
- Should we call it for EVERY market data bar, or only when basket has a position?

---

### Q1.4: Contract matching - logical vs specific
We allocate baskets with **logical contracts**: `_allocate(0, b'DCE', b'i<00>', capital, 1.0)`

But market data arrives with **specific contracts**: `DCE/i2501`, `DCE/i2505`, etc.

**Question**: How does the framework match them?
- Does `_allocate()` with `b'i<00>'` automatically match all `i****` contracts?
- Do we need to manually extract the commodity (`i` from `i2501`) and route?
- Do we need to track the active contract and update `basket.code` when rolling?

---

## SECTION 2: Basket Trading and PV Updates

### Q2.1: How does `basket._signal()` work?
When we call `basket._signal(price, timetag, signal)`:

**Question A**: Does it **immediately** execute a trade, or just set an intention?

**Question B**: Does it update `basket.pv` automatically based on:
- `basket.cash`
- `basket.price` (current market price)
- Position size calculated from available capital?

**Question C**: Does `basket._signal()` require `basket.price` to be set BEFORE calling it?

**Current behavior**:
```
LONG basket 2: DCE/m, contracts=84, price=2958.00
  -> Basket 2 after entry: cash=¥250,000, pv=¥250,000, signal=1
```
The `basket.signal` changes to 1 (correct), but `basket.pv` stays at ¥250,000 (wrong - should reflect position value).

---

### Q2.2: What updates basket.pv?
Which operations cause `basket.pv` to recalculate?

**Option A**: Automatically on every bar if basket has a position
**Option B**: Only when `basket._signal()` is called
**Option C**: Only when `basket.on_price_updated()` is called
**Option D**: Only when `_save()` and `_sync()` are called at composite level
**Option E**: Must be manually calculated and set

---

### Q2.3: Why is basket.pv staying at initial allocation?
After trades execute:
- `basket.signal = 1` (position is open)
- `basket.cash = ¥250,000` (unchanged)
- `basket.pv = ¥250,000` (unchanged - WRONG!)

**Expected**: `basket.pv = basket.cash + position_value`

**Question**: What's missing? Is it:
- Market data not reaching basket?
- `basket.price` not being set (currently 0.00)?
- `basket._signal()` not working as expected?
- Need to call a refresh method?

---

### Q2.4: How does composite self.pv get calculated?
Is `self.pv` (composite portfolio value):

**Option A**: Automatically calculated by framework as `sum(basket.pv for basket in self.strategies) + self.cash`

**Option B**: Must be manually calculated in `_update_portfolio_metrics()`

**Option C**: Only updates when we call `_save()` and `_sync()`

**Currently**: `self.pv` stays frozen at ¥1,000,000 despite trades executing.

---

## SECTION 3: The _allocate() Method

### Q3.1: What does _allocate() actually do?
When we call `self._allocate(basket_idx, market, code, capital, leverage)`:

**Does it**:
- Transfer cash from `self.cash` to `basket.cash`? ✓ (we see this working)
- Set `basket.market` and `basket.code`? (need confirmation)
- Subscribe basket to market data for that market/code? (unclear)
- Initialize basket for trading that instrument? (unclear)

---

### Q3.2: What code format should be passed to _allocate()?
Which is correct for Iron Ore on DCE?

**Option A**: `b'i<00>'` (logical contract, current implementation)
**Option B**: `b'i2501'` (specific active contract)
**Option C**: `b'i'` (just the commodity symbol)
**Option D**: Something else?

**Follow-up**: If we use logical contracts, how does basket receive data for specific contracts like `i2501`?

---

### Q3.3: Does _allocate() create automatic routing?
After calling `_allocate(0, b'DCE', b'i<00>', 250000, 1.0)`:

**Question**: Will market data for `DCE/i2501`, `DCE/i2505`, etc. automatically reach basket 0?
- If YES: Why isn't `basket.price` updating in our implementation?
- If NO: How do we manually route market data to that basket?

---

## SECTION 4: Data Export and Serialization

### Q4.1: Why is analysis.ipynb showing "Empty data"?
The notebook fetches:
```python
reader = svr3.sv_reader(
    meta_name="CompositeStrategy",
    markets=["COMPOSITE"],
    codes=["PORTFOLIO"],
    namespace=pc.namespace_private,
    ...
)
```

But gets empty results.

**Question**: Is the CompositeStrategy writing any output?
- Does `ready_to_serialize()` need to return True? (currently returns `self.initialized and self.bar_index > 0`)
- Does the framework call `sv_copy()` and write the data automatically?
- Is there a missing step to publish data to the server?

---

### Q4.2: What triggers data export?
When does the framework actually write CompositeStrategy data?

**Option A**: Every time `on_bar()` returns a non-empty list
**Option B**: At cycle boundaries when `ready_to_serialize()` returns True
**Option C**: Only when `_save()` is called
**Option D**: Something else?

---

### Q4.3: Is uout.json configuration correct?
Our `CompositeStrategy/uout.json`:
```json
{
  "private": {
    "markets": ["COMPOSITE"],
    "securities": [["PORTFOLIO"]],
    "export": {
      "XXX": {
        "fields": ["bar_index", "active_positions", "total_signals_processed", ...]
      }
    }
  }
}
```

**Question**: Is this correct? Should we:
- Use a different market/security code?
- Export to `global` instead of `private`?
- Use a different structure?

---

## SECTION 5: Complete Working Pattern

### Q5.1: Can you provide a MINIMAL working composite strategy?
Please provide the **absolute minimum** code for a composite strategy that:
1. Allocates 1 basket
2. Receives market data and updates `basket.price`
3. Executes 1 trade via `basket._signal()`
4. Shows `basket.pv` changing after trade

Just the core `on_bar()` logic - maybe 20-30 lines of code showing the correct pattern.

---

### Q5.2: What's the correct initialization sequence?
In what order should these happen?

```python
def __init__(self):
    super().__init__(initial_cash, BASKET_COUNT)  # Step ?
    self._initialize_baskets()                     # Step ?
    # When does metadata get loaded?
    # When do baskets become "ready"?
```

---

### Q5.3: Do baskets need initialization beyond _allocate()?
After `_allocate()`, do baskets need:
- Metadata loaded via `basket.load_def_from_dict()`?
- Global imports set via `basket.set_global_imports()`?
- A parser for SampleQuote?
- Manual subscription to market data?

---

## SECTION 6: Specific to Our Implementation

### Q6.1: Is our Tier-1 signal parsing correct?
We import from 3 Tier-1 indicators via `uin.json`:
```json
"private": {
  "imports": {
    "IronOreIndicatorRelaxed": {
      "fields": ["_preserved_field", "bar_index", "close", "signal", ...]
    }
  }
}
```

And parse with:
```python
parser = IronOreSignalParser()  # extends sv_object
parser.from_sv(bar)
signal_data = {'close': parser.close, 'signal': parser.signal, ...}
```

**Question**: Is this correct? Or should we access data differently?

---

### Q6.2: Can you review our _on_cycle_pass()?
```python
def _on_cycle_pass(self, time_tag: int):
    super()._on_cycle_pass(time_tag)
    self.risk_manager.update_peak_tracking(self.pv)
    # ... process signals, check rebalancing ...
    self._update_portfolio_metrics()
    self._save()
    self._sync()
    self._log_portfolio_state()
```

**Question**: Is this sequence correct? Are we missing any critical calls?

---

### Q6.3: Should we call super().on_bar() at all?
Currently we do:
```python
if ns == pc.namespace_global:
    super().on_bar(bar)  # Does this do anything useful?
```

**Question**: Should we:
- Keep calling `super().on_bar(bar)`?
- Remove it and handle everything manually?
- Call it but also do manual routing?

---

## SECTION 7: Debugging Questions

### Q7.1: How can we verify baskets are receiving data?
What attributes/methods can we check to confirm basket is getting market data?
- `basket.price` (currently 0.00)
- `basket.bar_count` or similar counter?
- `basket.last_update_time`?
- Something else?

---

### Q7.2: How can we verify basket trades executed?
After `basket._signal(price, tm, 1)`, what should we check?
- `basket.signal` (changes correctly to 1 ✓)
- `basket.position` or `basket.contracts`?
- `basket.entry_price`?
- `basket.pv` (currently doesn't change ✗)

---

### Q7.3: What's the relationship between basket.cash and basket.pv?
When basket has no position:
- `basket.pv == basket.cash` ? (currently TRUE)

When basket has a long position at profit:
- `basket.pv > basket.cash` ? (expected)
- `basket.cash` decreases by margin? Or stays the same?

**Current behavior**: Both stay at ¥250,000 even after entering position.

---

## SECTION 8: Critical Success Criteria

### Q8.1: What would SUCCESS look like in logs?
After implementing the correct pattern, logs should show:

```
BEFORE TRADE:
  basket.price = 2958.00 ✓ (currently 0.00 ✗)
  basket.pv = ¥250,000 ✓
  basket.signal = 0 ✓

AFTER EXECUTING: basket._signal(2958.00, tm, 1)
  basket.signal = 1 ✓ (we see this)
  basket.price = 2958.00 ✓ (currently 0.00 ✗)
  basket.pv = ¥??? ✓ (currently stays ¥250,000 ✗)

AFTER MARKET MOVES TO 2960.00:
  basket.price = 2960.00 ✓ (currently 0.00 ✗)
  basket.pv = basket.cash + (contracts * 2960.00) ✓ (currently ¥250,000 ✗)
  self.pv = sum(all basket.pv) + self.cash ✓ (currently frozen ✗)
```

**Question**: Is this the expected behavior?

---

## SECTION 9: Summary Questions

### Q9.1: Top 3 most likely causes of our bug?
Based on your experience with this framework, what are the top 3 most likely reasons for:
- `basket.price` staying 0.00
- `basket.pv` not updating after trades
- `self.pv` staying frozen at ¥1,000,000

---

### Q9.2: One critical missing piece?
If you had to guess ONE thing we're missing, what would it be?
- A method call?
- A configuration setting?
- A fundamental misunderstanding of how the framework works?

---

### Q9.3: Working reference implementation?
Can you point us to:
- File path of a working composite strategy in the codebase?
- Example code snippet that definitely works?
- Documentation page we might have missed?

---

## SECTION 10: Code Snippets for Review

### Current on_bar() implementation:
```python
def on_bar(self, bar: pc.StructValue) -> List[pc.StructValue]:
    market = bar.get_market()
    code = bar.get_stock_code()
    tm = bar.get_time_tag()
    ns = bar.get_namespace()

    if self.timetag is None:
        self.timetag = tm

    if self.timetag < tm:
        self._on_cycle_pass(tm)
        results = []
        if self.bar_index > 0:
            results.append(self.sv_copy())
        self.timetag = tm
        self.bar_index += 1
        return results

    if ns == pc.namespace_global:
        super().on_bar(bar)  # ← Does this do anything?

    elif ns == pc.namespace_private:
        self._process_tier1_signal(bar)

    return []
```

**Question**: What's wrong with this? What needs to change?

---

### Current _execute_entry() implementation:
```python
def _execute_entry(self, basket_idx: int, signal_data: Dict):
    basket = self.strategies[basket_idx]
    market, code = self.basket_to_instrument[basket_idx]

    allocated_capital = self.pv * self.base_allocation_pct
    current_price = signal_data['close']  # From Tier-1 signal
    contracts = int(allocated_capital / current_price)

    # Risk check...

    self.entry_prices[basket_idx] = current_price
    basket._signal(current_price, basket.timetag, signal)  # ← Execute trade

    logger.info(f"After entry: cash={basket.cash}, pv={basket.pv}, signal={basket.signal}")
```

**Question**: Is this correct? Why doesn't `basket.pv` change after `basket._signal()`?

---

## Thank You!

Please answer as many of these as possible. Even partial answers or pointers to the right direction would be extremely helpful. Our goal is to:

1. Get `basket.price` to update from market data
2. Get `basket.pv` to reflect position values
3. Get `self.pv` to update correctly
4. Get data exported so `analysis.ipynb` can visualize results

Thank you for your help!
