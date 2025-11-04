# Chapter 6: Backtest and Testing

**Learning objectives:**
- Run backtests with calculator3_test.py
- Test replay consistency with test_resuming_mode.py
- Interpret backtest results
- Use VSCode debug configurations
- Validate strategy performance

**Previous:** [05 - Stateless Design](05-stateless-design.md) | **Next:** [07 - Tier 1 Indicator](07-tier1-indicator.md)

---

## Overview

Backtesting validates your strategy against historical data before live deployment. The Wolverine framework provides tools for comprehensive testing including replay consistency validation.

## Quick Test (7-Day Run)

Fast iteration during development:

```bash
python /home/wolverine/bin/running/calculator3_test.py \
    --testcase ${PWD}/ \
    --algoname MyIndicator \
    --sourcefile MyIndicator.py \
    --start 20250703203204 \
    --end 20250710203204 \
    --granularity 300 \
    --tm wss://10.99.100.116:4433/tm \
    --tm-master 10.99.100.116:6102 \
    --rails https://10.99.100.116:4433/private-api/ \
    --token YOUR_TOKEN \
    --category 1 \
    --is-managed 1 \
    --restore-length 864000000 \
    --multiproc 1
```

**When to use:**
- Initial development
- Quick validation after changes
- Debug specific issues
- Parameter tuning

## Full Backtest (Multi-Year)

Comprehensive validation:

```bash
python /home/wolverine/bin/running/calculator3_test.py \
    --testcase ${PWD}/ \
    --algoname MyIndicator \
    --sourcefile MyIndicator.py \
    --start 20230101000000 \
    --end 20250925000000 \
    --granularity 300 \
    --tm wss://10.99.100.116:4433/tm \
    --tm-master 10.99.100.116:6102 \
    --rails https://10.99.100.116:4433/private-api/ \
    --token YOUR_TOKEN \
    --category 1 \
    --is-managed 1 \
    --restore-length 864000000 \
    --multiproc 1
```

**When to use:**
- Final validation before deployment
- Performance across market conditions
- Statistical significance
- Overfitting detection

## Replay Consistency Test

**MANDATORY before production deployment:**

```bash
python test_resuming_mode.py
```

**With custom time range:**

```bash
python test_resuming_mode.py \
    --start 20250701203204 \
    --end 20250710203204 \
    --midpoint 20250705120000
```

**What it tests:**
1. Runs strategy continuously from start to end → Result A
2. Runs start to midpoint, stops, resumes midpoint to end → Result B
3. Compares Result A vs Result B (must be identical)

**Passing criteria:**
- All output fields must match exactly
- Timestamps must align
- State consistency verified

**If test fails:**
- Check for non-deterministic code (random, time-based)
- Verify all state is persisted
- Look for external dependencies
- Ensure proper initialization

## VSCode Debug Configurations

`.vscode/launch.json` for easy debugging:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Quick Test",
            "type": "debugpy",
            "request": "launch",
            "stopOnEntry": false,
            "python": "python",
            "program": "/home/wolverine/bin/running/calculator3_test.py",
            "args": [
                "--testcase", "${workspaceFolder}/",
                "--algoname", "MyIndicator",
                "--sourcefile", "MyIndicator.py",
                "--start", "20250703203204",
                "--end", "20250710203204",
                "--granularity", "300",
                "--tm", "wss://10.99.100.116:4433/tm",
                "--tm-master", "10.99.100.116:6102",
                "--rails", "https://10.99.100.116:4433/private-api/",
                "--token", "YOUR_TOKEN",
                "--category", "1",
                "--is-managed", "1",
                "--restore-length", "864000000",
                "--multiproc", "1"
            ],
            "cwd": "${workspaceFolder}",
            "console": "integratedTerminal"
        },
        {
            "name": "Full Backtest",
            "type": "debugpy",
            "request": "launch",
            "program": "/home/wolverine/bin/running/calculator3_test.py",
            "args": [
                "--testcase", "${workspaceFolder}/",
                "--algoname", "MyIndicator",
                "--sourcefile", "MyIndicator.py",
                "--start", "20230101000000",
                "--end", "20250925000000",
                "--granularity", "300",
                "--tm", "wss://10.99.100.116:4433/tm",
                "--tm-master", "10.99.100.116:6102",
                "--rails", "https://10.99.100.116:4433/private-api/",
                "--token", "YOUR_TOKEN",
                "--category", "1",
                "--is-managed", "1",
                "--restore-length", "864000000",
                "--multiproc", "1"
            ],
            "cwd": "${workspaceFolder}",
            "console": "integratedTerminal"
        },
        {
            "name": "Test Resuming Mode",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/test_resuming_mode.py",
            "cwd": "${workspaceFolder}",
            "console": "integratedTerminal"
        }
    ]
}
```

**Usage:**
1. Press F5 or Run → Start Debugging
2. Select configuration from dropdown
3. Set breakpoints by clicking line numbers
4. Use Debug Console to inspect variables

## Interpreting Results

### Tier 1 Indicator Output

Check output frequency and field values:

```python
logger.info(f"Bar {self.bar_index}: "
           f"TSI={self.tsi_value:.2f}, "
           f"VAI={self.vai_value:.2f}, "
           f"Signal={self.signal_strength:.3f}")
```

### Tier 2 Strategy Output

Monitor portfolio metrics:

```python
logger.info(f"Bar {self.bar_index}: "
           f"NV={self.nv:.4f}, "
           f"PV={self.pv:.2f}, "
           f"Dip={self.max_dip_percentage:.3f}, "
           f"Active={self.active_positions}")
```

**Key metrics:**
- **NV (Net Value)**: Should grow over time (>1.0)
- **Dip Percentage**: Maximum drawdown
- **Active Positions**: Number of open positions
- **PnL**: Profit and loss

## Common Issues

### Issue 1: No Output Generated

**Symptoms:**
- Strategy runs but produces no output
- Logs show bars processed but no StructValue created

**Causes:**
- `ready_to_serialize()` returns False
- Not calling `copy_to_sv()` in output path
- Returning None instead of list

**Solution:**

```python
def on_bar(self, bar):
    # Process data
    self._process(bar)

    # ALWAYS return list
    results = []
    if self.ready_to_serialize():
        results.append(self.copy_to_sv())
    return results
```

### Issue 2: Replay Test Fails

**Symptoms:**
- test_resuming_mode.py shows mismatched outputs
- Fields differ after resume

**Causes:**
- Non-deterministic computation (random, time)
- Incomplete state persistence
- External dependencies

**Solution:**

```python
# Remove non-determinism
# ❌ WRONG
import random
signal = self.signal + random.random()

# ✅ CORRECT
signal = self.signal  # Deterministic

# Persist all critical state
# ❌ WRONG - local variable
prev_close = close

# ✅ CORRECT - instance variable
self.prev_close = close
```

### Issue 3: Memory Growth

**Symptoms:**
- Memory usage grows during backtest
- Eventually crashes on long runs

**Causes:**
- Unbounded collections
- Growing history buffers

**Solution:**

```python
# Use bounded collections
from collections import deque

# ❌ WRONG
self.history = []  # Grows forever

# ✅ CORRECT
self.history = deque(maxlen=100)  # Fixed size
```

## Testing Workflow

### 1. Development Phase

```
1. Write initial code
2. Quick test (7 days)
3. Fix issues
4. Repeat 2-3 until working
```

### 2. Validation Phase

```
1. Run replay consistency test
2. Fix any determinism issues
3. Run full backtest (2+ years)
4. Analyze results
```

### 3. Optimization Phase

```
1. Identify parameter issues
2. Create visualization script
3. Tune parameters
4. Validate with new backtest
```

### 4. Deployment Phase

```
1. Final replay consistency test
2. Final full backtest
3. Code review
4. Deploy to production
```

## Best Practices

### Testing Checklist

Before production:
- [ ] Quick test passes
- [ ] Replay consistency test passes
- [ ] Full backtest completes
- [ ] Memory usage stable
- [ ] Output fields validated
- [ ] No warnings or errors in logs
- [ ] Performance metrics acceptable

### Logging Strategy

```python
# Development - verbose
logger.debug(f"Processing bar {self.bar_index}")
logger.debug(f"Price: {price:.2f}, Volume: {volume:.0f}")

# Production - essential only
logger.info(f"Bar {self.bar_index}: Signal={self.signal}")

# Always log errors
try:
    self.calculate()
except Exception as e:
    logger.error(f"Calculation failed: {e}")
```

### Performance Monitoring

```python
import time

class PerformanceMonitored(pcts3.sv_object):
    def __init__(self):
        super().__init__()
        self.total_time = 0.0
        self.bar_count = 0

    def on_bar(self, bar):
        start = time.time()

        # Process bar
        results = self._process(bar)

        # Track performance
        elapsed = time.time() - start
        self.total_time += elapsed
        self.bar_count += 1

        if self.bar_count % 1000 == 0:
            avg_time = self.total_time / self.bar_count
            logger.info(f"Average time per bar: {avg_time*1000:.2f}ms")

        return results
```

## Summary

This chapter covered:

1. **Quick Tests**: 7-day runs for rapid iteration
2. **Full Backtests**: Multi-year validation
3. **Replay Consistency**: Mandatory before production
4. **VSCode Integration**: Debug configurations
5. **Result Interpretation**: Key metrics to monitor
6. **Common Issues**: Problems and solutions
7. **Testing Workflow**: Development to deployment

**Key Takeaways:**

- Always run replay consistency test before production
- Use quick tests for development iteration
- Full backtests validate across market conditions
- VSCode debug configs streamline testing
- Monitor memory usage during long runs
- Log appropriately for each environment

**Next Steps:**

In the next chapter, we'll build a complete Tier 1 indicator from scratch, applying all the concepts learned so far.

---

**Previous:** [05 - Stateless Design](05-stateless-design.md) | **Next:** [07 - Tier 1 Indicator](07-tier1-indicator.md)
