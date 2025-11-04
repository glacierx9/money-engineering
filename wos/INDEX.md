# WOS Documentation Index

Quick reference index for the Wolverine Operating System documentation.

## By Topic

### Getting Started
- [Overview and Architecture](01-overview.md#overview) - Start here
- [Quick Start Guide](README.md#quick-start) - Learning path
- [Framework Installation](01-overview.md#ecosystem) - Setup requirements

### Configuration
- [uin.json Structure](02-uin-and-uout.md#uinjson-configuration) - Input configuration
- [uout.json Structure](02-uin-and-uout.md#uoutjson-configuration) - Output configuration
- [Array Alignment](02-uin-and-uout.md#critical-configuration-rules) - Configuration rules
- [Granularity Configuration](02-uin-and-uout.md#granularities) - Timeframe setup

### Programming Model
- [Base Classes](03-programming-basics-and-cli.md#base-classes-architecture) - pcts3.sv_object, sc3.strategy
- [Module Structure](03-programming-basics-and-cli.md#module-structure) - Required components
- [Framework Globals](03-programming-basics-and-cli.md#framework-global-variables) - Required globals
- [Callbacks](03-programming-basics-and-cli.md#required-callbacks) - on_init, on_bar, etc.

### Data Flow
- [StructValue Basics](04-structvalue-and-sv_object.md#structvalue-the-universal-data-container) - Data container
- [sv_object Pattern](04-structvalue-and-sv_object.md#sv_object-automatic-serialization) - Serialization
- [from_sv() Method](04-structvalue-and-sv_object.md#from_sv---loading-state) - Loading state
- [copy_to_sv() Method](04-structvalue-and-sv_object.md#copy_to_sv---saving-state) - Saving state
- [Metadata Management](04-structvalue-and-sv_object.md#metadata-management) - Schema handling

### Stateless Design
- [Stateless Design Concept](05-stateless-design.md#overview) - Why stateless matters
- [Online Algorithms](05-stateless-design.md#online-algorithms) - EMA, Welford's, ATR
- [Bounded Memory](05-stateless-design.md#stateless-design-principles) - Using deque
- [Replay Consistency](05-stateless-design.md#replay-consistency) - Testing determinism

### Testing
- [Quick Test](06-backtest.md#quick-test-7-day-run) - 7-day testing
- [Full Backtest](06-backtest.md#full-backtest-multi-year) - Multi-year validation
- [Replay Consistency Test](06-backtest.md#replay-consistency-test) - test_resuming_mode.py
- [VSCode Debugging](06-backtest.md#vscode-debug-configurations) - Debug setup

### Tier 1 Development
- [Simple Indicator](07-tier1-indicator.md#building-a-simple-indicator) - EMA example
- [Multi-Commodity Pattern](07-tier1-indicator.md#multi-commodity-pattern) - Multiple objects
- [Cycle Boundaries](07-tier1-indicator.md#cycle-boundary-handling) - Proper handling
- [Output Serialization](07-tier1-indicator.md#output-serialization) - Return list pattern

### Tier 2 Development
- [Composite Architecture](08-tier2-composite.md#tier-2-architecture) - Portfolio layer
- [Basket Management](08-tier2-composite.md#key-concepts) - Managing positions
- [Signal Aggregation](08-tier2-composite.md#signal-aggregation) - Combining signals
- [Capital Allocation](08-tier2-composite.md#capital-allocation) - Dynamic allocation

### Tier 3 Development
- [Execution Architecture](09-tier3-strategy.md#architecture) - Order execution
- [Signal Translation](09-tier3-strategy.md#signal-translation) - Signals to orders
- [Order Types](09-tier3-strategy.md#order-types) - Market, limit, best
- [Position Sync](09-tier3-strategy.md#position-synchronization) - Broker sync

### Analysis
- [Visualization Scripts](10-visualization.md#visualization-script-template) - Analysis tools
- [Performance Plots](10-visualization.md#key-visualizations) - Charts and graphs
- [Parameter Analysis](10-visualization.md#parameter-analysis) - Optimization
- [SVR3 API](10-visualization.md#overview) - Data fetching

### Optimization
- [Issue Identification](11-fine-tune-and-iterate.md#identifying-common-issues) - Finding problems
- [Parameter Optimization](11-fine-tune-and-iterate.md#parameter-optimization) - Grid search
- [A/B Testing](11-fine-tune-and-iterate.md#ab-testing) - Comparing versions
- [Avoiding Overfitting](11-fine-tune-and-iterate.md#avoiding-overfitting) - Best practices

### Complete Examples
- [Mean Reversion System](12-example-project.md#project-mean-reversion-system) - Full example
- [Tier 1 Implementation](12-example-project.md#phase-1-tier-1-indicator) - RSI indicator
- [Testing Workflow](12-example-project.md#phase-2-test-and-validate) - Validation
- [Production Deployment](12-example-project.md#phase-5-production-deployment) - Going live

## By Framework Component

### pycaitlyn (pc)
- [StructValue](04-structvalue-and-sv_object.md#structvalue-the-universal-data-container)
- [Namespaces](04-structvalue-and-sv_object.md#namespaces)
- [IndexSchema](03-programming-basics-and-cli.md#framework-global-variables)

### pycaitlynts3 (pcts3)
- [sv_object](04-structvalue-and-sv_object.md#sv_object-automatic-serialization)
- [from_sv()](04-structvalue-and-sv_object.md#from_sv---loading-state)
- [copy_to_sv()](04-structvalue-and-sv_object.md#copy_to_sv---saving-state)

### strategyc3 (sc3)
- [strategy class](03-programming-basics-and-cli.md#strategy-base-classes)
- [Position management](08-tier2-composite.md#basket-management)
- [PnL tracking](08-tier2-composite.md#key-concepts)

### composite_strategyc3 (csc3)
- [composite_strategy](08-tier2-composite.md#building-a-composite-strategy)
- [Basket management](08-tier2-composite.md#basket-management)
- [Capital allocation](08-tier2-composite.md#capital-allocation)

### pycaitlynutils3 (pcu3)
- [Logger](03-programming-basics-and-cli.md#logging-best-practices)
- [Timestamp parsing](05-stateless-design.md#practical-example-stateless-indicator)

## By Skill Level

### Beginner
1. [Overview](01-overview.md) - Understand the framework
2. [Configuration](02-uin-and-uout.md) - Set up uin.json/uout.json
3. [Programming Basics](03-programming-basics-and-cli.md) - Learn base patterns
4. [Simple Indicator](07-tier1-indicator.md) - Build first indicator

### Intermediate
1. [StructValue](04-structvalue-and-sv_object.md) - Master data flow
2. [Stateless Design](05-stateless-design.md) - Implement online algorithms
3. [Multi-Commodity](07-tier1-indicator.md#multi-commodity-pattern) - Handle multiple instruments
4. [Composite Strategy](08-tier2-composite.md) - Build portfolio manager

### Advanced
1. [Replay Consistency](05-stateless-design.md#replay-consistency) - Perfect determinism
2. [Optimization](11-fine-tune-and-iterate.md) - Parameter tuning
3. [Tier 3 Execution](09-tier3-strategy.md) - Live trading
4. [Production Deployment](12-example-project.md#phase-5-production-deployment) - Go live

## Critical Concepts

### The 4 Doctrines
1. [Multiple Indicator Objects](07-tier1-indicator.md#multi-commodity-pattern) - One per commodity
2. [No Fallback Logic](04-structvalue-and-sv_object.md#accessing-field-values) - Trust dependencies
3. [Always Return List](03-programming-basics-and-cli.md#data-processing-flow) - Framework contract
4. [Logical Contracts Only](07-tier1-indicator.md#best-practices) - Filter `<00>` contracts

### Design Patterns
- [Online Algorithms](05-stateless-design.md#online-algorithms) - Bounded memory
- [Cycle Boundaries](07-tier1-indicator.md#cycle-boundary-handling) - Time management
- [Signal Aggregation](08-tier2-composite.md#signal-aggregation) - Combining signals
- [State Persistence](05-stateless-design.md#state-persistence-patterns) - Serialization

### Best Practices
- [Bounded Collections](05-stateless-design.md#principle-1-bounded-memory) - Use deque
- [Deterministic Computation](05-stateless-design.md#principle-4-deterministic-computation) - No randomness
- [Trust Dependency Data](04-structvalue-and-sv_object.md#accessing-field-values) - No fallbacks
- [Test Replay Consistency](06-backtest.md#replay-consistency-test) - Validate determinism

## Troubleshooting

### Common Issues
- [No Output Generated](06-backtest.md#issue-1-no-output-generated)
- [Replay Test Fails](06-backtest.md#issue-2-replay-test-fails)
- [Memory Growth](06-backtest.md#issue-3-memory-growth)
- [Non-Determinism](05-stateless-design.md#issue-2-non-determinism)

### Solutions
- [Growing Memory](05-stateless-design.md#issue-1-growing-memory)
- [External State](05-stateless-design.md#issue-3-external-state)
- [Incomplete Persistence](05-stateless-design.md#issue-4-incomplete-persistence)
- [Signal Issues](11-fine-tune-and-iterate.md#identifying-common-issues)

## Quick Reference

### File Templates
- [Tier 1 Indicator](07-tier1-indicator.md#step-4-complete-indicator-implementation)
- [Tier 2 Composite](08-tier2-composite.md#step-3-complete-composite-implementation)
- [Tier 3 Executor](09-tier3-strategy.md#basic-executor-template)
- [Visualization Script](10-visualization.md#visualization-script-template)

### Configuration Templates
- [uin.json](02-uin-and-uout.md#complete-uinjson-example)
- [uout.json](02-uin-and-uout.md#complete-uoutjson-example)
- [launch.json](06-backtest.md#vscode-debug-configurations)

### Command Reference
- [Quick Test](06-backtest.md#quick-test-7-day-run)
- [Full Backtest](06-backtest.md#full-backtest-multi-year)
- [Replay Test](06-backtest.md#replay-consistency-test)

---

**Start Learning**: [Chapter 1 - Overview](01-overview.md)

**Get Help**: See [README](README.md) for support resources
