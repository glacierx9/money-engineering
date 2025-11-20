# Wolverine Operating System (WOS) Documentation

This directory contains the complete documentation for the Wolverine algorithmic trading framework. It is designed to be the single source of truth for developers building indicators and strategies.

## Learning Path

If you are new to the Wolverine framework, we recommend following this structured learning path:

1.  **Fundamentals First**: Read **Chapters 1-6** to understand the core architecture, data flow, programming model, and testing procedures. Mastering stateless design (**Chapter 5**) is critical.
2.  **Build a Tier-1 Indicator**: Follow **Chapter 7** to build your first complete technical indicator from scratch.
3.  **Build a Tier-2 Strategy**: Learn how to manage a portfolio of indicators in **Chapter 8**.
4.  **Analyze & Optimize**: Use the techniques in **Chapters 10 and 11** to visualize, analyze, and fine-tune your creations.
5.  **Review the Full Example**: **Chapter 12** provides a complete end-to-end project walkthrough.

## Table of Contents

| Chapter | Title                                               | Description                                                               |
| :------ | :-------------------------------------------------- | :------------------------------------------------------------------------ |
| 1       | [Overview](01-overview.md)                          | Framework introduction, architecture, and core concepts.                  |
| 2       | [uin.json and uout.json](02-uin-and-uout.md)          | Configuring data inputs and outputs for your strategy.                    |
| 3       | [Programming Basics & CLI](03-programming-basics-and-cli.md) | Base classes, module structure, and using the `calculator3_test.py` CLI.  |
| 4       | [StructValue & sv_object](04-structvalue-and-sv_object.md) | The universal data container and automatic state serialization pattern.     |
| 5       | [Stateless Design](05-stateless-design.md)          | **CRITICAL**: Replay consistency, online algorithms, and state management. |
| 6       | [Backtest and Testing](06-backtest.md)              | Running backtests, validating consistency, and debugging.                 |
| 7       | [Tier-1 Indicator](07-tier1-indicator.md)           | Building technical analysis indicators that process raw market data.      |
| 8       | [Tier-2 Composite Strategy](08-tier2-composite.md)  | Aggregating Tier-1 signals into a portfolio management strategy.          |
| 9       | [Tier-3 Execution Strategy](09-tier3-strategy.md)   | Translating portfolio signals into live market orders.                    |
| 10      | [Visualization & Analysis](10-visualization.md)      | Fetching calculated data and creating diagnostic plots for analysis.      |
| 11      | [Fine-tune and Iterate](11-fine-tune-and-iterate.md) | Parameter optimization, A/B testing, and avoiding overfitting.            |
| 12      | [Example Project](12-example-project.md)            | A complete, end-to-end walkthrough of building a trading system.          |

## Index by Topic

This index helps you quickly find information on specific topics.

### Getting Started
- [Overview and Architecture](01-overview.md#overview)
- [Quick Start Guide](../README.md#quick-start)
- [Framework Installation](01-overview.md#ecosystem)

### Configuration
- [uin.json Structure](02-uin-and-uout.md#uinjson-configuration)
- [uout.json Structure](02-uin-and-uout.md#uoutjson-configuration)
- [Array Alignment Rules](02-uin-and-uout.md#critical-configuration-rules)
- [Granularity Configuration](02-uin-and-uout.md#granularities)

### Programming Model
- [Base Classes (`sv_object`, `strategy`)](03-programming-basics-and-cli.md#base-classes-architecture)
- [Required Module Structure](03-programming-basics-and-cli.md#module-structure)
- [Framework Global Variables](03-programming-basics-and-cli.md#framework-global-variables)
- [Framework Callbacks (`on_init`, `on_bar`)](03-programming-basics-and-cli.md#required-callbacks)

### Data Flow & State
- [StructValue Data Container](04-structvalue-and-sv_object.md#structvalue-the-universal-data-container)
- [`sv_object` Serialization Pattern](04-structvalue-and-sv_object.md#sv_object-automatic-serialization)
- [`from_sv()` - Loading State](04-structvalue-and-sv_object.md#from_sv---loading-state)
- [`copy_to_sv()` - Saving State](04-structvalue-and-sv_object.md#copy_to_sv---saving-state)
- [Stateless Design Principles](05-stateless-design.md#overview)
- [Online Algorithms (EMA, Welford's)](05-stateless-design.md#online-algorithms)
- [Bounded Memory (`deque`)](05-stateless-design.md#stateless-design-principles)

### Testing & Debugging
- [Quick Test (7-day)](06-backtest.md#quick-test-7-day-run)
- [Full Backtest (Multi-year)](06-backtest.md#full-backtest-multi-year)
- [Replay Consistency Test](06-backtest.md#replay-consistency-test)
- [VSCode Debugging Setup](06-backtest.md#vscode-debug-configurations)

### Strategy Development
- **Tier 1**: [Building a Simple Indicator](07-tier1-indicator.md#building-a-simple-indicator)
- **Tier 1**: [Multi-Commodity Pattern](07-tier1-indicator.md#multi-commodity-pattern)
- **Tier 1**: [Cycle Boundary Handling](07-tier1-indicator.md#cycle-boundary-handling)
- **Tier 2**: [Composite Architecture](08-tier2-composite.md#tier-2-architecture)
- **Tier 2**: [Basket Management](08-tier2-composite.md#key-concepts)
- **Tier 2**: [Signal Aggregation](08-tier2-composite.md#signal-aggregation)
- **Tier 3**: [Execution & Order Management](09-tier3-strategy.md#architecture)

### Analysis & Optimization
- [Visualization Scripts (`svr3`)](10-visualization.md#visualization-script-template)
- [Performance Plots](10-visualization.md#key-visualizations)
- [Parameter Optimization](11-fine-tune-and-iterate.md#parameter-optimization)
- [Avoiding Overfitting](11-fine-tune-and-iterate.md#avoiding-overfitting)

---
**Start Learning**: [Chapter 1 - Overview](01-overview.md)