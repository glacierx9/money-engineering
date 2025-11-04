# Wolverine Operating System (WOS) Documentation

Complete guide to the Wolverine algorithmic trading framework.

## Table of Contents

### Part I: Fundamentals (Chapters 1-6)

1. **[Overview](01-overview.md)** - Framework introduction, architecture, and ecosystem
2. **[uin.json and uout.json](02-uin-and-uout.md)** - Data configuration and dependencies
3. **[Programming Basics and CLI](03-programming-basics-and-cli.md)** - Base classes, module structure, and calculator3_test.py
4. **[StructValue and sv_object](04-structvalue-and-sv_object.md)** - Data containers and automatic serialization
5. **[Stateless Design and State Management](05-stateless-design.md)** - Replay consistency and online algorithms
6. **[Backtest and Testing](06-backtest.md)** - Running backtests and validation

### Part II: Implementation (Chapters 7-9)

7. **[Tier 1 Indicator](07-tier1-indicator.md)** - Building technical analysis indicators
8. **[Tier 2 Composite Strategy](08-tier2-composite.md)** - Portfolio management and signal aggregation
9. **[Tier 3 Execution Strategy](09-tier3-strategy.md)** - Order execution and live trading

### Part III: Optimization (Chapters 10-12)

10. **[Visualization and Analysis](10-visualization.md)** - Creating analysis scripts and diagnostic plots
11. **[Fine-tune and Iterate](11-fine-tune-and-iterate.md)** - Parameter optimization and avoiding overfitting
12. **[Example Project](12-example-project.md)** - Complete end-to-end trading system

## Documentation Statistics

- **Total Pages**: 12 chapters
- **Total Lines**: 6,545 lines
- **Coverage**: Complete framework from basics to production

## Quick Start

New to Wolverine? Follow this learning path:

1. Read **Chapter 1** for overview and architecture
2. Understand **Chapters 2-4** for data flow and programming model
3. Master **Chapter 5** for stateless design (critical!)
4. Practice with **Chapter 6** for testing workflow
5. Build your first indicator with **Chapter 7**
6. Expand to portfolio with **Chapter 8**
7. Deploy with **Chapter 9** (if live trading)
8. Optimize using **Chapters 10-12**

## Key Concepts

### The 4 Critical Doctrines

1. **Multiple Indicator Objects Pattern** (Chapter 7) - One object per commodity
2. **No Fallback Logic for Dependencies** (Chapter 4) - Trust data sources
3. **Always Return List Pattern** (Chapter 3) - Framework contract
4. **Logical Contract Filtering** (Chapter 7) - Only process continuous contracts

### Framework Layers

```
┌─────────────────────────────────────┐
│  Tier 1: Technical Indicators       │
│  - OHLCV → Signals                  │
│  - Multi-timeframe analysis         │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Tier 2: Portfolio Management       │
│  - Signal aggregation               │
│  - Risk management                  │
│  - Capital allocation               │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Tier 3: Order Execution            │
│  - Signal → Orders                  │
│  - Broker integration               │
│  - Position sync                    │
└─────────────────────────────────────┘
```

## Contributing

Found an issue or want to improve the documentation?

1. Check existing issues
2. Create detailed bug report or improvement proposal
3. Submit pull request with changes

## Version

- **Version**: 1.0
- **Date**: October 2025
- **Framework**: Wolverine Caitlyn3
- **Python**: 3.8+

## License

This documentation is provided for educational and reference purposes.

## Support

For framework support:
- Internal documentation: `/home/wolverine/docs/`
- API reference: Framework source code
- Examples: `/workspaces/Margarita/` and `/workspaces/Margarita/MargaritaComposite/`

---

**Start your journey**: [Chapter 1 - Overview](01-overview.md)
