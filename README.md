# Wolverine Operating System (WOS) - Environment Generation & Guidance (EGG)

## Project Overview

This is the **WOS-EGG (Wolverine Operating System - Environment Generation & Guidance)** project. It is the official starting point for developing financial indicators and trading strategies on the Wolverine framework.

### What This Project Is

This repository serves as a comprehensive development kit:

*   **Documentation Base**: Contains the complete 12-chapter Wolverine Operating System (WOS) documentation, providing in-depth knowledge on every aspect of the framework.
*   **Project Generator**: Features a powerful CLI tool (`create_project.py`) that instantly scaffolds new, fully-configured indicator or strategy projects.
*   **Template System**: Includes ready-to-use templates for indicators, composite strategies, and visualization scripts.
*   **Development Guide**: Offers clear instructions and best practices for building, testing, and deploying high-quality trading logic.

**This is NOT a trading strategy or a live indicator.** It is the foundational toolkit used to **create** them.

## 🚀 Quick Start (2 Minutes)

Get a fully functional project running in three simple steps.

### Step 1: Create Your Project

The recommended method is to use the interactive CLI tool.

```bash
# Run the interactive project generator
./create_project.py --interactive
```

The tool will guide you through naming your project, selecting markets, securities, and timeframes. It generates a complete, ready-to-code project directory.

Alternatively, use the command-line for faster setup:
```bash
# Example: Create a basic indicator for iron ore on the DCE market
./create_project.py --name MyIronIndicator --market DCE --securities i
```
For more advanced usage, see the [CLI Usage Guide](CLI_USAGE.md).

### Step 2: Open in VS Code

Navigate into your newly created project and open it in Visual Studio Code.

```bash
cd MyIronIndicator
code .
```

### Step 3: Reopen in Container & Start Coding

VS Code will prompt you to "Reopen in Container". Click it. This launches the pre-configured Docker environment, ensuring all dependencies are correctly installed.

Once the container is running, open the main Python file (e.g., `MyIronIndicator.py`) and start implementing your logic in the `_on_cycle_pass()` method. Press `F5` to run a quick backtest using the pre-configured debugger.

## WOS Documentation

This project includes the complete **Wolverine Operating System (WOS)** documentation, which is the single source of truth for the framework. The documentation is symlinked into every project you create for easy access.

Start exploring the documentation here: **[./wos/README.md](wos/README.md)**

### Table of Contents

*   **Part I: Fundamentals** (Chapters 1-6)
    *   [Chapter 01: Wolverine Framework Overview](wos/01-overview.md)
    *   [Chapter 02: uin.json and uout.json](wos/02-uin-and-uout.md)
    *   [Chapter 03: Programming Basics and CLI](wos/03-programming-basics-and-cli.md)
    *   [Chapter 04: StructValue and sv_object](wos/04-structvalue-and-sv_object.md)
    *   [Chapter 05: Stateless Design and Replay Consistency](wos/05-stateless-design.md)
    *   [Chapter 06: Backtest and Testing](wos/06-backtest.md)
*   **Part II: Implementation** (Chapters 7-9)
    *   [Chapter 07: Tier 1 Indicator Development](wos/07-tier1-indicator.md)
    *   [Chapter 08: Tier 2 Composite Strategy](wos/08-tier2-composite.md)
    *   [Chapter 09: Tier 3 Execution Strategy](wos/09-tier3-strategy.md)
*   **Part III: Optimization & Deployment** (Chapters 10-12)
    *   [Chapter 10: Visualization and Analysis](wos/10-visualization.md)
    *   [Chapter 11: Fine-tune and Iterate](wos/11-fine-tune-and-iterate.md)
    *   [Chapter 12: Complete Example Project](wos/12-example-project.md)

## Development Workflow

The standard workflow for creating a new indicator is:

1.  **Create Project**: Use `./create_project.py` to generate the project structure.
2.  **Implement Logic**: Write your indicator's core calculations in the `_on_cycle_pass()` method.
3.  **Test**: Use the VS Code debugger (`F5`) to run quick tests and the provided `test_resuming_mode.py` script to ensure replay consistency.
4.  **Visualize**: Run the generated `*_viz.py` script to analyze your indicator's behavior and performance.
5.  **Iterate**: Fine-tune parameters based on analysis, re-test, and re-validate.
6.  **Deploy**: Once validated, deploy the indicator to the production environment.

## Critical Framework Doctrines

Adherence to these rules is mandatory for creating stable, production-ready strategies.

1.  **Multiple Indicator Objects**: Never reuse a single indicator object for multiple commodities. Create one instance per instrument to prevent state contamination. (See [Chapter 7](wos/07-tier1-indicator.md#multi-commodity-pattern)).
2.  **No Fallback Logic**: Never use fallback values (e.g., `value or default`) for dependency data. Trust the data pipeline and fail fast to detect issues. (See [Chapter 4](wos/04-structvalue-and-sv_object.md#data-access-patterns)).
3.  **Always Return a List**: All `on_bar` callbacks must return a list, even if it's empty. This is a strict framework requirement. (See [Chapter 3](wos/03-programming-basics-and-cli.md#data-processing-flow)).
4.  **Filter for Logical Contracts**: Indicators should almost always process logical/continuous contracts (e.g., `i<00>`) and filter out specific monthly contracts. (See [Chapter 7](wos/07-tier1-indicator.md#best-practices)).

## Contributing

If you develop useful indicators, templates, or improvements to this environment, please submit a pull request to the main repository after thorough testing and documentation.