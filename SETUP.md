# Wolverine Development Environment Setup Guide

## Quick Start (5 Minutes)

### Step 1: Copy the Egg Package

```bash
# Navigate to your projects directory
cd ~/Projects

# Copy the egg package to create your new project
cp -r /path/to/egg MyIndicatorProject
cd MyIndicatorProject
```

### Step 2: Open in VS Code

```bash
# Open VS Code in the project directory
code .
```

### Step 3: Reopen in Container

1. Press `Cmd/Ctrl+Shift+P`
2. Type "Dev Containers: Reopen in Container"
3. Select it and wait for container to build (first time: 5-10 minutes)

### Step 4: Verify Installation

Once inside the container terminal:

```bash
# Check Python
python --version

# Check framework
python -c "import pycaitlyn as pc; print('Framework loaded!')"

# Check wolverine binaries
ls /home/wolverine/bin/running/calculator3_test.py
```

### Step 5: Create Your First Indicator with Claude Code

```bash
# Install Claude Code in the container if not already installed
# Then prompt Claude Code:

claude "Create a basic indicator project called SimpleMA that calculates moving averages for iron ore (i) on DCE at 15-minute granularity"
```

Claude Code will:
- Create project folder structure
- Generate all configuration files
- Set up debug configurations
- Create visualization script template
- Generate CLAUDE.md with project-specific guidance

### Step 6: Start Developing

```bash
# Open the created indicator file
code SimpleMA/SimpleMA.py

# Press F5 to start debugging with "Quick Test" configuration
# Or run manually:
python /home/wolverine/bin/running/calculator3_test.py \
    --testcase ./SimpleMA/ \
    --algoname SimpleMA \
    --sourcefile SimpleMA.py \
    --start 20250703203204 \
    --end 20250710203204 \
    --granularity 900 \
    --tm wss://10.99.100.116:4433/tm \
    --tm-master 10.99.100.116:6102 \
    --rails https://10.99.100.116:4433/private-api/ \
    --token YOUR_TOKEN \
    --category 1 \
    --is-managed 1 \
    --restore-length 864000000 \
    --multiproc 1
```

## Detailed Setup Instructions

### Prerequisites

Before starting, ensure you have:

1. **Docker Desktop** installed and running
   ```bash
   docker --version  # Should show Docker version
   ```

2. **VS Code** with Dev Containers extension
   - Install: `code --install-extension ms-vscode-remote.remote-containers`

3. **System Requirements**
   - 8GB RAM minimum (16GB recommended)
   - 20GB free disk space
   - Linux/macOS/Windows with WSL2

4. **Network Access**
   - Access to Docker Hub (for pulling wos-prod-arm64 image)
   - Access to Wolverine Time Machine server (for market data)

### Understanding the Egg Package Structure

```
egg/
├── README.md                    # Main documentation and quick start
├── SETUP.md                     # This file - detailed setup guide
├── .devcontainer/
│   └── devcontainer.json       # Container configuration (pre-configured)
├── wos/                        # Wolverine Operating System knowledge base
│   ├── 01-overview.md          # Framework architecture overview
│   ├── 02-uin-and-uout.md      # Configuration files guide
│   ├── 03-programming-basics-and-cli.md
│   ├── 04-structvalue-and-sv_object.md
│   ├── 05-stateless-design.md  # Stateless design principles
│   ├── 06-backtest.md          # Testing and validation
│   ├── 07-tier1-indicator.md   # Basic indicator development
│   ├── 08-tier2-composite.md   # Portfolio management
│   ├── 09-tier3-strategy.md    # Execution strategies
│   ├── 10-visualization.md     # Analysis and validation
│   ├── 11-fine-tune-and-iterate.md
│   └── 12-example-project.md   # End-to-end example
└── templates/                   # Project templates
    ├── .vscode/                # VS Code configuration template
    ├── basic_indicator/        # Tier 1 indicator template
    └── composite_strategy/     # Tier 2 composite template
```

### Container Environment Details

**Image**: `glacierx/wos-prod-arm64`
- Based on Ubuntu with Python 3.x
- Pre-installed Wolverine framework (`pycaitlyn`, `pycaitlynts3`, `strategyc3`)
- Framework binaries in `/home/wolverine/bin/running/`
- User: `wolverine`

**Environment Variables**:
- `PYTHONPATH=/home/wolverine/bin/running:$PYTHONPATH`
- Automatically configured in devcontainer.json

**VS Code Extensions** (auto-installed):
- ms-python.python (Python support)
- ms-python.vscode-pylance (Type checking)
- ms-python.debugpy (Python debugging)
- ms-toolsai.jupyter (Jupyter notebooks)

## Creating Your First Project

### Method 1: Using Claude Code (Recommended)

Claude Code understands the WOS knowledge base and can create complete projects automatically.

```bash
# Example 1: Basic moving average indicator
claude "Create indicator MomentumMA for copper (cu) on SHFE with 5min, 15min, 1H timeframes"

# Example 2: Multi-commodity indicator
claude "Create indicator VolumeProfile for DCE commodities i and j at 15min granularity"

# Example 3: Composite strategy
claude "Create Tier 2 composite strategy MultiStrategyPortfolio that manages 10 baskets"
```

**What Claude Code Will Do:**
1. Create project directory: `./ProjectName/`
2. Generate `ProjectName.py` with sv_object structure
3. Create `uin.json` with correct markets/securities
4. Create `uout.json` with output field definitions
5. Set up `.vscode/launch.json` with debug configurations
6. Generate `test_resuming_mode.py` for replay consistency
7. Create `projectname_viz.py` visualization script
8. Generate `CLAUDE.md` with project-specific AI guidance

### Method 2: Manual Template-Based Creation

If you prefer manual control:

```bash
# 1. Copy template
cp -r templates/basic_indicator MyIndicator
cd MyIndicator

# 2. Rename files
mv Indicator.py MyIndicator.py

# 3. Update metadata in MyIndicator.py
# Edit: self.meta_name = "MyIndicator"

# 4. Configure uin.json
# Update markets, securities, granularities

# 5. Configure uout.json
# Update fields, defs to match your output

# 6. Set up debug configuration
cp ../templates/.vscode/launch.json.template .vscode/launch.json
# Replace {{PROJECT_NAME}}, {{GRANULARITY}}, {{API_TOKEN}}, {{CATEGORY}}

# 7. Generate CLAUDE.md
claude "Generate CLAUDE.md for my MyIndicator project"
```

### Method 3: Learn by Example

Study the working Margarita project:

```bash
# Examine the indicator implementation
cat ../Margarita.py

# Study the configuration
cat ../uin.json
cat ../uout.json

# Check debug setup
cat ../.vscode/launch.json

# Review the composite strategy
cat ../MargaritaComposite/MargaritaComposite.py
```

## Configuration Customization

### API Token Configuration

Get your API token from your Wolverine administrator, then:

**Option 1: Update launch.json** (per-project)
```json
{
  "args": [
    ...
    "--token", "YOUR_ACTUAL_TOKEN_HERE",
    ...
  ]
}
```

**Option 2: Use environment variable** (all projects)
```bash
# Add to ~/.bashrc or ~/.zshrc in container
export WOS_API_TOKEN="YOUR_ACTUAL_TOKEN_HERE"

# Then in launch.json:
"args": [..., "--token", "${env:WOS_API_TOKEN}", ...]
```

**Option 3: Use .env file** (recommended)
```bash
# Create ~/.env
echo 'WOS_API_TOKEN=YOUR_ACTUAL_TOKEN_HERE' >> ~/.env

# launch.json already configured to use envFile: "~/.env"
```

### Market and Security Configuration

Edit `uin.json` and `uout.json` to customize:

```json
{
  "markets": ["DCE"],              // Your target market(s)
  "security_categories": [[1, 2, 3]],
  "securities": [["i", "j"]]       // Your target commodities
}
```

**Remember:**
- DCE/SHFE securities: lowercase (`["i", "j", "cu", "al"]`)
- CZCE securities: UPPERCASE (`["TA", "MA", "FG"]`)
- Arrays must align: markets, security_categories, securities

### Granularity Configuration

Edit `uin.json` to specify timeframes:

```json
{
  "granularities": [900]           // Single timeframe: 15min
}

// Or multi-timeframe:
{
  "granularities": [300, 900, 3600]  // 5min, 15min, 1H
}
```

**Available Granularities:**
- 60: 1 minute
- 300: 5 minutes
- 900: 15 minutes (most common)
- 1800: 30 minutes
- 3600: 1 hour
- 14400: 4 hours (use ZampleQuote)
- 86400: 1 day

## Development Workflow

### 1. Design Phase

Before writing code:
- Define indicator concept clearly
- Specify input requirements (markets, timeframes)
- Design output fields and their meanings
- Plan state management approach

### 2. Implementation Phase

```bash
# Create project structure
claude "Create indicator MyIndicator..."

# Implement logic in MyIndicator.py
code MyIndicator/MyIndicator.py

# Focus on:
# - sv_object initialization
# - on_bar() data routing
# - _on_cycle_pass() calculations
# - State serialization control
```

### 3. Testing Phase

```bash
# Quick test (7 days)
# Press F5 in VS Code, select "Quick Test"

# Or run manually:
python /home/wolverine/bin/running/calculator3_test.py \
    --testcase ./MyIndicator/ \
    --start 20250703203204 \
    --end 20250710203204 \
    # ... other args

# Replay consistency test (MANDATORY)
cd MyIndicator
python test_resuming_mode.py
```

### 4. Visualization Phase

```bash
# Run visualization script
cd MyIndicator
python myindicator_viz.py

# Or in Jupyter:
jupyter notebook myindicator_viz.py
```

### 5. Optimization Phase

Based on visualization insights:
- Adjust EMA alpha parameters
- Tune threshold values
- Optimize weight distributions
- Test parameter variations

### 6. Full Backtest

```bash
# Run full historical backtest
# Press F5, select "Full Backtest"
# Wait for completion (may take 10-60 minutes)
```

### 7. Deployment

Once validated:
- Final replay consistency check
- Document all parameters in CLAUDE.md
- Set `overwrite = False` for production
- Deploy to production servers (follow org procedures)

## Debugging Tips

### Using VS Code Debugger

1. **Set Breakpoints**: Click left of line number
2. **Start Debugging**: Press F5
3. **Step Through**:
   - F10: Step over
   - F11: Step into
   - Shift+F11: Step out
4. **Inspect Variables**: Hover over variables or use Variables pane
5. **Debug Console**: Execute Python expressions during debugging

### Common Debug Scenarios

**Data not arriving:**
```python
# Set breakpoint in on_bar()
async def on_bar(self, bar: pc.StructValue) -> List[pc.StructValue]:
    ret = []
    market = bar.get_market()  # <- Set breakpoint here
    code = bar.get_stock_code()
    # Check what data is arriving
```

**State not persisting:**
```python
# Check ready_to_serialize()
def ready_to_serialize(self) -> bool:
    result = self.bar_index > 0  # <- Set breakpoint
    return result  # Verify this returns True
```

**Calculation errors:**
```python
# Set breakpoint in calculation logic
def _on_cycle_pass(self, time_tag):
    current_price = float(self.sq.close)  # <- Check value
    self.ema = self.alpha * current_price + (1 - self.alpha) * self.ema
    # Verify intermediate values
```

### Viewing Framework Logs

```bash
# In container terminal during backtest
# Logs appear in console showing:
# - Bar processing
# - Worker assignments
# - State serialization events
# - Errors and warnings
```

## Troubleshooting

### Issue: Container fails to start

**Solution:**
```bash
# Check Docker is running
docker ps

# Rebuild container
# In VS Code: Cmd/Ctrl+Shift+P -> "Dev Containers: Rebuild Container"

# Check devcontainer.json for syntax errors
```

### Issue: Framework imports fail

**Solution:**
```bash
# Verify PYTHONPATH
echo $PYTHONPATH  # Should include /home/wolverine/bin/running

# If missing, add to ~/.bashrc:
export PYTHONPATH="/home/wolverine/bin/running:$PYTHONPATH"
source ~/.bashrc
```

### Issue: No market data received

**Solution:**
```bash
# Check network connectivity to Time Machine server
ping 10.99.100.116

# Verify token is correct in launch.json
# Check date range has data for your commodities
# Ensure uin.json markets/securities are correct
```

### Issue: Replay consistency test fails

**Common Causes:**
1. Using random values or system time
2. Non-deterministic calculations
3. State not fully persisted in output fields
4. Growing collections (lists) not bounded

**Solution:** Review Chapter 05 (Singularity) in WOS docs

### Issue: AttributeError when accessing fields

**Solution:**
```python
# Ensure from_sv() called before accessing:
if self.sq.namespace == ns and self.sq.meta_id == meta_id:
    self.sq.from_sv(bar)  # MUST call this first
    price = float(self.sq.close)  # Now safe to access
```

## Learning Resources

### WOS Knowledge Base

Read in order:
1. **01-overview.md** - Start here for framework understanding
2. **02-uin-and-uout.md** - Configuration files
3. **03-programming-basics-and-cli.md** - Base classes and CLI
4. **07-tier1-indicator.md** - Basic indicator development
5. **10-visualization.md** - Analysis and validation
6. **12-example-project.md** - Complete working example

### Working Examples

Study these files in parent directory:
- `../Margarita.py` - Complete Tier 1 indicator
- `../MargaritaComposite/` - Tier 2 composite strategy
- `../margarita_viz.py` - Visualization script
- `../uin.json` and `../uout.json` - Configuration examples

### Getting Help

1. **Claude Code**: Your first resource for questions
   ```bash
   claude "How do I calculate EMA with 20-period in Wolverine?"
   claude "Debug: AttributeError when accessing self.sq.close"
   ```

2. **WOS Documentation**: Comprehensive framework guide
   ```bash
   # Open any chapter
   code wos/07-tier1-indicator.md
   ```

3. **Framework Source**: When in doubt, read the source
   ```bash
   # View framework implementation
   cat /home/wolverine/bin/running/pycaitlynts3.py
   ```

## Next Steps

After successful setup:

1. **Read WOS Docs**: Start with chapters 01-04
2. **Create Test Indicator**: Simple moving average on single commodity
3. **Run Quick Test**: Verify everything works
4. **Study Examples**: Examine Margarita implementation
5. **Build Real Indicator**: Implement your actual strategy
6. **Visualize Results**: Use visualization scripts
7. **Optimize Parameters**: Based on visualization insights
8. **Full Backtest**: Validate over historical data
9. **Deploy**: Follow your organization's deployment process

## Support and Community

- **Documentation**: Complete guides in `wos/` directory
- **Templates**: Ready-to-use templates in `templates/`
- **Claude Code**: AI assistant for development questions
- **Examples**: Working implementations in parent directory

---

**You're Ready!** Start building your indicator:

```bash
claude "Create a basic indicator project called MyFirstIndicator for iron ore on DCE at 15-minute granularity"
```

Happy coding! 🚀
