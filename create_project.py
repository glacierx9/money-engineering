#!/usr/bin/env python3
"""
Wolverine Project Creator - CLI tool to create new indicator/strategy projects

Usage:
    ./create_project.py --name MyIndicator --type indicator --market DCE --securities i,j --granularity 900
    ./create_project.py --name MyComposite --type composite
    ./create_project.py --interactive
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path


class ProjectCreator:
    """Create new Wolverine indicator/strategy projects"""

    VALID_MARKETS = {
        'DCE': ['i', 'j', 'm', 'y', 'p', 'c', 'a', 'b', 'v', 'l', 'pp'],
        'SHFE': ['cu', 'al', 'zn', 'pb', 'ni', 'sn', 'au', 'ag', 'rb', 'wr', 'hc', 'fu', 'bu', 'ru', 'sc'],
        'CZCE': ['TA', 'MA', 'FG', 'SR', 'CF', 'RM', 'OI', 'WH', 'AP', 'CJ'],
        'CFFEX': ['IF', 'IC', 'IH', 'T', 'TF']
    }

    GRANULARITIES = {
        '1min': 60,
        '5min': 300,
        '15min': 900,
        '30min': 1800,
        '1h': 3600,
        '4h': 14400,
        '1d': 86400
    }

    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.templates_dir = self.base_dir / 'templates'

    def _load_template(self, template_name, replacements):
        """Load a template file and replace placeholders

        Args:
            template_name: Name of template file (e.g., 'indicator.py.template')
            replacements: Dict of {placeholder: value} pairs

        Returns:
            Template content with placeholders replaced
        """
        template_path = self.templates_dir / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        content = template_path.read_text()

        # Replace all placeholders
        for placeholder, value in replacements.items():
            content = content.replace(f"{{{{{placeholder}}}}}", str(value))

        return content

    def create_project(self, name, project_type='indicator', markets=None, securities=None,
                      granularities=None, token='YOUR_TOKEN_HERE', interactive=False):
        """Create a new project with specified configuration"""

        if interactive:
            return self._interactive_create()

        # Validate inputs
        if not name:
            raise ValueError("Project name is required")

        if not name.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Project name must be alphanumeric (underscores and hyphens allowed)")

        # Set defaults
        if markets is None:
            markets = ['DCE']
        if securities is None:
            securities = {'DCE': ['i']}
        if granularities is None:
            granularities = [900]

        # Create project directory
        project_dir = self.base_dir / name
        if project_dir.exists():
            response = input(f"Project '{name}' already exists. Overwrite? (yes/no): ")
            if response.lower() != 'yes':
                print("Aborted.")
                return None
            shutil.rmtree(project_dir)

        print(f"Creating project: {name}")
        project_dir.mkdir(parents=True)

        # Determine category based on type
        category = '1' if project_type == 'indicator' else '2' if project_type == 'composite' else '3'

        # Create project structure
        self._create_structure(project_dir, name, project_type, markets, securities,
                              granularities, token, category)

        print(f"\n✅ Project '{name}' created successfully!")
        print(f"\n📁 Location: {project_dir}")
        print(f"\n📚 Resources included:")
        print(f"   - WOS documentation linked at ./wos/ (readonly reference)")
        print(f"   - CLAUDE.md with project-specific guidance")
        print(f"   - Complete debug configurations")
        print(f"\n🚀 Next steps:")
        print(f"   1. cd {name}")
        print(f"   2. code .")
        print(f"   3. Press Cmd/Ctrl+Shift+P → 'Dev Containers: Reopen in Container'")
        print(f"   4. Read CLAUDE.md and ./wos/07-tier1-indicator.md")
        print(f"   5. Edit {name}.py to implement your logic")
        print(f"   6. Press F5 to run 'Quick Test' debug configuration")

        return project_dir

    def _interactive_create(self):
        """Interactive project creation wizard"""
        print("=" * 60)
        print("Wolverine Project Creator - Interactive Mode")
        print("=" * 60)
        print()

        # Project name
        while True:
            name = input("Project name (e.g., MyIndicator): ").strip()
            if name and name.replace('_', '').replace('-', '').isalnum():
                break
            print("❌ Invalid name. Use alphanumeric characters, underscores, or hyphens.")

        # Project type
        print("\nProject type:")
        print("  1. Indicator (Tier 1) - Technical analysis indicator")
        print("  2. Composite (Tier 2) - Multi-strategy portfolio manager")
        print("  3. Strategy (Tier 3) - Execution strategy")
        type_choice = input("Select type [1/2/3] (default: 1): ").strip() or '1'
        project_type = {
            '1': 'indicator',
            '2': 'composite',
            '3': 'strategy'
        }.get(type_choice, 'indicator')

        # Markets and securities
        print("\nSelect markets (comma-separated, e.g., DCE,SHFE):")
        print("  Available: DCE, SHFE, CZCE, CFFEX")
        markets_input = input("Markets (default: DCE): ").strip().upper() or 'DCE'
        markets = [m.strip() for m in markets_input.split(',')]

        securities = {}
        for market in markets:
            print(f"\nSecurities for {market}:")
            print(f"  Available: {', '.join(self.VALID_MARKETS.get(market, []))}")
            secs_input = input(f"  Enter securities (comma-separated, default: {self.VALID_MARKETS[market][0] if market in self.VALID_MARKETS else 'all'}): ").strip()

            if secs_input:
                # Split and validate
                secs = [s.strip() for s in secs_input.split(',')]
                # Validate securities
                valid_secs = []
                for sec in secs:
                    if sec in self.VALID_MARKETS.get(market, []):
                        valid_secs.append(sec)
                    else:
                        print(f"  ⚠️  Warning: '{sec}' not in standard {market} securities, including anyway")
                        valid_secs.append(sec)
                securities[market] = valid_secs
            else:
                securities[market] = [self.VALID_MARKETS[market][0]] if market in self.VALID_MARKETS else []

        # Granularities
        print("\nGranularities:")
        print("  Available: 1min, 5min, 15min, 30min, 1h, 4h, 1d")
        gran_input = input("Enter granularities (comma-separated, default: 15min): ").strip() or '15min'
        gran_list = [g.strip() for g in gran_input.split(',')]
        granularities = [self.GRANULARITIES.get(g, 900) for g in gran_list]

        # API Token
        print("\nAPI Token:")
        token = input("Enter API token (or press Enter to use placeholder): ").strip() or 'YOUR_TOKEN_HERE'

        # Confirm
        print("\n" + "=" * 60)
        print("Project Configuration:")
        print("=" * 60)
        print(f"Name:          {name}")
        print(f"Type:          {project_type}")
        print(f"Markets:       {', '.join(markets)}")
        for market, secs in securities.items():
            print(f"  {market:12} {', '.join(secs)}")
        print(f"Granularities: {', '.join([str(g) for g in granularities])}")
        print(f"Token:         {token if token != 'YOUR_TOKEN_HERE' else '(placeholder)'}")
        print("=" * 60)

        confirm = input("\nCreate project? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("Aborted.")
            return None

        return self.create_project(name, project_type, markets, securities, granularities, token, False)

    def _create_structure(self, project_dir, name, project_type, markets, securities,
                         granularities, token, category):
        """Create complete project structure"""

        # Create directories
        (project_dir / '.vscode').mkdir(exist_ok=True)
        (project_dir / '.devcontainer').mkdir(exist_ok=True)

        # Create symlink to WOS documentation (readonly reference)
        wos_source = self.base_dir / 'wos'
        wos_link = project_dir / 'wos'
        if wos_source.exists() and not wos_link.exists():
            try:
                # Create relative symlink so it works across different paths
                wos_link.symlink_to(os.path.relpath(wos_source, project_dir))
                print(f"   ✓ Linked WOS documentation (readonly reference)")
            except Exception as e:
                print(f"   ⚠ Warning: Could not create WOS symlink: {e}")
                print(f"     You can still access docs at: {wos_source}")

        # Create main indicator/strategy file
        self._create_main_file(project_dir, name, project_type, markets, securities)

        # Create uin.json
        self._create_uin_json(project_dir, markets, securities, granularities)

        # Create uout.json
        self._create_uout_json(project_dir, markets, securities, granularities, project_type)

        # Create test_resuming_mode.py
        self._create_test_resuming_mode(project_dir, name, category)

        # Create .vscode/launch.json
        self._create_launch_json(project_dir, name, granularities[0], token, category)

        # Create .devcontainer/devcontainer.json
        self._create_devcontainer_json(project_dir, name)

        # Create CLAUDE.md
        self._create_claude_md(project_dir, name, project_type, markets, securities, granularities)

        # Create visualization script template
        self._create_viz_script(project_dir, name)

        # Create README.md
        self._create_project_readme(project_dir, name, project_type, markets, securities)

    def _create_main_file(self, project_dir, name, project_type, markets, securities):
        """Create main indicator/strategy Python file"""

        if project_type == 'indicator':
            template = self._get_indicator_template(name, markets, securities)
        elif project_type == 'composite':
            template = self._get_composite_template(name)
        else:
            template = self._get_strategy_template(name)

        file_path = project_dir / f"{name}.py"
        file_path.write_text(template)
        os.chmod(file_path, 0o755)

    def _get_indicator_template(self, name, markets, securities):
        """Generate indicator template"""

        # Get first market and security for example
        first_market = list(markets)[0] if markets else 'DCE'
        first_sec = securities.get(first_market, ['i'])[0] if securities else 'i'

        return self._load_template('indicator.py.template', {
            'NAME': name,
            'FIRST_MARKET': first_market,
            'FIRST_SEC': first_sec
        })

    def _get_composite_template(self, name):
        """Generate composite strategy template"""
        return self._load_template('composite.py.template', {
            'NAME': name
        })

    def _get_strategy_template(self, name):
        """Generate execution strategy template"""
        return self._load_template('strategy.py.template', {
            'NAME': name
        })

    def _create_uin_json(self, project_dir, markets, securities, granularities):
        """Create uin.json configuration"""

        # Build securities array aligned with markets
        market_list = list(markets)
        securities_list = []
        sec_cats_list = []

        for market in market_list:
            secs = securities.get(market, [])
            securities_list.append(secs)
            sec_cats_list.append([1, 2, 3])

        config = {
            "global": {
                "imports": {
                    "SampleQuote": {
                        "fields": [
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "turnover"
                        ],
                        "granularities": granularities,
                        "revision": 4294967295,
                        "markets": market_list,
                        "security_categories": sec_cats_list,
                        "securities": securities_list
                    }
                }
            }
        }

        file_path = project_dir / "uin.json"
        file_path.write_text(json.dumps(config, indent=2))

    def _create_uout_json(self, project_dir, markets, securities, granularities, project_type):
        """Create uout.json configuration"""

        # Build securities array aligned with markets
        market_list = list(markets)
        securities_list = []
        sec_cats_list = []

        for market in market_list:
            secs = securities.get(market, [])
            securities_list.append(secs)
            sec_cats_list.append([1, 2, 3])

        # Define output fields based on type
        if project_type == 'indicator':
            fields = [
                "_preserved_field",
                "bar_index",
                "indicator_value",
                "signal"
            ]
            defs = [
                {
                    "field_type": "int64",
                    "display_name": "Preserved Field",
                    "precision": 0,
                    "sample_type": 0,
                    "multiple": 1
                },
                {
                    "field_type": "integer",
                    "display_name": "Bar Index",
                    "precision": 0,
                    "sample_type": 0,
                    "multiple": 1
                },
                {
                    "field_type": "double",
                    "display_name": "Indicator Value",
                    "precision": 6,
                    "sample_type": 0,
                    "multiple": 1
                },
                {
                    "field_type": "integer",
                    "display_name": "Signal",
                    "precision": 0,
                    "sample_type": 0,
                    "multiple": 1
                }
            ]
        else:
            # Composite/strategy fields
            fields = [
                "_preserved_field",
                "portfolio_value",
                "cash",
                "num_positions"
            ]
            defs = [
                {
                    "field_type": "int64",
                    "display_name": "Preserved Field",
                    "precision": 0,
                    "sample_type": 0,
                    "multiple": 1
                },
                {
                    "field_type": "double",
                    "display_name": "Portfolio Value",
                    "precision": 2,
                    "sample_type": 0,
                    "multiple": 1
                },
                {
                    "field_type": "double",
                    "display_name": "Cash",
                    "precision": 2,
                    "sample_type": 0,
                    "multiple": 1
                },
                {
                    "field_type": "integer",
                    "display_name": "Number of Positions",
                    "precision": 0,
                    "sample_type": 0,
                    "multiple": 1
                }
            ]

        config = {
            "private": {
                "markets": market_list,
                "security_categories": sec_cats_list,
                "securities": securities_list,
                "sample_granularities": {
                    "type": "min",
                    "cycles": [granularities[0]],
                    "cycle_lengths": [0]
                },
                "export": {
                    "XXX": {
                        "fields": fields,
                        "defs": defs,
                        "revision": -1
                    }
                }
            }
        }

        file_path = project_dir / "uout.json"
        file_path.write_text(json.dumps(config, indent=2))

    def _create_test_resuming_mode(self, project_dir, name, category):
        """Create test_resuming_mode.py"""
        content = self._load_template('test_resuming_mode.py.template', {
            'NAME': name,
            'CATEGORY': category
        })
        file_path = project_dir / "test_resuming_mode.py"
        file_path.write_text(content)
        os.chmod(file_path, 0o755)

    def _create_launch_json(self, project_dir, name, granularity, token, category):
        """Create .vscode/launch.json"""
        content = self._load_template('launch.json.template', {
            'NAME': name,
            'GRANULARITY': str(granularity),
            'TOKEN': token,
            'CATEGORY': category
        })

        vscode_dir = project_dir / ".vscode"
        vscode_dir.mkdir(exist_ok=True)
        file_path = vscode_dir / "launch.json"
        file_path.write_text(content)

    def _create_devcontainer_json(self, project_dir, name):
        """Create .devcontainer/devcontainer.json"""
        content = self._load_template('devcontainer.json.template', {
            'NAME': name,
            'NAME_LOWER': name.lower()
        })

        devcontainer_dir = project_dir / ".devcontainer"
        devcontainer_dir.mkdir(exist_ok=True)
        file_path = devcontainer_dir / "devcontainer.json"
        file_path.write_text(content)

    def _create_claude_md(self, project_dir, name, project_type, markets, securities, granularities):
        """Create CLAUDE.md project guidance"""

        import datetime
        today = datetime.datetime.now().strftime('%Y-%m-%d')

        market_str = ', '.join(markets)
        sec_str = ', '.join([f"{m}: {', '.join(secs)}" for m, secs in securities.items()])
        gran_str = ', '.join([f"{g}s" for g in granularities])

        content = self._load_template('CLAUDE.md.template', {
            'NAME': name,
            'TYPE': project_type,
            'MARKETS': market_str,
            'SECURITIES': sec_str,
            'GRANULARITIES': gran_str,
            'NAME_LOWER': name.lower(),
            'TODAY': today
        })
        file_path = project_dir / "CLAUDE.md"
        file_path.write_text(content)

    def _create_viz_script(self, project_dir, name):
        """Create visualization script template"""
        # Use the template with proper placeholder replacement
        content = self._load_template('indicator_viz.py.template', {
            'NAME': name
        })
        file_path = project_dir / f"{name.lower()}_viz.py"
        file_path.write_text(content)
        os.chmod(file_path, 0o755)

    def _create_project_readme(self, project_dir, name, project_type, markets, securities):
        """Create project README.md"""

        market_str = ', '.join(markets)
        sec_list = []
        for m, secs in securities.items():
            sec_list.append(f"- **{m}**: {', '.join(secs)}")
        sec_str = '\n'.join(sec_list)

        content = self._load_template('README.md.template', {
            'NAME': name,
            'TYPE': project_type.title(),
            'MARKETS': market_str,
            'SECURITIES': sec_str
        })
        file_path = project_dir / "README.md"
        file_path.write_text(content)


def main():
    parser = argparse.ArgumentParser(
        description="Create new Wolverine indicator/strategy project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  ./create_project.py --interactive

  # Create basic indicator
  ./create_project.py --name MyIndicator --type indicator --market DCE --securities i,j

  # Create multi-market indicator
  ./create_project.py --name MultiMarket --type indicator --market DCE,SHFE --securities-DCE i,j --securities-SHFE cu,al

  # Create composite strategy
  ./create_project.py --name MyComposite --type composite
        """
    )

    parser.add_argument('--name', help='Project name')
    parser.add_argument('--type', choices=['indicator', 'composite', 'strategy'],
                       default='indicator', help='Project type (default: indicator)')
    parser.add_argument('--market', '--markets', help='Comma-separated markets (e.g., DCE,SHFE)')
    parser.add_argument('--securities', help='Comma-separated securities for single market')
    parser.add_argument('--granularity', '--granularities',
                       help='Comma-separated granularities in seconds (e.g., 300,900,3600)')
    parser.add_argument('--token', default='YOUR_TOKEN_HERE', help='API token')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive mode (recommended for first time)')

    # Support per-market securities
    for market in ['DCE', 'SHFE', 'CZCE', 'CFFEX']:
        parser.add_argument(f'--securities-{market}',
                           help=f'Comma-separated securities for {market}')

    args = parser.parse_args()

    creator = ProjectCreator()

    if args.interactive or not args.name:
        creator._interactive_create()
        return

    # Parse markets
    markets = [m.strip().upper() for m in args.market.split(',')] if args.market else ['DCE']

    # Parse securities
    securities = {}
    if args.securities:
        # Single market shorthand
        securities[markets[0]] = [s.strip() for s in args.securities.split(',')]

    # Parse per-market securities
    for market in markets:
        sec_arg = getattr(args, f'securities_{market}', None)
        if sec_arg:
            securities[market] = [s.strip() for s in sec_arg.split(',')]
        elif market not in securities:
            # Use first available security for market
            securities[market] = [creator.VALID_MARKETS.get(market, ['i'])[0]]

    # Parse granularities
    if args.granularity:
        granularities = [int(g.strip()) for g in args.granularity.split(',')]
    else:
        granularities = [900]

    creator.create_project(
        name=args.name,
        project_type=args.type,
        markets=markets,
        securities=securities,
        granularities=granularities,
        token=args.token,
        interactive=False
    )


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
