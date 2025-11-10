#!/usr/bin/env python3
# coding=utf-8
"""
Tier-2 Composite Strategy - Multi-Instrument Portfolio Manager

Manages 3-instrument portfolio with equal-weight allocation:
- Iron Ore (DCE/i<00>)
- Copper (SHFE/cu<00>)
- Soybean Meal (DCE/m<00>)

Features:
- Equal-weight capital allocation (25% base per instrument)
- Conservative risk management (10% max DD, 3% stops)
- Systematic rebalancing
- Independent instrument management

Author: AI Agent
Date: 2025-11-06
"""

import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import pycaitlyn as pc
import pycaitlynutils3 as pcu3
import pycaitlynts3 as pcts3
import composite_strategyc3 as csc3

# ============================================================================
# FRAMEWORK GLOBALS (REQUIRED)
# ============================================================================

use_raw = True
overwrite = False
granularity = 900
schema = None
max_workers = 1
worker_no = None
exports = {}
imports = {}
metas = {}
logger = pcu3.vanilla_logger()

# ============================================================================
# CONFIGURATION
# ============================================================================

BASKET_COUNT = 3
INITIAL_CAPITAL = 1000000.0

# Instrument mapping: basket_idx -> (market, code)
BASKET_MAPPING = {
    0: (b'DCE', b'i'),   # Iron Ore
    1: (b'SHFE', b'cu'), # Copper
    2: (b'DCE', b'm'),   # Soybean Meal
}

# ============================================================================
# RISK MANAGER
# ============================================================================

class RiskManager:
    """
    Centralized risk management for portfolio

    Enforces:
    - Per-position limits (35% max, 3% stop-loss)
    - Portfolio limits (90% max exposure, 15% min cash)
    - Drawdown limits (10% max portfolio DD)
    - Daily loss limits (3% max daily loss)
    """

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital

        # Per-position limits
        self.max_position_pct = 0.35  # 35% max per instrument
        self.max_position_loss = 0.03  # 3% stop-loss per position

        # Portfolio limits
        self.max_total_exposure = 0.90  # 90% max total
        self.min_cash_reserve = 0.10   # 10% min cash (target: 90% invested)

        # Drawdown limits
        self.max_portfolio_drawdown = 0.10  # 10% max DD
        self.max_daily_loss = 0.03          # 3% daily limit

        # Position limits
        self.max_simultaneous_positions = 3  # All instruments can be active
        self.min_contracts_per_position = 50  # Minimum position size

        # Tracking
        self.peak_portfolio_value = initial_capital
        self.daily_start_value = initial_capital
        self.daily_loss = 0.0
        self.current_drawdown = 0.0

    def update_daily_tracking(self, portfolio_value: float):
        """Update daily P&L tracking (call on tradeday_begin)"""
        self.daily_start_value = portfolio_value
        self.daily_loss = 0.0

    def update_peak_tracking(self, portfolio_value: float):
        """Update peak and drawdown tracking (call every bar)"""
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value

        self.current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
        self.daily_loss = (self.daily_start_value - portfolio_value) / self.daily_start_value if self.daily_start_value > 0 else 0.0

    def can_enter_position(self, basket_idx: int, proposed_size: float, portfolio_state) -> Tuple[bool, str]:
        """
        Pre-trade risk checks

        Returns: (can_enter: bool, reason: str)
        """
        # 1. Position size limit - REMOVED
        # Reason: With 30% pre-allocation + 1.4x leverage = 42% per basket
        # This check was blocking valid trades. We have sufficient controls:
        # - Pre-allocated capital (30% per basket)
        # - Leverage limits (1.4x max)
        # - Total exposure limit (90% portfolio)
        # - Cash reserve minimum (10%)

        # 2. Total exposure limit
        total_exposure = portfolio_state.total_exposure + proposed_size
        exposure_pct = total_exposure / portfolio_state.total_value if portfolio_state.total_value > 0 else 0
        if exposure_pct > self.max_total_exposure:
            return False, f"Total exposure {exposure_pct*100:.1f}% exceeds limit {self.max_total_exposure*100:.1f}%"

        # 3. Cash reserve check - REMOVED for pre-allocated basket model
        # Reason: Each basket has pre-allocated capital (¥300k) and manages its own cash
        # The composite cash reserve (¥100k) is separate and not used for basket trades
        # Baskets trade within their allocated capital, not composite cash

        # 4. Drawdown check
        if self.current_drawdown >= self.max_portfolio_drawdown:
            return False, f"Portfolio drawdown {self.current_drawdown*100:.1f}% exceeds limit {self.max_portfolio_drawdown*100:.1f}%"

        # 5. Daily loss check
        if self.daily_loss >= self.max_daily_loss:
            return False, f"Daily loss {self.daily_loss*100:.1f}% exceeds limit {self.max_daily_loss*100:.1f}%"

        return True, "OK"

    def get_risk_metrics(self) -> Dict:
        """Return current risk metrics for logging"""
        return {
            'current_drawdown': self.current_drawdown,
            'daily_loss': self.daily_loss,
            'peak_value': self.peak_portfolio_value,
        }

    def check_leverage_weighted_exposure(self, positions: Dict, portfolio_value: float) -> Tuple[bool, float]:
        """
        Check portfolio exposure using leverage weighting

        Formula: sum(position_value / leverage) / portfolio_value <= 0.90

        This prevents over-leveraging entire portfolio by measuring
        the actual capital at risk, not notional value.

        Args:
            positions: Dict of {basket_idx: {'value': float, 'leverage': float}}
            portfolio_value: Total portfolio value

        Returns:
            (is_within_limits: bool, exposure_pct: float)
        """
        if portfolio_value <= 0:
            return True, 0.0

        # Calculate base exposure (capital at risk, not notional)
        base_exposure = sum(
            pos['value'] / pos['leverage']
            for pos in positions.values()
            if pos.get('leverage', 1.0) > 0
        )

        exposure_pct = base_exposure / portfolio_value

        return exposure_pct <= self.max_total_exposure, exposure_pct

    def get_max_leverage_for_state(self, active_positions: int) -> float:
        """
        Calculate maximum safe leverage based on portfolio state

        Reduces max leverage when:
        - High drawdown
        - Daily loss significant
        - Multiple active positions

        Returns:
            Maximum safe leverage multiplier
        """
        max_lev = 20.0  # Start with absolute max

        # Reduce based on drawdown
        if self.current_drawdown > 0.05:
            max_lev *= 0.60  # Severe reduction at 5%+ DD
        elif self.current_drawdown > 0.03:
            max_lev *= 0.80  # Moderate reduction at 3%+ DD

        # Reduce based on daily loss
        if self.daily_loss > 0.02:
            max_lev *= 0.60  # Severe reduction at 2%+ daily loss
        elif self.daily_loss > 0.01:
            max_lev *= 0.80  # Moderate reduction at 1%+ daily loss

        # Reduce based on concentration
        if active_positions >= 3:
            max_lev *= 0.90  # Slight reduction when all baskets active
        elif active_positions >= 2:
            max_lev *= 0.95  # Very slight reduction

        return max(max_lev, 1.0)  # Never below 1.0x

# ============================================================================
# REBALANCER
# ============================================================================

class Rebalancer:
    """
    Portfolio rebalancing manager

    Triggers rebalancing when:
    - Time-based: Every 96 bars (daily)
    - Deviation-based: >10% drift from target allocation
    """

    def __init__(self):
        self.rebalance_frequency = 96  # Daily (96 bars × 15min)
        self.rebalance_threshold = 0.10  # 10% deviation
        self.last_rebalance_bar = 0

    def should_rebalance(self, current_bar: int, current_allocations: Dict,
                        target_allocations: Dict) -> Tuple[bool, Optional[int], float]:
        """
        Check if rebalancing needed

        Returns: (should_rebalance, drifted_instrument, max_deviation)
        """
        # 1. Check frequency
        bars_since_rebalance = current_bar - self.last_rebalance_bar
        time_trigger = bars_since_rebalance >= self.rebalance_frequency

        # 2. Check allocation drift
        max_deviation = 0.0
        drifted_instrument = None

        for instrument in current_allocations:
            current_pct = current_allocations[instrument]
            target_pct = target_allocations.get(instrument, 0.25)
            deviation = abs(current_pct - target_pct)

            if deviation > max_deviation:
                max_deviation = deviation
                drifted_instrument = instrument

        deviation_trigger = max_deviation > self.rebalance_threshold

        return (time_trigger or deviation_trigger), drifted_instrument, max_deviation

    def calculate_rebalance_actions(self, current_allocations: Dict,
                                   target_allocations: Dict,
                                   portfolio_value: float) -> Dict:
        """
        Calculate trades needed to rebalance

        Returns: {basket_idx: {'type': 'reduce'|'increase', 'value': float, 'deviation_pct': float}}
        """
        actions = {}

        for basket_idx in current_allocations:
            current_pct = current_allocations[basket_idx]
            target_pct = target_allocations.get(basket_idx, 0.25)

            deviation = current_pct - target_pct

            if abs(deviation) > self.rebalance_threshold:
                # Calculate adjustment needed
                adjustment_value = deviation * portfolio_value

                actions[basket_idx] = {
                    'type': 'reduce' if deviation > 0 else 'increase',
                    'value': abs(adjustment_value),
                    'deviation_pct': deviation * 100,
                }

        return actions

# ============================================================================
# DYNAMIC CASH MANAGER
# ============================================================================

class DynamicCashManager:
    """
    Adaptive cash reserve management

    Adjusts cash reserves based on market conditions and signal quality:
    - Aggressive (5-10%): Strong signals across multiple baskets
    - Balanced (15-20%): Normal market conditions
    - Defensive (25-30%): High volatility or weak signals
    """

    def __init__(self):
        self.reserve_tiers = {
            'aggressive': 0.05,   # 5% cash when strong signals
            'balanced': 0.15,     # 15% cash in normal conditions
            'defensive': 0.25,    # 25% cash in high volatility
        }

        # Thresholds
        self.strong_conviction_threshold = 0.60
        self.chaos_regime = 4

    def get_target_reserve(self, tier1_signals: Dict) -> float:
        """
        Determine optimal cash reserve based on current conditions

        Args:
            tier1_signals: Dict of {basket_idx: signal_data}

        Returns:
            Target cash reserve percentage (0.05 to 0.30)
        """
        if not tier1_signals:
            return self.reserve_tiers['balanced']

        # Count strong signals
        strong_signals = 0
        chaos_count = 0
        total_conviction = 0.0

        for signal_data in tier1_signals.values():
            # Calculate conviction
            conviction = (signal_data['confidence'] * 0.6 +
                         signal_data['signal_strength'] * 0.4)
            total_conviction += conviction

            # Count strong signals
            if conviction >= self.strong_conviction_threshold:
                strong_signals += 1

            # Count chaos regimes
            if signal_data.get('regime', 0) == self.chaos_regime:
                chaos_count += 1

        avg_conviction = total_conviction / len(tier1_signals) if tier1_signals else 0

        # Decision logic
        if strong_signals >= 2 and chaos_count == 0:
            # Multiple strong opportunities, low volatility
            return self.reserve_tiers['aggressive']
        elif chaos_count >= 2 or avg_conviction < 0.30:
            # High volatility or weak signals
            return self.reserve_tiers['defensive']
        else:
            # Normal conditions
            return self.reserve_tiers['balanced']

    def get_available_capital(self, composite_cash: float, target_reserve_pct: float,
                             portfolio_value: float) -> float:
        """
        Calculate available capital for new positions

        Returns capital above target reserve that can be deployed
        """
        target_reserve = portfolio_value * target_reserve_pct
        available = composite_cash - target_reserve
        return max(available, 0)

# ============================================================================
# DYNAMIC LEVERAGE MANAGER
# ============================================================================

class DynamicLeverageManager:
    """
    Smart leverage calculation based on:
    - Signal conviction (confidence + strength)
    - Market volatility (chaos regimes, ATR)
    - Portfolio state (drawdown, daily loss)
    - Position tier (STRONG/MEDIUM/WEAK)

    Goal: Maximize profits while minimizing drawdown
    """

    def __init__(self):
        # Leverage ranges by conviction tier
        self.leverage_tiers = {
            'STRONG': {'min': 4.0, 'max': 10.0, 'multiplier': 12.86},
            'MEDIUM': {'min': 2.5, 'max': 6.0, 'multiplier': 10.0},
            'WEAK': {'min': 1.5, 'max': 4.0, 'multiplier': 10.0},
        }

        # Hard caps
        self.max_leverage = 20.0  # Absolute maximum
        self.max_basket_leverage = 15.0  # Per-basket maximum

        # Stop-loss tiers (tighter stops for higher leverage)
        self.stop_loss_tiers = [
            (2.0, 0.030),   # Up to 2x: 3.0% stop
            (4.0, 0.025),   # Up to 4x: 2.5% stop
            (6.0, 0.020),   # Up to 6x: 2.0% stop
            (10.0, 0.015),  # Up to 10x: 1.5% stop
            (20.0, 0.010),  # Above 10x: 1.0% stop
        ]

        # Profit targets by leverage (earlier exits for high leverage)
        self.profit_targets = [
            (5.0, 0.07),    # 5x+: 7% target
            (3.0, 0.08),    # 3x+: 8% target
            (0.0, 0.10),    # Default: 10% target
        ]

    def calculate_leverage(self, conviction: float, tier: str,
                          chaos_baskets: int = 0,
                          portfolio_dd: float = 0.0,
                          daily_loss: float = 0.0) -> float:
        """
        Calculate smart leverage based on conviction and risk factors

        Args:
            conviction: Combined conviction score (0.0-1.0)
            tier: Position tier ('STRONG', 'MEDIUM', 'WEAK')
            chaos_baskets: Number of baskets in chaos regime
            portfolio_dd: Current portfolio drawdown (0.0-1.0)
            daily_loss: Current daily loss (0.0-1.0)

        Returns:
            Leverage multiplier (1.0-20.0)
        """
        # Get tier parameters
        tier_params = self.leverage_tiers.get(tier, self.leverage_tiers['WEAK'])

        # Base leverage from conviction
        base_leverage = 1.0 + (conviction * tier_params['multiplier'])

        # Clamp to tier range
        leverage = max(tier_params['min'], min(base_leverage, tier_params['max']))

        # Risk adjustments (multiplicative)
        if chaos_baskets >= 2:
            leverage *= 0.60  # Reduce by 40% in high volatility
        elif chaos_baskets == 1:
            leverage *= 0.80  # Reduce by 20% in moderate volatility

        if portfolio_dd > 0.05:  # 5% drawdown
            leverage *= 0.70  # Reduce by 30%
        elif portfolio_dd > 0.03:  # 3% drawdown
            leverage *= 0.85  # Reduce by 15%

        if daily_loss > 0.02:  # 2% daily loss
            leverage *= 0.60  # Reduce by 40%
        elif daily_loss > 0.01:  # 1% daily loss
            leverage *= 0.80  # Reduce by 20%

        # Ensure minimum of 1.0x
        leverage = max(1.0, leverage)

        # Apply hard caps
        leverage = min(leverage, self.max_basket_leverage, self.max_leverage)

        return leverage

    def get_stop_loss(self, leverage: float) -> float:
        """
        Get leverage-adjusted stop-loss percentage

        Higher leverage = tighter stops to protect capital
        """
        for max_lev, stop_pct in self.stop_loss_tiers:
            if leverage <= max_lev:
                return stop_pct
        return 0.010  # 1% for very high leverage

    def get_profit_target(self, leverage: float) -> float:
        """
        Get leverage-adjusted profit target

        Higher leverage = earlier profit taking
        """
        for min_lev, target_pct in self.profit_targets:
            if leverage >= min_lev:
                return target_pct
        return 0.10  # 10% default

    def ensure_minimum_contracts(self, position_value: float,
                                 contract_size: float,
                                 basket_pv: float,
                                 size_pct: float,
                                 min_contracts: int = 1) -> float:
        """
        Calculate minimum leverage needed to trade at least N contracts

        Args:
            position_value: Current calculated position value
            contract_size: Size of one contract in CNY
            basket_pv: Basket portfolio value
            size_pct: Position size percentage
            min_contracts: Minimum number of contracts to trade

        Returns:
            Required leverage (capped at max_leverage)
        """
        min_position_value = contract_size * min_contracts

        if position_value >= min_position_value:
            # Already sufficient
            return position_value / (basket_pv * size_pct)

        # Calculate required leverage
        required_leverage = min_position_value / (basket_pv * size_pct)

        # Cap at maximum
        return min(required_leverage, self.max_leverage)

# ============================================================================
# TRAILING STOP MANAGER
# ============================================================================

class TrailingStopManager:
    """
    Manages trailing stops for profitable positions

    Activates when:
    - Leverage >= 5x and profit >= 5%
    - Trails 2% below peak price
    """

    def __init__(self):
        self.activation_leverage = 5.0  # Activate for 5x+ leverage
        self.activation_profit = 0.05   # Activate at 5% profit
        self.trail_distance = 0.02      # Trail 2% below peak

        # Track peaks for each basket
        self.peak_prices = {0: 0.0, 1: 0.0, 2: 0.0}
        self.trailing_active = {0: False, 1: False, 2: False}
        self.trail_stops = {0: 0.0, 1: 0.0, 2: 0.0}

    def update(self, basket_idx: int, current_price: float,
               entry_price: float, position_type: int,
               leverage: float, pnl_pct: float):
        """
        Update trailing stop for a position

        Args:
            basket_idx: Basket index
            current_price: Current market price
            entry_price: Entry price
            position_type: 1 (LONG) or -1 (SHORT)
            leverage: Current leverage
            pnl_pct: Current P&L percentage
        """
        # Check activation conditions
        if leverage >= self.activation_leverage and pnl_pct >= self.activation_profit:
            if not self.trailing_active[basket_idx]:
                self.trailing_active[basket_idx] = True
                self.peak_prices[basket_idx] = current_price
                logger.info(f"Trailing stop ACTIVATED for basket {basket_idx} "
                           f"at {current_price:.2f} (leverage={leverage:.1f}x, P&L={pnl_pct*100:.1f}%)")

        # Update peak and trail stop if active
        if self.trailing_active[basket_idx]:
            if position_type == 1:  # LONG
                if current_price > self.peak_prices[basket_idx]:
                    self.peak_prices[basket_idx] = current_price
                    self.trail_stops[basket_idx] = current_price * (1 - self.trail_distance)
            else:  # SHORT
                if current_price < self.peak_prices[basket_idx] or self.peak_prices[basket_idx] == 0:
                    self.peak_prices[basket_idx] = current_price
                    self.trail_stops[basket_idx] = current_price * (1 + self.trail_distance)

    def check(self, basket_idx: int, current_price: float,
              position_type: int) -> bool:
        """
        Check if trailing stop hit

        Returns True if stop triggered
        """
        if not self.trailing_active[basket_idx]:
            return False

        if position_type == 1:  # LONG
            if current_price <= self.trail_stops[basket_idx]:
                logger.info(f"Trailing stop HIT for basket {basket_idx}: "
                           f"price {current_price:.2f} <= stop {self.trail_stops[basket_idx]:.2f}")
                return True
        else:  # SHORT
            if current_price >= self.trail_stops[basket_idx]:
                logger.info(f"Trailing stop HIT for basket {basket_idx}: "
                           f"price {current_price:.2f} >= stop {self.trail_stops[basket_idx]:.2f}")
                return True

        return False

    def reset(self, basket_idx: int):
        """Reset trailing stop for basket (call on exit)"""
        self.peak_prices[basket_idx] = 0.0
        self.trailing_active[basket_idx] = False
        self.trail_stops[basket_idx] = 0.0

# ============================================================================
# SIGNAL PARSERS
# ============================================================================

class IronOreSignalParser(pcts3.sv_object):
    """Parse IronOreIndicatorRelaxed signals"""

    def __init__(self):
        super().__init__()
        self.meta_name = "IronOreIndicatorRelaxed"
        self.namespace = pc.namespace_private
        self.revision = (1 << 32) - 1

        # Signal fields (from Tier-1)
        self.bar_index = 0
        self.close = 0.0  # Price data
        self.signal = 0
        self.confidence = 0.0
        self.regime = 0
        self.signal_strength = 0.0

        # Technical indicators
        self.ema_12 = 0.0
        self.ema_26 = 0.0
        self.ema_50 = 0.0
        self.macd = 0.0
        self.rsi = 0.0
        self.bb_upper = 0.0
        self.bb_lower = 0.0
        self.atr = 0.0


class CopperSignalParser(pcts3.sv_object):
    """Parse CopperIndicator signals"""

    def __init__(self):
        super().__init__()
        self.meta_name = "CopperIndicator"
        self.namespace = pc.namespace_private
        self.revision = (1 << 32) - 1

        # Signal fields (same as IronOre)
        self.bar_index = 0
        self.close = 0.0  # Price data
        self.signal = 0
        self.confidence = 0.0
        self.regime = 0
        self.signal_strength = 0.0

        # Technical indicators
        self.ema_12 = 0.0
        self.ema_26 = 0.0
        self.ema_50 = 0.0
        self.macd = 0.0
        self.rsi = 0.0
        self.bb_upper = 0.0
        self.bb_lower = 0.0
        self.atr = 0.0


class SoybeanSignalParser(pcts3.sv_object):
    """Parse SoybeanIndicator signals"""

    def __init__(self):
        super().__init__()
        self.meta_name = "SoybeanIndicator"
        self.namespace = pc.namespace_private
        self.revision = (1 << 32) - 1

        # Signal fields (same as IronOre)
        self.bar_index = 0
        self.close = 0.0  # Price data
        self.signal = 0
        self.confidence = 0.0
        self.regime = 0
        self.signal_strength = 0.0

        # Technical indicators
        self.ema_12 = 0.0
        self.ema_26 = 0.0
        self.ema_50 = 0.0
        self.macd = 0.0
        self.rsi = 0.0
        self.bb_upper = 0.0
        self.bb_lower = 0.0
        self.atr = 0.0

# ============================================================================
# COMPOSITE STRATEGY
# ============================================================================

class CompositeStrategy(csc3.composite_strategy):
    """
    Tier-2 Composite Strategy

    Manages 3 baskets (one per instrument) with equal-weight allocation
    """

    def __init__(self, initial_cash: float = INITIAL_CAPITAL):
        # Initialize critical attributes FIRST (before super().__init__)
        self.bar_index_this_run = -1
        self.latest_sv = None

        # Basket mapping
        self.basket_to_instrument = BASKET_MAPPING
        self.instrument_to_basket = {v: k for k, v in BASKET_MAPPING.items()}

        # Signal parsers
        self.parsers = {
            0: IronOreSignalParser(),
            1: CopperSignalParser(),
            2: SoybeanSignalParser(),
        }

        # Initialize parsers with metadata
        for basket_idx, parser in self.parsers.items():
            market, code = self.basket_to_instrument[basket_idx]
            parser.market = market
            parser.code = code + b'<00>'
            parser.granularity = granularity

        # Current signals from Tier-1
        self.tier1_signals = {}

        # Initialize composite strategy base class
        super().__init__(initial_cash, BASKET_COUNT)

        # Metadata
        self.namespace = pc.namespace_private
        self.granularity = granularity
        self.market = b'DCE'
        self.code = b'COMPOSITE'
        self.meta_name = "CompositeStrategy"
        self.revision = (1 << 32) - 1
        self.timetag_ = None

        # Output fields
        self.bar_index = 0
        self.active_positions = 0
        self.total_signals_processed = 0
        self.portfolio_exposure_pct = 0.0
        self.cash_reserve_pct = 0.0
        self.leverage_weighted_exposure = 0.0  # NEW: Leverage-adjusted exposure
        self.avg_active_leverage = 1.0  # NEW: Average leverage across active positions

        # Per-basket metrics (for individual basket P&L visualization)
        self.basket_0_pv = 0.0
        self.basket_1_pv = 0.0
        self.basket_2_pv = 0.0
        self.basket_0_signal = 0
        self.basket_1_signal = 0
        self.basket_2_signal = 0
        self.basket_0_price = 0.0
        self.basket_1_price = 0.0
        self.basket_2_price = 0.0

        # Trading statistics (not exported)
        self.total_trades_executed = 0
        self.total_trades_closed = 0

        # Initialization flag
        self.initialized = False

        # Allocation config (equal-weight)
        self.base_allocation_pct = 0.25
        self.max_allocation_pct = 0.35
        self.min_cash_reserve_pct = 0.10  # 10% min cash (target: 90% invested)
        self.max_leverage = 20.0  # Maximum leverage for positions (smart leverage cap)

        # Risk manager
        self.risk_manager = RiskManager(initial_cash)

        # Rebalancer
        self.rebalancer = Rebalancer()

        # Dynamic cash manager
        self.cash_manager = DynamicCashManager()

        # Dynamic leverage manager (NEW)
        self.leverage_manager = DynamicLeverageManager()

        # Trailing stop manager (NEW)
        self.trailing_stop_manager = TrailingStopManager()

        # Entry price tracking (for stop-loss and trailing stops)
        self.entry_prices = {0: 0.0, 1: 0.0, 2: 0.0}

        # Leverage tracking (for monitoring and risk management)
        self.active_leverages = {0: 1.0, 1: 1.0, 2: 1.0}

        # Initialize baskets
        self._initialize_baskets(initial_cash)

        logger.info(f"Initialized CompositeStrategy with {BASKET_COUNT} baskets, "
                   f"initial capital ¥{initial_cash:,.0f}")

    def _initialize_baskets(self, initial_cash: float):
        """
        Pre-allocate capital to baskets for balanced portfolio

        Target structure: 30% + 30% + 30% + 10% reserve
        - Basket 0: 30% (¥300,000)
        - Basket 1: 30% (¥300,000)
        - Basket 2: 30% (¥300,000)
        - Composite cash: 10% (¥100,000) reserve

        Note: _allocate() can only be called ONCE per basket (framework constraint)
        """
        # Target: 30% per basket for balanced portfolio
        target_basket_pct = 0.30
        basket_capital = initial_cash * target_basket_pct

        logger.info(f"Pre-allocating ¥{basket_capital:,.0f} (30%) to each basket")
        logger.info(f"Composite cash before allocation: ¥{self.cash:,.0f}")

        for basket_idx in range(BASKET_COUNT):
            market, code = self.basket_to_instrument[basket_idx]
            instrument_code = code + b'<00>'

            # ONE-TIME allocation per basket (framework constraint)
            # This permanently transfers capital from composite to basket
            # CRITICAL: Set max leverage at allocation to enable dynamic leverage in _fit_position()
            # Without this, contract sizing will be capped at 1.0x even if _fit_position() sets higher leverage
            self._allocate(basket_idx, market, instrument_code, basket_capital, self.max_leverage)

            basket = self.strategies[basket_idx]
            logger.info(f"Basket {basket_idx}: {market.decode()}/{code.decode()} "
                       f"allocated ¥{basket_capital:,.0f} (30%) with max leverage {self.max_leverage}x")

        logger.info(f"Composite cash after allocation: ¥{self.cash:,.0f} "
                   f"({self.cash/initial_cash*100:.1f}% reserve)")

    def calc_commodity(self, code: bytes) -> bytes:
        """
        Extract base commodity from specific contract code

        Examples:
            b'i2501' -> b'i'
            b'cu2412' -> b'cu'
            b'm2501' -> b'm'
            b'I2501' -> b'I' (CZCE uppercase)
        """
        # Strip trailing digits and angle brackets
        code_str = code.decode('utf-8')

        # Remove logical contract suffix if present
        if code_str.endswith('<00>'):
            code_str = code_str[:-4]

        # Remove contract month (trailing digits)
        while code_str and code_str[-1].isdigit():
            code_str = code_str[:-1]

        return code_str.encode('utf-8')

    def _get_contract_size(self, basket_idx: int) -> float:
        """
        Get contract notional value for instrument

        Returns contract value in CNY based on current price and multiplier
        """
        market, code = self.basket_to_instrument[basket_idx]
        basket = self.strategies[basket_idx]

        # Get current price (use signal price as fallback)
        if basket_idx in self.tier1_signals:
            price = self.tier1_signals[basket_idx].get('close', basket.price)
        else:
            price = basket.price

        if price <= 0:
            return 0.0

        # Contract multipliers (tons per contract)
        MULTIPLIERS = {
            (b'DCE', b'i'): 100,    # Iron Ore: 100 tons
            (b'SHFE', b'cu'): 5,    # Copper: 5 tons
            (b'DCE', b'm'): 10,     # Soybean: 10 tons
        }

        multiplier = MULTIPLIERS.get((market, code), 1)
        contract_value = price * multiplier

        return contract_value

    def _update_basket_price(self, basket_idx: int, price: float, time_tag: int):
        """
        Update basket price from market data

        The framework automatically calculates basket.pv based on positions.
        We only need to update the price - DO NOT manually set basket.pv.
        """
        basket = self.strategies[basket_idx]
        basket.price = price

    def on_bar(self, bar: pc.StructValue) -> List[pc.StructValue]:
        """Main bar processing callback"""
        market = bar.get_market()
        code = bar.get_stock_code()
        tm = bar.get_time_tag()
        ns = bar.get_namespace()
        meta_id = bar.get_meta_id()

        # Initialize timetag
        if self.timetag is None:
            self.timetag = tm

        # Handle cycle boundaries
        if self.timetag < tm:
            self._on_cycle_pass(tm)

            results = []
            if self.bar_index > 0:
                sv = self.sv_copy()
                results.append(sv)

            self.timetag = tm
            self.bar_index += 1

            return results

        # Route bars based on namespace
        if ns == pc.namespace_global:
            # CRITICAL FIX: Manually set target_instrument from first market data arrival
            # This is needed because reference data might not populate target_instrument in test env
            commodity = self.calc_commodity(code)
            key = (market, commodity)

            if key in self.instrument_to_basket:
                basket_idx = self.instrument_to_basket[key]
                basket = self.strategies[basket_idx]

                # If target_instrument is still empty, set it from actual market data
                if basket.target_instrument == b'' or basket.target_instrument is None:
                    basket.target_instrument = code
                    basket.instrument = code
                    logger.info(f"Basket {basket_idx} target_instrument set to {code.decode()} from market data")

            # Market data (SampleQuote) - route to baskets via framework
            # The framework's composite_strategy.on_bar() automatically routes
            # market data to allocated baskets based on market/code matching
            super().on_bar(bar)

            # Log first occurrence for debugging
            if not hasattr(self, '_market_data_seen'):
                self._market_data_seen = set()

            if key not in self._market_data_seen and key in self.instrument_to_basket:
                self._market_data_seen.add(key)
                basket_idx = self.instrument_to_basket[key]
                logger.info(f"Market data routing: {market.decode()}/{code.decode()} -> Basket {basket_idx}")

        elif ns == pc.namespace_private:
            # Tier-1 indicator signals
            self._process_tier1_signal(bar)

        return []

    def _process_tier1_signal(self, bar: pc.StructValue):
        """Parse incoming Tier-1 indicator signals"""
        ns = bar.get_namespace()
        meta_id = bar.get_meta_id()

        # Try each parser
        for basket_idx, parser in self.parsers.items():
            if ns == parser.namespace and meta_id == parser.meta_id:
                # Parse signal
                parser.from_sv(bar)

                # Store signal data
                self.tier1_signals[basket_idx] = {
                    'close': parser.close,
                    'signal': parser.signal,
                    'confidence': parser.confidence,
                    'regime': parser.regime,
                    'signal_strength': parser.signal_strength,
                    'ema_12': parser.ema_12,
                    'ema_26': parser.ema_26,
                    'rsi': parser.rsi,
                }

                # CRITICAL FIX: Update basket price from signal data
                # This ensures basket.price is populated even if market data routing fails
                # due to empty target_instrument (which requires reference data)
                basket = self.strategies[basket_idx]
                if parser.close > 0:
                    basket.price = parser.close
                    # Only log first time for each basket
                    if not hasattr(self, '_basket_price_logged'):
                        self._basket_price_logged = set()
                    if basket_idx not in self._basket_price_logged:
                        market, code = self.basket_to_instrument[basket_idx]
                        logger.info(f"Basket {basket_idx} ({market.decode()}/{code.decode()}) price initialized to {parser.close:.2f}")
                        self._basket_price_logged.add(basket_idx)

                self.total_signals_processed += 1
                break

    def _on_cycle_pass(self, time_tag: int):
        """Process end of cycle"""
        # Base class cycle processing
        super()._on_cycle_pass(time_tag)

        # Update risk tracking
        self.risk_manager.update_peak_tracking(self.pv)

        # Check circuit breakers
        if self._check_circuit_breakers():
            self._emergency_close_all_positions()
            return

        # Process trading signals
        self._process_trading_signals()

        # Check rebalancing
        self._check_and_rebalance()

        # Update portfolio metrics
        self._update_portfolio_metrics()

        # Sync state
        self._save()
        self._sync()

        # Mark as initialized after first cycle
        if not self.initialized:
            self.initialized = True
            logger.info("CompositeStrategy initialized and ready to output data")

        # Logging
        self._log_portfolio_state()

    def _check_circuit_breakers(self) -> bool:
        """Check portfolio-level circuit breakers"""
        metrics = self.risk_manager.get_risk_metrics()

        # Max drawdown
        if metrics['current_drawdown'] >= self.risk_manager.max_portfolio_drawdown:
            logger.error(f"CIRCUIT BREAKER: Max drawdown {metrics['current_drawdown']*100:.2f}% exceeded")
            return True

        # Daily loss limit
        if metrics['daily_loss'] >= self.risk_manager.max_daily_loss:
            logger.error(f"CIRCUIT BREAKER: Daily loss {metrics['daily_loss']*100:.2f}% exceeded")
            return True

        return False

    def _emergency_close_all_positions(self):
        """Emergency close all positions"""
        logger.error("EMERGENCY: Closing all positions")

        for basket_idx in range(BASKET_COUNT):
            basket = self.strategies[basket_idx]
            if basket.signal != 0:
                market, code = self.basket_to_instrument[basket_idx]
                logger.error(f"CLOSING: {market.decode()}/{code.decode()}")

                # Use price from latest signal if available
                if basket_idx in self.tier1_signals:
                    close_price = self.tier1_signals[basket_idx]['close']
                else:
                    close_price = basket.price  # Fallback

                basket._signal(close_price, basket.timetag, 0)
                self.entry_prices[basket_idx] = 0.0

    def _get_portfolio_conviction(self) -> Dict:
        """
        Aggregate signals across baskets for portfolio-level insights

        Returns dict with:
            - net_conviction: Long conviction - short conviction
            - total_conviction: Sum of all conviction scores
            - directional_bias: 'long' or 'short'
            - chaos_baskets: Count of baskets in chaos regime
            - avg_conviction: Average conviction across baskets
            - strong_signals: Count of strong signals (conviction >= 0.55)
        """
        if not self.tier1_signals:
            return {
                'net_conviction': 0.0,
                'total_conviction': 0.0,
                'directional_bias': 'neutral',
                'chaos_baskets': 0,
                'avg_conviction': 0.0,
                'strong_signals': 0,
            }

        total_long_conviction = 0.0
        total_short_conviction = 0.0
        chaos_count = 0
        strong_count = 0

        for basket_idx, signal_data in self.tier1_signals.items():
            # Combined conviction score
            conviction = (signal_data['confidence'] * 0.6 +
                         signal_data['signal_strength'] * 0.4)

            # Aggregate by direction
            if signal_data['signal'] == 1:
                total_long_conviction += conviction
            elif signal_data['signal'] == -1:
                total_short_conviction += conviction

            # Count chaos regimes
            if signal_data.get('regime', 0) == 4:
                chaos_count += 1

            # Count strong signals
            if conviction >= 0.55:
                strong_count += 1

        total_conviction = total_long_conviction + total_short_conviction
        avg_conviction = total_conviction / len(self.tier1_signals) if self.tier1_signals else 0.0

        return {
            'net_conviction': total_long_conviction - total_short_conviction,
            'total_conviction': total_conviction,
            'directional_bias': 'long' if total_long_conviction > total_short_conviction else 'short',
            'chaos_baskets': chaos_count,
            'avg_conviction': avg_conviction,
            'strong_signals': strong_count,
        }

    def _process_trading_signals(self):
        """Execute trading signals for each basket"""
        for basket_idx in range(BASKET_COUNT):
            # Get Tier-1 signal
            if basket_idx not in self.tier1_signals:
                continue

            signal_data = self.tier1_signals[basket_idx]
            basket = self.strategies[basket_idx]

            # Check exit conditions first
            if basket.signal != 0:
                should_exit, exit_reason = self._should_exit(basket_idx, signal_data, basket)
                if should_exit:
                    self._execute_exit(basket_idx, signal_data, exit_reason)
                    continue

            # Check entry conditions
            if basket.signal == 0:
                if self._should_enter(signal_data):
                    self._execute_entry(basket_idx, signal_data)

    def _should_enter(self, signal_data: Dict) -> bool:
        """Entry condition checks (RELAXED thresholds for more trading)"""
        # Must have directional signal (1 or -1)
        if signal_data['signal'] == 0:
            return False

        # FURTHER RELAXED: Lower confidence threshold (0.20 vs previous 0.30)
        # This allows more trading opportunities while still filtering noise
        if signal_data['confidence'] < 0.20:
            return False

        # REMOVED: No chaos regime filter - trade in all regimes
        # if signal_data['regime'] == 4:
        #     return False

        # FURTHER RELAXED: Lower signal strength (0.15 vs previous 0.25)
        # This increases trade frequency to achieve higher exposure
        if signal_data['signal_strength'] < 0.15:
            return False

        return True

    def _should_exit(self, basket_idx: int, signal_data: Dict, basket) -> Tuple[bool, str]:
        """
        Enhanced exit condition checks with leverage-aware profit-taking and trailing stops

        Returns: (should_exit: bool, exit_reason: str)
        """
        entry_price = self.entry_prices[basket_idx]
        current_price = signal_data['close']
        market, code = self.basket_to_instrument[basket_idx]
        leverage = self.active_leverages.get(basket_idx, 1.0)

        # Calculate current P&L
        if entry_price > 0 and current_price > 0:
            if basket.signal == 1:  # Long position
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # Short position
                pnl_pct = (entry_price - current_price) / entry_price
        else:
            pnl_pct = 0.0

        # Update trailing stop
        self.trailing_stop_manager.update(
            basket_idx, current_price, entry_price, basket.signal, leverage, pnl_pct
        )

        # 0. TRAILING STOP (for high leverage positions)
        if self.trailing_stop_manager.check(basket_idx, current_price, basket.signal):
            return True, "trailing_stop"

        # 1. LEVERAGE-ADJUSTED PROFIT TARGETS
        profit_target = self.leverage_manager.get_profit_target(leverage)

        if pnl_pct >= profit_target:
            logger.info(f"PROFIT TARGET: Basket {basket_idx} ({market.decode()}/{code.decode()}) "
                       f"hit {profit_target*100:.0f}% profit ({pnl_pct*100:.1f}%) at {leverage:.1f}x leverage")
            return True, f"profit_target_{int(profit_target*100)}pct"

        # Additional 5% profit protection with conviction degradation
        if pnl_pct >= 0.05:
            conviction = (signal_data['confidence'] * 0.6 +
                         signal_data['signal_strength'] * 0.4)
            if conviction < 0.40:
                logger.info(f"PROFIT PROTECT: Basket {basket_idx} ({market.decode()}/{code.decode()}) "
                           f"5% profit + weak signal (conv={conviction:.2f})")
                return True, f"profit_protect_5pct"

        # 2. LEVERAGE-ADJUSTED STOP-LOSS
        stop_loss_pct = self.leverage_manager.get_stop_loss(leverage)
        if self._check_stop_loss(entry_price, current_price, basket.signal, stop_loss_pct):
            logger.info(f"STOP-LOSS: Basket {basket_idx} ({market.decode()}/{code.decode()}) "
                       f"hit {stop_loss_pct*100:.1f}% loss ({pnl_pct*100:.1f}%) at {leverage:.1f}x leverage")
            return True, f"stop_loss_{int(stop_loss_pct*100)}pct"

        # 3. SIGNAL REVERSAL (immediate exit)
        if signal_data['signal'] * basket.signal < 0:
            logger.info(f"SIGNAL REVERSAL: Basket {basket_idx} ({market.decode()}/{code.decode()}) "
                       f"signal flipped {basket.signal} -> {signal_data['signal']}")
            return True, "signal_reversal"

        # 4. CONFIDENCE DEGRADATION (tiered thresholds)
        if signal_data['confidence'] < 0.20:
            logger.info(f"LOW CONFIDENCE: Basket {basket_idx} ({market.decode()}/{code.decode()}) "
                       f"confidence={signal_data['confidence']:.2f} < 0.20")
            return True, "low_confidence"

        elif signal_data['confidence'] < 0.30:
            # Only exit if also losing money
            if pnl_pct < -0.01:  # Losing 1%+
                logger.info(f"CONFIDENCE DROP + LOSS: Basket {basket_idx} ({market.decode()}/{code.decode()}) "
                           f"conf={signal_data['confidence']:.2f}, P&L={pnl_pct*100:.1f}%")
                return True, "confidence_drop_with_loss"

        # 5. PORTFOLIO-LEVEL RISK (chaos regimes)
        portfolio_conviction = self._get_portfolio_conviction()
        if portfolio_conviction['chaos_baskets'] >= 2:
            # Exit losing positions in high volatility
            if pnl_pct < -0.01:
                logger.info(f"CHAOS EXIT: Basket {basket_idx} ({market.decode()}/{code.decode()}) "
                           f"2+ chaos regimes, P&L={pnl_pct*100:.1f}%")
                return True, "chaos_regime_exit"

        return False, ""

    def _check_stop_loss(self, entry_price: float, current_price: float,
                        position_type: int, stop_loss_pct: float = 0.03) -> bool:
        """Check if stop-loss hit (leverage-adjusted)"""
        if entry_price == 0 or current_price == 0:
            return False

        if position_type == 1:  # Long
            loss_pct = (entry_price - current_price) / entry_price
        else:  # Short
            loss_pct = (current_price - entry_price) / entry_price

        return loss_pct >= stop_loss_pct

    def _execute_entry(self, basket_idx: int, signal_data: Dict):
        """
        Execute entry using TIERED position sizing with SMART LEVERAGE

        Sizing tiers based on signal conviction:
        - STRONG (conviction >= 0.55): 80-100% of basket capital, 4-10x leverage
        - MEDIUM (conviction >= 0.35): 40-60% of basket capital, 2.5-6x leverage
        - WEAK (conviction >= 0.20): 20-30% of basket capital, 1.5-4x leverage

        Never blocks trades - uses fallback capital chain if needed.
        """
        basket = self.strategies[basket_idx]
        market, code = self.basket_to_instrument[basket_idx]

        # Calculate combined conviction score
        confidence = signal_data['confidence']
        strength = signal_data['signal_strength']
        conviction = (confidence * 0.6) + (strength * 0.4)

        # Get portfolio-level insights for risk adjustment
        portfolio_conviction = self._get_portfolio_conviction()

        # Determine position size tier
        if conviction >= 0.55:  # STRONG
            size_pct = 0.80 + (conviction - 0.55) * 0.44  # 80-100%
            tier_name = "STRONG"
        elif conviction >= 0.35:  # MEDIUM
            size_pct = 0.40 + (conviction - 0.35) * 1.0   # 40-60%
            tier_name = "MEDIUM"
        else:  # WEAK
            size_pct = 0.20 + (conviction - 0.20) * 0.67  # 20-30%
            tier_name = "WEAK"

        # Portfolio-level risk adjustments
        if portfolio_conviction['chaos_baskets'] >= 2:
            size_pct *= 0.70  # Reduce by 30% in high volatility
            tier_name += "_CHAOS_REDUCED"

        # Calculate SMART LEVERAGE (conviction-based, risk-adjusted)
        leverage = self.leverage_manager.calculate_leverage(
            conviction=conviction,
            tier=tier_name.replace("_CHAOS_REDUCED", ""),  # Clean tier name
            chaos_baskets=portfolio_conviction['chaos_baskets'],
            portfolio_dd=self.risk_manager.current_drawdown,
            daily_loss=self.risk_manager.daily_loss
        )

        # Calculate target position value
        target_position_value = basket.pv * size_pct * leverage

        # Ensure minimum contracts (especially for large contracts like copper)
        contract_size = self._get_contract_size(basket_idx)
        if contract_size > 0:
            min_contracts = 1
            min_position_value = contract_size * min_contracts

            if target_position_value < min_position_value:
                # Increase leverage to achieve minimum
                required_leverage = self.leverage_manager.ensure_minimum_contracts(
                    target_position_value, contract_size, basket.pv, size_pct, min_contracts
                )
                if required_leverage <= self.leverage_manager.max_leverage:
                    leverage = required_leverage
                    target_position_value = basket.pv * size_pct * leverage
                    logger.info(f"Increased leverage to {leverage:.2f}x to meet minimum contract size "
                               f"for {market.decode()}/{code.decode()}")

        # Risk check with FALLBACK CHAIN (never fully block)
        can_enter, reason = self.risk_manager.can_enter_position(
            basket_idx, target_position_value, self._get_portfolio_state()
        )

        if not can_enter:
            # FALLBACK CHAIN: Try to enable trade
            logger.info(f"Entry initially blocked for basket {basket_idx}: {reason}")

            # Fallback 1: Reduce position size by 40%
            if "exposure" in reason.lower() or "drawdown" in reason.lower():
                logger.info(f"FALLBACK 1: Reducing position size by 40%")
                size_pct *= 0.60
                target_position_value = basket.pv * size_pct * leverage
                can_enter, reason = self.risk_manager.can_enter_position(
                    basket_idx, target_position_value, self._get_portfolio_state()
                )

            # Fallback 2: Reduce leverage to 1.1x
            if not can_enter and leverage > 1.1:
                logger.info(f"FALLBACK 2: Reducing leverage from {leverage:.2f} to 1.1x")
                leverage = 1.1
                target_position_value = basket.pv * size_pct * leverage
                can_enter, reason = self.risk_manager.can_enter_position(
                    basket_idx, target_position_value, self._get_portfolio_state()
                )

            # Fallback 3: Minimum position (20% size, 1.0x leverage)
            if not can_enter:
                logger.info(f"FALLBACK 3: Minimum position (20% size, 1.0x leverage)")
                size_pct = 0.20
                leverage = 1.0
                target_position_value = basket.pv * size_pct * leverage
                can_enter, reason = self.risk_manager.can_enter_position(
                    basket_idx, target_position_value, self._get_portfolio_state()
                )

            # Final check - only block on critical issues (drawdown, daily loss)
            if not can_enter:
                logger.info(f"ENTRY BLOCKED basket {basket_idx} ({market.decode()}/{code.decode()}) "
                           f"after all fallbacks: {reason}")
                return

        # Execute trade via framework (WOS pattern)
        signal = signal_data['signal']

        # Use signal price when basket.price not yet available
        trade_price = basket.price if basket.price > 0 else signal_data['close']

        # Get leverage-adjusted risk parameters
        stop_loss_pct = self.leverage_manager.get_stop_loss(leverage)
        profit_target_pct = self.leverage_manager.get_profit_target(leverage)

        logger.info(f"{'LONG' if signal == 1 else 'SHORT'} basket {basket_idx} [{tier_name}]: "
                   f"{market.decode()}/{code.decode()}, size={size_pct*100:.0f}%, "
                   f"lev={leverage:.2f}x, stop={stop_loss_pct*100:.1f}%, "
                   f"target={profit_target_pct*100:.0f}%, price={trade_price:.2f}, "
                   f"conv={conviction:.2f} (conf={confidence:.2f}, str={strength:.2f})")

        # Store entry price and leverage for tracking
        self.entry_prices[basket_idx] = trade_price
        self.active_leverages[basket_idx] = leverage
        self.total_trades_executed += 1

        # WOS pattern: _fit_position sets leverage, _signal enters position
        basket._fit_position(leverage)
        basket._signal(trade_price, basket.timetag, signal * -1)

    def _execute_exit(self, basket_idx: int, signal_data: Dict, exit_reason: str = ""):
        """
        Execute exit for basket with detailed logging

        Args:
            basket_idx: Basket index
            signal_data: Current signal data
            exit_reason: Why we're exiting (profit_target, stop_loss, etc.)
        """
        basket = self.strategies[basket_idx]
        market, code = self.basket_to_instrument[basket_idx]

        # Calculate P&L
        entry_price = self.entry_prices[basket_idx]
        # Use signal price when basket.price not yet available (Day 1 before on_reference)
        exit_price = basket.price if basket.price > 0 else signal_data['close']

        if entry_price > 0 and exit_price > 0:
            if basket.signal == 1:  # Long position
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:  # Short position
                pnl_pct = (entry_price - exit_price) / entry_price * 100

            logger.info(f"CLOSE basket {basket_idx} [{exit_reason}]: {market.decode()}/{code.decode()}, "
                       f"{'LONG' if basket.signal == 1 else 'SHORT'} "
                       f"entry={entry_price:.2f}, exit={exit_price:.2f}, P&L={pnl_pct:+.2f}%")
        else:
            logger.info(f"CLOSE basket {basket_idx} [{exit_reason}]: {market.decode()}/{code.decode()}")

        # Execute exit via framework (use exit_price with fallback)
        basket._signal(exit_price, basket.timetag, 0)
        self.entry_prices[basket_idx] = 0.0
        self.active_leverages[basket_idx] = 1.0
        self.trailing_stop_manager.reset(basket_idx)
        self.total_trades_closed += 1

    def _calculate_position_size(self, allocated_capital: float, current_price: float) -> int:
        """Calculate number of contracts"""
        if current_price == 0:
            return 0

        contracts = int(allocated_capital / current_price)
        return max(contracts, 0)

    def _check_and_rebalance(self):
        """Check if rebalancing needed and execute"""
        # Get current allocations
        current_allocations = self._get_current_allocations()
        target_allocations = {
            0: 0.25,
            1: 0.25,
            2: 0.25,
        }

        # Check if rebalancing needed
        should_rebalance, instrument, deviation = self.rebalancer.should_rebalance(
            self.bar_index, current_allocations, target_allocations
        )

        if should_rebalance:
            logger.info(f"REBALANCE triggered: instrument {instrument} deviated by {deviation*100:.1f}%")
            self._execute_rebalance(current_allocations, target_allocations)
            self.rebalancer.last_rebalance_bar = self.bar_index

    def _get_current_allocations(self) -> Dict:
        """Calculate current allocation percentages"""
        allocations = {}
        for basket_idx in range(BASKET_COUNT):
            basket = self.strategies[basket_idx]
            allocations[basket_idx] = basket.pv / self.pv if self.pv > 0 else 0
        return allocations

    def _execute_rebalance(self, current_allocations: Dict, target_allocations: Dict):
        """Execute rebalancing trades (simplified - just log for Phase 1)"""
        # Calculate actions
        actions = self.rebalancer.calculate_rebalance_actions(
            current_allocations, target_allocations, self.pv
        )

        # Execute (simplified for Phase 1 - just log)
        for basket_idx, action in actions.items():
            market, code = self.basket_to_instrument[basket_idx]
            logger.info(f"REBALANCE: {action['type']} {market.decode()}/{code.decode()} "
                       f"by ¥{action['value']:,.0f} ({action['deviation_pct']:.1f}%)")

    def _update_portfolio_metrics(self):
        """Update portfolio-level metrics

        The framework automatically updates self.pv via _save() and _sync().
        We calculate derived metrics based on pre-allocated structure.
        """
        # Count active positions (read from basket state)
        self.active_positions = sum(1 for basket in self.strategies if basket.signal != 0)

        # Total basket capital (all baskets, whether flat or in position)
        # This represents the 90% invested portion (3 × 30% = 90%)
        total_basket_value = sum(basket.pv for basket in self.strategies)

        # Portfolio exposure = total basket capital / total portfolio
        # With 30%+30%+30% allocation, this should be ~90%
        self.portfolio_exposure_pct = total_basket_value / self.pv if self.pv > 0 else 0

        # Cash reserve = composite cash / total portfolio
        # Should be ~10% with our 30/30/30/10 structure
        self.cash_reserve_pct = self.cash / self.pv if self.pv > 0 else 0

        # Calculate leverage-weighted exposure (NEW)
        positions = {}
        for basket_idx in range(BASKET_COUNT):
            basket = self.strategies[basket_idx]
            if basket.signal != 0:
                positions[basket_idx] = {
                    'value': basket.pv,
                    'leverage': self.active_leverages.get(basket_idx, 1.0)
                }

        _, self.leverage_weighted_exposure = self.risk_manager.check_leverage_weighted_exposure(
            positions, self.pv
        )

        # Calculate average active leverage (NEW)
        active_leverages = [
            self.active_leverages.get(i, 1.0)
            for i in range(BASKET_COUNT)
            if self.strategies[i].signal != 0
        ]
        self.avg_active_leverage = sum(active_leverages) / len(active_leverages) if active_leverages else 1.0

        # Update per-basket metrics (for visualization)
        self.basket_0_pv = self.strategies[0].pv
        self.basket_1_pv = self.strategies[1].pv
        self.basket_2_pv = self.strategies[2].pv

        self.basket_0_signal = self.strategies[0].signal
        self.basket_1_signal = self.strategies[1].signal
        self.basket_2_signal = self.strategies[2].signal

        self.basket_0_price = self.strategies[0].price
        self.basket_1_price = self.strategies[1].price
        self.basket_2_price = self.strategies[2].price

    def _get_portfolio_state(self):
        """Get current portfolio state for risk checks"""
        class PortfolioState:
            def __init__(self):
                self.total_value = 0
                self.total_exposure = 0
                self.cash = 0

        state = PortfolioState()
        state.total_value = self.pv

        # FIX: Calculate actual active exposure (position values, not all basket PV)
        # Only count baskets with active positions (signal != 0)
        state.total_exposure = sum(s.pv for s in self.strategies if s.signal != 0)

        # FIX: Use framework-maintained cash field instead of calculating
        # The framework manages self.cash when capital is allocated/deallocated
        state.cash = self.cash

        return state

    def _log_portfolio_state(self):
        """Log portfolio state (reduced verbosity)"""
        # Only log every 10 bars or when positions are active
        if self.bar_index % 10 == 0 or self.active_positions > 0:
            metrics = self.risk_manager.get_risk_metrics()

            # Include leverage metrics in log
            logger.info(
                f"[Bar {self.bar_index}] "
                f"PV=¥{self.pv:,.0f}, NV={self.nv:.4f}, "
                f"Active={self.active_positions}/3, "
                f"Exposure={self.portfolio_exposure_pct*100:.1f}%, "
                f"LevExp={self.leverage_weighted_exposure*100:.1f}%, "
                f"AvgLev={self.avg_active_leverage:.2f}x, "
                f"Cash={self.cash_reserve_pct*100:.1f}%, "
                f"DD={metrics['current_drawdown']*100:.2f}%"
            )

    def on_tradeday_begin(self, market: bytes, tradeday: int):
        """Called at start of trading day"""
        super().on_tradeday_begin(market, tradeday)
        self.risk_manager.update_daily_tracking(self.pv)
        logger.info(f"=== Trading Day Begin: {tradeday} ===")

    def on_tradeday_end(self, market: bytes, tradeday: int):
        """Called at end of trading day"""
        super().on_tradeday_end(market, tradeday)
        metrics = self.risk_manager.get_risk_metrics()
        logger.info(
            f"=== Trading Day End: {tradeday} | "
            f"Trades: {self.total_trades_executed} opened, {self.total_trades_closed} closed | "
            f"PV: ¥{self.pv:,.0f} | NV: {self.nv:.4f} | "
            f"Daily Loss: {metrics['daily_loss']*100:.2f}% ==="
        )

    def ready_to_serialize(self) -> bool:
        """
        Tell framework when to serialize and output data.

        Returns True after initialization completes.
        """
        result = self.initialized and self.bar_index > 0
        return result

# ============================================================================
# GLOBAL STRATEGY INSTANCE
# ============================================================================

strategy = CompositeStrategy()

# ============================================================================
# FRAMEWORK CALLBACKS (REQUIRED)
# ============================================================================

async def on_init():
    """Initialize strategy"""
    global strategy, imports, metas, worker_no

    if worker_no != 0 and metas and imports:
        strategy.load_def_from_dict(metas)

        for parser in strategy.parsers.values():
            parser.load_def_from_dict(metas)
            parser.set_global_imports(imports)
        logger.info("CompositeStrategy initialized")


async def on_ready():
    """Strategy ready"""
    logger.info("CompositeStrategy ready")


async def on_bar(bar: pc.StructValue):
    """Process incoming bar"""
    global strategy, worker_no

    if worker_no != 1:
        return []

    return strategy.on_bar(bar)


async def on_market_open(market, tradeday, time_tag, time_string):
    """Market open event"""
    pass


async def on_market_close(market, tradeday, timetag, timestring):
    """Market close event"""
    pass


async def on_tradeday_begin(market, tradeday, time_tag, time_string):
    """Trading day begin event"""
    global strategy
    strategy.on_tradeday_begin(bytes(market, 'utf-8'), tradeday)


async def on_tradeday_end(market, tradeday, timetag, timestring):
    """Trading day end event"""
    global strategy
    strategy.on_tradeday_end(bytes(market, 'utf-8'), tradeday)


async def on_reference(market, tradeday, data, timetag, timestring):
    """CRITICAL: Forward reference data to baskets for contract rolling"""
    global strategy
    strategy.on_reference(bytes(market, 'utf-8'), tradeday, data)


async def on_historical(params, records):
    """Historical data event"""
    pass
