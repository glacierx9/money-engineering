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
        self.min_cash_reserve = 0.15   # 15% min cash

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
        # 1. Position size limit
        position_pct = proposed_size / portfolio_state.total_value if portfolio_state.total_value > 0 else 0
        if position_pct > self.max_position_pct:
            return False, f"Position size {position_pct*100:.1f}% exceeds limit {self.max_position_pct*100:.1f}%"

        # 2. Total exposure limit
        total_exposure = portfolio_state.total_exposure + proposed_size
        exposure_pct = total_exposure / portfolio_state.total_value if portfolio_state.total_value > 0 else 0
        if exposure_pct > self.max_total_exposure:
            return False, f"Total exposure {exposure_pct*100:.1f}% exceeds limit {self.max_total_exposure*100:.1f}%"

        # 3. Cash reserve check
        remaining_cash = portfolio_state.cash - proposed_size
        cash_pct = remaining_cash / portfolio_state.total_value if portfolio_state.total_value > 0 else 0
        if cash_pct < self.min_cash_reserve:
            return False, f"Cash reserve {cash_pct*100:.1f}% below minimum {self.min_cash_reserve*100:.1f}%"

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
        self.min_cash_reserve_pct = 0.15
        self.max_leverage = 1.4  # Maximum leverage for positions

        # Risk manager
        self.risk_manager = RiskManager(initial_cash)

        # Rebalancer
        self.rebalancer = Rebalancer()

        # Entry price tracking (for stop-loss)
        self.entry_prices = {0: 0.0, 1: 0.0, 2: 0.0}

        # Initialize baskets
        self._initialize_baskets(initial_cash)

        logger.info(f"Initialized CompositeStrategy with {BASKET_COUNT} baskets, "
                   f"initial capital ¥{initial_cash:,.0f}")

    def _initialize_baskets(self, initial_cash: float):
        """Allocate baskets to instruments"""
        base_capital = initial_cash * self.base_allocation_pct

        logger.info(f"Initializing baskets with ¥{base_capital:,.0f} each")
        logger.info(f"Composite cash before allocation: ¥{self.cash:,.0f}")

        for basket_idx in range(BASKET_COUNT):
            market, code = self.basket_to_instrument[basket_idx]
            instrument_code = code + b'<00>'

            # Allocate basket
            self._allocate(basket_idx, market, instrument_code, base_capital, 1.0)

            basket = self.strategies[basket_idx]
            logger.info(f"Basket {basket_idx}: {market.decode()}/{code.decode()} "
                       f"allocated ¥{base_capital:,.0f}, basket.cash=¥{basket.cash:,.0f}, basket.pv=¥{basket.pv:,.0f}")

        logger.info(f"Composite cash after allocation: ¥{self.cash:,.0f}")

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
                if self._should_exit(basket_idx, signal_data, basket):
                    self._execute_exit(basket_idx, signal_data)
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

        # RELAXED: Moderate confidence (0.30 vs previous 0.50)
        if signal_data['confidence'] < 0.30:
            return False

        # REMOVED: No chaos regime filter - trade in all regimes
        # if signal_data['regime'] == 4:
        #     return False

        # RELAXED: Signal strength (0.25 vs previous 0.40)
        if signal_data['signal_strength'] < 0.25:
            return False

        return True

    def _should_exit(self, basket_idx: int, signal_data: Dict, basket) -> bool:
        """Exit condition checks"""
        # Stop-loss check
        entry_price = self.entry_prices[basket_idx]
        current_price = signal_data['close']

        if self._check_stop_loss(entry_price, current_price, basket.signal):
            market, code = self.basket_to_instrument[basket_idx]
            logger.info(f"STOP-LOSS: Basket {basket_idx} ({market.decode()}/{code.decode()})")
            return True

        # Signal reversal
        if signal_data['signal'] * basket.signal < 0:
            market, code = self.basket_to_instrument[basket_idx]
            logger.info(f"EXIT: Signal reversed for basket {basket_idx} ({market.decode()}/{code.decode()})")
            return True

        # Confidence dropped
        if signal_data['confidence'] < 0.30:
            market, code = self.basket_to_instrument[basket_idx]
            logger.info(f"EXIT: Confidence dropped for basket {basket_idx} ({market.decode()}/{code.decode()})")
            return True

        # REMOVED: No chaos regime exit filter - hold through all regimes
        # if signal_data['regime'] == 4:
        #     market, code = self.basket_to_instrument[basket_idx]
        #     logger.info(f"EXIT: Chaos regime for basket {basket_idx} ({market.decode()}/{code.decode()})")
        #     return True

        return False

    def _check_stop_loss(self, entry_price: float, current_price: float, position_type: int) -> bool:
        """Check if stop-loss hit (3%)"""
        if entry_price == 0 or current_price == 0:
            return False

        if position_type == 1:  # Long
            loss_pct = (entry_price - current_price) / entry_price
        else:  # Short
            loss_pct = (current_price - entry_price) / entry_price

        return loss_pct >= 0.03

    def _execute_entry(self, basket_idx: int, signal_data: Dict):
        """Execute entry for basket"""
        basket = self.strategies[basket_idx]
        market, code = self.basket_to_instrument[basket_idx]

        # Calculate leverage based on confidence
        confidence = signal_data['confidence']
        leverage = 1.0 + (confidence * 1.5)  # 1.0 to 2.5
        leverage = min(leverage, self.max_leverage)

        # Risk check
        allocated_capital = self.pv * self.base_allocation_pct
        can_enter, reason = self.risk_manager.can_enter_position(
            basket_idx, allocated_capital * leverage, self._get_portfolio_state()
        )

        if not can_enter:
            logger.info(f"ENTRY BLOCKED basket {basket_idx} ({market.decode()}/{code.decode()}): {reason}")
            return

        # Execute trade via framework (WOS pattern)
        signal = signal_data['signal']

        # Use signal price when basket.price not yet available (Day 1 before on_reference)
        trade_price = basket.price if basket.price > 0 else signal_data['close']

        logger.info(f"{'LONG' if signal == 1 else 'SHORT'} basket {basket_idx}: "
                   f"{market.decode()}/{code.decode()}, lev={leverage:.2f}, "
                   f"price={trade_price:.2f}, conf={confidence:.2f}")

        # Store entry price for stop-loss tracking
        self.entry_prices[basket_idx] = trade_price
        self.total_trades_executed += 1

        # WOS pattern: _fit_position BEFORE _signal, use trade_price, invert signal
        basket._fit_position(leverage)
        basket._signal(trade_price, basket.timetag, signal * -1)

    def _execute_exit(self, basket_idx: int, signal_data: Dict):
        """Execute exit for basket"""
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

            logger.info(f"CLOSE basket {basket_idx}: {market.decode()}/{code.decode()}, "
                       f"entry={entry_price:.2f}, exit={exit_price:.2f}, P&L={pnl_pct:+.2f}%")
        else:
            logger.info(f"CLOSE basket {basket_idx}: {market.decode()}/{code.decode()}")

        # Execute exit via framework (use exit_price with fallback)
        basket._signal(exit_price, basket.timetag, 0)
        self.entry_prices[basket_idx] = 0.0
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
        We only need to calculate derived metrics.
        """
        # Count active positions (read from basket state)
        self.active_positions = sum(1 for basket in self.strategies if basket.signal != 0)

        # Calculate exposure (baskets with active positions)
        total_position_value = sum(basket.pv for basket in self.strategies if basket.signal != 0)

        self.portfolio_exposure_pct = total_position_value / self.pv if self.pv > 0 else 0

        # Cash reserve
        self.cash_reserve_pct = 1.0 - self.portfolio_exposure_pct

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
        state.total_exposure = sum(s.pv for s in self.strategies if s.signal != 0)
        state.cash = self.pv - state.total_exposure

        return state

    def _log_portfolio_state(self):
        """Log portfolio state (reduced verbosity)"""
        # Only log every 10 bars or when positions are active
        if self.bar_index % 10 == 0 or self.active_positions > 0:
            metrics = self.risk_manager.get_risk_metrics()

            logger.info(
                f"[Bar {self.bar_index}] "
                f"PV=¥{self.pv:,.0f}, NV={self.nv:.4f}, "
                f"Active={self.active_positions}/3, "
                f"Exposure={self.portfolio_exposure_pct*100:.1f}%, "
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
