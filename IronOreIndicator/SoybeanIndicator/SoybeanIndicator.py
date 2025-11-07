#!/usr/bin/env python3
# coding=utf-8
"""
SoybeanIndicator - Short-Term Position Accumulation Strategy

SHORT-TERM TRADING STRATEGY with 7-Indicator Market Regime Detection:
- Triple EMA (15/30/60) - Slower EMAs for more stable agricultural commodity trading
- MACD (15/30/9) - Momentum confirmation & reversal detection
- RSI (14) - Mean reversion signals
- Bollinger Bands (20, 2σ) - Price extremes & volatility
- ATR (14) - Volatility measurement
- BB Width - Volatility confirmation
- Volume EMA (20) - Liquidity confirmation (1.1x-1.2x)

Market: DCE (Dalian Commodity Exchange)
Security: m (Soybean Meal)
Granularity: 900s (15 minutes)

Market Regimes & Trading Logic:
1. Strong Uptrend - BUY dips (3/5), SELL quickly (2/4) - Short holds
2. Strong Downtrend - BUY on REVERSALS (3/6), SELL aggressively (1/3)
3. Sideways/Ranging - BUY dips (2/4), SELL quickly (2/4) - Quick scalps
4. High Volatility Chaos - BUY on RECOVERY (3/5), SELL if in position

Key Features:
- LONG ONLY with POSITION ACCUMULATION
- 70% MAX ALLOCATION (30% cash reserve)
- SHORT-TERM FOCUSED (quick profit taking)
- BUY THE DIP + REVERSALS (add 50 contracts per signal)
- SELL ALL on exits (take profits quickly)
- MA CROSSOVER reversal detection
- PRICE ACTION patterns (bar position, bounces)
- 1-bar cooldown (aggressive re-entry)
- Average entry price tracking

Parameter Tuning vs Iron Ore:
- Slower EMAs (15/30/60 vs 12/26/50) - Agricultural commodities are less volatile
- Standard RSI (14-period) - Same as IronOre
- Other parameters same as CopperIndicator (proven system)
"""

import math
import pycaitlyn as pc
import pycaitlynts3 as pcts3
import pycaitlynutils3 as pcu3
from typing import List

# Framework globals (REQUIRED)
use_raw = True
overwrite = False  # Set to False for production
granularity = 900
schema = None
max_workers = 1
worker_no = None
exports = {}
imports = {}
metas = {}
logger = pcu3.vanilla_logger()


class SampleQuote(pcts3.sv_object):
    """Parse SampleQuote (OHLCV) data from global namespace"""

    def __init__(self):
        super().__init__()

        # Metadata - CONSTANTS
        self.meta_name = "SampleQuote"
        self.namespace = pc.namespace_global
        self.revision = (1 << 32) - 1
        self.granularity = 900

        # OHLCV fields (automatically populated by from_sv)
        self.open = None
        self.high = None
        self.low = None
        self.close = None
        self.volume = None
        self.turnover = None


class SoybeanIndicator(pcts3.sv_object):
    """
    Soybean Meal Multi-Indicator Confirmation System

    Generates regime-aware buy/sell signals using multi-layer confirmation
    Tuned for DCE Soybean Meal futures with slower parameters for agricultural commodities
    """

    def __init__(self):
        super().__init__()

        # Metadata - CONSTANTS (never change after initialization)
        self.meta_name = "SoybeanIndicator"
        self.namespace = pc.namespace_private
        self.granularity = 900
        self.market = b'DCE'
        self.code = b'm<00>'
        self.revision = (1 << 32) - 1

        # State variables (automatically persisted by framework)
        self.bar_index = 0
        self.timetag = None
        self.initialized = False

        # Alpha parameters (smoothing factors) - SLOWER for Soybean Meal
        self.alpha_12 = 2.0 / 16.0   # 15-period (SLOWER than Copper's 12)
        self.alpha_26 = 2.0 / 31.0   # 30-period (SLOWER than Copper's 24)
        self.alpha_50 = 2.0 / 61.0   # 60-period (SLOWER than Copper's 45)
        self.alpha_9 = 2.0 / 10.0    # 9-period (same)
        self.alpha_14 = 2.0 / 15.0   # 14-period RSI (same as IronOre)
        self.alpha_20 = 2.0 / 21.0   # 20-period (same)

        # Current bar OHLCV
        self.open = 0.0
        self.high = 0.0
        self.low = 0.0
        self.close = 0.0
        self.volume = 0.0

        # Triple EMA states
        self.ema_12 = 0.0
        self.ema_26 = 0.0  # Actually 30-period due to alpha_26 change
        self.ema_50 = 0.0  # Actually 60-period due to alpha_50 change

        # MACD states
        self.macd = 0.0
        self.macd_signal = 0.0
        self.macd_histogram = 0.0

        # RSI states (online algorithm via gain/loss EMAs)
        self.rsi = 50.0
        self.gain_ema = 0.0
        self.loss_ema = 0.0
        self.prev_close = 0.0

        # Bollinger Band states (Welford's online variance)
        self.bb_n = 0
        self.bb_mean = 0.0
        self.bb_m2 = 0.0  # Sum of squared differences
        self.bb_variance = 0.0
        self.bb_std_dev = 0.0
        self.bb_upper = 0.0
        self.bb_middle = 0.0
        self.bb_lower = 0.0
        self.bb_width = 0.0
        self.bb_width_pct = 0.0

        # ATR states
        self.atr = 0.0
        self.mean_atr = 0.0
        self.atr_count = 0
        self.prev_close_atr = 0.0

        # Volume EMA
        self.volume_ema = 0.0

        # Regime tracking
        self.regime = 3  # Default to ranging
        self.regime_prev = 3

        # Signal outputs
        self.signal = 0          # -1 (sell), 0 (neutral), 1 (buy)
        self.confidence = 0.0    # Signal confidence [0.0, 1.0]
        self.position_state = 0  # 0 = flat, 1 = in position
        self.last_exit_bar = -999  # Track last exit to prevent immediate re-entry
        self.cooldown_bars = 1   # Minimum bars to wait after exit (short-term focused)

        # Position accumulation fields
        self.signal_strength = 0.0    # 0.0-1.0 conviction level
        self.desired_contracts = 0    # Target position size
        self.contracts_to_add = 0     # Contracts to add this signal (50 or 100)

        # Portfolio tracking (simulation with $1,000,000 initial capital)
        self.initial_capital = 1000000.0  # Starting capital (CNY)
        self.cash = 1000000.0             # Current cash
        self.contracts_held = 0           # Number of contracts in position
        self.entry_price = 0.0            # Average entry price for current position
        self.portfolio_value = 1000000.0  # PV = cash + position_value
        self.contracts_per_trade = 50     # Contracts per signal (increased from 10)
        self.max_allocation = 0.70        # Max 70% of portfolio can be invested
        self.reserve_allocation = 0.30    # Keep 30% in reserve

        # Dependency sv_objects
        self.sq = SampleQuote()

        # Control persistence
        self.persistent = True

    def initialize(self, imports, metas):
        """Initialize metadata schemas for all sv_objects"""
        self.load_def_from_dict(metas)
        self.set_global_imports(imports)

        # Initialize dependencies
        self.sq.load_def_from_dict(metas)
        self.sq.set_global_imports(imports)

    def on_bar(self, bar: pc.StructValue) -> List[pc.StructValue]:
        """
        Process incoming market data bars

        Args:
            bar: StructValue containing market data

        Returns:
            List of StructValue outputs (empty list if no output this cycle)
        """
        ret = []  # ALWAYS return list

        # Extract metadata
        market = bar.get_market()
        code = bar.get_stock_code()
        tm = bar.get_time_tag()
        ns = bar.get_namespace()
        meta_id = bar.get_meta_id()

        # Filter for our market
        if market != self.market:
            return ret

        # Route to appropriate sv_object
        if self.sq.namespace == ns and self.sq.meta_id == meta_id:
            # Filter for logical contracts only (ending in <00>)
            if code.endswith(b'<00>'):
                # Set metadata before from_sv
                self.sq.market = market
                self.sq.code = code
                self.sq.granularity = bar.get_granularity()

                # Parse data into sv_object
                self.sq.from_sv(bar)

                # Handle cycle boundaries
                if self.timetag is None:
                    self.timetag = tm

                if self.timetag < tm:
                    # New cycle - process previous cycle's data
                    self._on_cycle_pass(tm)

                    # Serialize state if ready
                    if self.ready_to_serialize():
                        ret.append(self.copy_to_sv())

                    # Update for next cycle
                    self.timetag = tm
                    self.bar_index += 1

        return ret  # ALWAYS return list

    def _on_cycle_pass(self, time_tag):
        """
        Process cycle - calculate indicators and generate signals

        Pipeline:
        1. Extract OHLCV data
        2. Initialize on first bar
        3. Update all indicators (online algorithms)
        4. Detect market regime
        5. Generate trading signal
        6. Log signal if generated
        """

        # Extract OHLCV data
        self.open = float(self.sq.open)
        self.high = float(self.sq.high)
        self.low = float(self.sq.low)
        self.close = float(self.sq.close)
        self.volume = float(self.sq.volume)

        # Initialize on first bar
        if not self.initialized:
            self._initialize_state()
            return

        # Update indicators (order matters for dependencies)
        self._update_emas(self.close)
        self._update_macd()
        self._update_rsi(self.close)
        self._update_bollinger_bands(self.close)
        self._update_atr(self.high, self.low, self.close)
        self._update_volume_ema(self.volume)

        # Detect regime
        self._detect_regime()

        # Generate signal
        self._generate_signal()

        # Update portfolio based on signal and current prices
        self._update_portfolio()

        # Log regime every 100 bars (for debugging stuck issues)
        if self.bar_index % 100 == 0 or self.signal != 0:
            position_value = self.contracts_held * self.close
            allocation_pct = (position_value / self.portfolio_value * 100) if self.portfolio_value > 0 else 0
            logger.info(
                f"[Bar {self.bar_index}] Regime={self.regime}, "
                f"RSI={self.rsi:.2f}, "
                f"MACD={self.macd:.4f}, "
                f"EMA15={self.ema_12:.2f}, "
                f"EMA30={self.ema_26:.2f}, "
                f"Contracts={self.contracts_held}, "
                f"Allocation={allocation_pct:.1f}%, "
                f"Signal={self.signal}"
            )

        # Log signal generation with allocation info
        if self.signal != 0:
            position_value = self.contracts_held * self.close
            allocation_pct = (position_value / self.portfolio_value * 100) if self.portfolio_value > 0 else 0
            avg_entry = self.entry_price if self.contracts_held > 0 else 0
            unrealized_pnl = (self.close - avg_entry) * self.contracts_held if self.contracts_held > 0 else 0

            logger.info(
                f"Bar {self.bar_index}: Signal={self.signal}, "
                f"Confidence={self.confidence:.3f}, "
                f"Regime={self.regime}, "
                f"RSI={self.rsi:.2f}, "
                f"MACD={self.macd:.4f}, "
                f"Close={self.close:.2f}, "
                f"PV=¥{self.portfolio_value:,.2f}, "
                f"Cash=¥{self.cash:,.2f}, "
                f"Contracts={self.contracts_held}, "
                f"Allocation={allocation_pct:.1f}%, "
                f"AvgEntry={avg_entry:.2f}, "
                f"UnrealizedPnL=¥{unrealized_pnl:,.2f}"
            )

    def _initialize_state(self):
        """Initialize indicator state on first bar"""

        # Initialize EMAs
        self.ema_12 = self.close
        self.ema_26 = self.close
        self.ema_50 = self.close

        # Initialize MACD
        self.macd = 0.0
        self.macd_signal = 0.0
        self.macd_histogram = 0.0

        # Initialize RSI
        self.rsi = 50.0
        self.gain_ema = 0.0
        self.loss_ema = 0.0
        self.prev_close = self.close

        # Initialize Bollinger Bands
        self.bb_n = 1
        self.bb_mean = self.close
        self.bb_m2 = 0.0
        self.bb_variance = 0.0
        self.bb_std_dev = 0.0
        self.bb_upper = self.close
        self.bb_middle = self.close
        self.bb_lower = self.close
        self.bb_width = 0.0
        self.bb_width_pct = 0.0

        # Initialize ATR
        self.atr = self.high - self.low if self.high > self.low else 1.0
        self.mean_atr = self.atr
        self.atr_count = 1
        self.prev_close_atr = self.close

        # Initialize Volume
        self.volume_ema = self.volume

        # Initialize signals
        self.signal = 0
        self.confidence = 0.0
        self.regime = 3

        self.initialized = True

        logger.info(
            f"Initialized: close={self.close:.2f}, "
            f"volume={self.volume:.2f}, "
            f"atr={self.atr:.2f}"
        )

    def _update_emas(self, close):
        """Update triple EMA (online algorithm)"""
        self.ema_12 = self.alpha_12 * close + (1 - self.alpha_12) * self.ema_12
        self.ema_26 = self.alpha_26 * close + (1 - self.alpha_26) * self.ema_26
        self.ema_50 = self.alpha_50 * close + (1 - self.alpha_50) * self.ema_50

    def _update_macd(self):
        """Update MACD and signal line"""
        # MACD line (already have EMA 15 and 30)
        self.macd = self.ema_12 - self.ema_26

        # Signal line (9-period EMA of MACD)
        self.macd_signal = (self.alpha_9 * self.macd +
                            (1 - self.alpha_9) * self.macd_signal)

        # Histogram
        self.macd_histogram = self.macd - self.macd_signal

    def _update_rsi(self, close):
        """Update RSI (online algorithm via gain/loss EMAs)"""

        # Price change
        change = close - self.prev_close
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        # Update gain/loss EMAs (using 14-period alpha)
        self.gain_ema = self.alpha_14 * gain + (1 - self.alpha_14) * self.gain_ema
        self.loss_ema = self.alpha_14 * loss + (1 - self.alpha_14) * self.loss_ema

        # Calculate RSI
        if self.loss_ema > 0:
            rs = self.gain_ema / self.loss_ema
            self.rsi = 100.0 - (100.0 / (1.0 + rs))
        else:
            self.rsi = 100.0  # No losses = max RSI

        # Update previous close for next iteration
        self.prev_close = close

    def _update_bollinger_bands(self, close):
        """Update Bollinger Bands using Welford's online variance algorithm"""

        self.bb_n += 1
        n = min(self.bb_n, 20)  # Cap at period

        # Welford's algorithm for mean and variance
        delta = close - self.bb_mean
        self.bb_mean += delta / n
        delta2 = close - self.bb_mean
        self.bb_m2 += delta * delta2

        # Calculate variance and std dev
        if n > 1:
            self.bb_variance = self.bb_m2 / (n - 1)
            self.bb_std_dev = math.sqrt(self.bb_variance)
        else:
            self.bb_variance = 0.0
            self.bb_std_dev = 0.0

        # Calculate bands
        self.bb_middle = self.bb_mean
        self.bb_upper = self.bb_mean + (2.0 * self.bb_std_dev)
        self.bb_lower = self.bb_mean - (2.0 * self.bb_std_dev)
        self.bb_width = self.bb_upper - self.bb_lower

        # Width percentage
        if self.bb_middle > 0:
            self.bb_width_pct = (self.bb_width / self.bb_middle) * 100.0
        else:
            self.bb_width_pct = 0.0

    def _update_atr(self, high, low, close):
        """Update ATR (online algorithm)"""

        # True Range
        tr1 = high - low
        tr2 = abs(high - self.prev_close_atr) if self.prev_close_atr > 0 else 0
        tr3 = abs(low - self.prev_close_atr) if self.prev_close_atr > 0 else 0
        tr = max(tr1, tr2, tr3)

        # ATR (EMA of True Range)
        self.atr = self.alpha_14 * tr + (1 - self.alpha_14) * self.atr

        # Update mean ATR (online mean)
        self.atr_count += 1
        delta = self.atr - self.mean_atr
        self.mean_atr += delta / min(self.atr_count, 100)

        # Update previous close
        self.prev_close_atr = close

    def _update_volume_ema(self, volume):
        """Update volume EMA"""
        self.volume_ema = (self.alpha_20 * volume +
                           (1 - self.alpha_20) * self.volume_ema)

    def _detect_regime(self):
        """
        Detect current market regime (1/2/3/4)

        1 = Strong Uptrend
        2 = Strong Downtrend
        3 = Sideways/Ranging
        4 = High Volatility Chaos
        """

        # 1. Check for chaos FIRST (highest priority)
        if self._is_high_volatility_chaos():
            self.regime = 4
            return

        # 2. Check for strong trends
        if self._is_strong_uptrend():
            self.regime = 1
            return

        if self._is_strong_downtrend():
            self.regime = 2
            return

        # 3. Default to ranging (everything else)
        self.regime = 3

    def _is_strong_uptrend(self):
        """Check Strong Uptrend criteria"""
        if self.mean_atr == 0:
            return False

        trend_aligned = (self.ema_12 > self.ema_26 > self.ema_50)
        momentum_bullish = (self.macd > self.macd_signal) and (self.macd_histogram > 0)
        price_above = self.close > self.ema_26
        volatility_normal = self.atr <= (self.mean_atr * 1.2)

        return trend_aligned and momentum_bullish and price_above and volatility_normal

    def _is_strong_downtrend(self):
        """Check Strong Downtrend criteria"""
        if self.mean_atr == 0:
            return False

        trend_aligned = (self.ema_12 < self.ema_26 < self.ema_50)
        momentum_bearish = (self.macd < self.macd_signal) and (self.macd_histogram < 0)
        price_below = self.close < self.ema_26
        volatility_normal = self.atr <= (self.mean_atr * 1.2)

        return trend_aligned and momentum_bearish and price_below and volatility_normal

    def _is_high_volatility_chaos(self):
        """Check High Volatility Chaos criteria"""
        if self.mean_atr == 0:
            return False

        extreme_volatility = self.atr > (self.mean_atr * 1.5)
        bb_expansion = self.bb_width_pct > 5.0

        return extreme_volatility or bb_expansion

    def _generate_signal(self):
        """
        IMPROVED: Intelligent Position Accumulation with Aggressive Dip Buying

        Key Features:
        - Calculate signal strength (0.0-1.0) from multi-factor analysis
        - Easier thresholds for adding to existing positions (2/5 vs 3/5)
        - Aggressive dip buying: 100 contracts on strong dips (RSI < 35)
        - Standard buying: 50 contracts on medium dips (RSI < 45)
        - Exit only on CLEAR reversals (not minor weakness)
        - Track desired_contracts and contracts_to_add for visibility

        Signal=1: ADD contracts (50 or 100 based on dip strength)
        Signal=-1: CLOSE ALL positions (clear reversal detected)
        Signal=0: HOLD current state
        """

        # Sync position_state with actual portfolio state
        if self.contracts_held > 0:
            self.position_state = 1
        else:
            self.position_state = 0

        # Calculate current allocation
        current_position_value = self.contracts_held * self.close
        self.portfolio_value = self.cash + current_position_value
        current_allocation = current_position_value / self.portfolio_value if self.portfolio_value > 0 else 0

        # Calculate SIGNAL STRENGTH from multiple factors
        self._calculate_signal_strength()

        # Determine contracts to add (based on dip strength)
        if self.rsi < 35:
            # STRONG dip - aggressive buying
            self.contracts_to_add = 100
        elif self.rsi < 45:
            # MEDIUM dip - standard buying
            self.contracts_to_add = 50
        else:
            # NO dip - standard entry
            self.contracts_to_add = 50

        # Check if we can add contracts (70% limit + cash check)
        proposed_position_value = (self.contracts_held + self.contracts_to_add) * self.close
        proposed_allocation = proposed_position_value / self.portfolio_value if self.portfolio_value > 0 else 0
        can_add_position = (proposed_allocation <= self.max_allocation and
                           self.cash >= (self.contracts_to_add * self.close))

        # Update desired_contracts for tracking
        if self.position_state == 1 and can_add_position:
            self.desired_contracts = self.contracts_held + self.contracts_to_add
        elif can_add_position:
            self.desired_contracts = self.contracts_to_add
        else:
            self.desired_contracts = self.contracts_held

        # Priority 1: Check chaos regime (force exits if in position)
        if self.regime == 4:
            if self.position_state == 1:
                # Only exit chaos if CLEAR danger signals
                if self._check_chaos_force_exit():
                    self.signal = -1
                    self.confidence = 1.0
                    self.signal_strength = 0.0
                    self.last_exit_bar = self.bar_index
                    return

        # Check cooldown (only when flat)
        bars_since_exit = self.bar_index - self.last_exit_bar
        in_cooldown = (bars_since_exit < self.cooldown_bars) and (self.position_state == 0)

        # Priority 2: Check EXIT conditions (STRICTER - only clear reversals)
        if self.position_state == 1:
            should_exit = False

            if self.regime == 1:  # Uptrend - hold through noise
                if self._check_uptrend_exit_strict():
                    should_exit = True

            elif self.regime == 2:  # Downtrend - exit on continuation
                if self._check_downtrend_exit_strict():
                    should_exit = True

            elif self.regime == 3:  # Ranging - exit on clear weakness
                if self._check_ranging_exit_strict():
                    should_exit = True

            if should_exit:
                self.signal = -1
                self.confidence = 0.8
                self.signal_strength = 0.0
                self.last_exit_bar = self.bar_index
                return

        # Priority 3: Check BUY conditions with EASIER thresholds when in position
        if not in_cooldown and can_add_position:

            # Determine required score based on position state
            if self.position_state == 1:
                # EASIER: Only need 2/5 to add to existing position
                required_score_entry = 2
            else:
                # INITIAL: Need 3/5 for first entry
                required_score_entry = 3

            if self.regime == 1:  # Strong Uptrend - buy dips aggressively
                buy_score = self._calculate_uptrend_buy_score()
                if buy_score >= required_score_entry:
                    self.signal = 1
                    self.confidence = max(0.0, min(1.0, (45.0 - self.rsi) / 45.0))
                    return

            elif self.regime == 3:  # Ranging - buy dips
                buy_score = self._calculate_ranging_buy_score()
                if buy_score >= required_score_entry:
                    self.signal = 1
                    bb_range = self.bb_upper - self.bb_lower
                    if bb_range > 0:
                        distance = self.bb_middle - self.close
                        self.confidence = max(0.0, min(abs(distance) / bb_range, 1.0))
                    else:
                        self.confidence = 0.5
                    return

            elif self.regime == 2:  # Downtrend - reversal buys
                buy_score = self._calculate_downtrend_reversal_score()
                if buy_score >= required_score_entry:
                    self.signal = 1
                    self.confidence = 0.6
                    return

            elif self.regime == 4:  # Chaos - recovery buys
                buy_score = self._calculate_chaos_recovery_score()
                if buy_score >= required_score_entry:
                    self.signal = 1
                    self.confidence = 0.5
                    return

        # Default: HOLD
        self.signal = 0
        self.confidence = 0.0

    def _calculate_signal_strength(self):
        """
        Calculate signal strength (0.0-1.0) from multiple factors:
        - RSI distance from extremes
        - MACD strength
        - Trend alignment
        - BB position
        """
        strength = 0.0
        count = 0

        # Factor 1: RSI strength (how oversold/overbought)
        if self.rsi < 50:
            # Oversold = bullish
            rsi_strength = (50.0 - self.rsi) / 50.0  # 0-1
            strength += rsi_strength
            count += 1

        # Factor 2: MACD strength
        if self.macd > self.macd_signal:
            # Positive histogram
            macd_strength = min(abs(self.macd_histogram) / 10.0, 1.0)
            strength += macd_strength
            count += 1

        # Factor 3: Trend alignment (EMA 15 vs 30)
        if self.ema_12 > self.ema_26:
            ema_distance = (self.ema_12 / self.ema_26 - 1.0) * 100  # Percentage
            trend_strength = min(abs(ema_distance) / 2.0, 1.0)  # Normalize
            strength += trend_strength
            count += 1

        # Factor 4: BB position (distance from middle)
        bb_range = self.bb_upper - self.bb_lower
        if bb_range > 0:
            distance_from_middle = abs(self.bb_middle - self.close)
            bb_strength = min(distance_from_middle / bb_range, 1.0)
            strength += bb_strength
            count += 1

        # Average the strength
        if count > 0:
            self.signal_strength = strength / count
        else:
            self.signal_strength = 0.0

    def _calculate_uptrend_buy_score(self):
        """Calculate BUY score for uptrend (returns 0-5)"""
        score = 0

        # 1. Trend aligned
        if self.ema_12 > self.ema_26:
            score += 1

        # 2. Momentum bullish
        if self.macd > self.macd_signal:
            score += 1

        # 3. RSI shows dip (increased from 40 to 50 for more opportunities)
        if self.rsi < 50:
            score += 1

        # 4. Volume confirmation (relaxed to 1.1x)
        if self.volume > (self.volume_ema * 1.1):
            score += 1

        # 5. Price showing dip (within 30% of band width from lower)
        bb_range = self.bb_upper - self.bb_lower
        if bb_range > 0 and self.close <= (self.bb_lower + bb_range * 0.3):
            score += 1

        return score

    def _check_uptrend_exit_strict(self):
        """STRICTER uptrend exit - only clear reversals (need 3/4)"""
        exit_score = 0

        # 1. RSI very overbought
        if self.rsi > 65:  # Stricter threshold
            exit_score += 1

        # 2. MACD clear bearish cross
        if self.macd < self.macd_signal and self.macd_histogram < -0.5:
            exit_score += 1

        # 3. Price breaks below BOTH short EMAs
        if self.close < self.ema_12 and self.close < self.ema_26:
            exit_score += 1

        # 4. Price showing clear weakness (close in bottom 30%)
        if self.high > self.low:
            bar_position = (self.close - self.low) / (self.high - self.low)
            if bar_position < 0.3:
                exit_score += 1

        return exit_score >= 3  # Need 3/4 for exit

    def _check_downtrend_exit_strict(self):
        """STRICTER downtrend exit - confirmed downtrend continuation (need 2/3)"""
        exit_score = 0

        if self.ema_12 < self.ema_26:  # Bearish trend
            exit_score += 1

        if self.macd < self.macd_signal:  # Bearish momentum
            exit_score += 1

        if self.close < self.bb_lower:  # Price very weak (not just middle)
            exit_score += 1

        return exit_score >= 2  # Need 2/3 to exit

    def _calculate_downtrend_reversal_score(self):
        """Calculate reversal BUY score in downtrend (returns 0-6)"""
        score = 0

        # 1. EMA15 crossing above EMA30 (reversal signal)
        if self.ema_12 > self.ema_26:
            score += 1

        # 2. MACD bullish
        if self.macd > self.macd_signal:
            score += 1

        # 3. RSI oversold (increased to 45 for more opportunities)
        if self.rsi < 45:
            score += 1

        # 4. Price at lower BB (oversold)
        bb_range = self.bb_upper - self.bb_lower
        if bb_range > 0 and self.close <= (self.bb_lower + bb_range * 0.3):
            score += 1

        # 5. Volume confirmation
        if self.volume > (self.volume_ema * 1.1):
            score += 1

        # 6. Price action showing bounce
        if self.high > self.low:
            bar_position = (self.close - self.low) / (self.high - self.low)
            if bar_position > 0.5:
                score += 1

        return score

    def _calculate_chaos_recovery_score(self):
        """Calculate chaos recovery BUY score (returns 0-5)"""
        score = 0

        # 1. Volatility calming
        if self.atr < self.mean_atr * 1.3:
            score += 1

        # 2. Trend direction emerging
        if self.ema_12 > self.ema_26:
            score += 1

        # 3. RSI in tradable zone (widened from 25-45 to 25-50)
        if 25 < self.rsi < 50:
            score += 1

        # 4. MACD momentum
        if self.macd > self.macd_signal:
            score += 1

        # 5. Price action strength
        if self.high > self.low:
            bar_position = (self.close - self.low) / (self.high - self.low)
            if bar_position > 0.6:
                score += 1

        return score

    def _check_chaos_force_exit(self):
        """Check if chaos is dangerous enough to force exit"""
        # Only exit if REALLY dangerous (need 3/4)
        danger_score = 0

        if self.atr > self.mean_atr * 2.0:  # Extreme volatility
            danger_score += 1

        if self.bb_width_pct > 6.0:  # Extreme BB expansion
            danger_score += 1

        if self.close < self.bb_lower:  # Price collapsing
            danger_score += 1

        if self.macd_histogram < -1.0:  # Strong negative momentum
            danger_score += 1

        return danger_score >= 3

    def _calculate_ranging_buy_score(self):
        """Calculate ranging BUY score (returns 0-4)"""
        score = 0

        # 1. Near lower band (widened to 40%)
        bb_range = self.bb_upper - self.bb_lower
        if bb_range > 0 and self.close <= (self.bb_lower + bb_range * 0.4):
            score += 1

        # 2. RSI shows dip (increased to 50)
        if self.rsi < 50:
            score += 1

        # 3. Volume confirmation (relaxed)
        if self.volume > (self.volume_ema * 1.1):
            score += 1

        # 4. Price showing bounce potential
        if self.close > self.low:
            score += 1

        return score

    def _check_ranging_exit_strict(self):
        """STRICTER ranging exit - clear weakness (need 3/4)"""
        score = 0

        # 1. Price near upper band (tightened to 20%)
        bb_range = self.bb_upper - self.bb_lower
        if bb_range > 0 and self.close >= (self.bb_upper - bb_range * 0.2):
            score += 1

        # 2. RSI very overbought
        if self.rsi > 65:  # Stricter threshold
            score += 1

        # 3. MACD clearly turning down
        if self.macd < self.macd_signal and self.macd_histogram < -0.3:
            score += 1

        # 4. Price action weak
        if self.high > self.low:
            bar_position = (self.close - self.low) / (self.high - self.low)
            if bar_position < 0.4:  # Close in bottom 40%
                score += 1

        return score >= 3  # Need 3/4 to exit

    def _update_portfolio(self):
        """
        Update portfolio value based on current signal and position.
        Supports DYNAMIC position accumulation (50 or 100 contracts)
        """

        # Calculate current position value
        if self.contracts_held > 0:
            position_value = self.contracts_held * self.close
        else:
            position_value = 0.0

        # Process signal to update portfolio
        if self.signal == 1:
            # BUY signal - ADD contracts (DYNAMIC: 50 or 100 based on dip)
            # Use contracts_to_add (calculated in _generate_signal)
            cost = self.contracts_to_add * self.close
            if self.cash >= cost:
                # Calculate new average entry price
                total_cost_before = self.contracts_held * self.entry_price
                new_cost = self.contracts_to_add * self.close
                new_contracts_total = self.contracts_held + self.contracts_to_add

                if new_contracts_total > 0:
                    self.entry_price = (total_cost_before + new_cost) / new_contracts_total

                # Add contracts and deduct cash
                self.contracts_held += self.contracts_to_add
                self.cash -= cost

                logger.info(f"[Bar {self.bar_index}] BUY: Added {self.contracts_to_add} contracts @ {self.close:.2f}. Total contracts: {self.contracts_held}")

        elif self.signal == -1 and self.contracts_held > 0:
            # SELL signal - CLOSE ALL positions
            self.cash += self.contracts_held * self.close
            logger.info(f"[Bar {self.bar_index}] SELL: Closed {self.contracts_held} contracts @ {self.close:.2f}")
            self.contracts_held = 0
            self.entry_price = 0.0

        # Calculate final portfolio value
        if self.contracts_held > 0:
            position_value = self.contracts_held * self.close
        else:
            position_value = 0.0

        self.portfolio_value = self.cash + position_value

    def ready_to_serialize(self) -> bool:
        """Determine if state should be serialized"""
        return self.bar_index > 0 and self.initialized


# Global instance
indicator = SoybeanIndicator()


# Framework callbacks (REQUIRED)

async def on_init():
    """Initialize indicator with metadata schemas"""
    global indicator, imports, metas, worker_no
    if worker_no != 0 and metas and imports:
        indicator.initialize(imports, metas)
        logger.info("SoybeanIndicator initialized")


async def on_bar(bar: pc.StructValue):
    """Process incoming bars"""
    global indicator, worker_no
    if worker_no != 1:
        return []
    return indicator.on_bar(bar)


async def on_ready():
    """Called when framework is ready"""
    logger.info("SoybeanIndicator ready")


async def on_market_open(market, tradeday, time_tag, time_string):
    """Called on market open"""
    pass


async def on_market_close(market, tradeday, timetag, timestring):
    """Called on market close"""
    pass


async def on_reference(market, tradeday, data, timetag, timestring):
    """Called on reference data"""
    pass


async def on_tradeday_begin(market, tradeday, time_tag, time_string):
    """Called at beginning of trade day"""
    pass


async def on_tradeday_end(market, tradeday, timetag, timestring):
    """Called at end of trade day"""
    pass


async def on_historical(params, records):
    """Called on historical data"""
    pass
