#!/usr/bin/env python3
# coding=utf-8
"""
IronOreIndicatorRelaxed - Pure Signal Generator (Tier 1 Indicator)

PURE SIGNAL GENERATION with 7-Indicator Market Regime Detection:
- Triple EMA (12/26/50) - Trend identification & MA crossover signals
- MACD (12/26/9) - Momentum confirmation & reversal detection
- RSI (14) - Mean reversion signals (relaxed thresholds)
- Bollinger Bands (20, 2σ) - Price extremes & volatility
- ATR (14) - Volatility measurement
- BB Width - Volatility confirmation
- Volume EMA (20) - Liquidity confirmation

Market Regimes & Signal Logic:
1. Strong Uptrend - LONG on dips (RSI < 45), SHORT on overbought (RSI > 55)
2. Strong Downtrend - LONG on reversals, SHORT on bearish rallies
3. Sideways/Ranging - LONG at lower BB, SHORT at upper BB (mean reversion)
4. High Volatility Chaos - LONG on recovery, SHORT on extreme danger

Key Features:
- NO TRADING LOGIC - Pure signal generation only
- Outputs: signal (-1/0/1), confidence (0-1), signal_strength (0-1), regime (1-4)
- RELAXED thresholds for more frequent signals
- Supports both LONG (signal=1) and SHORT (signal=-1) signals
- Technical indicators exported for analysis
- Designed for Tier 2 composite strategy consumption
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


class IronOreIndicator(pcts3.sv_object):
    """
    Iron Ore Multi-Indicator Confirmation System

    Generates regime-aware buy/sell signals using multi-layer confirmation
    """

    def __init__(self):
        super().__init__()

        # Metadata - CONSTANTS (never change after initialization)
        self.meta_name = "IronOreIndicatorRelaxed"
        self.namespace = pc.namespace_private
        self.granularity = 900
        self.market = b'DCE'
        self.code = b'i<00>'
        self.revision = (1 << 32) - 1

        # State variables (automatically persisted by framework)
        self.bar_index = 0
        self.timetag = None
        self.initialized = False

        # Alpha parameters (smoothing factors)
        self.alpha_12 = 2.0 / 13.0   # 0.1538
        self.alpha_26 = 2.0 / 27.0   # 0.0741
        self.alpha_50 = 2.0 / 51.0   # 0.0392
        self.alpha_9 = 2.0 / 10.0    # 0.2000
        self.alpha_14 = 2.0 / 15.0   # 0.1333
        self.alpha_20 = 2.0 / 21.0   # 0.0952

        # Current bar OHLCV
        self.open = 0.0
        self.high = 0.0
        self.low = 0.0
        self.close = 0.0
        self.volume = 0.0

        # Triple EMA states
        self.ema_12 = 0.0
        self.ema_26 = 0.0
        self.ema_50 = 0.0

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

        # Signal outputs (pure signal generation - no trading logic)
        self.signal = 0          # -1 (bearish), 0 (neutral), 1 (bullish)
        self.confidence = 0.0    # Signal confidence [0.0, 1.0]
        self.signal_strength = 0.0    # 0.0-1.0 conviction level

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

        # Log regime every 100 bars (for debugging)
        if self.bar_index % 100 == 0 or self.signal != 0:
            logger.info(
                f"[Bar {self.bar_index}] Regime={self.regime}, "
                f"RSI={self.rsi:.2f}, "
                f"MACD={self.macd:.4f}, "
                f"EMA12={self.ema_12:.2f}, "
                f"EMA26={self.ema_26:.2f}, "
                f"Signal={self.signal}, "
                f"Confidence={self.confidence:.3f}, "
                f"Strength={self.signal_strength:.3f}"
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
        # MACD line (already have EMA 12 and 26)
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

        # Update gain/loss EMAs
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
        Pure Signal Generation (no trading logic)

        Generates directional market signals based on multi-indicator analysis:
        - signal = 1: Bullish opportunity (composite should consider LONG)
        - signal = -1: Bearish opportunity (composite should consider SHORT)
        - signal = 0: Neutral (no clear direction)

        Confidence and signal_strength provide conviction level (0.0-1.0)
        """

        # Calculate SIGNAL STRENGTH from multiple factors
        self._calculate_signal_strength()

        # Default to neutral
        self.signal = 0
        self.confidence = 0.0

        # Generate signals based on regime with RELAXED thresholds
        if self.regime == 1:  # Strong Uptrend
            self._check_uptrend_signals()
        elif self.regime == 2:  # Strong Downtrend
            self._check_downtrend_signals()
        elif self.regime == 3:  # Ranging/Sideways
            self._check_ranging_signals()
        elif self.regime == 4:  # High Volatility Chaos
            self._check_chaos_signals()

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

        # Factor 3: Trend alignment (EMA 12 vs 26)
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

    def _check_uptrend_signals(self):
        """Generate signals for Strong Uptrend regime (RELAXED)"""
        # BULLISH (BUY) signals - look for dips to enter long
        if self.rsi < 45 and self.macd > self.macd_signal:
            # Dip in uptrend with momentum support
            self.signal = 1
            self.confidence = max(0.0, min(1.0, (45.0 - self.rsi) / 45.0))
            return

        # BEARISH (SELL) signals - look for overbought conditions
        if self.rsi > 55:
            bb_range = self.bb_upper - self.bb_lower
            if bb_range > 0 and self.close >= (self.bb_upper - bb_range * 0.3):
                # Overbought near upper band
                self.signal = -1
                self.confidence = max(0.0, min(1.0, (self.rsi - 55.0) / 45.0))
                return

    def _check_downtrend_signals(self):
        """Generate signals for Strong Downtrend regime (RELAXED)"""
        # BULLISH (BUY) signals - look for reversal
        if self.ema_12 > self.ema_26 and self.rsi < 45:
            # Trend reversal signal
            self.signal = 1
            self.confidence = 0.6
            return

        # BEARISH (SELL) signals - downtrend continuation
        if self.rsi > 55 and self.macd < self.macd_signal:
            # Bearish rally in downtrend
            self.signal = -1
            self.confidence = 0.7
            return

    def _check_ranging_signals(self):
        """Generate signals for Ranging/Sideways regime (RELAXED)"""
        bb_range = self.bb_upper - self.bb_lower

        # BULLISH (BUY) signals - mean reversion from lower band
        if bb_range > 0 and self.close <= (self.bb_lower + bb_range * 0.4):
            if self.rsi < 50:
                # Price near lower band with RSI support
                self.signal = 1
                distance = self.bb_middle - self.close
                self.confidence = max(0.0, min(abs(distance) / bb_range, 1.0))
                return

        # BEARISH (SELL) signals - mean reversion from upper band
        if bb_range > 0 and self.close >= (self.bb_upper - bb_range * 0.4):
            if self.rsi > 50:
                # Price near upper band with overbought RSI
                self.signal = -1
                distance = self.close - self.bb_middle
                self.confidence = max(0.0, min(abs(distance) / bb_range, 1.0))
                return

    def _check_chaos_signals(self):
        """Generate signals for High Volatility Chaos regime (RELAXED)"""
        # In chaos, look for recovery signals or strong momentum

        # BULLISH (BUY) signals - volatility calming with bullish momentum
        if self.atr < self.mean_atr * 1.3 and self.ema_12 > self.ema_26:
            if 25 < self.rsi < 50:
                # Volatility calming with bullish setup
                self.signal = 1
                self.confidence = 0.5
                return

        # BEARISH (SELL) signals - extreme chaos danger
        if self.atr > self.mean_atr * 2.0 or self.bb_width_pct > 6.0:
            if self.macd_histogram < -0.5:
                # Extreme volatility with bearish momentum
                self.signal = -1
                self.confidence = 0.6
                return

    def ready_to_serialize(self) -> bool:
        """Determine if state should be serialized"""
        return self.bar_index > 0 and self.initialized


# Global instance
indicator = IronOreIndicator()


# Framework callbacks (REQUIRED)

async def on_init():
    """Initialize indicator with metadata schemas"""
    global indicator, imports, metas, worker_no
    if worker_no != 0 and metas and imports:
        indicator.initialize(imports, metas)
        logger.info("IronOreIndicatorRelaxed initialized")


async def on_bar(bar: pc.StructValue):
    """Process incoming bars"""
    global indicator, worker_no
    if worker_no != 1:
        return []
    return indicator.on_bar(bar)


async def on_ready():
    """Called when framework is ready"""
    logger.info("IronOreIndicatorRelaxed ready")


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
