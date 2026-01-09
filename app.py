"""
================================================================================
GAMMA SCALPING BACKTEST - MARKET MAKER PERSPECTIVE (SHORT GAMMA)
================================================================================
Author: Quantitative Finance Implementation
References: 
    - "Option Volatility and Pricing" by Sheldon Natenberg (Ch. 8-10)
    - "Dynamic Hedging" by Nassim Taleb (Ch. 4-6, 12-14)
    - "Trading and Exchanges" by Larry Harris (Market Making)

MARKET MAKER PERSPECTIVE:

The previous script simulated a VOLATILITY BUYER (Long Straddle, Long Gamma).
This script flips the position to simulate a MARKET MAKER who:

1. SELLS the straddle (Short Call + Short Put) → COLLECTS PREMIUM
2. Is SHORT GAMMA (negative convexity) → LOSES from large moves
3. Is LONG THETA → EARNS from time decay
4. Must DELTA HEDGE to survive → But hedging COSTS money (buy high, sell low)

Key Insight (Taleb, "Dynamic Hedging" Ch. 12):
    "The market maker is essentially an insurance company. He collects 
    premium (theta) and pays out claims (gamma losses from hedging).
    He profits when realized volatility < implied volatility."

RISK PROFILE:
    - Long Straddle:  Limited loss (premium), Unlimited profit potential
    - Short Straddle: Limited profit (premium), UNLIMITED LOSS potential
    
This is why market makers MUST hedge continuously!
================================================================================
"""



import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Literal
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("✓ All dependencies loaded successfully")
print("📊 MODE: MARKET MAKER (Short Gamma)")

# ====================== CELL 2: CONFIGURATION ======================
class PositionType(Enum):
    """Position type for the straddle strategy."""
    LONG = "LONG"    # Buy straddle (long gamma, short theta)
    SHORT = "SHORT"  # Sell straddle (short gamma, long theta) - MARKET MAKER

@dataclass
class MarketMakerConfig:
    """
    Configuration for Market Maker Gamma Scalping.
    
    MARKET MAKER SPECIFICS:
    - We SELL options → Collect premium upfront
    - We are SHORT GAMMA → Hedging costs money (buy high, sell low)
    - We are LONG THETA → Time decay works FOR us
    - We want REALIZED VOL < IMPLIED VOL (quiet markets)
    
    Risk Management is CRITICAL for short gamma positions!
    """
    # Data Parameters
    ticker: str = "NVDA"
    interval: str = "5m"
    lookback_days: int = 5
    
    # Position Type - THE KEY DIFFERENCE
    position_type: PositionType = PositionType.SHORT  # MARKET MAKER = SHORT
    
    # Option Parameters
    time_to_expiry_days: int = 30
    risk_free_rate: float = 0.05
    
    # Implied Volatility (what we SELL the options at)
    # This is typically HIGHER than realized vol → Market Maker edge
    implied_vol_premium: float = 0.05  # IV = RV + 5% premium (volatility risk premium)
    
    # Volatility Calculation
    realized_vol_window: int = 78
    vol_annualization_factor: int = 252 * 78
    
    # Delta Hedging - More aggressive for short gamma (RISK MANAGEMENT)
    delta_threshold: float = 0.05  # Tighter threshold for short gamma!
    min_shares_to_hedge: int = 5
    
    # Position Sizing (smaller for short gamma due to risk)
    num_straddles: int = 5  # Fewer contracts due to unlimited risk
    contract_multiplier: int = 100
    
    # Transaction Costs
    stock_commission_bps: float = 1.0
    slippage_bps: float = 2.0
    
    # RISK MANAGEMENT - Critical for Market Makers
    max_loss_limit: float = 50000  # Stop loss limit
    margin_requirement_pct: float = 0.20  # 20% of notional as margin
    var_confidence: float = 0.99  # 99% VaR
    
    @property
    def total_tc_bps(self) -> float:
        return self.stock_commission_bps + self.slippage_bps
    
    @property
    def position_sign(self) -> int:
        """Returns +1 for long, -1 for short."""
        return 1 if self.position_type == PositionType.LONG else -1

config = MarketMakerConfig()

print(f"\n{'='*60}")
print(f"⚠️  MARKET MAKER MODE: SHORT STRADDLE (SHORT GAMMA)")
print(f"{'='*60}")
print(f"""
Position:           {config.position_type.value} STRADDLE
Ticker:             {config.ticker}
Contracts:          {config.num_straddles} straddles
Delta Threshold:    ±{config.delta_threshold} (tighter for risk management)

PROFIT FROM:        Time decay (Theta) + Volatility Risk Premium
LOSE FROM:          Large price moves (Gamma) + Hedging costs

⚠️  WARNING: Short gamma has UNLIMITED LOSS potential!
    Max Loss Limit set to: ${config.max_loss_limit:,}
""")

# ====================== CELL 3: BLACK-SCHOLES ENGINE (Same as before) ======================
class BlackScholesModel:
    """
    Black-Scholes-Merton Option Pricing Model.
    (Same implementation, but now used for SHORT positions)
    """
    
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float):
        self.S = S
        self.K = K
        self.T = max(T, 1e-10)
        self.r = r
        self.sigma = max(sigma, 1e-10)
        self._compute_d1_d2()
    
    def _compute_d1_d2(self) -> None:
        sqrt_T = np.sqrt(self.T)
        self.d1 = (np.log(self.S / self.K) + 
                   (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * sqrt_T)
        self.d2 = self.d1 - self.sigma * sqrt_T
    
    def call_price(self) -> float:
        return (self.S * norm.cdf(self.d1) - 
                self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))
    
    def put_price(self) -> float:
        return (self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - 
                self.S * norm.cdf(-self.d1))
    
    def call_delta(self) -> float:
        return norm.cdf(self.d1)
    
    def put_delta(self) -> float:
        return norm.cdf(self.d1) - 1.0
    
    def gamma(self) -> float:
        return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
    
    def theta_call(self) -> float:
        """Daily theta for call."""
        sqrt_T = np.sqrt(self.T)
        term1 = -(self.S * norm.pdf(self.d1) * self.sigma) / (2 * sqrt_T)
        term2 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        return (term1 + term2) / 365.0
    
    def theta_put(self) -> float:
        """Daily theta for put."""
        sqrt_T = np.sqrt(self.T)
        term1 = -(self.S * norm.pdf(self.d1) * self.sigma) / (2 * sqrt_T)
        term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
        return (term1 + term2) / 365.0
    
    def vega(self) -> float:
        """Vega per 1% vol move."""
        return self.S * np.sqrt(self.T) * norm.pdf(self.d1) / 100.0
    
    def straddle_price(self) -> float:
        return self.call_price() + self.put_price()
    
    def straddle_delta(self) -> float:
        return self.call_delta() + self.put_delta()
    
    def straddle_gamma(self) -> float:
        return 2.0 * self.gamma()
    
    def straddle_theta(self) -> float:
        return self.theta_call() + self.theta_put()
    
    def straddle_vega(self) -> float:
        return 2.0 * self.vega()


def bsm_vectorized(S: pd.Series, K: float, T: pd.Series, r: float, 
                   sigma: pd.Series, position_sign: int = 1) -> pd.DataFrame:
    """
    Vectorized BSM with position sign adjustment.
    
    position_sign: +1 for LONG, -1 for SHORT
    
    For SHORT positions:
    - Price represents LIABILITY (negative to holder, positive to seller)
    - Delta is REVERSED (we need opposite hedge)
    - Gamma is still positive magnitude but represents RISK not opportunity
    - Theta is POSITIVE (we EARN from decay)
    """
    T = T.clip(lower=1e-10)
    sigma = sigma.clip(lower=1e-10)
    
    sqrt_T = np.sqrt(T)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    N_neg_d1 = norm.cdf(-d1)
    N_neg_d2 = norm.cdf(-d2)
    n_d1 = norm.pdf(d1)
    
    discount = np.exp(-r * T)
    
    # Raw option prices (always positive)
    call_price = S * N_d1 - K * discount * N_d2
    put_price = K * discount * N_neg_d2 - S * N_neg_d1
    straddle_price = call_price + put_price
    
    # Greeks (raw, before position adjustment)
    call_delta = N_d1
    put_delta = N_d1 - 1
    straddle_delta = call_delta + put_delta
    gamma = n_d1 / (S * sigma * sqrt_T)
    
    theta_term1 = -(S * n_d1 * sigma) / (2 * sqrt_T)
    theta_call = (theta_term1 - r * K * discount * N_d2) / 365
    theta_put = (theta_term1 + r * K * discount * N_neg_d2) / 365
    straddle_theta = theta_call + theta_put
    
    vega = S * sqrt_T * n_d1 / 100
    
    return pd.DataFrame({
        # Raw prices (for premium calculation)
        'call_price_raw': call_price,
        'put_price_raw': put_price,
        'straddle_price_raw': straddle_price,
        
        # Position-adjusted values
        # For SHORT: negative price = we received premium (positive P&L at inception)
        'call_price': call_price * position_sign,
        'put_price': put_price * position_sign,
        'straddle_price': straddle_price * position_sign,
        
        # Delta adjustment for hedging
        # SHORT straddle with positive delta → we need to SELL stock (opposite of long)
        'call_delta': call_delta * position_sign,
        'put_delta': put_delta * position_sign,
        'straddle_delta': straddle_delta * position_sign,
        
        # Gamma (always shown as magnitude, sign indicates risk)
        'gamma': gamma,
        'straddle_gamma': 2 * gamma,
        'position_gamma_sign': position_sign,  # Negative for short = risk
        
        # Theta (positive for short = we earn)
        'theta_call': theta_call * position_sign * -1,  # Flip sign: short earns theta
        'theta_put': theta_put * position_sign * -1,
        'straddle_theta': straddle_theta * position_sign * -1,
        
        'vega': vega,
        'straddle_vega': 2 * vega,
        'd1': d1,
        'd2': d2,
        'sigma_used': sigma
    })

print("✓ Black-Scholes Engine loaded (Market Maker mode)")

# ====================== CELL 4: DATA INGESTION ======================
class DataIngestion:
    """Fetches and preprocesses intraday data."""
    
    def __init__(self, config: MarketMakerConfig):
        self.config = config
    
    def fetch_data(self) -> pd.DataFrame:
        print(f"\n📡 Fetching {self.config.interval} data for {self.config.ticker}...")
        
        ticker = yf.Ticker(self.config.ticker)
        df = ticker.history(
            period=f"{self.config.lookback_days}d",
            interval=self.config.interval,
            prepost=False
        )
        
        if df.empty:
            raise ValueError(f"No data returned for {self.config.ticker}")
        
        print(f"  → Retrieved {len(df)} bars")
        return df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        initial_len = len(df)
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        if len(df) < initial_len:
            print(f"  ⚠ Removed {initial_len - len(df)} NaN rows")
        
        df['price'] = df['Close']
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        
        # Realized Volatility
        df['realized_vol'] = (
            df['log_return']
            .rolling(window=self.config.realized_vol_window, min_periods=20)
            .std() * np.sqrt(self.config.vol_annualization_factor)
        )
        
        first_valid_vol = df['realized_vol'].dropna().iloc[0] if df['realized_vol'].dropna().any() else 0.30
        df['realized_vol'] = df['realized_vol'].fillna(first_valid_vol).clip(0.05, 2.0)
        
        # IMPLIED VOLATILITY (what we SELL at)
        # Market makers typically sell at IV > RV (volatility risk premium)
        df['implied_vol'] = df['realized_vol'] + self.config.implied_vol_premium
        
        df['bar_index'] = range(len(df))
        
        print(f"  ✓ Processed {len(df)} bars")
        print(f"  ✓ Avg Realized Vol: {df['realized_vol'].mean():.1%}")
        print(f"  ✓ Avg Implied Vol (sell at): {df['implied_vol'].mean():.1%}")
        print(f"  ✓ Vol Risk Premium: {self.config.implied_vol_premium:.1%}")
        
        return df
    
    def get_data(self) -> pd.DataFrame:
        raw = self.fetch_data()
        return self.preprocess(raw)

data_loader = DataIngestion(config)
market_data = data_loader.get_data()

print("\n📊 Sample Data:")
display(market_data[['price', 'log_return', 'realized_vol', 'implied_vol']].tail(5))

# ====================== CELL 5: MARKET MAKER BACKTEST ENGINE ======================
@dataclass
class HedgeTrade:
    """Record of a hedge transaction."""
    timestamp: datetime
    action: str
    shares: int
    price: float
    delta_before: float
    delta_after: float
    transaction_cost: float
    reason: str = ""

@dataclass
class RiskMetrics:
    """Real-time risk metrics for the position."""
    current_pnl: float
    max_drawdown: float
    current_delta: float
    current_gamma_dollars: float  # Gamma in dollar terms
    var_95: float
    margin_used: float
    margin_available: float

@dataclass
class BacktestResults:
    df: pd.DataFrame
    trades: List[HedgeTrade]
    summary: dict
    risk_events: List[dict]


class MarketMakerBacktest:
    """
    Market Maker Gamma Scalping Backtest Engine.
    
    CRITICAL DIFFERENCES FROM LONG STRADDLE:
    
    1. PREMIUM COLLECTION
       - We SELL the straddle → Receive premium upfront
       - Initial P&L = +Premium received
       - This is our "edge" if vol stays low
    
    2. REVERSED DELTA HEDGE
       - Long straddle: Price up → Delta positive → Sell stock
       - Short straddle: Price up → Delta NEGATIVE (for us) → BUY stock
       - We are always "chasing" the market (buy high, sell low)
    
    3. THETA WORKS FOR US
       - Each day that passes, options lose value
       - Since we're SHORT, we PROFIT from this decay
       - This is our "carry" income
    
    4. GAMMA WORKS AGAINST US
       - Large moves hurt us (negative convexity)
       - We lose more as moves get bigger
       - Hedging COSTS money, doesn't make money
    
    5. RISK MANAGEMENT IS CRITICAL
       - Unlimited loss potential
       - Must have stop-losses
       - Position sizing matters enormously
    
    P&L FORMULA FOR MARKET MAKER:
       P&L = Premium Received + Theta Earned - Gamma Losses - Hedge Costs - TC
       
    PROFIT CONDITION:
       Implied Volatility > Realized Volatility
       (We sold expensive insurance, market was calm)
    """
    
    def __init__(self, config: MarketMakerConfig, market_data: pd.DataFrame):
        self.config = config
        self.data = market_data.copy()
        self.trades: List[HedgeTrade] = []
        self.risk_events: List[dict] = []
        
        self.strike = None
        self.stock_position = 0
        self.premium_received = 0  # Key for market maker
        self.position_sign = config.position_sign  # -1 for SHORT
        
        # Risk tracking
        self.max_pnl = 0
        self.max_drawdown = 0
        self.is_stopped_out = False
        
    def run(self) -> BacktestResults:
        """Execute full backtest."""
        print(f"\n🚀 Starting MARKET MAKER Backtest ({self.config.position_type.value} Straddle)...")
        
        self._initialize_position()
        self._compute_option_values()
        self._simulate_hedging()
        self._calculate_pnl()
        summary = self._generate_summary()
        
        print("\n✅ Backtest Complete!")
        return BacktestResults(
            df=self.data,
            trades=self.trades,
            summary=summary,
            risk_events=self.risk_events
        )
    
    def _initialize_position(self) -> None:
        """Initialize the SHORT straddle position."""
        first_price = self.data['price'].iloc[0]
        first_iv = self.data['implied_vol'].iloc[0]
        
        self.strike = round(first_price)
        
        # Time to expiry calculation
        bars_per_day = 78
        self.data['time_to_expiry_years'] = (
            self.config.time_to_expiry_days / 365.0 - 
            self.data['bar_index'] / (bars_per_day * 252)
        ).clip(lower=1e-6)
        
        # Calculate initial premium using IMPLIED VOL (what we SELL at)
        bsm = BlackScholesModel(
            S=first_price,
            K=self.strike,
            T=self.config.time_to_expiry_days / 365.0,
            r=self.config.risk_free_rate,
            sigma=first_iv  # Use IMPLIED vol for pricing
        )
        
        straddle_price_per_contract = bsm.straddle_price()
        multiplier = self.config.num_straddles * self.config.contract_multiplier
        
        # PREMIUM RECEIVED (positive for short)
        self.premium_received = straddle_price_per_contract * multiplier
        
        # Calculate margin requirement
        notional = self.strike * multiplier
        self.margin_required = notional * self.config.margin_requirement_pct
        
        print(f"\n{'='*60}")
        print(f"📋 SHORT STRADDLE INITIATED (Market Maker)")
        print(f"{'='*60}")
        print(f"""
Strike:                 ${self.strike}
Underlying:             ${first_price:.2f}
Position:               SHORT {self.config.num_straddles} straddles

IMPLIED Vol (sell at):  {first_iv:.1%}
REALIZED Vol (actual):  {self.data['realized_vol'].iloc[0]:.1%}
Vol Edge:               {(first_iv - self.data['realized_vol'].iloc[0]):.1%}

💰 PREMIUM RECEIVED:    ${self.premium_received:,.2f}
   (This is our MAX PROFIT if options expire worthless)

📊 Risk Metrics:
   Notional Exposure:   ${notional:,.2f}
   Margin Required:     ${self.margin_required:,.2f}
   Max Loss Limit:      ${self.config.max_loss_limit:,.2f}
""")
    
    def _compute_option_values(self) -> None:
        """Compute BSM values - using REALIZED vol for true value."""
        print("⚙ Computing option values (marking to REALIZED vol)...")
        
        # Price options at REALIZED vol for mark-to-market
        # (In practice, would use market IV, but we simulate with RV)
        bsm_results = bsm_vectorized(
            S=self.data['price'],
            K=self.strike,
            T=self.data['time_to_expiry_years'],
            r=self.config.risk_free_rate,
            sigma=self.data['realized_vol'],  # Mark-to-market at realized vol
            position_sign=self.position_sign  # -1 for SHORT
        )
        
        for col in bsm_results.columns:
            self.data[col] = bsm_results[col].values
        
        multiplier = self.config.num_straddles * self.config.contract_multiplier
        
        # Position values (negative for short = liability)
        self.data['position_delta'] = self.data['straddle_delta'] * multiplier
        self.data['position_gamma'] = self.data['straddle_gamma'] * multiplier
        self.data['position_theta'] = self.data['straddle_theta'] * multiplier
        self.data['position_vega'] = self.data['straddle_vega'] * multiplier
        
        # Current liability (what we'd pay to close)
        self.data['current_liability'] = self.data['straddle_price_raw'] * multiplier
        
        # Gamma in dollar terms (how much we lose per 1% move)
        self.data['gamma_dollars'] = 0.5 * self.data['position_gamma'] * (self.data['price'] * 0.01)**2
        
        print(f"   Initial position delta: {self.data['position_delta'].iloc[0]:.2f}")
        print(f"   Position gamma (risk): {self.data['position_gamma'].iloc[0]:.4f}")
        print(f"   Daily theta (income): ${self.data['position_theta'].iloc[0]:.2f}")
    
    def _simulate_hedging(self) -> None:
        """
        Simulate delta hedging for SHORT gamma position.
        
        KEY DIFFERENCE FROM LONG:
        - When price RISES, our short call becomes more valuable (bad for us)
        - Our delta becomes MORE NEGATIVE (we're more short)
        - To hedge, we BUY stock (at HIGH price) - this COSTS money
        
        - When price FALLS, our short put becomes more valuable
        - Our delta becomes MORE POSITIVE (we're more long)
        - To hedge, we SELL stock (at LOW price) - this COSTS money
        
        SHORT GAMMA = ALWAYS CHASING THE MARKET = BUYING HIGH, SELLING LOW
        """
        print("\n📈 Simulating Delta Hedging (Short Gamma)...")
        
        # Initialize columns
        self.data['stock_position'] = 0.0
        self.data['net_delta'] = 0.0
        self.data['hedge_trade_shares'] = 0
        self.data['hedge_trade_action'] = ''
        self.data['cumulative_hedge_cost'] = 0.0  # Renamed to reflect cost, not profit
        self.data['cumulative_tc'] = 0.0
        self.data['running_pnl'] = 0.0
        
        stock_pos = 0
        total_cash_spent_on_stock = 0  # Track net cash outflow
        cumulative_tc = 0.0
        
        multiplier = self.config.num_straddles * self.config.contract_multiplier
        
        for i in range(len(self.data)):
            idx = self.data.index[i]
            row = self.data.loc[idx]
            price = row['price']
            
            # Current option delta (already adjusted for SHORT in bsm_vectorized)
            option_delta = row['straddle_delta'] * multiplier
            
            # Net delta = option delta + stock position
            net_delta = option_delta + stock_pos
            
            self.data.loc[idx, 'stock_position'] = stock_pos
            self.data.loc[idx, 'net_delta'] = net_delta
            
            # Check for stop loss
            current_liability = row['current_liability']
            current_pnl = self.premium_received - current_liability - cumulative_tc
            self.data.loc[idx, 'running_pnl'] = current_pnl
            
            # Update max drawdown
            if current_pnl > self.max_pnl:
                self.max_pnl = current_pnl
            drawdown = self.max_pnl - current_pnl
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
            
            # STOP LOSS CHECK
            if current_pnl < -self.config.max_loss_limit and not self.is_stopped_out:
                self.is_stopped_out = True
                self.risk_events.append({
                    'timestamp': idx,
                    'event': 'STOP_LOSS_TRIGGERED',
                    'pnl': current_pnl,
                    'price': price
                })
                print(f"\n⚠️ STOP LOSS TRIGGERED at {idx}")
                print(f"   P&L: ${current_pnl:,.2f}")
                # In practice, would close position here
            
            # Hedge decision
            threshold_shares = self.config.delta_threshold * multiplier
            
            if abs(net_delta) > threshold_shares and not self.is_stopped_out:
                shares_to_trade = -int(round(net_delta))
                
                if abs(shares_to_trade) >= self.config.min_shares_to_hedge:
                    if shares_to_trade > 0:
                        action = 'BUY'
                        reason = 'Delta negative (short exposure), buying to hedge'
                    else:
                        action = 'SELL'
                        reason = 'Delta positive (long exposure), selling to hedge'
                    
                    stock_pos += shares_to_trade
                    total_cash_spent_on_stock += shares_to_trade * price  # Negative for sales
                    
                    tc = abs(shares_to_trade) * price * (self.config.total_tc_bps / 10000)
                    cumulative_tc += tc
                    
                    trade = HedgeTrade(
                        timestamp=idx,
                        action=action,
                        shares=abs(shares_to_trade),
                        price=price,
                        delta_before=net_delta,
                        delta_after=option_delta + stock_pos,
                        transaction_cost=tc,
                        reason=reason
                    )
                    self.trades.append(trade)
                    
                    self.data.loc[idx, 'hedge_trade_shares'] = shares_to_trade
                    self.data.loc[idx, 'hedge_trade_action'] = action
                    self.data.loc[idx, 'stock_position'] = stock_pos
                    self.data.loc[idx, 'net_delta'] = option_delta + stock_pos
            
            self.data.loc[idx, 'cumulative_hedge_cost'] = total_cash_spent_on_stock
            self.data.loc[idx, 'cumulative_tc'] = cumulative_tc
        
        self.stock_position = stock_pos
        
        print(f"   Total hedge trades: {len(self.trades)}")
        print(f"   Final stock position: {stock_pos} shares")
        print(f"   Total transaction costs: ${cumulative_tc:,.2f}")
        if self.is_stopped_out:
            print(f"   ⚠️ POSITION WAS STOPPED OUT")
    
    def _calculate_pnl(self) -> None:
        """
        Calculate P&L for Market Maker.
        
        MARKET MAKER P&L BREAKDOWN:
        
        1. PREMIUM COLLECTED: Positive at start (what we received)
        
        2. OPTION MTM CHANGE: How much the option liability changed
           - If option value went DOWN → Good for us (liability decreased)
           - If option value went UP → Bad for us (liability increased)
        
        3. HEDGE P&L: Always a COST for short gamma
           - We systematically buy high and sell low
           - This is the PRICE we pay for delta neutrality
        
        4. THETA INCOME: We earn this every day
        
        FORMULA:
           Total P&L = Premium - (Current Liability - Initial Liability) 
                       + Stock P&L - Transaction Costs
                     = Premium - Current Liability + Hedge P&L - TC
        """
        print("\n💰 Calculating P&L (Market Maker Perspective)...")
        
        # Initial liability (what straddle was worth when we sold)
        initial_liability = self.data['current_liability'].iloc[0]
        
        # Option P&L from perspective of SHORT position
        # We profit when liability DECREASES (option loses value)
        self.data['option_mtm_pnl'] = initial_liability - self.data['current_liability']
        
        # Stock hedge P&L
        # Cash spent on stock + current value of stock position
        self.data['stock_value'] = self.data['stock_position'] * self.data['price']
        self.data['hedge_pnl'] = -self.data['cumulative_hedge_cost'] + self.data['stock_value']
        
        # Total P&L = Premium + Option MTM Change + Hedge P&L - TC
        self.data['total_pnl'] = (
            self.premium_received + 
            self.data['option_mtm_pnl'] + 
            self.data['hedge_pnl'] - 
            self.data['cumulative_tc']
        )
        
        # Theta income (cumulative)
        self.data['cumulative_theta'] = self.data['position_theta'].cumsum()
        
        # Gamma cost (theoretical)
        # Gamma P&L ≈ -0.5 × Γ × (ΔS)² for each period
        price_changes = self.data['price'].diff().fillna(0)
        self.data['instantaneous_gamma_pnl'] = -0.5 * self.data['position_gamma'] * price_changes**2
        self.data['cumulative_gamma_cost'] = self.data['instantaneous_gamma_pnl'].cumsum()
        
        final = self.data.iloc[-1]
        
        print(f"""
   💵 PREMIUM RECEIVED:     ${self.premium_received:+,.2f}
   📊 Option MTM P&L:       ${final['option_mtm_pnl']:+,.2f}
   🔄 Hedge P&L:            ${final['hedge_pnl']:+,.2f}
   💸 Transaction Costs:    ${final['cumulative_tc']:,.2f}
   {'─'*40}
   📈 TOTAL P&L:            ${final['total_pnl']:+,.2f}
   
   Theory Breakdown:
   ├─ Cumulative Theta:     ${final['cumulative_theta']:+,.2f} (earned)
   └─ Cumulative Gamma:     ${final['cumulative_gamma_cost']:+,.2f} (cost)
""")
    
    def _generate_summary(self) -> dict:
        """Generate comprehensive summary."""
        final = self.data.iloc[-1]
        
        # Calculate realized vol over the period
        period_returns = self.data['log_return'].dropna()
        realized_vol_period = period_returns.std() * np.sqrt(self.config.vol_annualization_factor)
        
        # Vol edge: IV we sold at minus RV that occurred
        initial_iv = self.data['implied_vol'].iloc[0]
        vol_edge = initial_iv - realized_vol_period
        
        return {
            'position_type': self.config.position_type.value,
            'ticker': self.config.ticker,
            'strike': self.strike,
            'initial_price': self.data['price'].iloc[0],
            'final_price': final['price'],
            'price_change_pct': (final['price'] / self.data['price'].iloc[0] - 1) * 100,
            
            # Volatility metrics
            'implied_vol_sold': initial_iv,
            'realized_vol_actual': realized_vol_period,
            'vol_edge': vol_edge,
            'vol_edge_favorable': vol_edge > 0,
            
            # P&L components
            'premium_received': self.premium_received,
            'option_mtm_pnl': final['option_mtm_pnl'],
            'hedge_pnl': final['hedge_pnl'],
            'transaction_costs': final['cumulative_tc'],
            'total_pnl': final['total_pnl'],
            
            # Greek P&L
            'cumulative_theta': final['cumulative_theta'],
            'cumulative_gamma_cost': final['cumulative_gamma_cost'],
            
            # Trading stats
            'num_trades': len(self.trades),
            'final_stock_position': final['stock_position'],
            
            # Risk metrics
            'max_drawdown': self.max_drawdown,
            'was_stopped_out': self.is_stopped_out,
            'margin_required': self.margin_required,
            'return_on_margin': (final['total_pnl'] / self.margin_required * 100) if self.margin_required > 0 else 0
        }

# Execute backtest
backtest = MarketMakerBacktest(config, market_data)
results = backtest.run()

# ====================== CELL 6: RESULTS DISPLAY ======================
print("\n" + "="*70)
print("📊 MARKET MAKER BACKTEST RESULTS")
print("="*70)

s = results.summary

# Determine if trade was profitable
profit_status = "✅ PROFITABLE" if s['total_pnl'] > 0 else "❌ LOSS"
vol_status = "✅ FAVORABLE" if s['vol_edge_favorable'] else "❌ UNFAVORABLE"

print(f"""
Position:               {s['position_type']} STRADDLE (Market Maker)
Ticker:                 {s['ticker']}
Strike:                 ${s['strike']}

PRICE ACTION:
  Initial:              ${s['initial_price']:.2f}
  Final:                ${s['final_price']:.2f}
  Change:               {s['price_change_pct']:+.2f}%

VOLATILITY ANALYSIS:
  Implied Vol (sold):   {s['implied_vol_sold']:.1%}
  Realized Vol:         {s['realized_vol_actual']:.1%}
  Vol Edge:             {s['vol_edge']:+.1%} {vol_status}

{'='*50}
                    P&L ATTRIBUTION
{'='*50}
  💵 Premium Received:  ${s['premium_received']:+,.2f}
  📊 Option MTM:        ${s['option_mtm_pnl']:+,.2f}
  🔄 Hedge P&L:         ${s['hedge_pnl']:+,.2f}
  💸 Trans. Costs:      -${s['transaction_costs']:,.2f}
  {'─'*46}
  📈 TOTAL P&L:         ${s['total_pnl']:+,.2f} {profit_status}

GREEK ATTRIBUTION:
  θ Earned:             ${s['cumulative_theta']:+,.2f}
  Γ Cost:               ${s['cumulative_gamma_cost']:+,.2f}

RISK METRICS:
  Max Drawdown:         ${s['max_drawdown']:,.2f}
  Margin Used:          ${s['margin_required']:,.2f}
  Return on Margin:     {s['return_on_margin']:+.2f}%
  Stopped Out:          {'YES ⚠️' if s['was_stopped_out'] else 'No'}
  Hedge Trades:         {s['num_trades']}
""")

# Trade log
if results.trades:
    print("\n📝 Hedge Trade Log (Last 10):")
    trade_df = pd.DataFrame([{
        'Time': t.timestamp.strftime('%m/%d %H:%M'),
        'Action': t.action,
        'Shares': t.shares,
        'Price': f"${t.price:.2f}",
        'Δ Before': f"{t.delta_before:+.0f}",
        'Δ After': f"{t.delta_after:+.0f}",
    } for t in results.trades[-10:]])
    display(trade_df)

# ====================== CELL 7: VISUALIZATION ======================
def create_market_maker_visualizations(results: BacktestResults, config: MarketMakerConfig):
    """Create comprehensive visualizations for market maker perspective."""
    df = results.df
    trades = results.trades
    summary = results.summary
    
    fig = plt.figure(figsize=(16, 24))
    
    colors = {
        'price': '#2C3E50',
        'buy': '#27AE60',
        'sell': '#E74C3C',
        'premium': '#F39C12',
        'option_pnl': '#3498DB',
        'hedge_pnl': '#9B59B6',
        'total_pnl': '#1ABC9C',
        'theta': '#27AE60',
        'gamma': '#E74C3C',
        'vol': '#8E44AD'
    }
    
    # ========== SUBPLOT 1: Price & Trades ==========
    ax1 = fig.add_subplot(5, 1, 1)
    
    ax1.plot(df.index, df['price'], color=colors['price'], linewidth=1.5, alpha=0.9)
    ax1.axhline(y=summary['strike'], color='orange', linestyle='--',
                alpha=0.7, label=f"Strike ${summary['strike']}")
    
    # Trade markers
    buy_trades = [t for t in trades if t.action == 'BUY']
    sell_trades = [t for t in trades if t.action == 'SELL']
    
    if buy_trades:
        ax1.scatter([t.timestamp for t in buy_trades],
                   [t.price for t in buy_trades],
                   c=colors['buy'], marker='^', s=80, alpha=0.7,
                   label=f'Buy Stock ({len(buy_trades)})', zorder=5)
    if sell_trades:
        ax1.scatter([t.timestamp for t in sell_trades],
                   [t.price for t in sell_trades],
                   c=colors['sell'], marker='v', s=80, alpha=0.7,
                   label=f'Sell Stock ({len(sell_trades)})', zorder=5)
    
    ax1.set_title(f'Market Maker: {config.ticker} SHORT Straddle - Price & Hedge Trades',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # ========== SUBPLOT 2: P&L Components ==========
    ax2 = fig.add_subplot(5, 1, 2)
    
    # Premium line (constant)
    ax2.axhline(y=summary['premium_received'], color=colors['premium'],
                linestyle='--', alpha=0.7, label=f"Premium ${summary['premium_received']:,.0f}")
    
    # P&L lines
    ax2.plot(df.index, df['total_pnl'], color=colors['total_pnl'],
             linewidth=2.5, label='Total P&L', zorder=5)
    ax2.plot(df.index, summary['premium_received'] + df['option_mtm_pnl'],
             color=colors['option_pnl'], linewidth=1.5, linestyle='--',
             label='Premium + Option MTM', alpha=0.7)
    ax2.plot(df.index, df['hedge_pnl'], color=colors['hedge_pnl'],
             linewidth=1.5, linestyle=':', label='Hedge P&L', alpha=0.7)
    
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.fill_between(df.index, 0, df['total_pnl'],
                     where=(df['total_pnl'] > 0), color='green', alpha=0.2)
    ax2.fill_between(df.index, 0, df['total_pnl'],
                     where=(df['total_pnl'] <= 0), color='red', alpha=0.2)
    
    ax2.set_title('P&L Attribution (Market Maker)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('P&L ($)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    final_pnl = df['total_pnl'].iloc[-1]
    ax2.annotate(f'Final: ${final_pnl:+,.0f}', xy=(df.index[-1], final_pnl),
                xytext=(10, 0), textcoords='offset points', fontsize=11, fontweight='bold')
    
    # ========== SUBPLOT 3: Theta vs Gamma ==========
    ax3 = fig.add_subplot(5, 1, 3)
    
    ax3.plot(df.index, df['cumulative_theta'], color=colors['theta'],
             linewidth=2, label='Cumulative Theta (Earned)')
    ax3.plot(df.index, df['cumulative_gamma_cost'], color=colors['gamma'],
             linewidth=2, label='Cumulative Gamma (Cost)')
    ax3.plot(df.index, df['cumulative_theta'] + df['cumulative_gamma_cost'],
             color='purple', linewidth=2, linestyle='--', label='Net Greek P&L')
    
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.fill_between(df.index, df['cumulative_theta'], df['cumulative_gamma_cost'],
                     where=(df['cumulative_theta'] > -df['cumulative_gamma_cost']),
                     color='green', alpha=0.2, label='Theta > Gamma')
    ax3.fill_between(df.index, df['cumulative_theta'], df['cumulative_gamma_cost'],
                     where=(df['cumulative_theta'] <= -df['cumulative_gamma_cost']),
                     color='red', alpha=0.2, label='Gamma > Theta')
    
    ax3.set_title('Theta Income vs Gamma Cost (The Core Market Maker Tradeoff)',
                  fontsize=14, fontweight='bold')
    ax3.set_ylabel('Cumulative P&L ($)')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # ========== SUBPLOT 4: Delta & Stock Position ==========
    ax4 = fig.add_subplot(5, 1, 4)
    ax4_twin = ax4.twinx()
    
    ax4.plot(df.index, df['net_delta'], color='darkorange', linewidth=1.5)
    ax4.fill_between(df.index, 0, df['net_delta'], color='orange', alpha=0.3)
    ax4.axhline(y=0, color='black', linewidth=0.5)
    
    threshold = config.delta_threshold * config.num_straddles * config.contract_multiplier
    ax4.axhline(y=threshold, color='gray', linestyle=':', alpha=0.5)
    ax4.axhline(y=-threshold, color='gray', linestyle=':', alpha=0.5)
    
    ax4_twin.plot(df.index, df['stock_position'], color='blue',
                  linewidth=1.5, linestyle='--', alpha=0.7)
    
    ax4.set_title('Net Delta & Stock Hedge Position', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Net Delta', color='darkorange')
    ax4_twin.set_ylabel('Stock Position', color='blue')
    ax4.grid(True, alpha=0.3)
    
    # ========== SUBPLOT 5: Realized vs Implied Volatility ==========
    ax5 = fig.add_subplot(5, 1, 5)
    
    ax5.plot(df.index, df['realized_vol'] * 100, color='blue',
             linewidth=1.5, label='Realized Vol')
    ax5.plot(df.index, df['implied_vol'] * 100, color='red',
             linewidth=1.5, linestyle='--', label='Implied Vol (Sold At)')
    ax5.fill_between(df.index, df['realized_vol']*100, df['implied_vol']*100,
                     where=(df['implied_vol'] > df['realized_vol']),
                     color='green', alpha=0.3, label='Vol Edge (Favorable)')
    ax5.fill_between(df.index, df['realized_vol']*100, df['implied_vol']*100,
                     where=(df['implied_vol'] <= df['realized_vol']),
                     color='red', alpha=0.3, label='Vol Edge (Unfavorable)')
    
    ax5.set_title('Implied vs Realized Volatility (Market Maker Edge)',
                  fontsize=14, fontweight='bold')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Volatility (%)')
    ax5.legend(loc='upper left')
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # ========== SUMMARY CHART ==========
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # P&L waterfall
    components = ['Premium\nReceived', 'Option\nMTM', 'Hedge\nP&L', 'Trans.\nCosts', 'TOTAL\nP&L']
    values = [
        summary['premium_received'],
        summary['option_mtm_pnl'],
        summary['hedge_pnl'],
        -summary['transaction_costs'],
        summary['total_pnl']
    ]
    colors_bar = ['green', 'blue' if values[1]>0 else 'red',
                  'blue' if values[2]>0 else 'red', 'red',
                  'green' if values[4]>0 else 'red']
    
    axes[0].bar(components, values, color=colors_bar, alpha=0.8, edgecolor='black')
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].set_title('P&L Waterfall', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('P&L ($)')
    for i, v in enumerate(values):
        axes[0].text(i, v + (500 if v>0 else -500), f'${v:+,.0f}',
                    ha='center', va='bottom' if v>0 else 'top', fontsize=10)
    
    # Greek breakdown
    greek_labels = ['Theta\n(Earned)', 'Gamma\n(Cost)', 'Net\nGreek P&L']
    greek_values = [
        summary['cumulative_theta'],
        summary['cumulative_gamma_cost'],
        summary['cumulative_theta'] + summary['cumulative_gamma_cost']
    ]
    greek_colors = ['green', 'red', 'green' if greek_values[2]>0 else 'red']
    
    axes[1].bar(greek_labels, greek_values, color=greek_colors, alpha=0.8, edgecolor='black')
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_title('Greek P&L Attribution', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('P&L ($)')
    for i, v in enumerate(greek_values):
        axes[1].text(i, v + (300 if v>0 else -300), f'${v:+,.0f}',
                    ha='center', va='bottom' if v>0 else 'top', fontsize=10)
    
    plt.tight_layout()
    plt.show()

create_market_maker_visualizations(results, config)

# ====================== CELL 8: COMPARISON - LONG vs SHORT ======================
print("\n" + "="*70)
print("📊 COMPARISON: Long Straddle vs Short Straddle (Market Maker)")
print("="*70)

# Run long straddle for comparison
config_long = MarketMakerConfig(
    ticker=config.ticker,
    position_type=PositionType.LONG,
    delta_threshold=0.10
)

backtest_long = MarketMakerBacktest(config_long, market_data)
results_long = backtest_long.run()

# Comparison table
comparison_data = {
    'Metric': [
        'Position Type',
        'Premium Flow',
        'Gamma Exposure',
        'Theta Exposure',
        'Hedge Trades',
        'Total P&L',
        'Vol Edge Needed'
    ],
    'Long Straddle': [
        'BUY options',
        f"-${results_long.summary['premium_received']:,.0f} (paid)",
        'LONG (profit from moves)',
        'SHORT (pay decay)',
        results_long.summary['num_trades'],
        f"${results_long.summary['total_pnl']:+,.0f}",
        'RV > IV'
    ],
    'Short Straddle (MM)': [
        'SELL options',
        f"+${results.summary['premium_received']:,.0f} (received)",
        'SHORT (lose from moves)',
        'LONG (earn decay)',
        results.summary['num_trades'],
        f"${results.summary['total_pnl']:+,.0f}",
        'IV > RV'
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n")
display(comparison_df.style.set_properties(**{'text-align': 'center'}))

# Plot comparison
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(results_long.df.index, results_long.df['total_pnl'],
        color='blue', linewidth=2, label='Long Straddle (Vol Buyer)')
ax.plot(results.df.index, results.df['total_pnl'],
        color='red', linewidth=2, label='Short Straddle (Market Maker)')

ax.axhline(y=0, color='black', linewidth=0.5)
ax.fill_between(results.df.index, results_long.df['total_pnl'], results.df['total_pnl'],
                alpha=0.3, color='gray')

ax.set_title('Long vs Short Straddle: Mirror Image P&L', fontsize=14, fontweight='bold')
ax.set_xlabel('Time')
ax.set_ylabel('Total P&L ($)')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Annotate
ax.annotate(f"Long: ${results_long.summary['total_pnl']:+,.0f}",
           xy=(results.df.index[-1], results_long.df['total_pnl'].iloc[-1]),
           xytext=(10, 10), textcoords='offset points', color='blue', fontweight='bold')
ax.annotate(f"Short: ${results.summary['total_pnl']:+,.0f}",
           xy=(results.df.index[-1], results.df['total_pnl'].iloc[-1]),
           xytext=(10, -15), textcoords='offset points', color='red', fontweight='bold')

plt.tight_layout()
plt.show()

# ====================== CELL 9: EDUCATIONAL SUMMARY ======================
print("\n" + "="*70)
print("📚 EDUCATIONAL SUMMARY: Market Maker Perspective")
print("="*70)

print("""
KEY CONCEPTS - MARKET MAKER (SHORT GAMMA):

1. PREMIUM COLLECTION (Income)
   ├─ We SELL options at IMPLIED volatility
   ├─ Collect premium upfront (our max profit if calm)
   └─ This is the "insurance premium" we earn

2. THETA DECAY (Our Friend)
   ├─ Every day, options lose value
   ├─ Since we're SHORT, this is INCOME
   ├─ θ/day × days = Cumulative theta income
   └─ Works best in calm markets

3. GAMMA EXPOSURE (Our Enemy)
   ├─ We are SHORT gamma (negative convexity)
   ├─ Large moves HURT us disproportionately
   ├─ Γ × (ΔS)² = Instantaneous gamma loss
   └─ Hedging COSTS money (buy high, sell low)

4. DELTA HEDGING (Mandatory for Survival)
   ├─ Price UP → Delta becomes negative → BUY stock (at high)
   ├─ Price DOWN → Delta becomes positive → SELL stock (at low)
   ├─ We're always "chasing" the market
   └─ Hedging is a COST, not a profit center

5. VOLATILITY RISK PREMIUM (The Edge)
   ├─ Market tends to price IV > RV (fear premium)
   ├─ We SELL at IV, reality is RV
   ├─ If IV > RV → We profit (sold expensive insurance, few claims)
   └─ If RV > IV → We lose (sold cheap insurance, many claims)

MARKET MAKER P&L FORMULA:
   P&L = Premium + θ×Time - ½×Γ×(ΔS)² - Transaction Costs
   
   WHERE:
   • Premium = what we collected upfront
   • θ×Time = theta decay earned
   • ½×Γ×(ΔS)² = gamma losses from hedging
   • TC = transaction costs from frequent hedging

PROFITABILITY CONDITION:
   ┌────────────────────────────────────────┐
   │  IMPLIED VOL > REALIZED VOL            │
   │  (We sold volatility expensive,        │
   │   market was calmer than expected)     │
   └────────────────────────────────────────┘

RISK MANAGEMENT ESSENTIALS:
   • Position sizing (this has UNLIMITED risk!)
   • Stop-loss limits
   • Tighter delta thresholds (hedge more frequently)
   • Diversification across strikes/expirations
   • Avoid earnings/events (vol spikes)

COMPARISON TO LONG STRADDLE:
   ┌──────────────────┬─────────────────┬─────────────────┐
   │ Factor           │ Long Straddle   │ Short Straddle  │
   ├──────────────────┼─────────────────┼─────────────────┤
   │ Premium          │ PAY (cost)      │ RECEIVE (income)│
   │ Gamma            │ LONG (want vol) │ SHORT (hate vol)│
   │ Theta            │ PAY (decay)     │ EARN (decay)    │
   │ Hedge P&L        │ EARN (scalp)    │ COST (chase)    │
   │ Max Loss         │ Limited         │ UNLIMITED       │
   │ Max Profit       │ Unlimited       │ Limited         │
   │ Win when         │ RV > IV         │ IV > RV         │
   └──────────────────┴─────────────────┴─────────────────┘
""")

print("\n✅ Market Maker backtest complete!")
print("   Adjust 'implied_vol_premium' in config to simulate different vol regimes.")