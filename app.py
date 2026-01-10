import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, skew, kurtosis
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import warnings
from fpdf import FPDF
import tempfile
import os

warnings.filterwarnings('ignore')

# ==========================================
# 1. KONFIGURASI & CSS (ENHANCED)
# ==========================================
st.set_page_config(
    page_title="The Little Quant Terminal | Institutional Edition",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {background-color: #0e1117;}
    .metric-card {background-color: #262730; border: 1px solid #4e4f52; padding: 15px; border-radius: 5px; color: white;}
    .buy-signal {color: #00FF00; font-weight: bold;}
    .sell-signal {color: #FF3333; font-weight: bold;}
    
    /* NEW INSTITUTIONAL STYLES */
    .quant-header {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 20px; border-radius: 10px; margin-bottom: 20px;
        border: 1px solid #e94560;
    }
    .risk-critical {background-color: #ff1744; color: white; padding: 10px; border-radius: 5px; font-weight: bold;}
    .risk-warning {background-color: #ff9100; color: black; padding: 10px; border-radius: 5px;}
    .risk-normal {background-color: #00e676; color: black; padding: 10px; border-radius: 5px;}
    .institutional-badge {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 5px 15px; border-radius: 20px; font-size: 12px;
    }
    .var-box {
        background-color: #1e1e2f; border: 2px solid #ff6b6b; 
        padding: 15px; border-radius: 8px; margin: 5px 0;
    }
    .monte-carlo-box {
        background: linear-gradient(135deg, #0c0c1e 0%, #1a1a3e 100%);
        border: 1px solid #4ecdc4; padding: 15px; border-radius: 8px;
    }
    .correlation-matrix {border: 1px solid #45b7d1; border-radius: 8px; padding: 10px;}
    .stress-test-card {
        background-color: #2d2d44; border-left: 4px solid #e94560;
        padding: 15px; margin: 10px 0; border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Session State Init (PRESERVED + NEW)
if 'analyzed' not in st.session_state: st.session_state.analyzed = False
if 'pdf_bytes' not in st.session_state: st.session_state.pdf_bytes = None
if 'ticker_file_name' not in st.session_state: st.session_state.ticker_file_name = "report"
if 'radar_results' not in st.session_state: st.session_state.radar_results = pd.DataFrame()
if 'signal_results' not in st.session_state: st.session_state.signal_results = pd.DataFrame()

# NEW SESSION STATES FOR INSTITUTIONAL FEATURES
if 'var_results' not in st.session_state: st.session_state.var_results = None
if 'monte_carlo_paths' not in st.session_state: st.session_state.monte_carlo_paths = None
if 'correlation_matrix' not in st.session_state: st.session_state.correlation_matrix = None
if 'portfolio_weights' not in st.session_state: st.session_state.portfolio_weights = None
if 'stress_results' not in st.session_state: st.session_state.stress_results = None
if 'regime_analysis' not in st.session_state: st.session_state.regime_analysis = None
if 'ml_predictions' not in st.session_state: st.session_state.ml_predictions = None

# ==========================================
# 2. MESIN 1: GAMMA SCALPING (AUDIT ENGINE) - PRESERVED
# ==========================================
class PositionType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

@dataclass
class MarketMakerConfig:
    ticker: str = "NVDA"
    interval: str = "5m"
    lookback_days: int = 5
    position_type: PositionType = PositionType.SHORT 
    time_to_expiry_days: int = 30
    risk_free_rate: float = 0.05
    implied_vol_premium: float = 0.05 
    realized_vol_window: int = 78
    vol_annualization_factor: int = 252 * 78
    delta_threshold: float = 0.05 
    min_shares_to_hedge: int = 5
    num_straddles: int = 5
    contract_multiplier: int = 100
    stock_commission_bps: float = 1.0
    slippage_bps: float = 2.0
    
    @property
    def total_tc_bps(self) -> float: return self.stock_commission_bps + self.slippage_bps
    @property
    def position_sign(self) -> int: return 1 if self.position_type == PositionType.LONG else -1

def bsm_vectorized(S, K, T, r, sigma, position_sign=1):
    T = T.clip(lower=1e-10)
    sigma = sigma.clip(lower=1e-10)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    N_d1 = norm.cdf(d1); N_d2 = norm.cdf(d2)
    N_neg_d1 = norm.cdf(-d1); N_neg_d2 = norm.cdf(-d2)
    n_d1 = norm.pdf(d1)
    discount = np.exp(-r * T)
    call_price = S * N_d1 - K * discount * N_d2
    put_price = K * discount * N_neg_d2 - S * N_neg_d1
    straddle_price = call_price + put_price
    call_delta = N_d1; put_delta = N_d1 - 1
    straddle_delta = call_delta + put_delta
    gamma = n_d1 / (S * sigma * sqrt_T)
    theta_term1 = -(S * n_d1 * sigma) / (2 * sqrt_T)
    theta_call = (theta_term1 - r * K * discount * N_d2) / 365
    theta_put = (theta_term1 + r * K * discount * N_neg_d2) / 365
    straddle_theta = theta_call + theta_put
    return pd.DataFrame({
        'price': S, 'straddle_price_raw': straddle_price,
        'pos_value': straddle_price * position_sign,
        'pos_delta': straddle_delta * position_sign,
        'pos_gamma': 2 * gamma, 'pos_theta': straddle_theta * position_sign * -1,
    })

class DataIngestion:
    def __init__(self, config): self.config = config
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_data(_self, ticker, period, interval):
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        return data
    def process(self, df):
        df = df.dropna(subset=['Close']).copy()
        df['price'] = df['Close']
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        ann_factor = self.config.vol_annualization_factor
        df['realized_vol'] = df['log_return'].rolling(window=self.config.realized_vol_window).std() * np.sqrt(ann_factor)
        df['realized_vol'] = df['realized_vol'].fillna(method='bfill').fillna(0.30).clip(0.05, 2.0)
        df['implied_vol'] = df['realized_vol'] + self.config.implied_vol_premium
        df['bar_index'] = range(len(df))
        return df

@dataclass
class HedgeTrade:
    timestamp: datetime; action: str; shares: int; price: float; transaction_cost: float

class MarketMakerBacktest:
    def __init__(self, config, market_data): self.config = config; self.data = market_data.copy(); self.trades = []
    def run(self):
        first_price = self.data['price'].iloc[0]; strike = round(first_price)
        bars_per_day = 78
        self.data['T'] = (self.config.time_to_expiry_days/365 - self.data['bar_index']/(bars_per_day*252)).clip(1e-6)
        bsm_res = bsm_vectorized(self.data['price'], strike, self.data['T'], self.config.risk_free_rate, self.data['realized_vol'], self.config.position_sign)
        mult = self.config.num_straddles * self.config.contract_multiplier
        self.data['pos_delta'] = bsm_res['pos_delta'] * mult
        self.data['pos_gamma'] = bsm_res['pos_gamma'] * mult
        self.data['pos_theta'] = bsm_res['pos_theta'] * mult
        self.data['liability'] = bsm_res['straddle_price_raw'] * mult
        
        initial_iv = self.data['implied_vol'].iloc[0]
        initial_bsm = bsm_vectorized(pd.Series([first_price]), strike, pd.Series([self.config.time_to_expiry_days/365]), 0.05, pd.Series([initial_iv]))
        premium_total = initial_bsm['straddle_price_raw'].iloc[0] * mult
        
        stock_pos = 0; cash_flow_stock = 0.0; cum_tc = 0.0
        stock_hist, pnl_hist = [], []
        threshold = self.config.delta_threshold * mult
        
        for idx, row in self.data.iterrows():
            net_delta = row['pos_delta'] + stock_pos
            if abs(net_delta) > threshold:
                shares = -int(round(net_delta))
                if abs(shares) >= self.config.min_shares_to_hedge:
                    price = row['price']
                    cost = abs(shares) * price * (self.config.total_tc_bps/10000)
                    stock_pos += shares; cash_flow_stock -= (shares * price); cum_tc += cost
                    self.trades.append(HedgeTrade(idx, "BUY" if shares>0 else "SELL", abs(shares), price, cost))
            stock_hist.append(stock_pos)
            current_stock_val = stock_pos * row['price']
            hedge_pnl = cash_flow_stock + current_stock_val
            opt_pnl = row['liability'] - premium_total if self.config.position_type == PositionType.LONG else premium_total - row['liability']
            pnl_hist.append(opt_pnl + hedge_pnl - cum_tc)
        
        self.data['stock_pos'] = stock_hist; self.data['total_pnl'] = pnl_hist; self.data['cum_tc'] = cum_tc
        self.data['opt_pnl'] = [premium_total - l if self.config.position_type == PositionType.SHORT else l - premium_total for l in self.data['liability']]
        return self.data, self.trades, premium_total

# --- PDF GENERATOR (FULL) - PRESERVED ---
class PDFReport(FPDF):
    def header(self): self.set_font('Arial', 'B', 15); self.cell(0, 10, "The Little Quant | Institutional Analysis Report", 0, 1, 'C'); self.ln(5)
    def footer(self): self.set_y(-15); self.set_font('Arial', 'I', 8); self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf(ticker, strategy_name, final_metrics, trades_df, figures):
    pdf = PDFReport(); pdf.add_page(); pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 12); pdf.cell(0, 10, f"Ticker: {ticker} | Strategi: {strategy_name}", 0, 1)
    pdf.set_font("Arial", "", 10); pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1); pdf.ln(5)
    pdf.set_fill_color(240, 240, 240); pdf.cell(0, 10, "Executive Summary:", 0, 1, 'L', fill=True)
    pdf.cell(95, 10, f"Net P&L: {final_metrics['pnl_str']}", 1); pdf.cell(95, 10, f"Trades: {final_metrics['trades_count']}", 1, 1); pdf.ln(10)
    pdf.cell(0, 10, "Risk & Performance Visuals:", 0, 1)
    for fig in figures:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            fig.savefig(tmp.name, bbox_inches='tight', dpi=100); pdf.image(tmp.name, x=10, w=190); pdf.ln(5)
    
    pdf.add_page(); pdf.set_font("Arial", "B", 12); pdf.cell(0, 10, "Transaction Log (Last 20)", 0, 1)
    pdf.set_font("Arial", "", 8); pdf.cell(40, 8, "Timestamp", 1); pdf.cell(20, 8, "Action", 1); pdf.cell(30, 8, "Price", 1); pdf.cell(30, 8, "Shares", 1); pdf.cell(30, 8, "Cost", 1, 1)
    if not trades_df.empty:
        for idx, row in trades_df.tail(20).iterrows():
            pdf.cell(40, 8, str(row['timestamp']), 1); pdf.cell(20, 8, row['action'], 1); pdf.cell(30, 8, str(row['price']), 1); pdf.cell(30, 8, str(row['shares']), 1); pdf.cell(30, 8, str(row['transaction_cost']), 1, 1)
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 3. MESIN 2: SMART WHALE RADAR - PRESERVED
# ==========================================
def whale_radar_scanner(tickers):
    radar_data = []
    status_text = st.empty()
    status_text.info("📡 Menyalakan Radar... Memindai Jejak Institusi...")
    prog = st.progress(0)
    clean_tickers = list(set([t.strip().upper() for t in tickers if t.strip() != ""]))
    
    for i, ticker in enumerate(clean_tickers):
        try:
            df = yf.download(ticker, period="3mo", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 20: 
                continue
            df = df.dropna()
            last_close = float(df['Close'].iloc[-1])
            last_open = float(df['Open'].iloc[-1])
            last_vol = float(df['Volume'].iloc[-1])
            if last_vol == 0: 
                continue
            history = df.iloc[:-1]
            vol_mean = history['Volume'].tail(20).mean()
            vol_std = history['Volume'].tail(20).std()
            vol_z = (last_vol - vol_mean) / vol_std if vol_std > 0 else 0
            rvol = last_vol / vol_mean if vol_mean > 0 else 0
            body_pct = abs(last_close - last_open) / last_open if last_open > 0 else 0
            
            status = "Normal"
            score = 0
            
            if vol_z > 2.0 and body_pct < 0.005: 
                status = "🛡️ ABSORPTION (Tembok)"
                score = 3
            elif vol_z > 2.0 and last_close > last_open: 
                status = "🚀 MARK-UP (Akumulasi)"
                score = 3
            elif vol_z > 2.0 and last_close < last_open: 
                status = "🔻 DISTRIBUTION (Guyur)"
                score = 3
            elif body_pct > 0.03:
                status = "🌪️ VOLATILE (Gerak Liar)"
                score = 2
            elif vol_z > 1.2: 
                status = "👀 High Vol"
                score = 1
            
            if score > 0:
                radar_data.append({
                    "Ticker": ticker, 
                    "Price": last_close, 
                    "Z-Score": vol_z, 
                    "RVOL": rvol, 
                    "Status": status, 
                    "_score": score
                })
        except Exception as e:
            continue
        prog.progress((i+1)/len(clean_tickers))
        
    status_text.empty()
    prog.empty()
    
    if not radar_data: 
        return pd.DataFrame()
    return pd.DataFrame(radar_data).sort_values(by=["_score", "Z-Score"], ascending=False).drop(columns=["_score"])

# ==========================================
# 4. MESIN 3: SIGNAL GENERATOR - PRESERVED
# ==========================================
def generate_signals(tickers):
    signals = []
    status_text = st.empty(); status_text.info("🎯 Calculating Probabilities..."); prog = st.progress(0)
    clean_tickers = list(set([t.strip().upper() for t in tickers if t.strip() != ""]))
    
    for i, ticker in enumerate(clean_tickers):
        try:
            df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if len(df) < 50: continue
            close = df['Close']
            
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss; rsi = 100 - (100 / (1 + rs)); curr_rsi = rsi.iloc[-1]
            
            sma20 = close.rolling(20).mean(); std20 = close.rolling(20).std()
            upper_bb = sma20 + (2 * std20); lower_bb = sma20 - (2 * std20)
            curr_price = close.iloc[-1]
            
            sma50 = close.rolling(50).mean().iloc[-1]; trend = "BULL" if curr_price > sma50 else "BEAR"
            
            action = "WAIT"; reason = "-"; confidence = "Low"
            
            if curr_price < lower_bb.iloc[-1] and curr_rsi < 35:
                action = "BUY (Reversal)"; reason = f"Oversold (RSI {curr_rsi:.0f}) + Tembus BB Bawah"
            elif curr_price > upper_bb.iloc[-1] and trend == "BULL":
                action = "BUY (Breakout)"; reason = "Momentum kuat tembus BB Atas"
            elif curr_rsi > 75:
                action = "SELL/TP"; reason = f"Overbought (RSI {curr_rsi:.0f})"
            elif trend == "BEAR" and curr_rsi > 60:
                action = "SHORT"; reason = "Pantulan di Tren Turun (Bearish)"

            if action != "WAIT":
                signals.append({"Ticker": ticker, "Price": curr_price, "Action": action, "Reason": reason, "Trend": trend, "RSI": f"{curr_rsi:.1f}"})
        except: continue
        prog.progress((i+1)/len(clean_tickers))
        
    status_text.empty(); prog.empty()
    if not signals: return pd.DataFrame()
    return pd.DataFrame(signals).sort_values(by=["Action"])

# ==========================================
# 5. NEW MESIN 4: INSTITUTIONAL VAR ENGINE
# ==========================================
class InstitutionalVaREngine:
    """
    Institutional-Grade Value at Risk Engine
    Supports: Historical, Parametric, Monte Carlo, Cornish-Fisher VaR
    """
    
    def __init__(self, returns: pd.Series, confidence_levels: List[float] = [0.95, 0.99]):
        self.returns = returns.dropna()
        self.confidence_levels = confidence_levels
        
    def historical_var(self) -> Dict[float, float]:
        """Historical Simulation VaR"""
        results = {}
        for cl in self.confidence_levels:
            results[cl] = np.percentile(self.returns, (1 - cl) * 100)
        return results
    
    def parametric_var(self) -> Dict[float, float]:
        """Parametric (Gaussian) VaR"""
        mu = self.returns.mean()
        sigma = self.returns.std()
        results = {}
        for cl in self.confidence_levels:
            z_score = norm.ppf(1 - cl)
            results[cl] = mu + z_score * sigma
        return results
    
    def cornish_fisher_var(self) -> Dict[float, float]:
        """Cornish-Fisher VaR (Adjusts for Skewness & Kurtosis)"""
        mu = self.returns.mean()
        sigma = self.returns.std()
        s = skew(self.returns)
        k = kurtosis(self.returns)
        
        results = {}
        for cl in self.confidence_levels:
            z = norm.ppf(1 - cl)
            # Cornish-Fisher expansion
            z_cf = (z + (z**2 - 1) * s / 6 + 
                    (z**3 - 3*z) * (k - 3) / 24 - 
                    (2*z**3 - 5*z) * s**2 / 36)
            results[cl] = mu + z_cf * sigma
        return results
    
    def expected_shortfall(self) -> Dict[float, float]:
        """Expected Shortfall (CVaR) - Average loss beyond VaR"""
        results = {}
        for cl in self.confidence_levels:
            var = np.percentile(self.returns, (1 - cl) * 100)
            results[cl] = self.returns[self.returns <= var].mean()
        return results
    
    def monte_carlo_var(self, num_simulations: int = 10000, horizon: int = 1) -> Tuple[Dict[float, float], np.ndarray]:
        """Monte Carlo VaR with GBM simulation"""
        mu = self.returns.mean()
        sigma = self.returns.std()
        
        # Simulate paths
        np.random.seed(42)
        simulated_returns = np.random.normal(mu * horizon, sigma * np.sqrt(horizon), num_simulations)
        
        results = {}
        for cl in self.confidence_levels:
            results[cl] = np.percentile(simulated_returns, (1 - cl) * 100)
        
        return results, simulated_returns
    
    def full_risk_report(self) -> pd.DataFrame:
        """Generate comprehensive VaR report"""
        hist_var = self.historical_var()
        param_var = self.parametric_var()
        cf_var = self.cornish_fisher_var()
        es = self.expected_shortfall()
        mc_var, _ = self.monte_carlo_var()
        
        data = []
        for cl in self.confidence_levels:
            data.append({
                'Confidence Level': f'{cl*100:.0f}%',
                'Historical VaR': f'{hist_var[cl]*100:.2f}%',
                'Parametric VaR': f'{param_var[cl]*100:.2f}%',
                'Cornish-Fisher VaR': f'{cf_var[cl]*100:.2f}%',
                'Monte Carlo VaR': f'{mc_var[cl]*100:.2f}%',
                'Expected Shortfall': f'{es[cl]*100:.2f}%'
            })
        
        return pd.DataFrame(data)

# ==========================================
# 6. NEW MESIN 5: MONTE CARLO SIMULATOR
# ==========================================
class MonteCarloSimulator:
    """
    Advanced Monte Carlo Price Path Simulator
    Supports: GBM, Jump Diffusion, Heston Model approximation
    """
    
    def __init__(self, current_price: float, mu: float, sigma: float):
        self.S0 = current_price
        self.mu = mu
        self.sigma = sigma
    
    def simulate_gbm(self, days: int = 252, num_paths: int = 1000) -> np.ndarray:
        """Geometric Brownian Motion simulation"""
        dt = 1/252
        np.random.seed(42)
        
        paths = np.zeros((days + 1, num_paths))
        paths[0] = self.S0
        
        for t in range(1, days + 1):
            z = np.random.standard_normal(num_paths)
            paths[t] = paths[t-1] * np.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z)
        
        return paths
    
    def simulate_jump_diffusion(self, days: int = 252, num_paths: int = 1000, 
                                  lambda_j: float = 0.1, mu_j: float = -0.02, sigma_j: float = 0.1) -> np.ndarray:
        """Merton Jump Diffusion Model"""
        dt = 1/252
        np.random.seed(42)
        
        paths = np.zeros((days + 1, num_paths))
        paths[0] = self.S0
        
        for t in range(1, days + 1):
            z = np.random.standard_normal(num_paths)
            jumps = np.random.poisson(lambda_j * dt, num_paths)
            jump_sizes = np.random.normal(mu_j, sigma_j, num_paths) * jumps
            
            paths[t] = paths[t-1] * np.exp(
                (self.mu - 0.5 * self.sigma**2) * dt + 
                self.sigma * np.sqrt(dt) * z + 
                jump_sizes
            )
        
        return paths
    
    def get_statistics(self, paths: np.ndarray) -> Dict:
        """Calculate path statistics"""
        final_prices = paths[-1]
        
        return {
            'mean_price': np.mean(final_prices),
            'median_price': np.median(final_prices),
            'std_price': np.std(final_prices),
            'percentile_5': np.percentile(final_prices, 5),
            'percentile_25': np.percentile(final_prices, 25),
            'percentile_75': np.percentile(final_prices, 75),
            'percentile_95': np.percentile(final_prices, 95),
            'prob_profit': np.mean(final_prices > self.S0) * 100,
            'max_price': np.max(final_prices),
            'min_price': np.min(final_prices)
        }

# ==========================================
# 7. NEW MESIN 6: PORTFOLIO OPTIMIZER
# ==========================================
class PortfolioOptimizer:
    """
    Institutional Portfolio Optimization Engine
    Supports: Mean-Variance, Risk Parity, Maximum Sharpe, Minimum Variance
    """
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        self.returns = returns.dropna()
        self.rf = risk_free_rate / 252  # Daily risk-free rate
        self.n_assets = len(returns.columns)
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        
    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        ret = np.sum(self.mean_returns * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        sharpe = (ret - self.rf * 252) / vol
        return ret, vol, sharpe
    
    def optimize_sharpe(self) -> Dict:
        """Maximum Sharpe Ratio Portfolio"""
        def neg_sharpe(weights):
            return -self.portfolio_performance(weights)[2]
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(neg_sharpe, initial, method='SLSQP', bounds=bounds, constraints=constraints)
        
        ret, vol, sharpe = self.portfolio_performance(result.x)
        return {
            'weights': result.x,
            'return': ret,
            'volatility': vol,
            'sharpe': sharpe,
            'type': 'Maximum Sharpe'
        }
    
    def optimize_min_variance(self) -> Dict:
        """Minimum Variance Portfolio"""
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.cov_matrix * 252, weights))
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(portfolio_variance, initial, method='SLSQP', bounds=bounds, constraints=constraints)
        
        ret, vol, sharpe = self.portfolio_performance(result.x)
        return {
            'weights': result.x,
            'return': ret,
            'volatility': vol,
            'sharpe': sharpe,
            'type': 'Minimum Variance'
        }
    
    def risk_parity(self) -> Dict:
        """Risk Parity Portfolio - Equal Risk Contribution"""
        def risk_budget_objective(weights):
            vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
            marginal_contrib = np.dot(self.cov_matrix * 252, weights)
            risk_contrib = weights * marginal_contrib / vol
            target_risk = vol / self.n_assets
            return np.sum((risk_contrib - target_risk)**2)
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.01, 1) for _ in range(self.n_assets))
        initial = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(risk_budget_objective, initial, method='SLSQP', bounds=bounds, constraints=constraints)
        
        ret, vol, sharpe = self.portfolio_performance(result.x)
        return {
            'weights': result.x,
            'return': ret,
            'volatility': vol,
            'sharpe': sharpe,
            'type': 'Risk Parity'
        }
    
    def efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """Calculate Efficient Frontier"""
        target_returns = np.linspace(self.mean_returns.min() * 252, self.mean_returns.max() * 252, n_points)
        frontier = []
        
        for target in target_returns:
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(self.cov_matrix * 252, weights))
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(self.mean_returns * x) * 252 - target}
            ]
            bounds = tuple((0, 1) for _ in range(self.n_assets))
            initial = np.array([1/self.n_assets] * self.n_assets)
            
            try:
                result = minimize(portfolio_variance, initial, method='SLSQP', bounds=bounds, constraints=constraints)
                if result.success:
                    vol = np.sqrt(result.fun)
                    frontier.append({'Return': target, 'Volatility': vol})
            except:
                continue
        
        return pd.DataFrame(frontier)

# ==========================================
# 8. NEW MESIN 7: STRESS TESTING ENGINE
# ==========================================
class StressTestEngine:
    """
    Institutional Stress Testing & Scenario Analysis
    """
    
    HISTORICAL_SCENARIOS = {
        'COVID Crash (Mar 2020)': -0.34,
        'GFC (2008)': -0.57,
        'Dot-com Crash (2000-02)': -0.49,
        'Black Monday (1987)': -0.22,
        'Asian Crisis (1997)': -0.25,
        'Flash Crash (2010)': -0.09,
        'Brexit (2016)': -0.08,
        'Volmageddon (Feb 2018)': -0.12
    }
    
    def __init__(self, portfolio_value: float, current_var_95: float):
        self.portfolio_value = portfolio_value
        self.var_95 = current_var_95
    
    def historical_stress_test(self) -> pd.DataFrame:
        """Apply historical stress scenarios"""
        results = []
        for scenario, shock in self.HISTORICAL_SCENARIOS.items():
            loss = self.portfolio_value * shock
            results.append({
                'Scenario': scenario,
                'Shock': f'{shock*100:.1f}%',
                'Estimated Loss': loss,
                'Surviving Value': self.portfolio_value + loss,
                'VaR Multiplier': abs(shock / self.var_95) if self.var_95 != 0 else 0
            })
        return pd.DataFrame(results)
    
    def custom_stress_test(self, custom_shocks: Dict[str, float]) -> pd.DataFrame:
        """Apply custom stress scenarios"""
        results = []
        for name, shock in custom_shocks.items():
            loss = self.portfolio_value * shock
            results.append({
                'Scenario': name,
                'Shock': f'{shock*100:.1f}%',
                'Estimated Loss': loss,
                'Surviving Value': self.portfolio_value + loss
            })
        return pd.DataFrame(results)

# ==========================================
# 9. NEW MESIN 8: REGIME DETECTION
# ==========================================
class RegimeDetector:
    """
    Market Regime Detection using Volatility Clustering
    """
    
    def __init__(self, returns: pd.Series):
        self.returns = returns.dropna()
    
    def detect_volatility_regime(self, short_window: int = 20, long_window: int = 60) -> pd.DataFrame:
        """Detect volatility regimes using rolling volatility ratio"""
        short_vol = self.returns.rolling(short_window).std() * np.sqrt(252)
        long_vol = self.returns.rolling(long_window).std() * np.sqrt(252)
        
        vol_ratio = short_vol / long_vol
        
        df = pd.DataFrame({
            'returns': self.returns,
            'short_vol': short_vol,
            'long_vol': long_vol,
            'vol_ratio': vol_ratio
        })
        
        # Classify regimes
        def classify_regime(row):
            if pd.isna(row['vol_ratio']):
                return 'Unknown'
            elif row['vol_ratio'] > 1.5:
                return 'High Volatility'
            elif row['vol_ratio'] > 1.1:
                return 'Rising Volatility'
            elif row['vol_ratio'] < 0.7:
                return 'Low Volatility'
            else:
                return 'Normal'
        
        df['regime'] = df.apply(classify_regime, axis=1)
        return df
    
    def get_current_regime(self) -> str:
        """Get current market regime"""
        df = self.detect_volatility_regime()
        return df['regime'].iloc[-1]
    
    def regime_statistics(self) -> pd.DataFrame:
        """Calculate statistics for each regime"""
        df = self.detect_volatility_regime()
        
        stats = df.groupby('regime').agg({
            'returns': ['mean', 'std', 'count'],
            'short_vol': 'mean'
        }).round(4)
        
        stats.columns = ['Avg Return', 'Return Std', 'Days', 'Avg Volatility']
        stats['Annualized Return'] = stats['Avg Return'] * 252
        
        return stats

# ==========================================
# 10. NEW MESIN 9: TECHNICAL PATTERN RECOGNITION
# ==========================================
class TechnicalPatternEngine:
    """
    Advanced Technical Analysis & Pattern Recognition
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare technical indicators"""
        close = self.df['Close']
        high = self.df['High']
        low = self.df['Low']
        
        # Moving Averages
        self.df['SMA_20'] = close.rolling(20).mean()
        self.df['SMA_50'] = close.rolling(50).mean()
        self.df['SMA_200'] = close.rolling(200).mean()
        self.df['EMA_12'] = close.ewm(span=12, adjust=False).mean()
        self.df['EMA_26'] = close.ewm(span=26, adjust=False).mean()
        
        # MACD
        self.df['MACD'] = self.df['EMA_12'] - self.df['EMA_26']
        self.df['MACD_Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['MACD_Hist'] = self.df['MACD'] - self.df['MACD_Signal']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        self.df['BB_Mid'] = close.rolling(20).mean()
        std = close.rolling(20).std()
        self.df['BB_Upper'] = self.df['BB_Mid'] + (2 * std)
        self.df['BB_Lower'] = self.df['BB_Mid'] - (2 * std)
        self.df['BB_Width'] = (self.df['BB_Upper'] - self.df['BB_Lower']) / self.df['BB_Mid']
        
        # ATR
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        self.df['ATR'] = tr.rolling(14).mean()
        
        # Support & Resistance (Pivot Points)
        self.df['Pivot'] = (high + low + close) / 3
        self.df['R1'] = 2 * self.df['Pivot'] - low
        self.df['S1'] = 2 * self.df['Pivot'] - high
        self.df['R2'] = self.df['Pivot'] + (high - low)
        self.df['S2'] = self.df['Pivot'] - (high - low)
        
    def detect_patterns(self) -> List[Dict]:
        """Detect candlestick and chart patterns"""
        patterns = []
        
        if len(self.df) < 5:
            return patterns
        
        close = self.df['Close'].values
        open_p = self.df['Open'].values
        high = self.df['High'].values
        low = self.df['Low'].values
        
        # Golden Cross / Death Cross
        if len(self.df) > 50:
            sma_50 = self.df['SMA_50'].values
            sma_200 = self.df['SMA_200'].values
            
            if not np.isnan(sma_50[-1]) and not np.isnan(sma_200[-1]):
                if sma_50[-2] < sma_200[-2] and sma_50[-1] > sma_200[-1]:
                    patterns.append({'pattern': 'Golden Cross', 'signal': 'BULLISH', 'strength': 'Strong'})
                elif sma_50[-2] > sma_200[-2] and sma_50[-1] < sma_200[-1]:
                    patterns.append({'pattern': 'Death Cross', 'signal': 'BEARISH', 'strength': 'Strong'})
        
        # MACD Crossover
        macd = self.df['MACD'].values
        macd_signal = self.df['MACD_Signal'].values
        
        if not np.isnan(macd[-1]) and not np.isnan(macd_signal[-1]):
            if macd[-2] < macd_signal[-2] and macd[-1] > macd_signal[-1]:
                patterns.append({'pattern': 'MACD Bullish Crossover', 'signal': 'BULLISH', 'strength': 'Medium'})
            elif macd[-2] > macd_signal[-2] and macd[-1] < macd_signal[-1]:
                patterns.append({'pattern': 'MACD Bearish Crossover', 'signal': 'BEARISH', 'strength': 'Medium'})
        
        # RSI Divergence
        rsi = self.df['RSI'].values
        if not np.isnan(rsi[-1]):
            if rsi[-1] < 30:
                patterns.append({'pattern': 'RSI Oversold', 'signal': 'BULLISH', 'strength': 'Medium'})
            elif rsi[-1] > 70:
                patterns.append({'pattern': 'RSI Overbought', 'signal': 'BEARISH', 'strength': 'Medium'})
        
        # Bollinger Band Squeeze
        bb_width = self.df['BB_Width'].values
        if len(bb_width) > 20 and not np.isnan(bb_width[-1]):
            avg_width = np.nanmean(bb_width[-20:])
            if bb_width[-1] < avg_width * 0.5:
                patterns.append({'pattern': 'Bollinger Squeeze', 'signal': 'NEUTRAL', 'strength': 'Breakout Imminent'})
        
        # Doji Detection
        body = abs(close[-1] - open_p[-1])
        wick = high[-1] - low[-1]
        if wick > 0 and body / wick < 0.1:
            patterns.append({'pattern': 'Doji', 'signal': 'NEUTRAL', 'strength': 'Reversal Possible'})
        
        # Hammer Detection
        if len(self.df) >= 2:
            body = abs(close[-1] - open_p[-1])
            lower_wick = min(close[-1], open_p[-1]) - low[-1]
            upper_wick = high[-1] - max(close[-1], open_p[-1])
            
            if lower_wick > 2 * body and upper_wick < body:
                if close[-2] > close[-1]:  # Downtrend
                    patterns.append({'pattern': 'Hammer', 'signal': 'BULLISH', 'strength': 'Medium'})
        
        return patterns
    
    def get_support_resistance(self) -> Dict:
        """Get current support and resistance levels"""
        last = self.df.iloc[-1]
        return {
            'current_price': last['Close'],
            'pivot': last['Pivot'],
            'resistance_1': last['R1'],
            'resistance_2': last['R2'],
            'support_1': last['S1'],
            'support_2': last['S2'],
            'atr': last['ATR']
        }

# ==========================================
# 11. NEW MESIN 10: LIQUIDITY ANALYZER
# ==========================================
class LiquidityAnalyzer:
    """
    Market Liquidity & Microstructure Analysis
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
    def calculate_metrics(self) -> Dict:
        """Calculate liquidity metrics"""
        close = self.df['Close']
        volume = self.df['Volume']
        high = self.df['High']
        low = self.df['Low']
        
        # Average Daily Volume
        adv_20 = volume.tail(20).mean()
        adv_60 = volume.tail(60).mean()
        
        # Dollar Volume
        dollar_volume = (close * volume).tail(20).mean()
        
        # Spread Proxy (High-Low as % of Close)
        spread_proxy = ((high - low) / close * 100).tail(20).mean()
        
        # Amihud Illiquidity Ratio
        returns = close.pct_change().abs()
        amihud = (returns / (volume * close)).tail(20).mean() * 1e6
        
        # Volume Volatility
        vol_volatility = volume.tail(20).std() / volume.tail(20).mean()
        
        # Turnover Trend
        vol_trend = volume.tail(5).mean() / volume.tail(20).mean()
        
        return {
            'ADV_20': adv_20,
            'ADV_60': adv_60,
            'Dollar_Volume': dollar_volume,
            'Spread_Proxy_Pct': spread_proxy,
            'Amihud_Illiquidity': amihud,
            'Volume_Volatility': vol_volatility,
            'Turnover_Trend': vol_trend,
            'Liquidity_Score': self._calculate_liquidity_score(adv_20, spread_proxy, amihud)
        }
    
    def _calculate_liquidity_score(self, adv: float, spread: float, amihud: float) -> str:
        """Score liquidity from 1-5"""
        score = 3  # Default medium
        
        if adv > 1000000:
            score += 1
        elif adv < 100000:
            score -= 1
            
        if spread < 1:
            score += 0.5
        elif spread > 3:
            score -= 0.5
        
        score = max(1, min(5, score))
        
        labels = {1: 'Very Low', 2: 'Low', 3: 'Medium', 4: 'High', 5: 'Very High'}
        return labels.get(int(round(score)), 'Medium')

# ==========================================
# 12. NEW MESIN 11: CORRELATION ENGINE
# ==========================================
class CorrelationEngine:
    """
    Multi-Asset Correlation & Beta Analysis
    """
    
    def __init__(self, tickers: List[str], period: str = "1y"):
        self.tickers = tickers
        self.period = period
        self.data = None
        self.returns = None
        
    def fetch_data(self) -> bool:
        """Fetch price data for all tickers"""
        try:
            data = yf.download(self.tickers, period=self.period, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                self.data = data['Close']
            else:
                self.data = data[['Close']]
                self.data.columns = self.tickers
            
            self.returns = self.data.pct_change().dropna()
            return True
        except:
            return False
    
    def correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix"""
        if self.returns is None:
            self.fetch_data()
        return self.returns.corr()
    
    def rolling_correlation(self, asset1: str, asset2: str, window: int = 60) -> pd.Series:
        """Calculate rolling correlation between two assets"""
        if self.returns is None:
            self.fetch_data()
        return self.returns[asset1].rolling(window).corr(self.returns[asset2])
    
    def calculate_betas(self, benchmark: str = 'SPY') -> pd.DataFrame:
        """Calculate beta against benchmark"""
        if self.returns is None:
            self.fetch_data()
        
        betas = []
        bench_var = self.returns[benchmark].var() if benchmark in self.returns.columns else 1
        
        for ticker in self.tickers:
            if ticker != benchmark and ticker in self.returns.columns:
                cov = self.returns[ticker].cov(self.returns.get(benchmark, self.returns[ticker]))
                beta = cov / bench_var if bench_var > 0 else 1
                betas.append({'Ticker': ticker, 'Beta': beta})
        
        return pd.DataFrame(betas)

# ==========================================
# 13. UI UTAMA (ENHANCED DASHBOARD)
# ==========================================

# Sidebar (PRESERVED + ENHANCED)
with st.sidebar:
    st.markdown('<span class="institutional-badge">INSTITUTIONAL EDITION</span>', unsafe_allow_html=True)
    st.header("🎛️ Control Panel")
    
    client_mode = st.radio("Mode:", ["Retail (Long)", "Market Maker (Short)"])
    st.divider()
    
    currency = st.selectbox("Mata Uang:", ["USD", "IDR"])
    sym = "$" if currency == "USD" else "Rp"
    
    ticker_in = st.text_input("Ticker Audit:", "NVDA" if currency=="USD" else "BBCA.JK")
    period = st.selectbox("Durasi:", ["5d", "1mo", "3mo", "6mo", "1y"])
    
    st.caption("Risk Parameters")
    d_thresh = st.slider("Delta Threshold", 0.05, 0.50, 0.10, 0.05)
    lot = st.number_input("Lot", 1, 100, 10)
    
    st.divider()
    st.caption("🔬 Quant Lab Settings")
    var_confidence = st.selectbox("VaR Confidence:", [0.95, 0.99])
    mc_simulations = st.number_input("Monte Carlo Paths:", 1000, 50000, 10000, 1000)
    
    st.divider()
    st.caption("📊 Portfolio Settings")
    portfolio_value = st.number_input(f"Portfolio Value ({currency}):", 10000, 100000000, 1000000, 10000)

# --- TAB NAVIGASI (ENHANCED) ---
tab_audit, tab_radar, tab_signal, tab_quant_lab, tab_portfolio, tab_stress, tab_patterns = st.tabs([
    "📊 Audit Strategi", 
    "🐘 Whale Radar", 
    "🎯 Signal Generator",
    "🔬 Quant Lab",
    "📈 Portfolio Analytics",
    "⚡ Stress Testing",
    "🎨 Pattern Recognition"
])

# ==========================================
# TAB 1: AUDIT (PRESERVED)
# ==========================================
with tab_audit:
    if st.button("JALANKAN AUDIT", type="primary", key="audit_btn"):
        with st.spinner("Processing..."):
            pos = PositionType.SHORT if "Short" in client_mode else PositionType.LONG
            cfg = MarketMakerConfig(ticker=ticker_in, position_type=pos, delta_threshold=d_thresh, num_straddles=lot, lookback_days=5 if period=="5d" else 60)
            ldr = DataIngestion(cfg); raw = ldr.fetch_data(ticker_in, period, "5m")
            if not raw.empty:
                df = ldr.process(raw); eng = MarketMakerBacktest(cfg, df)
                res, trd, prem = eng.run()
                st.session_state.analyzed=True; st.session_state.res_df=res; st.session_state.trades=trd
                st.session_state.pos=pos; st.session_state.tick=ticker_in
                
                met = {"pnl_str": f"{sym} {res['total_pnl'].iloc[-1]:,.2f}", "opt_str": f"{sym} {res['opt_pnl'].iloc[-1]:,.2f}", 
                       "hed_str": f"{sym} {(res['total_pnl'].iloc[-1]-res['opt_pnl'].iloc[-1]):,.2f}", "trades_count": str(len(trd))}
                figs=[]; f1,ax=plt.subplots(figsize=(10,4)); ax.plot(res.index, res['total_pnl']); ax.set_title("P&L"); figs.append(f1)
                f2,(axa,axb)=plt.subplots(2,1,figsize=(10,8)); axa.plot(res['pos_delta']+res['stock_pos']); axb.plot(res['pos_gamma']); figs.append(f2)
                st.session_state.pdf_bytes = create_pdf(ticker_in, pos.value, met, pd.DataFrame([vars(t) for t in trd]) if trd else pd.DataFrame(), figs)

    if st.session_state.analyzed:
        fin = st.session_state.res_df.iloc[-1]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Net P&L", f"{sym} {fin['total_pnl']:,.2f}", delta_color="normal")
        c2.metric("Option Income", f"{sym} {fin['opt_pnl']:,.2f}")
        c3.metric("Hedge Cost", f"{sym} {(fin['total_pnl'] - fin['opt_pnl']):,.2f}")
        c4.metric("Trades", len(st.session_state.trades))
        
        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📈 Kurva Ekuitas", "⚠️ Risiko Gamma", "📝 Log Transaksi"])
        
        with sub_tab1:
            st.subheader("Kurva P&L")
            fig1, ax = plt.subplots(figsize=(10, 4))
            ax.plot(st.session_state.res_df.index, st.session_state.res_df['total_pnl'], color='#00FF00', linewidth=1.5)
            ax.axhline(0, color='white', linestyle='--', linewidth=0.5)
            ax.set_facecolor('#0e1117'); fig1.patch.set_facecolor('#0e1117')
            ax.tick_params(colors='white'); ax.spines['bottom'].set_color('white'); ax.spines['left'].set_color('white')
            st.pyplot(fig1)

        with sub_tab2:
            st.subheader("Audit Profil Risiko")
            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown("#### Net Delta (Arah)")
                st.line_chart(st.session_state.res_df['pos_delta'] + st.session_state.res_df['stock_pos'])
            with c_right:
                st.markdown("#### Gamma Risk (Percepatan)")
                st.line_chart(st.session_state.res_df['pos_gamma'])

        with sub_tab3:
            st.subheader("Rekaman Transaksi Robot")
            if st.session_state.trades:
                trade_df = pd.DataFrame([vars(t) for t in st.session_state.trades])
                fmt_df = trade_df.copy()
                fmt_df['price'] = fmt_df['price'].apply(lambda x: f"{sym} {x:,.2f}")
                st.dataframe(fmt_df, use_container_width=True)
            else:
                st.info("Tidak ada transaksi hedging.")

        st.divider()
        if st.session_state.pdf_bytes: 
            st.download_button("📄 Download PDF Laporan Lengkap", st.session_state.pdf_bytes, f"{st.session_state.ticker_file_name}.pdf", "application/pdf")

# ==========================================
# TAB 2: WHALE RADAR (PRESERVED)
# ==========================================
with tab_radar:
    col_rad1, col_rad2 = st.columns([3,1])
    def_tick = "BBCA.JK, BBRI.JK, TLKM.JK, BUMI.JK, GOTO.JK, MINA.JK, INET.JK, ANTM.JK"
    usr_tick = col_rad1.text_area("Watchlist Radar:", def_tick, height=70)
    if col_rad2.button("SCAN GAJAH", key="whale_btn"):
        st.session_state.radar_results = whale_radar_scanner(usr_tick.split(","))
    
    if not st.session_state.radar_results.empty:
        def highlight_row(row):
            if "ABSORPTION" in row['Status']: return ['background-color: #4a148c; color: white'] * len(row)
            elif "MARK-UP" in row['Status']: return ['background-color: #1b5e20; color: white'] * len(row)
            elif "DISTRIBUTION" in row['Status']: return ['background-color: #b71c1c; color: white'] * len(row)
            return [''] * len(row)
            
        st.dataframe(st.session_state.radar_results.style.apply(highlight_row, axis=1).format({"Price":"{:,.0f}", "Z-Score":"{:.2f}", "RVOL":"{:.1f}x"}), use_container_width=True)

# ==========================================
# TAB 3: SIGNAL GENERATOR (PRESERVED)
# ==========================================
with tab_signal:
    st.subheader("Mesin Pencari Sinyal (The Alpha Scanner)")
    col_sig1, col_sig2 = st.columns([3,1])
    sig_tickers = col_sig1.text_area("Watchlist Sinyal:", "BBCA.JK, BBRI.JK, TLKM.JK, ADRO.JK, UNTR.JK, BTC-USD, EURUSD=X", height=70)
    if col_sig2.button("CARI PELUANG", type="primary", key="signal_btn"):
        st.session_state.signal_results = generate_signals(sig_tickers.split(","))
        
    if not st.session_state.signal_results.empty:
        res = st.session_state.signal_results
        bull = res[res['Action'].str.contains("BUY")]; bear = res[res['Action'].str.contains("SELL") | res['Action'].str.contains("SHORT")]
        
        c_bull, c_bear = st.columns(2)
        with c_bull:
            st.success(f"🟢 Peluang Beli ({len(bull)})")
            if not bull.empty: st.dataframe(bull[['Ticker', 'Price', 'Action', 'Reason']].style.format({"Price":"{:,.2f}"}), use_container_width=True)
        with c_bear:
            st.error(f"🔴 Peluang Jual ({len(bear)})")
            if not bear.empty: st.dataframe(bear[['Ticker', 'Price', 'Action', 'Reason']].style.format({"Price":"{:,.2f}"}), use_container_width=True)

# ==========================================
# TAB 4: QUANT LAB (NEW)
# ==========================================
with tab_quant_lab:
    st.markdown('<div class="quant-header"><h2>🔬 Institutional Quant Lab</h2><p>Advanced Risk Analytics & Monte Carlo Simulation</p></div>', unsafe_allow_html=True)
    
    quant_ticker = st.text_input("Ticker untuk Analisis:", ticker_in, key="quant_ticker")
    quant_period = st.selectbox("Periode Data:", ["6mo", "1y", "2y"], key="quant_period")
    
    col_var, col_mc = st.columns(2)
    
    with col_var:
        if st.button("📊 Calculate VaR Suite", type="primary", key="var_btn"):
            with st.spinner("Calculating institutional VaR metrics..."):
                try:
                    df = yf.download(quant_ticker, period=quant_period, progress=False)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    
                    returns = df['Close'].pct_change().dropna()
                    
                    var_engine = InstitutionalVaREngine(returns, [0.95, 0.99])
                    st.session_state.var_results = var_engine.full_risk_report()
                    
                    # Additional metrics
                    st.session_state.var_extra = {
                        'skewness': skew(returns),
                        'kurtosis': kurtosis(returns),
                        'mean_return': returns.mean() * 252,
                        'volatility': returns.std() * np.sqrt(252),
                        'max_drawdown': (df['Close'] / df['Close'].cummax() - 1).min()
                    }
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.session_state.var_results is not None:
            st.markdown("### 📉 Value at Risk Report")
            st.dataframe(st.session_state.var_results, use_container_width=True)
            
            st.markdown("### 📈 Distribution Metrics")
            extra = st.session_state.var_extra
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Skewness", f"{extra['skewness']:.3f}")
            mc2.metric("Kurtosis", f"{extra['kurtosis']:.3f}")
            mc3.metric("Max Drawdown", f"{extra['max_drawdown']*100:.2f}%")
            
            mc4, mc5 = st.columns(2)
            mc4.metric("Ann. Return", f"{extra['mean_return']*100:.2f}%")
            mc5.metric("Ann. Volatility", f"{extra['volatility']*100:.2f}%")
    
    with col_mc:
        if st.button("🎲 Run Monte Carlo Simulation", type="primary", key="mc_btn"):
            with st.spinner("Running Monte Carlo paths..."):
                try:
                    df = yf.download(quant_ticker, period=quant_period, progress=False)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    
                    returns = df['Close'].pct_change().dropna()
                    current_price = df['Close'].iloc[-1]
                    
                    mc_sim = MonteCarloSimulator(
                        current_price=current_price,
                        mu=returns.mean() * 252,
                        sigma=returns.std() * np.sqrt(252)
                    )
                    
                    paths = mc_sim.simulate_gbm(days=252, num_paths=int(mc_simulations))
                    stats = mc_sim.get_statistics(paths)
                    
                    st.session_state.monte_carlo_paths = paths
                    st.session_state.mc_stats = stats
                    st.session_state.mc_current_price = current_price
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.session_state.monte_carlo_paths is not None:
            st.markdown("### 🎲 Monte Carlo Results")
            
            stats = st.session_state.mc_stats
            curr = st.session_state.mc_current_price
            
            mcc1, mcc2 = st.columns(2)
            mcc1.metric("Current Price", f"{sym} {curr:,.2f}")
            mcc2.metric("Prob of Profit (1Y)", f"{stats['prob_profit']:.1f}%")
            
            mcc3, mcc4, mcc5 = st.columns(3)
            mcc3.metric("5th Percentile", f"{sym} {stats['percentile_5']:,.2f}")
            mcc4.metric("Median", f"{sym} {stats['median_price']:,.2f}")
            mcc5.metric("95th Percentile", f"{sym} {stats['percentile_95']:,.2f}")
            
            # Plot paths
            fig_mc, ax_mc = plt.subplots(figsize=(10, 5))
            paths = st.session_state.monte_carlo_paths
            
            # Plot subset of paths
            for i in range(min(100, paths.shape[1])):
                ax_mc.plot(paths[:, i], alpha=0.1, color='cyan', linewidth=0.5)
            
            ax_mc.axhline(curr, color='yellow', linestyle='--', linewidth=2, label='Current Price')
            ax_mc.axhline(stats['percentile_5'], color='red', linestyle='--', linewidth=1.5, label='5th Percentile')
            ax_mc.axhline(stats['percentile_95'], color='green', linestyle='--', linewidth=1.5, label='95th Percentile')
            
            ax_mc.set_facecolor('#0e1117')
            fig_mc.patch.set_facecolor('#0e1117')
            ax_mc.tick_params(colors='white')
            ax_mc.set_xlabel('Days', color='white')
            ax_mc.set_ylabel('Price', color='white')
            ax_mc.legend(facecolor='#262730', labelcolor='white')
            ax_mc.set_title(f'Monte Carlo Simulation ({mc_simulations:,} paths)', color='white')
            
            st.pyplot(fig_mc)

    st.divider()
    
    # Correlation Matrix Section
    st.markdown("### 🔗 Multi-Asset Correlation Matrix")
    corr_tickers = st.text_input("Tickers (comma separated):", "AAPL, MSFT, GOOGL, AMZN, META, NVDA, SPY", key="corr_tickers")
    
    if st.button("Generate Correlation Matrix", key="corr_btn"):
        with st.spinner("Calculating correlations..."):
            ticker_list = [t.strip() for t in corr_tickers.split(",")]
            corr_engine = CorrelationEngine(ticker_list, "1y")
            
            if corr_engine.fetch_data():
                corr_matrix = corr_engine.correlation_matrix()
                st.session_state.correlation_matrix = corr_matrix
                
                # Heatmap
                fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                im = ax_corr.imshow(corr_matrix.values, cmap='RdYlGn', vmin=-1, vmax=1)
                
                ax_corr.set_xticks(range(len(corr_matrix.columns)))
                ax_corr.set_yticks(range(len(corr_matrix.columns)))
                ax_corr.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', color='white')
                ax_corr.set_yticklabels(corr_matrix.columns, color='white')
                
                # Add correlation values
                for i in range(len(corr_matrix.columns)):
                    for j in range(len(corr_matrix.columns)):
                        text = ax_corr.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                          ha='center', va='center', color='black', fontsize=9)
                
                fig_corr.colorbar(im)
                ax_corr.set_facecolor('#0e1117')
                fig_corr.patch.set_facecolor('#0e1117')
                ax_corr.set_title('Asset Correlation Matrix', color='white')
                
                st.pyplot(fig_corr)
                
                # Beta calculation
                if 'SPY' in ticker_list:
                    betas = corr_engine.calculate_betas('SPY')
                    st.markdown("### 📊 Beta vs SPY")
                    st.dataframe(betas.style.format({"Beta": "{:.2f}"}), use_container_width=True)

# ==========================================
# TAB 5: PORTFOLIO ANALYTICS (NEW)
# ==========================================
with tab_portfolio:
    st.markdown('<div class="quant-header"><h2>📈 Portfolio Optimization Engine</h2><p>Mean-Variance, Risk Parity & Efficient Frontier</p></div>', unsafe_allow_html=True)
    
    port_tickers = st.text_input("Portfolio Assets:", "AAPL, MSFT, GOOGL, AMZN, BRK-B, JPM, JNJ, V, PG, HD", key="port_tickers")
    
    if st.button("🚀 Optimize Portfolio", type="primary", key="optimize_btn"):
        with st.spinner("Running optimization algorithms..."):
            try:
                ticker_list = [t.strip() for t in port_tickers.split(",")]
                data = yf.download(ticker_list, period="2y", progress=False)
                
                if isinstance(data.columns, pd.MultiIndex):
                    closes = data['Close']
                else:
                    closes = data[['Close']]
                    closes.columns = ticker_list
                
                returns = closes.pct_change().dropna()
                
                optimizer = PortfolioOptimizer(returns)
                
                # Run all optimizations
                max_sharpe = optimizer.optimize_sharpe()
                min_var = optimizer.optimize_min_variance()
                risk_parity = optimizer.risk_parity()
                
                st.session_state.portfolio_weights = {
                    'max_sharpe': max_sharpe,
                    'min_var': min_var,
                    'risk_parity': risk_parity,
                    'tickers': ticker_list
                }
                
                # Efficient Frontier
                frontier = optimizer.efficient_frontier()
                st.session_state.efficient_frontier = frontier
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    if st.session_state.portfolio_weights is not None:
        weights = st.session_state.portfolio_weights
        tickers = weights['tickers']
        
        # Display optimized portfolios
        col_p1, col_p2, col_p3 = st.columns(3)
        
        with col_p1:
            st.markdown("#### 🎯 Maximum Sharpe")
            ms = weights['max_sharpe']
            st.metric("Sharpe Ratio", f"{ms['sharpe']:.3f}")
            st.metric("Expected Return", f"{ms['return']*100:.2f}%")
            st.metric("Volatility", f"{ms['volatility']*100:.2f}%")
            
            # Weights chart
            fig_w1, ax_w1 = plt.subplots(figsize=(6, 4))
            ax_w1.barh(tickers, ms['weights'], color='#00ff88')
            ax_w1.set_facecolor('#0e1117')
            fig_w1.patch.set_facecolor('#0e1117')
            ax_w1.tick_params(colors='white')
            ax_w1.set_xlabel('Weight', color='white')
            st.pyplot(fig_w1)
        
        with col_p2:
            st.markdown("#### 🛡️ Minimum Variance")
            mv = weights['min_var']
            st.metric("Sharpe Ratio", f"{mv['sharpe']:.3f}")
            st.metric("Expected Return", f"{mv['return']*100:.2f}%")
            st.metric("Volatility", f"{mv['volatility']*100:.2f}%")
            
            fig_w2, ax_w2 = plt.subplots(figsize=(6, 4))
            ax_w2.barh(tickers, mv['weights'], color='#ff6b6b')
            ax_w2.set_facecolor('#0e1117')
            fig_w2.patch.set_facecolor('#0e1117')
            ax_w2.tick_params(colors='white')
            ax_w2.set_xlabel('Weight', color='white')
            st.pyplot(fig_w2)
        
        with col_p3:
            st.markdown("#### ⚖️ Risk Parity")
            rp = weights['risk_parity']
            st.metric("Sharpe Ratio", f"{rp['sharpe']:.3f}")
            st.metric("Expected Return", f"{rp['return']*100:.2f}%")
            st.metric("Volatility", f"{rp['volatility']*100:.2f}%")
            
            fig_w3, ax_w3 = plt.subplots(figsize=(6, 4))
            ax_w3.barh(tickers, rp['weights'], color='#4ecdc4')
            ax_w3.set_facecolor('#0e1117')
            fig_w3.patch.set_facecolor('#0e1117')
            ax_w3.tick_params(colors='white')
            ax_w3.set_xlabel('Weight', color='white')
            st.pyplot(fig_w3)
        
        # Efficient Frontier Plot
        if st.session_state.efficient_frontier is not None:
            st.divider()
            st.markdown("### 📈 Efficient Frontier")
            
            frontier = st.session_state.efficient_frontier
            
            fig_ef, ax_ef = plt.subplots(figsize=(12, 6))
            ax_ef.plot(frontier['Volatility'] * 100, frontier['Return'] * 100, 
                      'b-', linewidth=2, label='Efficient Frontier')
            
            # Mark optimal portfolios
            ax_ef.scatter(ms['volatility'] * 100, ms['return'] * 100, 
                         color='gold', s=200, marker='*', label='Max Sharpe', zorder=5)
            ax_ef.scatter(mv['volatility'] * 100, mv['return'] * 100, 
                         color='red', s=150, marker='s', label='Min Variance', zorder=5)
            ax_ef.scatter(rp['volatility'] * 100, rp['return'] * 100, 
                         color='cyan', s=150, marker='^', label='Risk Parity', zorder=5)
            
            ax_ef.set_facecolor('#0e1117')
            fig_ef.patch.set_facecolor('#0e1117')
            ax_ef.tick_params(colors='white')
            ax_ef.set_xlabel('Volatility (%)', color='white', fontsize=12)
            ax_ef.set_ylabel('Expected Return (%)', color='white', fontsize=12)
            ax_ef.set_title('Markowitz Efficient Frontier', color='white', fontsize=14)
            ax_ef.legend(facecolor='#262730', labelcolor='white')
            ax_ef.grid(True, alpha=0.3)
            
            st.pyplot(fig_ef)

# ==========================================
# TAB 6: STRESS TESTING (NEW)
# ==========================================
with tab_stress:
    st.markdown('<div class="quant-header"><h2>⚡ Stress Testing & Scenario Analysis</h2><p>Historical Scenarios & Custom Shock Analysis</p></div>', unsafe_allow_html=True)
    
    stress_value = st.number_input(f"Portfolio Value ({currency}):", 100000, 100000000, portfolio_value, 10000, key="stress_val")
    
    col_s1, col_s2 = st.columns([2, 1])
    
    with col_s1:
        if st.button("🔥 Run Historical Stress Test", type="primary", key="stress_btn"):
            # Get current VaR for reference
            try:
                df = yf.download(ticker_in, period="1y", progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                returns = df['Close'].pct_change().dropna()
                var_95 = np.percentile(returns, 5)
            except:
                var_95 = -0.02
            
            stress_engine = StressTestEngine(stress_value, var_95)
            st.session_state.stress_results = stress_engine.historical_stress_test()
        
        if st.session_state.stress_results is not None:
            st.markdown("### 📊 Historical Stress Scenarios")
            
            results = st.session_state.stress_results
            
            def color_loss(val):
                if isinstance(val, (int, float)):
                    if val < 0:
                        intensity = min(abs(val) / stress_value * 2, 1)
                        return f'background-color: rgba(255, 0, 0, {intensity}); color: white'
                return ''
            
            st.dataframe(
                results.style.format({
                    'Estimated Loss': lambda x: f'{sym} {x:,.0f}',
                    'Surviving Value': lambda x: f'{sym} {x:,.0f}',
                    'VaR Multiplier': '{:.1f}x'
                }).applymap(color_loss, subset=['Estimated Loss']),
                use_container_width=True
            )
            
            # Visualization
            fig_stress, ax_stress = plt.subplots(figsize=(12, 6))
            scenarios = results['Scenario'].values
            losses = results['Estimated Loss'].values
            
            colors = ['#ff1744' if l < -stress_value * 0.3 else '#ff9100' if l < -stress_value * 0.15 else '#ffeb3b' for l in losses]
            
            bars = ax_stress.barh(scenarios, losses, color=colors)
            ax_stress.axvline(0, color='white', linewidth=1)
            ax_stress.set_facecolor('#0e1117')
            fig_stress.patch.set_facecolor('#0e1117')
            ax_stress.tick_params(colors='white')
            ax_stress.set_xlabel(f'Estimated Loss ({currency})', color='white')
            ax_stress.set_title('Historical Stress Test Results', color='white')
            
            # Add value labels
            for bar, val in zip(bars, losses):
                ax_stress.text(val - (stress_value * 0.02), bar.get_y() + bar.get_height()/2, 
                              f'{sym}{val/1000:.0f}K', va='center', ha='right', color='white', fontsize=9)
            
            st.pyplot(fig_stress)
    
    with col_s2:
        st.markdown("### 🎚️ Custom Scenarios")
        
        custom_shock_1 = st.slider("Scenario 1: Market Correction", -50, 0, -15, 1)
        custom_shock_2 = st.slider("Scenario 2: Flash Crash", -50, 0, -25, 1)
        custom_shock_3 = st.slider("Scenario 3: Black Swan", -80, 0, -40, 1)
        
        if st.button("Apply Custom Shocks", key="custom_shock_btn"):
            custom_shocks = {
                'Market Correction': custom_shock_1 / 100,
                'Flash Crash': custom_shock_2 / 100,
                'Black Swan Event': custom_shock_3 / 100
            }
            
            stress_engine = StressTestEngine(stress_value, 0)
            custom_results = stress_engine.custom_stress_test(custom_shocks)
            
            st.markdown("#### Custom Shock Results")
            for _, row in custom_results.iterrows():
                st.markdown(f"""
                <div class="stress-test-card">
                    <strong>{row['Scenario']}</strong><br>
                    Shock: {row['Shock']} | Loss: {sym} {row['Estimated Loss']:,.0f}<br>
                    Surviving: {sym} {row['Surviving Value']:,.0f}
                </div>
                """, unsafe_allow_html=True)
    
    st.divider()
    
    # Regime Analysis
    st.markdown("### 📊 Market Regime Analysis")
    
    if st.button("Analyze Current Regime", key="regime_btn"):
        with st.spinner("Analyzing market regime..."):
            try:
                df = yf.download(ticker_in, period="1y", progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                returns = df['Close'].pct_change().dropna()
                
                detector = RegimeDetector(returns)
                regime_df = detector.detect_volatility_regime()
                current_regime = detector.get_current_regime()
                regime_stats = detector.regime_statistics()
                
                st.session_state.regime_analysis = {
                    'df': regime_df,
                    'current': current_regime,
                    'stats': regime_stats
                }
            except Exception as e:
                st.error(f"Error: {e}")
    
    if st.session_state.regime_analysis is not None:
        regime = st.session_state.regime_analysis
        
        # Current regime display
        regime_colors = {
            'High Volatility': '🔴',
            'Rising Volatility': '🟠',
            'Normal': '🟡',
            'Low Volatility': '🟢',
            'Unknown': '⚪'
        }
        
        st.markdown(f"### Current Regime: {regime_colors.get(regime['current'], '⚪')} {regime['current']}")
        
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            st.markdown("#### Regime Statistics")
            st.dataframe(regime['stats'], use_container_width=True)
        
        with col_r2:
            # Volatility chart
            fig_vol, ax_vol = plt.subplots(figsize=(10, 4))
            ax_vol.plot(regime['df'].index, regime['df']['short_vol'], label='20-day Vol', color='cyan', linewidth=1)
            ax_vol.plot(regime['df'].index, regime['df']['long_vol'], label='60-day Vol', color='orange', linewidth=1)
            ax_vol.set_facecolor('#0e1117')
            fig_vol.patch.set_facecolor('#0e1117')
            ax_vol.tick_params(colors='white')
            ax_vol.legend(facecolor='#262730', labelcolor='white')
            ax_vol.set_title('Volatility Regime', color='white')
            st.pyplot(fig_vol)

# ==========================================
# TAB 7: PATTERN RECOGNITION (NEW)
# ==========================================
with tab_patterns:
    st.markdown('<div class="quant-header"><h2>🎨 Technical Pattern Recognition</h2><p>Chart Patterns, Support/Resistance & Technical Indicators</p></div>', unsafe_allow_html=True)
    
    pattern_ticker = st.text_input("Ticker:", ticker_in, key="pattern_ticker")
    pattern_period = st.selectbox("Period:", ["3mo", "6mo", "1y"], key="pattern_period")
    
    if st.button("🔍 Analyze Patterns", type="primary", key="pattern_btn"):
        with st.spinner("Scanning for patterns..."):
            try:
                df = yf.download(pattern_ticker, period=pattern_period, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                pattern_engine = TechnicalPatternEngine(df)
                patterns = pattern_engine.detect_patterns()
                sr_levels = pattern_engine.get_support_resistance()
                
                # Liquidity Analysis
                liquidity = LiquidityAnalyzer(df)
                liq_metrics = liquidity.calculate_metrics()
                
                st.session_state.pattern_data = {
                    'df': pattern_engine.df,
                    'patterns': patterns,
                    'sr': sr_levels,
                    'liquidity': liq_metrics
                }
            except Exception as e:
                st.error(f"Error: {e}")
    
    if st.session_state.get('pattern_data') is not None:
        data = st.session_state.pattern_data
        
        # Detected Patterns
        st.markdown("### 🎯 Detected Patterns")
        
        if data['patterns']:
            for p in data['patterns']:
                signal_color = '🟢' if p['signal'] == 'BULLISH' else '🔴' if p['signal'] == 'BEARISH' else '🟡'
                st.markdown(f"""
                <div class="stress-test-card" style="border-left-color: {'#00ff88' if p['signal'] == 'BULLISH' else '#ff6b6b' if p['signal'] == 'BEARISH' else '#ffeb3b'}">
                    <strong>{signal_color} {p['pattern']}</strong><br>
                    Signal: {p['signal']} | Strength: {p['strength']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No significant patterns detected.")
        
        # Support & Resistance
        col_sr1, col_sr2 = st.columns(2)
        
        with col_sr1:
            st.markdown("### 📊 Support & Resistance Levels")
            sr = data['sr']
            
            st.metric("Current Price", f"{sym} {sr['current_price']:,.2f}")
 
            sr_col1, sr_col2 = st.columns(2)
            with sr_col1:
                st.markdown("**Resistance Levels**")
                st.write(f"R2: {sym} {sr['resistance_2']:,.2f}")
                st.write(f"R1: {sym} {sr['resistance_1']:,.2f}")
                st.write(f"Pivot: {sym} {sr['pivot']:,.2f}")
            with sr_col2:
                st.markdown("**Support Levels**")
                st.write(f"S1: {sym} {sr['support_1']:,.2f}")
                st.write(f"S2: {sym} {sr['support_2']:,.2f}")
                st.write(f"ATR: {sr['atr']:,.2f}")
        
        with col_sr2:
            st.markdown("### 💧 Liquidity Analysis")
            liq = data['liquidity']
            
            liq_score_color = {
                'Very High': '🟢',
                'High': '🟢', 
                'Medium': '🟡',
                'Low': '🟠',
                'Very Low': '🔴'
            }
            
            st.metric("Liquidity Score", f"{liq_score_color.get(liq['Liquidity_Score'], '⚪')} {liq['Liquidity_Score']}")
            st.metric("Avg Daily Volume (20d)", f"{liq['ADV_20']:,.0f}")
            st.metric("Dollar Volume", f"{sym} {liq['Dollar_Volume']:,.0f}")
            st.metric("Spread Proxy", f"{liq['Spread_Proxy_Pct']:.2f}%")
            st.metric("Volume Trend", f"{liq['Turnover_Trend']:.2f}x")
        
        st.divider()
        
        # Technical Chart
        st.markdown("### 📈 Technical Analysis Chart")
        
        df_plot = data['df'].tail(120)  # Last 120 periods
        
        fig_tech, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Price Chart with Bollinger Bands
        ax1 = axes[0]
        ax1.plot(df_plot.index, df_plot['Close'], label='Close', color='white', linewidth=1.5)
        ax1.plot(df_plot.index, df_plot['SMA_20'], label='SMA 20', color='#00ff88', linewidth=1, alpha=0.7)
        ax1.plot(df_plot.index, df_plot['SMA_50'], label='SMA 50', color='#ff6b6b', linewidth=1, alpha=0.7)
        ax1.fill_between(df_plot.index, df_plot['BB_Lower'], df_plot['BB_Upper'], 
                        alpha=0.2, color='cyan', label='Bollinger Bands')
        
        # Support & Resistance lines
        current_price = sr['current_price']
        ax1.axhline(sr['resistance_1'], color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax1.axhline(sr['support_1'], color='green', linestyle='--', alpha=0.5, linewidth=1)
        
        ax1.set_facecolor('#0e1117')
        ax1.tick_params(colors='white')
        ax1.legend(loc='upper left', facecolor='#262730', labelcolor='white', fontsize=8)
        ax1.set_title(f'{pattern_ticker} - Technical Analysis', color='white', fontsize=12)
        ax1.set_ylabel('Price', color='white')
        
        # RSI
        ax2 = axes[1]
        ax2.plot(df_plot.index, df_plot['RSI'], color='#ff9f1c', linewidth=1)
        ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax2.fill_between(df_plot.index, 30, 70, alpha=0.1, color='gray')
        ax2.set_facecolor('#0e1117')
        ax2.tick_params(colors='white')
        ax2.set_ylabel('RSI', color='white')
        ax2.set_ylim(0, 100)
        
        # MACD
        ax3 = axes[2]
        ax3.plot(df_plot.index, df_plot['MACD'], label='MACD', color='cyan', linewidth=1)
        ax3.plot(df_plot.index, df_plot['MACD_Signal'], label='Signal', color='orange', linewidth=1)
        ax3.bar(df_plot.index, df_plot['MACD_Hist'], 
               color=['#00ff88' if x >= 0 else '#ff6b6b' for x in df_plot['MACD_Hist']], alpha=0.5)
        ax3.axhline(0, color='white', linestyle='-', alpha=0.3)
        ax3.set_facecolor('#0e1117')
        ax3.tick_params(colors='white')
        ax3.set_ylabel('MACD', color='white')
        ax3.legend(loc='upper left', facecolor='#262730', labelcolor='white', fontsize=8)
        
        fig_tech.patch.set_facecolor('#0e1117')
        plt.tight_layout()
        st.pyplot(fig_tech)
        
        # Indicator Summary Table
        st.markdown("### 📋 Technical Indicator Summary")
        
        last_row = data['df'].iloc[-1]
        
        indicator_data = {
            'Indicator': ['RSI (14)', 'MACD', 'MACD Signal', 'BB Width', 'Price vs SMA20', 'Price vs SMA50'],
            'Value': [
                f"{last_row['RSI']:.2f}",
                f"{last_row['MACD']:.4f}",
                f"{last_row['MACD_Signal']:.4f}",
                f"{last_row['BB_Width']*100:.2f}%",
                f"{((last_row['Close']/last_row['SMA_20'])-1)*100:.2f}%",
                f"{((last_row['Close']/last_row['SMA_50'])-1)*100:.2f}%" if not pd.isna(last_row['SMA_50']) else 'N/A'
            ],
            'Signal': [
                '🔴 Overbought' if last_row['RSI'] > 70 else '🟢 Oversold' if last_row['RSI'] < 30 else '🟡 Neutral',
                '🟢 Bullish' if last_row['MACD'] > last_row['MACD_Signal'] else '🔴 Bearish',
                '🟢 Bullish' if last_row['MACD'] > 0 else '🔴 Bearish',
                '⚠️ Squeeze' if last_row['BB_Width'] < 0.1 else '🟡 Normal',
                '🟢 Above' if last_row['Close'] > last_row['SMA_20'] else '🔴 Below',
                '🟢 Above' if not pd.isna(last_row['SMA_50']) and last_row['Close'] > last_row['SMA_50'] else '🔴 Below' if not pd.isna(last_row['SMA_50']) else 'N/A'
            ]
        }
        
        indicator_df = pd.DataFrame(indicator_data)
        st.dataframe(indicator_df, use_container_width=True, hide_index=True)

# ==========================================
# FOOTER
# ==========================================
st.divider()
st.markdown("""
<div style="text-align: center; padding: 20px; color: #666;">
    <p><strong>The Little Quant Terminal</strong> | Institutional Edition v2.0</p>
    <p style="font-size: 12px;">
        ⚠️ <em>Disclaimer: This tool is for educational and research purposes only. 
        Not financial advice. Past performance does not guarantee future results.</em>
    </p>
    <p style="font-size: 11px; color: #444;">
        Built with ❤️ using Streamlit, yfinance, NumPy, SciPy & Matplotlib
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# ADDITIONAL UTILITY FUNCTIONS
# ==========================================

def export_analysis_to_excel(ticker: str, var_results: pd.DataFrame, portfolio_weights: Dict, 
                              stress_results: pd.DataFrame, pattern_data: Dict) -> bytes:
    """Export all analysis to Excel file"""
    import io
    
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # VaR Results
        if var_results is not None:
            var_results.to_excel(writer, sheet_name='VaR Analysis', index=False)
        
        # Portfolio Weights
        if portfolio_weights is not None:
            weights_data = []
            for opt_type in ['max_sharpe', 'min_var', 'risk_parity']:
                if opt_type in portfolio_weights:
                    opt = portfolio_weights[opt_type]
                    for i, ticker in enumerate(portfolio_weights['tickers']):
                        weights_data.append({
                            'Optimization': opt['type'],
                            'Ticker': ticker,
                            'Weight': opt['weights'][i],
                            'Return': opt['return'],
                            'Volatility': opt['volatility'],
                            'Sharpe': opt['sharpe']
                        })
            if weights_data:
                pd.DataFrame(weights_data).to_excel(writer, sheet_name='Portfolio Optimization', index=False)
        
        # Stress Test Results
        if stress_results is not None:
            stress_results.to_excel(writer, sheet_name='Stress Testing', index=False)
        
        # Pattern Data
        if pattern_data is not None and 'patterns' in pattern_data:
            if pattern_data['patterns']:
                pd.DataFrame(pattern_data['patterns']).to_excel(writer, sheet_name='Patterns', index=False)
    
    return buffer.getvalue()

# Export Button (in sidebar)
with st.sidebar:
    st.divider()
    st.markdown("### 📥 Export")
    
    if st.button("📊 Export All to Excel", key="export_excel"):
        try:
            excel_data = export_analysis_to_excel(
                ticker_in,
                st.session_state.get('var_results'),
                st.session_state.get('portfolio_weights'),
                st.session_state.get('stress_results'),
                st.session_state.get('pattern_data')
            )
            
            st.download_button(
                label="⬇️ Download Excel Report",
                data=excel_data,
                file_name=f"quant_analysis_{ticker_in}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Export error: {e}")
    
    st.divider()
    st.markdown("### ℹ️ Quick Reference")
    
    with st.expander("📖 VaR Interpretation"):
        st.markdown("""
        - **Historical VaR**: Based on actual past returns
        - **Parametric VaR**: Assumes normal distribution
        - **Cornish-Fisher**: Adjusts for skewness/kurtosis
        - **Expected Shortfall**: Average loss beyond VaR
        """)
    
    with st.expander("📖 Portfolio Optimization"):
        st.markdown("""
        - **Max Sharpe**: Highest risk-adjusted return
        - **Min Variance**: Lowest portfolio volatility
        - **Risk Parity**: Equal risk contribution
        """)
    
    with st.expander("📖 Regime Detection"):
        st.markdown("""
        - **High Vol**: Short-term vol > 1.5x long-term
        - **Rising Vol**: Increasing volatility trend
        - **Normal**: Stable volatility regime
        - **Low Vol**: Compressed volatility (breakout coming?)
        """)