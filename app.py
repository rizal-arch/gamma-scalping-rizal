import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from dataclasses import dataclass
from typing import List
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
from fpdf import FPDF
import tempfile
import os

warnings.filterwarnings('ignore')

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS
# ==========================================
st.set_page_config(
    page_title="Don's Quantitative Terminal",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom untuk Tampilan Institusional
st.markdown("""
<style>
    .stApp {background-color: #0e1117;}
    .metric-card {
        background-color: #262730;
        border: 1px solid #4e4f52;
        padding: 15px;
        border-radius: 5px;
        color: white;
    }
    /* Mempercantik Tab */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e2127;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e4f52;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOGIC MESIN ANALISIS
# ==========================================

class PositionType(Enum):
    LONG = "LONG"    # Retail Trader
    SHORT = "SHORT"  # Market Maker

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
    def total_tc_bps(self) -> float:
        return self.stock_commission_bps + self.slippage_bps
    
    @property
    def position_sign(self) -> int:
        return 1 if self.position_type == PositionType.LONG else -1

# --- BSM ENGINE ---
def bsm_vectorized(S, K, T, r, sigma, position_sign=1):
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
    
    call_price = S * N_d1 - K * discount * N_d2
    put_price = K * discount * N_neg_d2 - S * N_neg_d1
    straddle_price = call_price + put_price
    
    call_delta = N_d1
    put_delta = N_d1 - 1
    straddle_delta = call_delta + put_delta
    gamma = n_d1 / (S * sigma * sqrt_T)
    
    theta_term1 = -(S * n_d1 * sigma) / (2 * sqrt_T)
    theta_call = (theta_term1 - r * K * discount * N_d2) / 365
    theta_put = (theta_term1 + r * K * discount * N_neg_d2) / 365
    straddle_theta = theta_call + theta_put
    
    return pd.DataFrame({
        'price': S,
        'straddle_price_raw': straddle_price,
        'pos_value': straddle_price * position_sign,
        'pos_delta': straddle_delta * position_sign,
        'pos_gamma': 2 * gamma,
        'pos_theta': straddle_theta * position_sign * -1,
    })

# --- DATA HANDLER ---
class DataIngestion:
    def __init__(self, config):
        self.config = config
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_data(_self, ticker, period, interval):
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
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

# --- BACKTEST ENGINE ---
@dataclass
class HedgeTrade:
    timestamp: datetime
    action: str
    shares: int
    price: float
    transaction_cost: float

class MarketMakerBacktest:
    def __init__(self, config, market_data):
        self.config = config
        self.data = market_data.copy()
        self.trades = []
        
    def run(self):
        first_price = self.data['price'].iloc[0]
        strike = round(first_price)
        bars_per_day = 78
        
        self.data['T'] = (self.config.time_to_expiry_days/365 - self.data['bar_index']/(bars_per_day*252)).clip(1e-6)
        
        bsm_res = bsm_vectorized(
            self.data['price'], strike, self.data['T'], 
            self.config.risk_free_rate, 
            self.data['realized_vol'], 
            self.config.position_sign
        )
        
        mult = self.config.num_straddles * self.config.contract_multiplier
        self.data['pos_delta'] = bsm_res['pos_delta'] * mult
        self.data['pos_gamma'] = bsm_res['pos_gamma'] * mult
        self.data['pos_theta'] = bsm_res['pos_theta'] * mult
        self.data['liability'] = bsm_res['straddle_price_raw'] * mult
        
        initial_iv = self.data['implied_vol'].iloc[0]
        initial_bsm = bsm_vectorized(pd.Series([first_price]), strike, pd.Series([self.config.time_to_expiry_days/365]), 0.05, pd.Series([initial_iv]))
        premium_total = initial_bsm['straddle_price_raw'].iloc[0] * mult
        
        stock_pos = 0
        cash_flow_stock = 0.0
        cum_tc = 0.0
        stock_hist, pnl_hist = [], []
        
        threshold = self.config.delta_threshold * mult
        
        for idx, row in self.data.iterrows():
            net_delta = row['pos_delta'] + stock_pos
            
            if abs(net_delta) > threshold:
                shares = -int(round(net_delta))
                
                if abs(shares) >= self.config.min_shares_to_hedge:
                    price = row['price']
                    cost = abs(shares) * price * (self.config.total_tc_bps/10000)
                    
                    stock_pos += shares
                    cash_flow_stock -= (shares * price) 
                    cum_tc += cost
                    
                    self.trades.append(HedgeTrade(idx, "BUY" if shares>0 else "SELL", abs(shares), price, cost))
            
            stock_hist.append(stock_pos)
            
            current_stock_val = stock_pos * row['price']
            hedge_pnl = cash_flow_stock + current_stock_val
            
            if self.config.position_type == PositionType.LONG:
                 opt_pnl = row['liability'] - premium_total
                 total_pnl = opt_pnl + hedge_pnl - cum_tc
            else:
                 opt_pnl = premium_total - row['liability']
                 total_pnl = opt_pnl + hedge_pnl - cum_tc

            pnl_hist.append(total_pnl)

        self.data['stock_pos'] = stock_hist
        self.data['total_pnl'] = pnl_hist
        self.data['cum_tc'] = cum_tc
        self.data['opt_pnl'] = [premium_total - l if self.config.position_type == PositionType.SHORT else l - premium_total for l in self.data['liability']]
        
        return self.data, self.trades, premium_total

# ==========================================
# 3. PDF REPORT GENERATOR
# ==========================================
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, "Don's Quantitative Terminal | Analysis Report", 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf(ticker, strategy_name, final_metrics, trades_df, figures):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Judul
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Ticker: {ticker} | Strategi: {strategy_name}", 0, 1)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1)
    pdf.ln(5)
    
    # Metrics Table
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, "Executive Summary:", 0, 1, 'L', fill=True)
    
    pdf.cell(95, 10, f"Total Net P&L: {final_metrics['pnl_str']}", 1)
    pdf.cell(95, 10, f"Option Income: {final_metrics['opt_str']}", 1, 1)
    pdf.cell(95, 10, f"Scalping P&L:  {final_metrics['hed_str']}", 1)
    pdf.cell(95, 10, f"Total Trades:  {final_metrics['trades_count']}", 1, 1)
    pdf.ln(10)
    
    # Charts
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Visualisasi Kinerja & Risiko", 0, 1)
    
    for i, fig in enumerate(figures):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig.savefig(tmpfile.name, bbox_inches='tight', dpi=100)
            pdf.image(tmpfile.name, x=10, w=190)
            pdf.ln(5)
            
    # Trade Log
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Log Transaksi (20 Terakhir)", 0, 1)
    pdf.set_font("Arial", "", 8)
    
    pdf.cell(40, 8, "Timestamp", 1)
    pdf.cell(20, 8, "Action", 1)
    pdf.cell(30, 8, "Price", 1)
    pdf.cell(30, 8, "Shares", 1)
    pdf.cell(30, 8, "Cost", 1, 1)
    
    if not trades_df.empty:
        for idx, row in trades_df.tail(20).iterrows():
            pdf.cell(40, 8, str(row['timestamp']), 1)
            pdf.cell(20, 8, row['action'], 1)
            pdf.cell(30, 8, str(row['price']), 1)
            pdf.cell(30, 8, str(row['shares']), 1)
            pdf.cell(30, 8, str(row['transaction_cost']), 1, 1)
    else:
        pdf.cell(0, 8, "Tidak ada transaksi hedging tereksekusi.", 1, 1)

    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 4. DASHBOARD UI
# ==========================================

with st.sidebar:
    st.header("🎛️ Control Panel")
    
    client_mode = st.radio("Mode Klien:", ["Retail Trader (Long)", "Market Maker (Short)"])
    
    st.divider()
    
    # Currency Switcher
    col_curr1, col_curr2 = st.columns(2)
    with col_curr1:
        currency_code = st.selectbox("Mata Uang:", ["USD", "IDR"])
    with col_curr2:
        curr_sym = "$" if currency_code == "USD" else "Rp"
        st.write(f"Simbol: **{curr_sym}**")

    ticker_default = "NVDA" if currency_code == "USD" else "BBCA.JK"
    ticker = st.text_input("Ticker Saham:", ticker_default)
    period = st.selectbox("Durasi Data:", ["5d", "1mo", "3mo"])
    
    st.divider()
    
    st.caption("Risk Management")
    delta_thresh = st.slider("Delta Threshold (Risk Tolerance)", 0.05, 0.50, 0.10, 0.05)
    contracts = st.number_input("Jumlah Lot (Contracts)", 1, 100, 10)
    
    if "Market Maker" in client_mode:
        iv_premium = st.slider("Vol Premium Edge (%)", 0.0, 0.20, 0.05, 0.01)
    else:
        iv_premium = 0.0

if st.button("JALANKAN AUDIT", type="primary", use_container_width=True):
    with st.spinner("Mengunduh data pasar & kalkulasi Greeks..."):
        
        pos_type = PositionType.SHORT if "Market Maker" in client_mode else PositionType.LONG
        
        config = MarketMakerConfig(
            ticker=ticker,
            position_type=pos_type,
            delta_threshold=delta_thresh,
            num_straddles=contracts,
            implied_vol_premium=iv_premium,
            lookback_days=5 if period=="5d" else 60
        )
        
        loader = DataIngestion(config)
        raw_df = loader.fetch_data(ticker, period, "5m")
        
        if not raw_df.empty:
            df = loader.process(raw_df)
            engine = MarketMakerBacktest(config, df)
            res_df, trades, premium = engine.run()
            
            # --- DASHBOARD HEADER ---
            st.title(f"♟️ Laporan Analisis: {ticker}")
            st.markdown(f"**Strategi:** `{pos_type.value} STRADDLE` | **Status:** `{'Ready' if res_df['total_pnl'].iloc[-1] > 0 else 'Risk Warning'}`")
            
            # --- METRICS CARDS ---
            final = res_df.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            pnl_color = "normal" if final['total_pnl'] > 0 else "inverse"
            
            c1.metric("Total Net P&L", f"{curr_sym} {final['total_pnl']:,.2f}", delta=f"{final['total_pnl']:,.2f}", delta_color=pnl_color)
            c2.metric("P&L Opsi (Income)", f"{curr_sym} {final['opt_pnl']:,.2f}")
            c3.metric("P&L Hedging (Cost)", f"{curr_sym} {(final['total_pnl'] - final['opt_pnl']):,.2f}")
            c4.metric("Jumlah Trade", len(trades))
            
            # --- TABS VISUALISASI ---
            figures_to_print = [] # Container untuk PDF
            
            tab1, tab2, tab3 = st.tabs(["📈 Kinerja Keuangan", "⚠️ Audit Risiko", "📝 Log Transaksi"])
            
            with tab1:
                st.subheader("Kurva Ekuitas")
                # Chart 1: P&L
                fig1, ax = plt.subplots(figsize=(10, 4))
                ax.plot(res_df.index, res_df['total_pnl'], label='Total P&L', color='#00FF00' if final['total_pnl']>0 else '#FF3333', linewidth=2)
                ax.fill_between(res_df.index, 0, res_df['total_pnl'], alpha=0.1, color='gray')
                ax.axhline(0, color='white', linestyle='--', linewidth=0.5)
                ax.set_ylabel(f"P&L ({currency_code})")
                
                # Dark Mode Styling
                ax.set_facecolor('#0e1117')
                fig1.patch.set_facecolor('#0e1117')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.spines['left'].set_color('white')
                
                st.pyplot(fig1)
                figures_to_print.append(fig1) # Save for PDF

            with tab2:
                # Chart 2: Risk Profile (Dual Plot for PDF consistency)
                fig2, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                
                # Delta
                net_delta = res_df['pos_delta'] + res_df['stock_pos']
                ax_a.plot(res_df.index, net_delta, color='#F39C12')
                ax_a.set_title("Net Delta Exposure (Directional Risk)")
                ax_a.axhline(0, color='white', linestyle='--', linewidth=0.5)
                ax_a.set_facecolor('#0e1117')
                
                # Gamma
                ax_b.plot(res_df.index, res_df['pos_gamma'], color='#9B59B6')
                ax_b.set_title("Gamma Sensitivity (Acceleration Risk)")
                ax_b.set_facecolor('#0e1117')
                
                # Styling
                fig2.patch.set_facecolor('#0e1117')
                for ax in [ax_a, ax_b]:
                    ax.tick_params(colors='white')
                    ax.title.set_color('white')
                    ax.spines['bottom'].set_color('white')
                    ax.spines['top'].set_color('white')
                    ax.spines['right'].set_color('white')
                    ax.spines['left'].set_color('white')
                
                st.pyplot(fig2)
                figures_to_print.append(fig2) # Save for PDF

            with tab3:
                if trades:
                    trade_df = pd.DataFrame([vars(t) for t in trades])
                    # Formatting tampilan
                    display_df = trade_df.copy()
                    display_df['price'] = display_df['price'].apply(lambda x: f"{curr_sym} {x:,.2f}")
                    display_df['transaction_cost'] = display_df['transaction_cost'].apply(lambda x: f"{curr_sym} {x:,.2f}")
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.write("Pasar tenang. Tidak ada hedging yang tereksekusi.")

            # --- PDF DOWNLOAD SECTION ---
            st.divider()
            st.subheader("🖨️ Ekspor Laporan Klien")
            
            col_d1, col_d2 = st.columns([1, 4])
            with col_d1:
                if st.button("Generate PDF Report"):
                    with st.spinner("Mencetak dokumen institusional..."):
                        metrics_data = {
                            "pnl_str": f"{curr_sym} {final['total_pnl']:,.2f}",
                            "opt_str": f"{curr_sym} {final['opt_pnl']:,.2f}",
                            "hed_str": f"{curr_sym} {(final['total_pnl'] - final['opt_pnl']):,.2f}",
                            "trades_count": str(len(trades))
                        }
                        
                        pdf_bytes = create_pdf(
                            ticker=ticker,
                            strategy_name=f"{pos_type.value} STRADDLE",
                            final_metrics=metrics_data,
                            trades_df=pd.DataFrame([vars(t) for t in trades]) if trades else pd.DataFrame(),
                            figures=figures_to_print
                        )
                        
                        st.download_button(
                            label="📄 Download PDF Lengkap",
                            data=pdf_bytes,
                            file_name=f"Quant_Report_{ticker}_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
            with col_d2:
                st.info("Laporan PDF akan menyertakan ringkasan eksekutif, grafik kinerja, profil risiko, dan log transaksi lengkap.")
                    
        else:
            st.error(f"Gagal mengambil data untuk {ticker}. Pastikan ticker benar (Contoh: 'BBCA.JK' untuk Indonesia, 'NVDA' untuk US).")
