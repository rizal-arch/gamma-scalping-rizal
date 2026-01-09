import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from dataclasses import dataclass
from typing import List
from enum import Enum
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from fpdf import FPDF
import tempfile
import os

warnings.filterwarnings('ignore')

# ==========================================
# 1. KONFIGURASI & CSS
# ==========================================
st.set_page_config(
    page_title="The Little Quant Terminal",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {background-color: #0e1117;}
    .metric-card {background-color: #262730; border: 1px solid #4e4f52; padding: 15px; border-radius: 5px; color: white;}
    /* Styling khusus untuk Sinyal */
    .buy-signal {color: #00FF00; font-weight: bold;}
    .sell-signal {color: #FF3333; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# Session State Init (Agar data tidak hilang saat klik tab lain)
if 'analyzed' not in st.session_state: st.session_state.analyzed = False
if 'pdf_bytes' not in st.session_state: st.session_state.pdf_bytes = None
if 'ticker_file_name' not in st.session_state: st.session_state.ticker_file_name = "report"
if 'radar_results' not in st.session_state: st.session_state.radar_results = pd.DataFrame()
if 'signal_results' not in st.session_state: st.session_state.signal_results = pd.DataFrame()

# ==========================================
# 2. MESIN 1: GAMMA SCALPING (AUDIT ENGINE)
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

# --- PDF GENERATOR (FULL) ---
class PDFReport(FPDF):
    def header(self): self.set_font('Arial', 'B', 15); self.cell(0, 10, "The Little Quant | Analysis Report", 0, 1, 'C'); self.ln(5)
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
    
    # Log Transaksi di PDF
    pdf.add_page(); pdf.set_font("Arial", "B", 12); pdf.cell(0, 10, "Transaction Log (Last 20)", 0, 1)
    pdf.set_font("Arial", "", 8); pdf.cell(40, 8, "Timestamp", 1); pdf.cell(20, 8, "Action", 1); pdf.cell(30, 8, "Price", 1); pdf.cell(30, 8, "Shares", 1); pdf.cell(30, 8, "Cost", 1, 1)
    if not trades_df.empty:
        for idx, row in trades_df.tail(20).iterrows():
            pdf.cell(40, 8, str(row['timestamp']), 1); pdf.cell(20, 8, row['action'], 1); pdf.cell(30, 8, str(row['price']), 1); pdf.cell(30, 8, str(row['shares']), 1); pdf.cell(30, 8, str(row['transaction_cost']), 1, 1)
    return pdf.output(dest='S').encode('latin-1')

# =========================================# ==========================================
# 3. MESIN 2: SMART WHALE RADAR (FIXED & UPGRADED)
# ==========================================
def whale_radar_scanner(tickers):
    radar_data = []
    
    # UI Feedback
    status_text = st.empty()
    status_text.info("📡 Menyalakan Radar... Memindai Jejak Institusi...")
    prog = st.progress(0)
    
    # Bersihkan input ticker
    clean_tickers = list(set([t.strip().upper() for t in tickers if t.strip() != ""]))
    
    for i, ticker in enumerate(clean_tickers):
        try:
            # Download data
            df = yf.download(ticker, period="3mo", progress=False, auto_adjust=True)
            
            # Handle MultiIndex column issue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Skip jika data kosong/kurang
            if df.empty or len(df) < 20: 
                continue
            
            df = df.dropna()
            
            # Ambil data terakhir
            last_close = float(df['Close'].iloc[-1])
            last_open = float(df['Open'].iloc[-1])
            last_vol = float(df['Volume'].iloc[-1])
            
            if last_vol == 0: 
                continue
            
            # Hitung Statistik 20 Hari Terakhir
            history = df.iloc[:-1]
            vol_mean = history['Volume'].tail(20).mean()
            vol_std = history['Volume'].tail(20).std()
            
            # Z-Score Volume
            vol_z = (last_vol - vol_mean) / vol_std if vol_std > 0 else 0
            rvol = last_vol / vol_mean if vol_mean > 0 else 0
            
            # Persentase Gerak Body Candle
            body_pct = abs(last_close - last_open) / last_open if last_open > 0 else 0
            
            # --- LOGIKA DETEKSI (THE BRAIN) ---
            status = "Normal"
            score = 0
            
            # 1. Gajah Asli (Volume Meledak)
            if vol_z > 2.0 and body_pct < 0.005: 
                status = "🛡️ ABSORPTION (Tembok)"
                score = 3
            elif vol_z > 2.0 and last_close > last_open: 
                status = "🚀 MARK-UP (Akumulasi)"
                score = 3
            elif vol_z > 2.0 and last_close < last_open: 
                status = "🔻 DISTRIBUTION (Guyur)"
                score = 3
            
            # 2. Badai Volatilitas (Tanpa Volume Gajah) - Kasus BULL.JK
            # Jika harga gerak liar (>3%) tapi volume biasa saja
            elif body_pct > 0.03:
                status = "🌪️ VOLATILE (Gerak Liar)"
                score = 2
            
            # 3. Aktivitas Sedang
            elif vol_z > 1.2: 
                status = "👀 High Vol"
                score = 1
            
            # Masukkan ke hasil jika ada score
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
        
        # Update progress bar
        prog.progress((i+1)/len(clean_tickers))
        
    status_text.empty()
    prog.empty()
    
    if not radar_data: 
        return pd.DataFrame()
    
    # Urutkan berdasarkan Score tertinggi
    return pd.DataFrame(radar_data).sort_values(by=["_score", "Z-Score"], ascending=False).drop(columns=["_score"])
 ==========================================
#
# 4. MESIN 3: SIGNAL GENERATOR
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
# 5. UI UTAMA (DASHBOARD)
# ==========================================
with st.sidebar:
    st.header("🎛️ Control Panel")
    client_mode = st.radio("Mode:", ["Retail (Long)", "Market Maker (Short)"])
    st.divider()
    currency = st.selectbox("Mata Uang:", ["USD", "IDR"])
    sym = "$" if currency == "USD" else "Rp"
    ticker_in = st.text_input("Ticker Audit:", "NVDA" if currency=="USD" else "BBCA.JK")
    period = st.selectbox("Durasi:", ["5d", "1mo", "3mo"])
    st.caption("Risk Param")
    d_thresh = st.slider("Delta Threshold", 0.05, 0.50, 0.10, 0.05)
    lot = st.number_input("Lot", 1, 100, 10)

# --- TAB NAVIGASI ---
tab_audit, tab_radar, tab_signal = st.tabs(["📊 Audit Strategi", "🐘 Whale Radar", "🎯 Signal Generator"])

# 1. TAB AUDIT (RESTORED FULL FEATURES)
with tab_audit:
    if st.button("JALANKAN AUDIT", type="primary"):
        with st.spinner("Processing..."):
            pos = PositionType.SHORT if "Short" in client_mode else PositionType.LONG
            cfg = MarketMakerConfig(ticker=ticker_in, position_type=pos, delta_threshold=d_thresh, num_straddles=lot, lookback_days=5 if period=="5d" else 60)
            ldr = DataIngestion(cfg); raw = ldr.fetch_data(ticker_in, period, "5m")
            if not raw.empty:
                df = ldr.process(raw); eng = MarketMakerBacktest(cfg, df)
                res, trd, prem = eng.run()
                st.session_state.analyzed=True; st.session_state.res_df=res; st.session_state.trades=trd
                st.session_state.pos=pos; st.session_state.tick=ticker_in
                
                # Metrics for PDF
                met = {"pnl_str": f"{sym} {res['total_pnl'].iloc[-1]:,.2f}", "opt_str": f"{sym} {res['opt_pnl'].iloc[-1]:,.2f}", 
                       "hed_str": f"{sym} {(res['total_pnl'].iloc[-1]-res['opt_pnl'].iloc[-1]):,.2f}", "trades_count": str(len(trd))}
                # Figs
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
        
        # --- RESTORED SUB-TABS (Fitur yang sempat hilang) ---
        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📈 Kurva Ekuitas", "⚠️ Risiko Gamma", "📝 Log Transaksi"])
        
        with sub_tab1:
            st.subheader("Kurva P&L")
            fig1, ax = plt.subplots(figsize=(10, 4))
            ax.plot(st.session_state.res_df.index, st.session_state.res_df['total_pnl'], color='#00FF00', linewidth=1.5)
            ax.axhline(0, color='white', linestyle='--', linewidth=0.5)
            # Dark mode friendly plot
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
            st.download_button("📄 Download PDF Laporan Lengkap", st.session_state.pdf_bytes, st.session_state.ticker_file_name, "application/pdf")

# 2. TAB WHALE RADAR
with tab_radar:
    col_rad1, col_rad2 = st.columns([3,1])
    def_tick = "BBCA.JK, BBRI.JK, TLKM.JK, BUMI.JK, GOTO.JK, MINA.JK, INET.JK, ANTM.JK"
    usr_tick = col_rad1.text_area("Watchlist Radar:", def_tick, height=70)
    if col_rad2.button("SCAN GAJAH"):
        st.session_state.radar_results = whale_radar_scanner(usr_tick.split(","))
    
    if not st.session_state.radar_results.empty:
        # Highlight Style
        def highlight_row(row):
            if "ABSORPTION" in row['Status']: return ['background-color: #4a148c; color: white'] * len(row)
            elif "MARK-UP" in row['Status']: return ['background-color: #1b5e20; color: white'] * len(row)
            elif "DISTRIBUTION" in row['Status']: return ['background-color: #b71c1c; color: white'] * len(row)
            return [''] * len(row)
            
        st.dataframe(st.session_state.radar_results.style.apply(highlight_row, axis=1).format({"Price":"{:,.0f}", "Z-Score":"{:.2f}", "RVOL":"{:.1f}x"}), use_container_width=True)

# 3. TAB SIGNAL GENERATOR
with tab_signal:
    st.subheader("Mesin Pencari Sinyal (The Alpha Scanner)")
    col_sig1, col_sig2 = st.columns([3,1])
    sig_tickers = col_sig1.text_area("Watchlist Sinyal:", "BBCA.JK, BBRI.JK, TLKM.JK, ADRO.JK, UNTR.JK, BTC-USD, EURUSD=X", height=70)
    if col_sig2.button("CARI PELUANG", type="primary"):
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
