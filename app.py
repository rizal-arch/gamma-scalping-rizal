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
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. KONFIGURASI HALAMAN (BRANDING DON)
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
    h1, h2, h3 {color: #FAFAFA;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOGIC MESIN (Diadaptasi dari 'Untuk market maker.txt')
# ==========================================

class PositionType(Enum):
    LONG = "LONG"    # Retail Trader (Beli Opsi)
    SHORT = "SHORT"  # Market Maker (Jual Opsi)

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
    
    # Raw Prices (Harga Pasar)
    call_price = S * N_d1 - K * discount * N_d2
    put_price = K * discount * N_neg_d2 - S * N_neg_d1
    straddle_price = call_price + put_price
    
    # Greeks
    call_delta = N_d1
    put_delta = N_d1 - 1
    straddle_delta = call_delta + put_delta
    gamma = n_d1 / (S * sigma * sqrt_T)
    
    # Theta Calculations
    theta_term1 = -(S * n_d1 * sigma) / (2 * sqrt_T)
    theta_call = (theta_term1 - r * K * discount * N_d2) / 365
    theta_put = (theta_term1 + r * K * discount * N_neg_d2) / 365
    straddle_theta = theta_call + theta_put
    
    # Output DataFrame dengan penyesuaian Posisi (Long/Short)
    return pd.DataFrame({
        'price': S,
        'straddle_price_raw': straddle_price, # Harga pasar murni
        'pos_value': straddle_price * position_sign, # Nilai bagi kita (Short = Liabilitas)
        'pos_delta': straddle_delta * position_sign, # Delta bagi kita
        'pos_gamma': 2 * gamma, # Gamma Magnitude
        'pos_theta': straddle_theta * position_sign * -1, # Short: Theta positif (Income)
    })

# --- DATA HANDLER ---
class DataIngestion:
    def __init__(self, config):
        self.config = config
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_data(_self, ticker, period, interval):
        # Menggunakan yfinance untuk tarik data
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        # Fix untuk yfinance versi baru yang kadang return MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data

    def process(self, df):
        df = df.dropna(subset=['Close']).copy()
        df['price'] = df['Close']
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        
        # Hitung Realized Volatility
        ann_factor = self.config.vol_annualization_factor
        df['realized_vol'] = df['log_return'].rolling(window=self.config.realized_vol_window).std() * np.sqrt(ann_factor)
        
        # Isi data kosong dengan forward fill atau nilai default
        df['realized_vol'] = df['realized_vol'].fillna(method='bfill').fillna(0.30).clip(0.05, 2.0)
        
        # Hitung Implied Vol (Market Maker Edge)
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
        # 1. Inisialisasi
        first_price = self.data['price'].iloc[0]
        strike = round(first_price)
        bars_per_day = 78
        
        # Hitung Time to Expiry (T)
        self.data['T'] = (self.config.time_to_expiry_days/365 - self.data['bar_index']/(bars_per_day*252)).clip(1e-6)
        
        # 2. Hitung Greeks (BSM)
        # Kita gunakan Implied Vol untuk menentukan harga Premium Awal (saat jual)
        # Tapi gunakan Realized Vol untuk Mark-to-Market harian
        bsm_res = bsm_vectorized(
            self.data['price'], strike, self.data['T'], 
            self.config.risk_free_rate, 
            self.data['realized_vol'], 
            self.config.position_sign
        )
        
        # Scaling dengan jumlah lot
        mult = self.config.num_straddles * self.config.contract_multiplier
        self.data['pos_delta'] = bsm_res['pos_delta'] * mult
        self.data['pos_gamma'] = bsm_res['pos_gamma'] * mult # Risk exposure
        self.data['pos_theta'] = bsm_res['pos_theta'] * mult # Income potential
        self.data['liability'] = bsm_res['straddle_price_raw'] * mult
        
        # Hitung Premium Awal (Uang yang diterima/dibayar di awal)
        initial_iv = self.data['implied_vol'].iloc[0]
        initial_bsm = bsm_vectorized(pd.Series([first_price]), strike, pd.Series([self.config.time_to_expiry_days/365]), 0.05, pd.Series([initial_iv]))
        premium_total = initial_bsm['straddle_price_raw'].iloc[0] * mult
        
        # 3. Simulasi Hedging
        stock_pos = 0
        cash_flow_stock = 0.0
        cum_tc = 0.0
        stock_hist, pnl_hist = [], []
        
        # Threshold dalam lembar saham
        threshold = self.config.delta_threshold * mult
        
        for idx, row in self.data.iterrows():
            net_delta = row['pos_delta'] + stock_pos
            
            # Logic Hedging: Jika Net Delta > Batas, lawan arah.
            if abs(net_delta) > threshold:
                shares = -int(round(net_delta))
                
                if abs(shares) >= self.config.min_shares_to_hedge:
                    price = row['price']
                    cost = abs(shares) * price * (self.config.total_tc_bps/10000)
                    
                    stock_pos += shares
                    # Cash Flow: Beli = Keluar Uang (-), Jual = Masuk Uang (+)
                    cash_flow_stock -= (shares * price) 
                    cum_tc += cost
                    
                    self.trades.append(HedgeTrade(idx, "BUY" if shares>0 else "SELL", abs(shares), price, cost))
            
            stock_hist.append(stock_pos)
            
            # 4. Perhitungan P&L Total
            current_stock_val = stock_pos * row['price']
            hedge_pnl = cash_flow_stock + current_stock_val # Realized + Unrealized Stock P&L
            
            if self.config.position_type == PositionType.LONG:
                 # Retail: (Nilai Opsi Sekarang - Biaya Beli Awal) + Profit Saham - Biaya Transaksi
                 opt_pnl = row['liability'] - premium_total
                 total_pnl = opt_pnl + hedge_pnl - cum_tc
            else:
                 # Market Maker: (Uang Diterima Awal - Kewajiban Bayar Sekarang) + Profit Saham - Biaya Transaksi
                 opt_pnl = premium_total - row['liability']
                 total_pnl = opt_pnl + hedge_pnl - cum_tc

            pnl_hist.append(total_pnl)

        self.data['stock_pos'] = stock_hist
        self.data['total_pnl'] = pnl_hist
        self.data['cum_tc'] = cum_tc
        # Kolom Opt PnL untuk visualisasi
        self.data['opt_pnl'] = [premium_total - l if self.config.position_type == PositionType.SHORT else l - premium_total for l in self.data['liability']]
        
        return self.data, self.trades, premium_total

# ==========================================
# 3. UI STREAMLIT (TAMPILAN KONSULTAN)
# ==========================================

with st.sidebar:
    st.header("🎛️ Parameter Simulasi")
    
    # Pilih Tipe Klien
    client_mode = st.radio("Mode Klien:", ["Retail Trader (Long)", "Market Maker (Short)"])
    
    st.divider()
    
    ticker = st.text_input("Ticker Saham:", "NVDA")
    period = st.selectbox("Durasi Data:", ["5d", "1mo", "3mo"])
    
    st.divider()
    
    st.caption("Pengaturan Risiko")
    # Slider menggantikan input manual
    delta_thresh = st.slider("Delta Threshold (Sensitivitas)", 0.05, 0.50, 0.10, 0.05)
    contracts = st.number_input("Jumlah Lot (Contracts)", 1, 100, 10)
    
    if "Market Maker" in client_mode:
        iv_premium = st.slider("Vol Premium Edge (%)", 0.0, 0.20, 0.05, 0.01)
    else:
        iv_premium = 0.0

if st.button("JALANKAN AUDIT", type="primary", use_container_width=True):
    with st.spinner("Sedang memproses data pasar & simulasi Greeks..."):
        
        # Tentukan Tipe Posisi
        pos_type = PositionType.SHORT if "Market Maker" in client_mode else PositionType.LONG
        
        # Konfigurasi Mesin
        config = MarketMakerConfig(
            ticker=ticker,
            position_type=pos_type,
            delta_threshold=delta_thresh,
            num_straddles=contracts,
            implied_vol_premium=iv_premium,
            lookback_days=5 if period=="5d" else 60
        )
        
        # Tarik Data
        loader = DataIngestion(config)
        raw_df = loader.fetch_data(ticker, period, "5m")
        
        if not raw_df.empty:
            df = loader.process(raw_df)
            
            # Jalankan Mesin
            engine = MarketMakerBacktest(config, df)
            res_df, trades, premium = engine.run()
            
            # --- TAMPILAN DASHBOARD ---
            
            # Judul
            st.title(f"♟️ Laporan Analisis: {ticker}")
            st.markdown(f"**Strategi:** `{pos_type.value} STRADDLE`")
            
            # Metrik Utama (Kartu)
            final = res_df.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            
            # Warna P&L
            pnl_color = "normal" if final['total_pnl'] > 0 else "inverse"
            
            c1.metric("Total Net P&L", f"${final['total_pnl']:,.2f}", delta=f"{final['total_pnl']:,.2f}", delta_color=pnl_color)
            c2.metric("P&L Opsi (Theta/Vega)", f"${final['opt_pnl']:,.2f}")
            c3.metric("P&L Hedging (Scalping)", f"${(final['total_pnl'] - final['opt_pnl']):,.2f}")
            c4.metric("Jumlah Trade", len(trades))
            
            # Tabs Visualisasi
            tab1, tab2, tab3 = st.tabs(["📈 Kinerja Keuangan", "⚠️ Audit Risiko", "📝 Log Transaksi"])
            
            with tab1:
                st.subheader("Kurva Ekuitas (Equity Curve)")
                # Grafik P&L menggunakan Matplotlib agar sama persis dengan gaya Don
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(res_df.index, res_df['total_pnl'], label='Total P&L', color='#00FF00' if final['total_pnl']>0 else '#FF3333', linewidth=2)
                ax.fill_between(res_df.index, 0, res_df['total_pnl'], alpha=0.1, color='gray')
                ax.axhline(0, color='white', linestyle='--', linewidth=0.5)
                
                # Styling Chart agar menyatu dengan Dark Mode Streamlit
                ax.set_facecolor('#0e1117')
                fig.patch.set_facecolor('#0e1117')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white') 
                ax.spines['right'].set_color('white')
                ax.spines['left'].set_color('white')
                
                st.pyplot(fig)
                
                st.info("Grafik ini menunjukkan pertumbuhan atau penyusutan modal bersih setelah dikurangi biaya hedging.")

            with tab2:
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("#### Eksposur Net Delta")
                    st.line_chart(res_df['pos_delta'] + res_df['stock_pos'])
                    st.caption("Garis ini harus mendekati nol. Lonjakan tajam menandakan risiko arah (Directional Risk).")
                    
                with col_right:
                    st.markdown("#### Risiko Gamma (Sensitivitas)")
                    st.line_chart(res_df['pos_gamma'])
                    st.caption("Semakin tinggi grafik ini, semakin cepat Anda rugi jika harga bergerak liar.")

            with tab3:
                if trades:
                    trade_df = pd.DataFrame([vars(t) for t in trades])
                    # Format kolom agar rapi
                    trade_df['price'] = trade_df['price'].map('${:,.2f}'.format)
                    trade_df['transaction_cost'] = trade_df['transaction_cost'].map('${:,.2f}'.format)
                    st.dataframe(trade_df, use_container_width=True)
                else:
                    st.write("Tidak ada aktivitas hedging (Pasar terlalu tenang).")
                    
        else:
            st.error(f"Gagal mengambil data untuk {ticker}. Coba simbol lain.")
