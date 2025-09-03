import os
import json
from datetime import datetime, time
from typing import Optional
import math

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pytz
from scipy.stats import norm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Optional auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh
    AUTORFR = True
except ImportError:
    AUTORFR = False

# Auto-refresh only during market hours
def is_market_open():
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist).time()
    return time(9, 15) <= now <= time(15, 30)

# Enhanced NSE data fetching with robust error handling
@st.cache_data(ttl=60)  # Cache for 1 minute
def robust_nse_fetch(url, timeout=30, max_retries=3):
    """
    Robust NSE data fetching with retries and proper headers
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Referer': 'https://www.nseindia.com/',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
    }
    
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        method_whitelist=["HEAD", "GET", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    try:
        # First, get cookies from main NSE page
        session.get("https://www.nseindia.com", headers=headers, timeout=timeout)
        
        # Then fetch the actual data
        response = session.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. NSE servers might be slow or unreachable.")
    except requests.exceptions.ConnectionError:
        raise Exception("Connection error. Please check your internet connection.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_fno_list():
    """
    Get F&O stock list with robust error handling
    """
    try:
        url = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
        data = robust_nse_fetch(url)
        
        if 'data' in data:
            symbols = [item['symbol'] for item in data['data'] if 'symbol' in item]
            # Add major indices
            indices = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']
            all_symbols = indices + [s for s in symbols if s not in indices]
            return sorted(all_symbols)
        else:
            # Fallback to a predefined list if API fails
            return get_fallback_symbols()
    except Exception as e:
        st.warning(f"Could not fetch live F&O list: {str(e)}")
        return get_fallback_symbols()

def get_fallback_symbols():
    """
    Fallback symbol list when API is unavailable
    """
    return [
        'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY',
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR',
        'ICICIBANK', 'KOTAKBANK', 'LT', 'ITC', 'SBIN',
        'BHARTIARTL', 'ASIANPAINT', 'MARUTI', 'AXISBANK',
        'TITAN', 'NESTLEIND', 'ULTRACEMCO', 'BAJFINANCE',
        'HCLTECH', 'WIPRO', 'SUNPHARMA', 'POWERGRID',
        'TATAMOTORS', 'NTPC', 'ONGC', 'COALINDIA',
        'GRASIM', 'BAJAJFINSV', 'M&M', 'TECHM',
        'DRREDDY', 'JSWSTEEL', 'CIPLA', 'EICHERMOT',
        'BRITANNIA', 'DIVISLAB', 'HEROMOTOCO', 'BPCL'
    ]

@st.cache_data(ttl=30)  # Cache for 30 seconds
def fetch_option_chain(symbol):
    """
    Fetch option chain data with robust error handling
    """
    try:
        if symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']:
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        else:
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
        
        data = robust_nse_fetch(url)
        return data
    except Exception as e:
        st.error(f"Failed to fetch option chain for {symbol}: {str(e)}")
        return None

# ----------------------------
# Black-Scholes Greeks Calculator
# ----------------------------
def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes Greeks
    S: Current stock price
    K: Strike price
    T: Time to expiration (in years)
    r: Risk-free rate
    sigma: Volatility (implied volatility)
    option_type: 'call' or 'put'
    """
    if T <= 0 or sigma <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        # Standard normal CDF and PDF
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)  # PDF for gamma, vega, theta calculations
        
        if option_type == 'call':
            delta = N_d1
            theta = (-S * n_d1 * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * N_d2) / 365
        else:  # put
            delta = N_d1 - 1
            theta = (-S * n_d1 * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
        
        # Gamma and Vega are same for calls and puts
        gamma = n_d1 / (S * sigma * math.sqrt(T))
        vega = S * n_d1 * math.sqrt(T) / 100  # Divided by 100 for percentage volatility
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega
        }
    except:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

def get_time_to_expiry(expiry_date):
    """Calculate time to expiry in years"""
    try:
        expiry = datetime.strptime(expiry_date, "%d-%b-%Y")
        now = datetime.now()
        days_to_expiry = (expiry - now).days
        return max(days_to_expiry / 365.0, 1/365)  # Minimum 1 day
    except:
        return 1/365  # Default to 1 day if parsing fails

# ----------------------------
# Utility functions
# ----------------------------
def format_number(num: float) -> str:
    """Format large numbers for summary display"""
    if num >= 10000000:  # 1 crore
        return f"{num/10000000:.2f}Cr"
    elif num >= 100000:  # 1 lakh
        return f"{num/100000:.2f}L"
    elif num >= 1000:
        return f"{num/1000:.2f}K"
    else:
        return str(int(num)) if num == int(num) else f"{num:.2f}"

def g(d, key):
    """Safe getter"""
    return d.get(key, 0) if isinstance(d, dict) else 0

def trend_badge(curr: float, prev: Optional[float]) -> str:
    if prev is None:
        return f"<span>{format_number(curr)}</span>"
    if curr > prev:
        return f"<span style='color:#16a34a;font-weight:600'>▲ {format_number(curr)}</span>"
    if curr < prev:
        return f"<span style='color:#dc2626;font-weight:600'>▼ {format_number(curr)}</span>"
    return f"<span style='color:#6b7280'>• {format_number(curr)}</span>"

def calculate_pcr(put_value: float, call_value: float, pcr_type: str = 'standard') -> float:
    """
    Calculate Put-Call Ratio with different calculation methods
    pcr_type: 'standard' for regular PCR, 'itm' for ITM PCR, 'otm' for OTM PCR
    """
    if call_value == 0:
        return 0
        
    if pcr_type == 'standard':
        return put_value / call_value
    elif pcr_type in ['itm', 'otm']:
        # For ITM and OTM ratios, we might want to weight the ratio differently
        # or apply different thresholds based on moneyness
        pcr = put_value / call_value
        return pcr
    else:
        return put_value / call_value

def get_position_signal(ltp: float, change: float, chg_oi: float) -> str:
    """Determine position type based on price change and change in OI"""
    if change == 0 and chg_oi == 0:
        return "No Change"
    
    price_up = change > 0
    price_down = change < 0
    oi_increase = chg_oi > 0
    oi_decrease = chg_oi < 0
    
    if price_up and oi_increase:
        return "Long Build"
    elif price_down and oi_decrease:
        return "Long Unwinding"
    elif price_down and oi_increase:
        return "Short Buildup"
    elif price_up and oi_decrease:
        return "Short Covering"
    elif oi_increase and change == 0:
        return "Fresh Positions"
    elif oi_decrease and change == 0:
        return "Position Unwinding"
    else:
        return "Mixed Activity"

def get_position_color(position: str) -> str:
    """Get color for position type"""
    colors = {
        "Long Build": "#4caf50",
        "Long Unwinding": "#ff5722", 
        "Short Buildup": "#f44336",
        "Short Covering": "#2196f3",
        "Fresh Positions": "#9c27b0",
        "Position Unwinding": "#ff9800",
        "Mixed Activity": "#795548",
        "No Change": "#6b7280",
        "No Data": "#9e9e9e"
    }
    return colors.get(position, "#6b7280")

def position_color_style(val):
    """Apply background color based on position type for styling"""
    color = get_position_color(val)
    return f'background-color: {color}20; color: {color}; font-weight: bold'

def get_pcr_signal(pcr_value: float, metric_type: str = "OI", ce_chgoi: float = 0, pe_chgoi: float = 0) -> str:
    """Get PCR signal based on value and metric type"""
    if metric_type == "OI":
        if pcr_value > 1.2:
            return "🔻 **Bearish**"
        elif pcr_value < 0.8:
            return "🔺 **Bullish**"
        else:
            return "➡️ **Neutral**"
    elif metric_type == "ChgOI":
        # New logic for Change in OI PCR
        if pcr_value < 0:  # Negative PCR
            if ce_chgoi < 0:  # Call side unwinding
                return "🔺 **Bullish (Call Unwinding)**"
            elif pe_chgoi < 0:  # Put side unwinding
                return "🔺 **Bullish (Put Unwinding)**"
            else:
                return "➡️ **Neutral**"
        else:  # Positive PCR
            if pcr_value > 1.5:
                return "🔺 **Strong Bullish**"
            elif pcr_value > 1.0:
                return "🔺 **Bullish**"
            elif pcr_value < 0.5:
                return "🔻 **Strong Bearish**"
            elif pcr_value < 1.0:
                return "🔻 **Bearish**"
            else:
                return "➡️ **Neutral**"
    else:  # Volume
        if pcr_value > 1.3:
            return "🔻 **Bearish**"
        elif pcr_value < 0.7:
            return "🔺 **Bullish**"
        else:
            return "➡️ **Neutral**"

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="NSE Option Chain", layout="wide")
st.title("📊 NSE Option Chain Dashboard")

# Initialize session state for symbol persistence
if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = "NIFTY"

if "prev_buckets" not in st.session_state:
    st.session_state.prev_buckets = None

# Show loading spinner while fetching symbol list
with st.spinner("Loading F&O symbols..."):
    fno_list = get_fno_list()

with st.sidebar:
    # Find the index of currently selected symbol
    try:
        current_index = fno_list.index(st.session_state.selected_symbol)
    except ValueError:
        current_index = 0
        st.session_state.selected_symbol = fno_list[0]
    
    # Symbol selection with persistent state
    symbol = st.selectbox(
        "Symbol", 
        fno_list, 
        index=current_index,
        key="symbol_selector"
    )
    
    # Update session state when symbol changes
    if symbol != st.session_state.selected_symbol:
        st.session_state.selected_symbol = symbol
    
    # Use the session state symbol
    symbol = st.session_state.selected_symbol
    
    itm_count = st.radio("ITM Strikes", [3, 5], index=1)
    refresh_sec = st.slider("Auto-Refresh (sec)", 10, 60, 30)
    risk_free_rate = st.number_input("Risk-free Rate (%)", value=5.84, min_value=0.0, max_value=15.0, step=0.1) / 100
    
    # Network status indicator
    st.markdown("---")
    st.markdown("**Connection Status**")
    try:
        test_response = requests.get("https://httpbin.org/status/200", timeout=5)
        if test_response.status_code == 200:
            st.success("✅ Network OK")
        else:
            st.warning("⚠️ Network Issues")
    except:
        st.error("❌ Network Down")
    
    st.caption("Install `streamlit-autorefresh` for auto refresh.")

if AUTORFR and is_market_open():
    st_autorefresh(interval=refresh_sec * 1000, key="oc_refresh")

# ----------------------------
# Fetch data with enhanced error handling
# ----------------------------
data = None
with st.spinner(f"Fetching option chain data for {symbol}..."):
    data = fetch_option_chain(symbol)

if not data:
    st.error("Failed to fetch option chain data. Please try:")
    st.markdown("""
    - **Refresh the page** - Network issues are often temporary
    - **Try a different symbol** - Some symbols may have better connectivity
    - **Check during market hours** - NSE servers are more responsive during trading hours
    - **Wait a few minutes** - NSE may be blocking requests temporarily
    """)
    st.stop()

# Safely extract data
try:
    spot = float(data["records"]["underlyingValue"])
    expiry_list = data["records"]["expiryDates"] or []
    
    if not expiry_list:
        st.error("No expiry dates available for this symbol.")
        st.stop()
        
except KeyError as e:
    st.error(f"Unexpected data format from NSE API. Missing key: {e}")
    st.json(data)  # Show raw data for debugging
    st.stop()
except Exception as e:
    st.error(f"Error processing NSE data: {e}")
    st.stop()

selected_expiry = st.selectbox("Select Expiry", expiry_list, index=0)

# Calculate time to expiry
time_to_expiry = get_time_to_expiry(selected_expiry)

# Save snapshot
os.makedirs("snapshots", exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join("snapshots", f"{symbol}_oc_{ts}.json")
with open(save_path, "w") as f:
    json.dump(data, f)

# Spot banner
st.markdown(
    f"""
<div style="background-color:#0A71E2;padding:12px;border-radius:12px;text-align:center;margin-bottom:12px">
  <div style="color:#fff;font-size:24px;font-weight:700;">{symbol} Spot: {spot}</div>
  <div style="color:#D8E9FF;font-size:16px;">Expiry: {selected_expiry} | Time to Expiry: {time_to_expiry*365:.0f} days</div>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Prepare data for table
# ----------------------------
rows_all = [r for r in data["records"]["data"] if r.get("expiryDate") == selected_expiry]
if not rows_all:
    st.warning("No rows for this expiry.")
    st.stop()

df = pd.DataFrame(rows_all)[["strikePrice", "CE", "PE"]].copy()
df.sort_values("strikePrice", inplace=True, ignore_index=True)

atm_idx = df["strikePrice"].sub(spot).abs().idxmin()
atm_strike = float(df.loc[atm_idx, "strikePrice"])

# Fixed strike selection to maintain consistent positioning
below_df = df[df["strikePrice"] < atm_strike].tail(itm_count)
above_df = df[df["strikePrice"] > atm_strike].head(itm_count)
atm_row = df.loc[[atm_idx]]

show_df = pd.concat([below_df, atm_row, above_df], axis=0).drop_duplicates(subset=["strikePrice"]).reset_index(drop=True)

flat = []
for _, r in show_df.iterrows():
    ce, pe = r["CE"], r["PE"]
    
    # Get price changes - using 0 if not available
    ce_change = float(g(ce, "change")) if g(ce, "change") else 0
    pe_change = float(g(pe, "change")) if g(pe, "change") else 0
    ce_chg_oi = float(g(ce, "changeinOpenInterest"))
    pe_chg_oi = float(g(pe, "changeinOpenInterest"))
    ce_ltp = float(g(ce, "lastPrice"))
    pe_ltp = float(g(pe, "lastPrice"))
    
    # Calculate position signals
    ce_position = get_position_signal(ce_ltp, ce_change, ce_chg_oi)
    pe_position = get_position_signal(pe_ltp, pe_change, pe_chg_oi)
    
    # Calculate Greeks using Black-Scholes
    strike_price = float(r["strikePrice"])
    ce_iv = float(g(ce, "impliedVolatility")) / 100 if g(ce, "impliedVolatility") else 0.2  # Default 20% if no IV
    pe_iv = float(g(pe, "impliedVolatility")) / 100 if g(pe, "impliedVolatility") else 0.2
    
    ce_greeks = calculate_greeks(spot, strike_price, time_to_expiry, risk_free_rate, ce_iv, 'call')
    pe_greeks = calculate_greeks(spot, strike_price, time_to_expiry, risk_free_rate, pe_iv, 'put')
    
    flat.append({
        "CE_OI": float(g(ce, "openInterest")),
        "CE_LTP": ce_ltp,
        "CE_Change": ce_change,
        "CE_Volume": float(g(ce, "totalTradedVolume")),
        "CE_ChgOI": ce_chg_oi,
        "CE_IV": ce_iv * 100,  # Convert back to percentage for display
        "CE_Delta": ce_greeks['delta'],
        "CE_Gamma": ce_greeks['gamma'],
        "CE_Theta": ce_greeks['theta'],
        "CE_Vega": ce_greeks['vega'],
        "CE_Position": ce_position,
        "Strike": strike_price,
        "PE_OI": float(g(pe, "openInterest")),
        "PE_LTP": pe_ltp,
        "PE_Change": pe_change,
        "PE_Volume": float(g(pe, "totalTradedVolume")),
        "PE_ChgOI": pe_chg_oi,
        "PE_IV": pe_iv * 100,  # Convert back to percentage for display
        "PE_Delta": pe_greeks['delta'],
        "PE_Gamma": pe_greeks['gamma'],
        "PE_Theta": pe_greeks['theta'],
        "PE_Vega": pe_greeks['vega'],
        "PE_Position": pe_position,
    })

table = pd.DataFrame(flat)

# Add PCR calculations for each strike
table["PCR_Strike_OI"] = table.apply(lambda row: calculate_pcr(row["PE_OI"], row["CE_OI"]), axis=1)
table["PCR_Volume"] = table.apply(lambda row: calculate_pcr(row["PE_Volume"], row["CE_Volume"]), axis=1)
table["PCR_ChgOI"] = table.apply(lambda row: calculate_pcr(row["PE_ChgOI"], row["CE_ChgOI"]), axis=1)

# ----------------------------
# Chart with dual axis
# ----------------------------
st.subheader("OI, ChgOI & Volume Distribution")
fig, ax1 = plt.subplots(figsize=(14, 6))

indices = np.arange(len(table))
bar_width = 0.2

# Plot for OI, ChgOI & Volume Distribution

# Left axis → OI & ChgOI
ax1.bar(indices - 0.2, table["CE_OI"], bar_width, color="#1f77b4", label="CE OI")
ax1.bar(indices, table["PE_OI"], bar_width, color="#2ca02c", label="PE OI")
ax1.bar(indices + 0.2, table["CE_ChgOI"], bar_width, color="#aec7e8", label="CE ChgOI")
ax1.bar(indices + 0.4, table["PE_ChgOI"], bar_width, color="#98df8a", label="PE ChgOI")

ax1.set_ylabel("OI / ChgOI")
ax1.set_xticks(indices)
ax1.set_xticklabels(table["Strike"], rotation=45)

# Right axis → Volume
ax2 = ax1.twinx()
ax2.plot(indices, table["CE_Volume"], marker="o", color="#03fc7f", label="CE Volume")
ax2.plot(indices, table["PE_Volume"], marker="o", color="#d62728", label="PE Volume")
ax2.set_ylabel("Volume")

# Spot line - fixed to show at ATM position
atm_position = table.index[table["Strike"] == atm_strike].tolist()
if atm_position:
    ax1.axvline(atm_position[0], color="red", linestyle="--", label=f"Spot {spot}")

# Merge legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, ncol=3)

ax1.set_title(f"OI, ChgOI & Volume for {symbol} ({itm_count} ITM each side)")
plt.tight_layout()
st.pyplot(fig)

# Quick display of the main table
st.subheader(f"Option Chain Table ({itm_count} ITM each side)")

# Basic table display first
display_columns = ["CE_OI", "CE_LTP", "CE_Change", "CE_Volume", "CE_ChgOI", "CE_Position", 
                   "Strike", "PE_Position", "PE_ChgOI", "PE_Volume", "PE_Change", "PE_LTP", "PE_OI"]

table_display = table[display_columns]

atm_row_idx = table.index[table["Strike"] == atm_strike]
atm_row_idx = int(atm_row_idx[0]) if len(atm_row_idx) else None

def row_highlight(row):
    if atm_row_idx is not None and row.name == atm_row_idx:
        return ["background-color: #FFF3B0"] * len(row)
    return [""] * len(row)

def cell_green(val, max_val):
    try:
        return "background-color: #C6F6D5" if float(val) == float(max_val) else ""
    except:
        return ""

# Format table numbers
def format_table_number(num):
    if pd.isna(num):
        return "0"
    num = float(num)
    if abs(num) >= 10000000:
        return f"{num/10000000:.2f}Cr"
    elif abs(num) >= 100000:
        return f"{num/100000:.2f}L"
    elif abs(num) >= 1000:
        return f"{num/1000:.2f}K"
    else:
        return f"{num:.0f}" if num == int(num) else f"{num:.2f}"

# Apply styling
styled = (
    table_display.style
    .apply(row_highlight, axis=1)
    .applymap(position_color_style, subset=["CE_Position", "PE_Position"])
    .format({
        "CE_OI": format_table_number,
        "CE_LTP": "{:,.2f}",
        "CE_Change": "{:+.2f}",
        "CE_Volume": format_table_number,
        "CE_ChgOI": format_table_number,
        "Strike": "{:,.0f}",
        "PE_ChgOI": format_table_number,
        "PE_Volume": format_table_number,
        "PE_Change": "{:+.2f}",
        "PE_LTP": "{:,.2f}",
        "PE_OI": format_table_number,
    })
)

st.dataframe(styled, use_container_width=True)

# Quick Stats
st.subheader("Quick Stats")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total CE OI", format_number(table["CE_OI"].sum()))
    
with col2:
    st.metric("Total PE OI", format_number(table["PE_OI"].sum()))
    
with col3:
    max_pain_idx = (table["CE_OI"] + table["PE_OI"]).idxmax()
    max_pain_strike = table.loc[max_pain_idx, "Strike"]
    st.metric("Max Pain Strike", f"{max_pain_strike:,.0f}")
    
with col4:
    total_ce_vol = table["CE_Volume"].sum()
    total_pe_vol = table["PE_Volume"].sum()
    volume_pcr = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0
    st.metric("Volume PCR", f"{volume_pcr:.3f}")

st.markdown("---")
st.caption(f"Data refreshed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Snapshot saved: {save_path}")
st.caption("**Note:** Greeks values are calculated using Black-Scholes formula with NSE implied volatility data.")

# Connection troubleshooting info
with st.expander("Connection Troubleshooting", expanded=False):
    st.markdown("""
    ### Common Issues and Solutions:
    
    **1. ReadTimeout Errors:**
    - NSE servers can be slow during high traffic periods
    - The app now includes retry logic and longer timeouts
    - Try refreshing the page if the error persists
    
    **2. Connection Blocked:**
    - NSE may temporarily block cloud server requests
    - This is more common during market hours
    - The app includes proper headers and session management
    
    **3. Network Issues:**
    - Check the network status indicator in the sidebar
    - If network is down, wait a few minutes and try again
    
    **4. Fallback Options:**
    - If symbol list fails to load, a predefined list is used
    - Core functionality remains available even with limited connectivity
    
    **5. Best Practices:**
    - Use the app during Indian market hours (9:15 AM - 3:30 PM IST) for best results
    - Avoid excessive refreshing which may trigger rate limits
    - Select different symbols if one isn't working
    """)
