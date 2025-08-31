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

from nsepython import nse_optionchain_scrapper

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

def calculate_pcr(put_value: float, call_value: float) -> float:
    """Calculate Put-Call Ratio"""
    return put_value / call_value if call_value != 0 else 0

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

def get_pcr_signal(pcr_value: float, metric_type: str = "OI") -> str:
    """Get PCR signal based on value and metric type"""
    if metric_type == "OI":
        if pcr_value > 1.2:
            return "🔻 **Bearish**"
        elif pcr_value < 0.8:
            return "🔺 **Bullish**"
        else:
            return "➡️ **Neutral**"
    elif metric_type == "ChgOI":
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

def calculate_comprehensive_sentiment_score(table_data, bucket_summary, pcr_data, spot_price) -> dict:
    """
    Comprehensive multi-factor sentiment analysis with weighted scoring
    Returns sentiment score from -100 (extremely bearish) to +100 (extremely bullish)
    """
    
    # Initialize scores
    scores = {
        "price_action": 0,
        "open_interest": 0, 
        "fresh_activity": 0,
        "position_distribution": 0
    }
    
    # 1. PRICE ACTION ANALYSIS (25% weight)
    price_score = 0
    
    # ATM strike analysis
    atm_strike = table_data.loc[table_data["Strike"].sub(spot_price).abs().idxmin(), "Strike"]
    strikes_above_spot = len(table_data[table_data["Strike"] > spot_price])
    strikes_below_spot = len(table_data[table_data["Strike"] < spot_price])
    
    if strikes_above_spot > strikes_below_spot:
        price_score += 20  # More room to rise
    elif strikes_below_spot > strikes_above_spot:
        price_score -= 20  # More room to fall
    
    # Price vs Max Pain analysis
    max_pain_strike = table_data.loc[table_data["CE_OI"].add(table_data["PE_OI"]).idxmax(), "Strike"]
    price_vs_max_pain = (spot_price - max_pain_strike) / max_pain_strike * 100
    
    if price_vs_max_pain > 2:
        price_score -= 30  # Price above max pain, downward pressure
    elif price_vs_max_pain < -2:
        price_score += 30  # Price below max pain, upward pressure
        
    scores["price_action"] = max(-100, min(100, price_score))
    
    # 2. OPEN INTEREST ANALYSIS (30% weight)
    oi_score = 0
    
    # OI PCR scoring
    oi_pcr = pcr_data['OVERALL_PCR_OI']
    if oi_pcr < 0.6:
        oi_score += 40  # Very bullish
    elif oi_pcr < 0.8:
        oi_score += 20  # Bullish
    elif oi_pcr > 1.4:
        oi_score -= 40  # Very bearish
    elif oi_pcr > 1.2:
        oi_score -= 20  # Bearish
    
    # OI concentration analysis
    total_ce_oi = bucket_summary["CE_ITM"]["OI"] + bucket_summary["CE_OTM"]["OI"]
    total_pe_oi = bucket_summary["PE_ITM"]["OI"] + bucket_summary["PE_OTM"]["OI"]
    
    # ITM vs OTM OI distribution
    ce_itm_dominance = bucket_summary["CE_ITM"]["OI"] / (total_ce_oi + 1)
    pe_itm_dominance = bucket_summary["PE_ITM"]["OI"] / (total_pe_oi + 1)
    
    if ce_itm_dominance > 0.6:  # Heavy CE ITM positions
        oi_score += 15
    if pe_itm_dominance > 0.6:  # Heavy PE ITM positions  
        oi_score -= 15
        
    scores["open_interest"] = max(-100, min(100, oi_score))
    
    # 3. FRESH ACTIVITY ANALYSIS (25% weight)
    activity_score = 0
    
    # Change in OI PCR (CRITICAL - reversed logic)
    chgoi_pcr = pcr_data['OVERALL_PCR_CHGOI']
    if chgoi_pcr > 2.0:
        activity_score += 50  # Heavy put unwinding = bullish
    elif chgoi_pcr > 1.5:
        activity_score += 30  # Moderate put unwinding = bullish
    elif chgoi_pcr > 1.0:
        activity_score += 10  # Slight put unwinding = slightly bullish
    elif chgoi_pcr < 0.3:
        activity_score -= 50  # Heavy call unwinding = bearish
    elif chgoi_pcr < 0.6:
        activity_score -= 30  # Moderate call unwinding = bearish
    elif chgoi_pcr < 1.0:
        activity_score -= 10  # Slight call unwinding = slightly bearish
    
    # Volume PCR analysis
    vol_pcr = pcr_data['OVERALL_PCR_VOLUME']
    if vol_pcr < 0.5:
        activity_score += 25  # Heavy call buying
    elif vol_pcr < 0.8:
        activity_score += 15  # Moderate call buying
    elif vol_pcr > 2.0:
        activity_score -= 25  # Heavy put buying
    elif vol_pcr > 1.3:
        activity_score -= 15  # Moderate put buying
        
    scores["fresh_activity"] = max(-100, min(100, activity_score))
    
    # 4. POSITION DISTRIBUTION ANALYSIS (20% weight)
    position_score = 0
    
    # Count different position types
    ce_positions = table_data['CE_Position'].value_counts()
    pe_positions = table_data['PE_Position'].value_counts()
    
    # Bullish positions
    bullish_ce = ce_positions.get("Long Build", 0) + ce_positions.get("Short Covering", 0)
    bullish_pe = pe_positions.get("Long Unwinding", 0) + pe_positions.get("Short Buildup", 0)
    
    # Bearish positions  
    bearish_ce = ce_positions.get("Short Buildup", 0) + ce_positions.get("Long Unwinding", 0)
    bearish_pe = pe_positions.get("Long Build", 0) + pe_positions.get("Short Covering", 0)
    
    total_strikes = len(table_data)
    
    # Calculate position bias
    net_bullish_activity = (bullish_ce - bearish_ce) + (bullish_pe - bearish_pe)
    position_bias_pct = (net_bullish_activity / total_strikes) * 100
    
    position_score = max(-100, min(100, position_bias_pct * 10))
    
    scores["position_distribution"] = position_score
    
    # 5. CALCULATE WEIGHTED FINAL SCORE
    weights = {
        "price_action": 0.25,
        "open_interest": 0.30,
        "fresh_activity": 0.25, 
        "position_distribution": 0.20
    }
    
    final_score = sum(scores[key] * weights[key] for key in scores.keys())
    
    # Determine sentiment category and confidence
    if final_score >= 60:
        sentiment = "STRONG BULLISH"
        confidence = "HIGH"
    elif final_score >= 30:
        sentiment = "BULLISH"
        confidence = "HIGH"
    elif final_score >= 15:
        sentiment = "BULLISH BIAS"
        confidence = "MEDIUM"
    elif final_score <= -60:
        sentiment = "STRONG BEARISH"
        confidence = "HIGH" 
    elif final_score <= -30:
        sentiment = "BEARISH"
        confidence = "HIGH"
    elif final_score <= -15:
        sentiment = "BEARISH BIAS"
        confidence = "MEDIUM"
    else:
        sentiment = "NEUTRAL"
        confidence = "MEDIUM"
    
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "final_score": final_score,
        "component_scores": scores,
        "score_breakdown": {
            "price_action": f"{scores['price_action']:.1f} (Weight: 25%)",
            "open_interest": f"{scores['open_interest']:.1f} (Weight: 30%)",
            "fresh_activity": f"{scores['fresh_activity']:.1f} (Weight: 25%)",
            "position_distribution": f"{scores['position_distribution']:.1f} (Weight: 20%)"
        }
    }

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

with st.sidebar:
    from nsepython import fnolist
    fno_list = [x for x in fnolist()]
    
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
    st.caption("Install `streamlit-autorefresh` for auto refresh.")

if AUTORFR and is_market_open():
    st_autorefresh(interval=refresh_sec * 1000, key="oc_refresh")

# ----------------------------
# Fetch data
# ----------------------------
@st.cache_data(ttl=30)
def fetch_oc(sym: str):
    return nse_optionchain_scrapper(sym)

data = fetch_oc(symbol)
if not data:
    st.error("Failed to fetch option chain.")
    st.stop()

spot = float(data["records"]["underlyingValue"])
expiry_list = data["records"]["expiryDates"] or []
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
st.pyplot(fig)

# ----------------------------
# Bucket summaries with calculated Greeks (No Tabs)
# ----------------------------
def flatten_block(block_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, r in block_df.iterrows():
        ce, pe = r["CE"], r["PE"]
        
        # Calculate Greeks for bucket analysis
        strike_price = float(r["strikePrice"])
        ce_iv = float(g(ce, "impliedVolatility")) / 100 if g(ce, "impliedVolatility") else 0.2
        pe_iv = float(g(pe, "impliedVolatility")) / 100 if g(pe, "impliedVolatility") else 0.2
        
        ce_greeks = calculate_greeks(spot, strike_price, time_to_expiry, risk_free_rate, ce_iv, 'call')
        pe_greeks = calculate_greeks(spot, strike_price, time_to_expiry, risk_free_rate, pe_iv, 'put')
        
        out.append({
            "Strike": strike_price,
            "CE_OI": float(g(ce, "openInterest")),
            "CE_ChgOI": float(g(ce, "changeinOpenInterest")),
            "CE_Volume": float(g(ce, "totalTradedVolume")),
            "CE_IV": ce_iv * 100,
            "CE_Delta": ce_greeks['delta'],
            "CE_Gamma": ce_greeks['gamma'],
            "CE_Theta": ce_greeks['theta'],
            "CE_Vega": ce_greeks['vega'],
            "PE_OI": float(g(pe, "openInterest")),
            "PE_ChgOI": float(g(pe, "changeinOpenInterest")),
            "PE_Volume": float(g(pe, "totalTradedVolume")),
            "PE_IV": pe_iv * 100,
            "PE_Delta": pe_greeks['delta'],
            "PE_Gamma": pe_greeks['gamma'],
            "PE_Theta": pe_greeks['theta'],
            "PE_Vega": pe_greeks['vega'],
        })
    return pd.DataFrame(out)

below_block = flatten_block(below_df)
above_block = flatten_block(above_df)

def agg_side_with_greeks(df_in: pd.DataFrame, side: str):
    if df_in.empty:
        return {"OI": 0, "ChgOI": 0, "Volume": 0, "IV": 0, "Delta": 0, "Gamma": 0, "Theta": 0, "Vega": 0}
    
    # Weight Greeks by OI for more meaningful aggregation
    total_oi = df_in[f"{side}_OI"].sum()
    if total_oi == 0:
        return {"OI": 0, "ChgOI": 0, "Volume": 0, "IV": 0, "Delta": 0, "Gamma": 0, "Theta": 0, "Vega": 0}
    
    weighted_delta = (df_in[f"{side}_Delta"] * df_in[f"{side}_OI"]).sum() / total_oi
    weighted_gamma = (df_in[f"{side}_Gamma"] * df_in[f"{side}_OI"]).sum() / total_oi
    weighted_theta = (df_in[f"{side}_Theta"] * df_in[f"{side}_OI"]).sum() / total_oi
    weighted_vega = (df_in[f"{side}_Vega"] * df_in[f"{side}_OI"]).sum() / total_oi
    
    return {
        "OI": df_in[f"{side}_OI"].sum(),
        "ChgOI": df_in[f"{side}_ChgOI"].sum(),
        "Volume": df_in[f"{side}_Volume"].sum(),
        "IV": df_in[f"{side}_IV"].mean() if not df_in[f"{side}_IV"].empty else 0,
        "Delta": weighted_delta,
        "Gamma": weighted_gamma,
        "Theta": weighted_theta,
        "Vega": weighted_vega,
    }

bucket_summary = {
    "CE_ITM": agg_side_with_greeks(below_block, "CE"),
    "CE_OTM": agg_side_with_greeks(above_block, "CE"),
    "PE_ITM": agg_side_with_greeks(above_block, "PE"),
    "PE_OTM": agg_side_with_greeks(below_block, "PE"),
}

# Calculate PCR values for OI, ChgOI, and Volume
pcr_data = {
    # OI PCRs
    "ITM_PCR_OI": calculate_pcr(bucket_summary["PE_ITM"]["OI"], bucket_summary["CE_ITM"]["OI"]),
    "OTM_PCR_OI": calculate_pcr(bucket_summary["PE_OTM"]["OI"], bucket_summary["CE_OTM"]["OI"]),
    "PUT_OTM_CALL_ITM_PCR_OI": calculate_pcr(bucket_summary["PE_OTM"]["OI"], bucket_summary["CE_ITM"]["OI"]),
    "PUT_ITM_CALL_OTM_PCR_OI": calculate_pcr(bucket_summary["PE_ITM"]["OI"], bucket_summary["CE_OTM"]["OI"]),
    "OVERALL_PCR_OI": calculate_pcr(
        bucket_summary["PE_ITM"]["OI"] + bucket_summary["PE_OTM"]["OI"],
        bucket_summary["CE_ITM"]["OI"] + bucket_summary["CE_OTM"]["OI"]
    ),
    
    # ChgOI PCRs
    "ITM_PCR_CHGOI": calculate_pcr(bucket_summary["PE_ITM"]["ChgOI"], bucket_summary["CE_ITM"]["ChgOI"]),
    "OTM_PCR_CHGOI": calculate_pcr(bucket_summary["PE_OTM"]["ChgOI"], bucket_summary["CE_OTM"]["ChgOI"]),
    "PUT_OTM_CALL_ITM_PCR_CHGOI": calculate_pcr(bucket_summary["PE_OTM"]["ChgOI"], bucket_summary["CE_ITM"]["ChgOI"]),
    "PUT_ITM_CALL_OTM_PCR_CHGOI": calculate_pcr(bucket_summary["PE_ITM"]["ChgOI"], bucket_summary["CE_OTM"]["ChgOI"]),
    "OVERALL_PCR_CHGOI": calculate_pcr(
        bucket_summary["PE_ITM"]["ChgOI"] + bucket_summary["PE_OTM"]["ChgOI"],
        bucket_summary["CE_ITM"]["ChgOI"] + bucket_summary["CE_OTM"]["ChgOI"]
    ),
    
    # Volume PCRs
    "ITM_PCR_VOLUME": calculate_pcr(bucket_summary["PE_ITM"]["Volume"], bucket_summary["CE_ITM"]["Volume"]),
    "OTM_PCR_VOLUME": calculate_pcr(bucket_summary["PE_OTM"]["Volume"], bucket_summary["CE_OTM"]["Volume"]),
    "PUT_OTM_CALL_ITM_PCR_VOLUME": calculate_pcr(bucket_summary["PE_OTM"]["Volume"], bucket_summary["CE_ITM"]["Volume"]),
    "PUT_ITM_CALL_OTM_PCR_VOLUME": calculate_pcr(bucket_summary["PE_ITM"]["Volume"], bucket_summary["CE_OTM"]["Volume"]),
    "OVERALL_PCR_VOLUME": calculate_pcr(
        bucket_summary["PE_ITM"]["Volume"] + bucket_summary["PE_OTM"]["Volume"],
        bucket_summary["CE_ITM"]["Volume"] + bucket_summary["CE_OTM"]["Volume"]
    )
}

# ----------------------------
# Enhanced Bucket Summaries with Greeks Analysis (No Tabs)
# ----------------------------
st.subheader(f"Enhanced Bucket Summaries with Greeks Analysis ({itm_count} ITM each side)")

# Basic Metrics and Greeks Analysis combined
left, middle, right = st.columns([1, 1, 1])

with left:
    st.markdown("### Calls (CE)")
    prev = st.session_state.prev_buckets["CE_ITM"] if st.session_state.prev_buckets else None
    st.markdown("**ITM (below spot)**")
    st.markdown(f"OI: {trend_badge(bucket_summary['CE_ITM']['OI'], None if not prev else prev['OI'])}", unsafe_allow_html=True)
    st.markdown(f"ChgOI: {trend_badge(bucket_summary['CE_ITM']['ChgOI'], None if not prev else prev['ChgOI'])}", unsafe_allow_html=True)
    st.markdown(f"Volume: {trend_badge(bucket_summary['CE_ITM']['Volume'], None if not prev else prev['Volume'])}", unsafe_allow_html=True)
    st.markdown(f"IV: {bucket_summary['CE_ITM']['IV']:.2f}%")
    st.markdown(f"Delta: {bucket_summary['CE_ITM']['Delta']:.4f}")
    st.markdown(f"Gamma: {bucket_summary['CE_ITM']['Gamma']:.4f}")
    st.markdown(f"Theta: {bucket_summary['CE_ITM']['Theta']:.4f}")
    st.markdown(f"Vega: {bucket_summary['CE_ITM']['Vega']:.4f}")

    prev = st.session_state.prev_buckets["CE_OTM"] if st.session_state.prev_buckets else None
    st.markdown("**OTM (above spot)**")
    st.markdown(f"OI: {trend_badge(bucket_summary['CE_OTM']['OI'], None if not prev else prev['OI'])}", unsafe_allow_html=True)
    st.markdown(f"ChgOI: {trend_badge(bucket_summary['CE_OTM']['ChgOI'], None if not prev else prev['ChgOI'])}", unsafe_allow_html=True)
    st.markdown(f"Volume: {trend_badge(bucket_summary['CE_OTM']['Volume'], None if not prev else prev['Volume'])}", unsafe_allow_html=True)
    st.markdown(f"IV: {bucket_summary['CE_OTM']['IV']:.2f}%")
    st.markdown(f"Delta: {bucket_summary['CE_OTM']['Delta']:.4f}")
    st.markdown(f"Gamma: {bucket_summary['CE_OTM']['Gamma']:.4f}")
    st.markdown(f"Theta: {bucket_summary['CE_OTM']['Theta']:.4f}")
    st.markdown(f"Vega: {bucket_summary['CE_OTM']['Vega']:.4f}")

with middle:
    st.markdown("### Puts (PE)")
    prev = st.session_state.prev_buckets["PE_ITM"] if st.session_state.prev_buckets else None
    st.markdown("**ITM (above spot)**")
    st.markdown(f"OI: {trend_badge(bucket_summary['PE_ITM']['OI'], None if not prev else prev['OI'])}", unsafe_allow_html=True)
    st.markdown(f"ChgOI: {trend_badge(bucket_summary['PE_ITM']['ChgOI'], None if not prev else prev['ChgOI'])}", unsafe_allow_html=True)
    st.markdown(f"Volume: {trend_badge(bucket_summary['PE_ITM']['Volume'], None if not prev else prev['Volume'])}", unsafe_allow_html=True)
    st.markdown(f"IV: {bucket_summary['PE_ITM']['IV']:.2f}%")
    st.markdown(f"Delta: {bucket_summary['PE_ITM']['Delta']:.4f}")
    st.markdown(f"Gamma: {bucket_summary['PE_ITM']['Gamma']:.4f}")
    st.markdown(f"Theta: {bucket_summary['PE_ITM']['Theta']:.4f}")
    st.markdown(f"Vega: {bucket_summary['PE_ITM']['Vega']:.4f}")

    prev = st.session_state.prev_buckets["PE_OTM"] if st.session_state.prev_buckets else None
    st.markdown("**OTM (below spot)**")
    st.markdown(f"OI: {trend_badge(bucket_summary['PE_OTM']['OI'], None if not prev else prev['OI'])}", unsafe_allow_html=True)
    st.markdown(f"ChgOI: {trend_badge(bucket_summary['PE_OTM']['ChgOI'], None if not prev else prev['ChgOI'])}", unsafe_allow_html=True)
    st.markdown(f"Volume: {trend_badge(bucket_summary['PE_OTM']['Volume'], None if not prev else prev['Volume'])}", unsafe_allow_html=True)
    st.markdown(f"IV: {bucket_summary['PE_OTM']['IV']:.2f}%")
    st.markdown(f"Delta: {bucket_summary['PE_OTM']['Delta']:.4f}")
    st.markdown(f"Gamma: {bucket_summary['PE_OTM']['Gamma']:.4f}")
    st.markdown(f"Theta: {bucket_summary['PE_OTM']['Theta']:.4f}")
    st.markdown(f"Vega: {bucket_summary['PE_OTM']['Vega']:.4f}")

with right:
    st.markdown("### PCR Analysis")
    
    st.markdown("""
    <div style="background-color:#f8f9fa;padding:10px;border-radius:8px;border-left:4px solid #0A71E2;">
        <div style="font-weight:600;color:#0A71E2;margin-bottom:5px;">Open Interest PCR</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"**Overall:** {pcr_data['OVERALL_PCR_OI']:.3f}")
    st.markdown(get_pcr_signal(pcr_data['OVERALL_PCR_OI'], "OI"))
    
    st.markdown(f"**ITM:** {pcr_data['ITM_PCR_OI']:.3f}")
    st.markdown(f"**OTM:** {pcr_data['OTM_PCR_OI']:.3f}")
    
    st.markdown("""
    <div style="background-color:#fff3e0;padding:10px;border-radius:8px;border-left:4px solid #ff9800;margin-top:10px;">
        <div style="font-weight:600;color:#ff9800;margin-bottom:5px;">Change in OI PCR</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"**Overall:** {pcr_data['OVERALL_PCR_CHGOI']:.3f}")
    st.markdown(get_pcr_signal(pcr_data['OVERALL_PCR_CHGOI'], "ChgOI"))
    
    st.markdown("""
    <div style="background-color:#e8f5e8;padding:10px;border-radius:8px;border-left:4px solid #4caf50;margin-top:10px;">
        <div style="font-weight:600;color:#4caf50;margin-bottom:5px;">Volume PCR</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"**Overall:** {pcr_data['OVERALL_PCR_VOLUME']:.3f}")
    st.markdown(get_pcr_signal(pcr_data['OVERALL_PCR_VOLUME'], "Volume"))


# ----------------------------
# Intelligent Sentiment Analysis (Simplified)
# ----------------------------
st.markdown("---")
st.subheader("🧠 Intelligent Market Sentiment Analysis")

# Get comprehensive sentiment analysis
sentiment_analysis = calculate_comprehensive_sentiment_score(table, bucket_summary, pcr_data, spot)

# Create sentiment display
sentiment_colors = {
    "STRONG BULLISH": "#4caf50",
    "BULLISH": "#8bc34a", 
    "BULLISH BIAS": "#cddc39",
    "NEUTRAL": "#6b7280",
    "BEARISH BIAS": "#ff9800",
    "BEARISH": "#ff5722",
    "STRONG BEARISH": "#f44336",
    "BEARISH REVERSAL": "#9c27b0",
    "CONSOLIDATION": "#2196f3",
    "MIXED SIGNALS": "#795548"
}

sentiment_icons = {
    "STRONG BULLISH": "🚀",
    "BULLISH": "📈", 
    "BULLISH BIAS": "📊",
    "NEUTRAL": "⚖️",
    "BEARISH BIAS": "📉",
    "BEARISH": "⬇️",
    "STRONG BEARISH": "💥",
    "BEARISH REVERSAL": "🔄",
    "CONSOLIDATION": "📦",
    "MIXED SIGNALS": "❓"
}

sentiment_color = sentiment_colors.get(sentiment_analysis["sentiment"], "#6b7280")
sentiment_icon = sentiment_icons.get(sentiment_analysis["sentiment"], "📊")

# Main sentiment card with score (Simplified)
score_color = sentiment_color
if sentiment_analysis["final_score"] > 0:
    score_bg = f"linear-gradient(90deg, #f0f0f0 50%, {sentiment_color}30 100%)"
else:
    score_bg = f"linear-gradient(90deg, {sentiment_color}30 0%, #f0f0f0 50%)"

st.markdown(f"""
<div style="background: linear-gradient(135deg, {sentiment_color}15 0%, {sentiment_color}05 100%);
            border: 2px solid {sentiment_color};
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;">
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 15px;">
        <div style="display: flex; align-items: center;">
            <div style="font-size: 2em; margin-right: 15px;">{sentiment_icon}</div>
            <div>
                <h2 style="color: {sentiment_color}; margin: 0; font-size: 1.8em;">
                    {sentiment_analysis["sentiment"]}
                </h2>
                <p style="margin: 5px 0 0 0; color: {sentiment_color}; font-weight: 600;">
                    Confidence: {sentiment_analysis["confidence"]}
                </p>
            </div>
        </div>
        <div style="text-align: center; padding: 15px; border-radius: 10px; 
                    background: {score_bg}; border: 2px solid {sentiment_color};">
            <h3 style="color: {sentiment_color}; margin: 0; font-size: 1.5em;">
                {sentiment_analysis["final_score"]:+.1f}
            </h3>
            <small style="color: #666; font-weight: 600;">Sentiment Score</small>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Score Breakdown Analysis (Simplified - No score scale)
st.markdown("### 📊 Score Breakdown Analysis")

st.markdown("#### Component Scores")
for component, description in sentiment_analysis["score_breakdown"].items():
    score_val = sentiment_analysis["component_scores"][component]
    
    if score_val > 20:
        bar_color = "#4caf50"
        text_color = "#2e7d32"
    elif score_val > 0:
        bar_color = "#8bc34a" 
        text_color = "#558b2f"
    elif score_val < -20:
        bar_color = "#f44336"
        text_color = "#c62828"
    elif score_val < 0:
        bar_color = "#ff5722"
        text_color = "#d84315"
    else:
        bar_color = "#9e9e9e"
        text_color = "#424242"
    
    # Create a visual bar for the score
    bar_width = abs(score_val)
    
    component_name = component.replace('_', ' ').title()
    
    st.markdown(f"""
    <div style="margin: 10px 0; padding: 10px; border-radius: 8px; background-color: #f8f9fa;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
            <strong style="color: {text_color};">{component_name}</strong>
            <span style="color: {text_color}; font-weight: 600;">{score_val:+.1f}</span>
        </div>
        <div style="background-color: #e0e0e0; height: 6px; border-radius: 3px; position: relative;">
            <div style="background-color: {bar_color}; height: 6px; border-radius: 3px; 
                       width: {bar_width}%; float: {'right' if score_val < 0 else 'left'};"></div>
        </div>
        <small style="color: #666;">{description.split('(Weight:')[1][:-1]} weight</small>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# Option Chain Table with Column Selection
# ----------------------------
st.subheader(f"Option Chain Table with Calculated Greeks ({itm_count} ITM each side)")

# Column selection interface
all_columns = [
    "CE_OI", "CE_LTP", "CE_Change", "CE_Volume", "CE_ChgOI", "CE_IV", 
    "CE_Delta", "CE_Gamma", "CE_Theta", "CE_Vega", "CE_Position",
    "Strike",
    "PE_Position", "PE_Vega", "PE_Theta", "PE_Gamma", "PE_Delta",
    "PE_IV", "PE_ChgOI", "PE_Volume", "PE_Change", "PE_LTP", "PE_OI",
    "PCR_Strike_OI", "PCR_Volume", "PCR_ChgOI"
]

default_columns = [
    "CE_OI", "CE_LTP", "CE_Change", "CE_Volume", "CE_ChgOI", "CE_Position",
    "Strike",
    "PE_Position", "PE_ChgOI", "PE_Volume", "PE_Change", "PE_LTP", "PE_OI"
]

# Column selection expander
with st.expander("🔧 Select Table Columns", expanded=False):
    col_selection_col1, col_selection_col2, col_selection_col3 = st.columns(3)
    
    with col_selection_col1:
        st.markdown("**Basic Metrics**")
        show_oi = st.checkbox("Open Interest", value=True)
        show_ltp = st.checkbox("Last Price", value=True)
        show_change = st.checkbox("Price Change", value=True)
        show_volume = st.checkbox("Volume", value=True)
        show_chgoi = st.checkbox("Change in OI", value=True)
        show_position = st.checkbox("Position Type", value=True)
        
    with col_selection_col2:
        st.markdown("**Greeks & IV**")
        show_iv = st.checkbox("Implied Volatility", value=False)
        show_delta = st.checkbox("Delta", value=False)
        show_gamma = st.checkbox("Gamma", value=False)
        show_theta = st.checkbox("Theta", value=False)
        show_vega = st.checkbox("Vega", value=False)
        
    with col_selection_col3:
        st.markdown("**PCR Ratios**")
        show_pcr_oi = st.checkbox("PCR Strike OI", value=False)
        show_pcr_volume = st.checkbox("PCR Volume", value=False)
        show_pcr_chgoi = st.checkbox("PCR ChgOI", value=False)

# Build selected columns list
selected_columns = []

# CE columns
if show_oi:
    selected_columns.append("CE_OI")
if show_ltp:
    selected_columns.append("CE_LTP")
if show_change:
    selected_columns.append("CE_Change")
if show_volume:
    selected_columns.append("CE_Volume")
if show_chgoi:
    selected_columns.append("CE_ChgOI")
if show_iv:
    selected_columns.append("CE_IV")
if show_delta:
    selected_columns.append("CE_Delta")
if show_gamma:
    selected_columns.append("CE_Gamma")
if show_theta:
    selected_columns.append("CE_Theta")
if show_vega:
    selected_columns.append("CE_Vega")
if show_position:
    selected_columns.append("CE_Position")

# Strike (always included)
selected_columns.append("Strike")

# PE columns (reverse order for better display)
if show_position:
    selected_columns.append("PE_Position")
if show_vega:
    selected_columns.append("PE_Vega")
if show_theta:
    selected_columns.append("PE_Theta")
if show_gamma:
    selected_columns.append("PE_Gamma")
if show_delta:
    selected_columns.append("PE_Delta")
if show_iv:
    selected_columns.append("PE_IV")
if show_chgoi:
    selected_columns.append("PE_ChgOI")
if show_volume:
    selected_columns.append("PE_Volume")
if show_change:
    selected_columns.append("PE_Change")
if show_ltp:
    selected_columns.append("PE_LTP")
if show_oi:
    selected_columns.append("PE_OI")

# PCR columns
if show_pcr_oi:
    selected_columns.append("PCR_Strike_OI")
if show_pcr_volume:
    selected_columns.append("PCR_Volume")
if show_pcr_chgoi:
    selected_columns.append("PCR_ChgOI")

atm_row_idx = table.index[table["Strike"] == atm_strike]
atm_row_idx = int(atm_row_idx[0]) if len(atm_row_idx) else None

ce_oi_max = table["CE_OI"].max()
ce_vol_max = table["CE_Volume"].max()
pe_oi_max = table["PE_OI"].max()
pe_vol_max = table["PE_Volume"].max()

def row_highlight(row):
    if atm_row_idx is not None and row.name == atm_row_idx:
        return ["background-color: #FFF3B0"] * len(row)
    return [""] * len(row)

def cell_green(val, max_val):
    try:
        return "background-color: #C6F6D5" if float(val) == float(max_val) else ""
    except:
        return ""

# Function to format numbers in K/L/Cr format for table
def format_table_number(num):
    """Format numbers for table display in K/L/Cr format"""
    if pd.isna(num):
        return "0"
    num = float(num)
    if abs(num) >= 10000000:  # 1 crore
        return f"{num/10000000:.2f}Cr"
    elif abs(num) >= 100000:  # 1 lakh
        return f"{num/100000:.2f}L"
    elif abs(num) >= 1000:
        return f"{num/1000:.2f}K"
    else:
        return f"{num:.0f}" if num == int(num) else f"{num:.2f}"

# Display table with selected columns
table_display = table[selected_columns]

styled = (
    table_display.style
    .apply(row_highlight, axis=1)
)

# Apply conditional formatting based on selected columns
if "CE_OI" in selected_columns:
    styled = styled.applymap(lambda v: cell_green(v, ce_oi_max), subset=["CE_OI"])
if "CE_Volume" in selected_columns:
    styled = styled.applymap(lambda v: cell_green(v, ce_vol_max), subset=["CE_Volume"])
if "PE_OI" in selected_columns:
    styled = styled.applymap(lambda v: cell_green(v, pe_oi_max), subset=["PE_OI"])
if "PE_Volume" in selected_columns:
    styled = styled.applymap(lambda v: cell_green(v, pe_vol_max), subset=["PE_Volume"])

# Apply position coloring if position columns are selected
position_columns = [col for col in selected_columns if col in ["CE_Position", "PE_Position"]]
if position_columns:
    styled = styled.applymap(position_color_style, subset=position_columns)

# Format columns based on selection
format_dict = {}
if "CE_OI" in selected_columns:
    format_dict["CE_OI"] = format_table_number
if "CE_LTP" in selected_columns:
    format_dict["CE_LTP"] = "{:,.2f}"
if "CE_Change" in selected_columns:
    format_dict["CE_Change"] = "{:+.2f}"
if "CE_Volume" in selected_columns:
    format_dict["CE_Volume"] = format_table_number
if "CE_ChgOI" in selected_columns:
    format_dict["CE_ChgOI"] = format_table_number
if "CE_IV" in selected_columns:
    format_dict["CE_IV"] = "{:.1f}%"
if "CE_Delta" in selected_columns:
    format_dict["CE_Delta"] = "{:.4f}"
if "CE_Gamma" in selected_columns:
    format_dict["CE_Gamma"] = "{:.4f}"
if "CE_Theta" in selected_columns:
    format_dict["CE_Theta"] = "{:.4f}"
if "CE_Vega" in selected_columns:
    format_dict["CE_Vega"] = "{:.4f}"

format_dict["Strike"] = "{:,.0f}"

if "PE_OI" in selected_columns:
    format_dict["PE_OI"] = format_table_number
if "PE_LTP" in selected_columns:
    format_dict["PE_LTP"] = "{:,.2f}"
if "PE_Change" in selected_columns:
    format_dict["PE_Change"] = "{:+.2f}"
if "PE_Volume" in selected_columns:
    format_dict["PE_Volume"] = format_table_number
if "PE_ChgOI" in selected_columns:
    format_dict["PE_ChgOI"] = format_table_number
if "PE_IV" in selected_columns:
    format_dict["PE_IV"] = "{:.1f}%"
if "PE_Delta" in selected_columns:
    format_dict["PE_Delta"] = "{:.4f}"
if "PE_Gamma" in selected_columns:
    format_dict["PE_Gamma"] = "{:.4f}"
if "PE_Theta" in selected_columns:
    format_dict["PE_Theta"] = "{:.4f}"
if "PE_Vega" in selected_columns:
    format_dict["PE_Vega"] = "{:.4f}"

if "PCR_Strike_OI" in selected_columns:
    format_dict["PCR_Strike_OI"] = "{:.3f}"
if "PCR_Volume" in selected_columns:
    format_dict["PCR_Volume"] = "{:.3f}"
if "PCR_ChgOI" in selected_columns:
    format_dict["PCR_ChgOI"] = "{:.3f}"

styled = styled.format(format_dict)

st.dataframe(styled, use_container_width=True)

# Enhanced Quick Stats
st.markdown("---")
st.subheader("Enhanced Quick Stats")

stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)

with stats_col1:
    st.metric("Total CE OI", format_number(table["CE_OI"].sum()))
    
with stats_col2:
    st.metric("Total PE OI", format_number(table["PE_OI"].sum()))
    
with stats_col3:
    max_pain_idx = (table["CE_OI"] + table["PE_OI"]).idxmax()
    max_pain_strike = table.loc[max_pain_idx, "Strike"]
    st.metric("Max Pain Strike", f"{max_pain_strike:,.0f}")
    
with stats_col4:
    total_ce_vol = table["CE_Volume"].sum()
    total_pe_vol = table["PE_Volume"].sum()
    volume_pcr = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0
    st.metric("Total Volume PCR", f"{volume_pcr:.3f}")

with stats_col5:
    # Calculate portfolio delta
    total_ce_oi = table["CE_OI"].sum()
    total_pe_oi = table["PE_OI"].sum()
    total_oi = total_ce_oi + total_pe_oi
    
    if total_oi > 0:
        portfolio_delta = ((table["CE_Delta"] * table["CE_OI"]).sum() + 
                          (table["PE_Delta"] * table["PE_OI"]).sum()) / total_oi
    else:
        portfolio_delta = 0
    
    st.metric("Portfolio Delta", f"{portfolio_delta:+.3f}")

# Update session state for next refresh
st.session_state.prev_buckets = bucket_summary

st.markdown("---")
st.caption(f"Data refreshed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Snapshot saved: {save_path}")
st.caption("**Note:** Greeks values are calculated using Black-Scholes formula with NSE implied volatility data.")
