import os
import json
from datetime import datetime, time
from typing import Optional

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pytz

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
        # REVERSED: For Change in OI, LOWER PCR indicates bearish (call unwinding/put building)
        # HIGHER PCR indicates bullish (put unwinding/call building)
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
        # For Volume, similar to ChgOI but slightly different thresholds
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
    bullish_pe = pe_positions.get("Long Unwinding", 0) + pe_positions.get("Short Buildup", 0)  # For puts, these are bearish for underlying
    
    # Bearish positions  
    bearish_ce = ce_positions.get("Short Buildup", 0) + ce_positions.get("Long Unwinding", 0)
    bearish_pe = pe_positions.get("Long Build", 0) + pe_positions.get("Short Covering", 0)  # For puts, these are bullish for underlying
    
    total_strikes = len(table_data)
    
    # Calculate position bias
    net_bullish_activity = (bullish_ce - bearish_ce) + (bullish_pe - bearish_pe)
    position_bias_pct = (net_bullish_activity / total_strikes) * 100
    
    position_score = max(-100, min(100, position_bias_pct * 10))  # Scale to -100 to +100
    
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
        description = "Multiple strong bullish indicators across price action, OI, and fresh activity"
        action = "Consider aggressive bullish strategies - ATM/OTM calls, bull spreads"
    elif final_score >= 30:
        sentiment = "BULLISH"
        confidence = "HIGH"
        description = "Clear bullish bias with supporting indicators"
        action = "Consider bullish strategies - ITM calls, bull call spreads"
    elif final_score >= 15:
        sentiment = "BULLISH BIAS"
        confidence = "MEDIUM"
        description = "Mild bullish tilt, some supporting factors"
        action = "Cautiously bullish - consider call spreads with risk management"
    elif final_score <= -60:
        sentiment = "STRONG BEARISH"
        confidence = "HIGH" 
        description = "Multiple strong bearish indicators across all parameters"
        action = "Consider aggressive bearish strategies - ATM/OTM puts, bear spreads"
    elif final_score <= -30:
        sentiment = "BEARISH"
        confidence = "HIGH"
        description = "Clear bearish bias with supporting indicators"
        action = "Consider bearish strategies - ITM puts, bear put spreads"
    elif final_score <= -15:
        sentiment = "BEARISH BIAS"
        confidence = "MEDIUM"
        description = "Mild bearish tilt, some supporting factors"
        action = "Cautiously bearish - consider put spreads with risk management"
    else:
        sentiment = "NEUTRAL"
        confidence = "MEDIUM"
        description = "Mixed signals or balanced market conditions"
        action = "Consider neutral strategies - straddles, strangles, iron condors"
    
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "description": description,
        "action_suggestion": action,
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
  <div style="color:#D8E9FF;font-size:16px;">Expiry: {selected_expiry}</div>
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

below_df = df[df["strikePrice"] < spot].tail(itm_count)
above_df = df[df["strikePrice"] > spot].head(itm_count)
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
    
    flat.append({
        "CE_OI": float(g(ce, "openInterest")),
        "CE_LTP": ce_ltp,
        "CE_Change": ce_change,
        "CE_Volume": float(g(ce, "totalTradedVolume")),
        "CE_ChgOI": ce_chg_oi,
        "CE_IV": float(g(ce, "impliedVolatility")),
        "CE_Position": ce_position,
        "Strike": float(r["strikePrice"]),
        "PE_OI": float(g(pe, "openInterest")),
        "PE_LTP": pe_ltp,
        "PE_Change": pe_change,
        "PE_Volume": float(g(pe, "totalTradedVolume")),
        "PE_ChgOI": pe_chg_oi,
        "PE_IV": float(g(pe, "impliedVolatility")),
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

# Spot line
ax1.axvline(len(indices)//2, color="red", linestyle="--", label=f"Spot {spot}")

# Merge legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, ncol=3)

ax1.set_title(f"OI, ChgOI & Volume for {symbol} ({itm_count} ITM each side)")
st.pyplot(fig)

# ----------------------------
# Bucket summaries
# ----------------------------
def flatten_block(block_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, r in block_df.iterrows():
        ce, pe = r["CE"], r["PE"]
        out.append({
            "Strike": float(r["strikePrice"]),
            "CE_OI": float(g(ce, "openInterest")),
            "CE_ChgOI": float(g(ce, "changeinOpenInterest")),
            "CE_Volume": float(g(ce, "totalTradedVolume")),
            "CE_IV": float(g(ce, "impliedVolatility")),
            "PE_OI": float(g(pe, "openInterest")),
            "PE_ChgOI": float(g(pe, "changeinOpenInterest")),
            "PE_Volume": float(g(pe, "totalTradedVolume")),
            "PE_IV": float(g(pe, "impliedVolatility")),
        })
    return pd.DataFrame(out)

below_block = flatten_block(below_df)
above_block = flatten_block(above_df)

def agg_side(df_in: pd.DataFrame, side: str):
    if df_in.empty:
        return {"OI": 0, "ChgOI": 0, "Volume": 0, "IV": 0}
    return {
        "OI": df_in[f"{side}_OI"].sum(),
        "ChgOI": df_in[f"{side}_ChgOI"].sum(),
        "Volume": df_in[f"{side}_Volume"].sum(),
        "IV": df_in[f"{side}_IV"].mean() if not df_in[f"{side}_IV"].empty else 0,
    }

bucket_summary = {
    "CE_ITM": agg_side(below_block, "CE"),
    "CE_OTM": agg_side(above_block, "CE"),
    "PE_ITM": agg_side(above_block, "PE"),
    "PE_OTM": agg_side(below_block, "PE"),
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
# Show summaries with enhanced PCR
# ----------------------------
st.subheader(f"Bucket Summaries ({itm_count} ITM each side)")
left, middle, right = st.columns([2, 2, 2])

with left:
    st.markdown("### Calls (CE)")
    prev = st.session_state.prev_buckets["CE_ITM"] if st.session_state.prev_buckets else None
    st.markdown("**ITM (below spot)**")
    st.markdown(f"OI: {trend_badge(bucket_summary['CE_ITM']['OI'], None if not prev else prev['OI'])}", unsafe_allow_html=True)
    st.markdown(f"ChgOI: {trend_badge(bucket_summary['CE_ITM']['ChgOI'], None if not prev else prev['ChgOI'])}", unsafe_allow_html=True)
    st.markdown(f"Volume: {trend_badge(bucket_summary['CE_ITM']['Volume'], None if not prev else prev['Volume'])}", unsafe_allow_html=True)
    st.markdown(f"IV: {bucket_summary['CE_ITM']['IV']:.2f}")

    prev = st.session_state.prev_buckets["CE_OTM"] if st.session_state.prev_buckets else None
    st.markdown("**OTM (above spot)**")
    st.markdown(f"OI: {trend_badge(bucket_summary['CE_OTM']['OI'], None if not prev else prev['OI'])}", unsafe_allow_html=True)
    st.markdown(f"ChgOI: {trend_badge(bucket_summary['CE_OTM']['ChgOI'], None if not prev else prev['ChgOI'])}", unsafe_allow_html=True)
    st.markdown(f"Volume: {trend_badge(bucket_summary['CE_OTM']['Volume'], None if not prev else prev['Volume'])}", unsafe_allow_html=True)
    st.markdown(f"IV: {bucket_summary['CE_OTM']['IV']:.2f}")

with middle:
    st.markdown("### Puts (PE)")
    prev = st.session_state.prev_buckets["PE_ITM"] if st.session_state.prev_buckets else None
    st.markdown("**ITM (above spot)**")
    st.markdown(f"OI: {trend_badge(bucket_summary['PE_ITM']['OI'], None if not prev else prev['OI'])}", unsafe_allow_html=True)
    st.markdown(f"ChgOI: {trend_badge(bucket_summary['PE_ITM']['ChgOI'], None if not prev else prev['ChgOI'])}", unsafe_allow_html=True)
    st.markdown(f"Volume: {trend_badge(bucket_summary['PE_ITM']['Volume'], None if not prev else prev['Volume'])}", unsafe_allow_html=True)
    st.markdown(f"IV: {bucket_summary['PE_ITM']['IV']:.2f}")

    prev = st.session_state.prev_buckets["PE_OTM"] if st.session_state.prev_buckets else None
    st.markdown("**OTM (below spot)**")
    st.markdown(f"OI: {trend_badge(bucket_summary['PE_OTM']['OI'], None if not prev else prev['OI'])}", unsafe_allow_html=True)
    st.markdown(f"ChgOI: {trend_badge(bucket_summary['PE_OTM']['ChgOI'], None if not prev else prev['ChgOI'])}", unsafe_allow_html=True)
    st.markdown(f"Volume: {trend_badge(bucket_summary['PE_OTM']['Volume'], None if not prev else prev['Volume'])}", unsafe_allow_html=True)
    st.markdown(f"IV: {bucket_summary['PE_OTM']['IV']:.2f}")

with right:
    st.markdown("### Enhanced PCR Analysis")
    
    # Create tabs for different PCR types
    tab1, tab2, tab3 = st.tabs(["OI PCR", "ChgOI PCR", "Volume PCR"])
    
    with tab1:
        st.markdown("""
        <div style="background-color:#f8f9fa;padding:10px;border-radius:8px;border-left:4px solid #0A71E2;">
            <div style="font-weight:600;color:#0A71E2;margin-bottom:5px;">Open Interest PCR</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Overall:** {pcr_data['OVERALL_PCR_OI']:.3f}")
        st.markdown(get_pcr_signal(pcr_data['OVERALL_PCR_OI'], "OI"))
        
        st.markdown(f"**ITM:** {pcr_data['ITM_PCR_OI']:.3f}")
        st.markdown(f"**OTM:** {pcr_data['OTM_PCR_OI']:.3f}")
        
        st.markdown("**Cross Ratios:**")
        st.caption(f"PUT OTM/CE ITM: {pcr_data['PUT_OTM_CALL_ITM_PCR_OI']:.3f}")
        st.caption(f"PUT ITM/CE OTM: {pcr_data['PUT_ITM_CALL_OTM_PCR_OI']:.3f}")
    
    with tab2:
        st.markdown("""
        <div style="background-color:#fff3e0;padding:10px;border-radius:8px;border-left:4px solid #ff9800;">
            <div style="font-weight:600;color:#ff9800;margin-bottom:5px;">Change in OI PCR</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Overall:** {pcr_data['OVERALL_PCR_CHGOI']:.3f}")
        st.markdown(get_pcr_signal(pcr_data['OVERALL_PCR_CHGOI'], "ChgOI"))
        
        st.markdown(f"**ITM:** {pcr_data['ITM_PCR_CHGOI']:.3f}")
        st.markdown(f"**OTM:** {pcr_data['OTM_PCR_CHGOI']:.3f}")
        
        st.markdown("**Cross Ratios:**")
        st.caption(f"PUT OTM/CE ITM: {pcr_data['PUT_OTM_CALL_ITM_PCR_CHGOI']:.3f}")
        st.caption(f"PUT ITM/CE OTM: {pcr_data['PUT_ITM_CALL_OTM_PCR_CHGOI']:.3f}")
        
        st.caption("📈 Shows recent position changes (Lower=Bearish, Higher=Bullish)")
    
    with tab3:
        st.markdown("""
        <div style="background-color:#e8f5e8;padding:10px;border-radius:8px;border-left:4px solid #4caf50;">
            <div style="font-weight:600;color:#4caf50;margin-bottom:5px;">Volume PCR</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Overall:** {pcr_data['OVERALL_PCR_VOLUME']:.3f}")
        st.markdown(get_pcr_signal(pcr_data['OVERALL_PCR_VOLUME'], "Volume"))
        
        st.markdown(f"**ITM:** {pcr_data['ITM_PCR_VOLUME']:.3f}")
        st.markdown(f"**OTM:** {pcr_data['OTM_PCR_VOLUME']:.3f}")
        
        st.markdown("**Cross Ratios:**")
        st.caption(f"PUT OTM/CE ITM: {pcr_data['PUT_OTM_CALL_ITM_PCR_VOLUME']:.3f}")
        st.caption(f"PUT ITM/CE OTM: {pcr_data['PUT_ITM_CALL_OTM_PCR_VOLUME']:.3f}")
        
        st.caption("📊 Shows current session activity")


# ----------------------------
# Intelligent Sentiment Analysis
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

# Main sentiment card with score
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
    <p style="font-size: 1.1em; margin: 10px 0; color: #444;">
        <strong>Analysis:</strong> {sentiment_analysis["description"]}
    </p>
    <p style="font-size: 1em; margin: 10px 0; color: #666; background-color: rgba(255,255,255,0.7); 
              padding: 10px; border-radius: 8px;">
        <strong>Strategy:</strong> {sentiment_analysis["action_suggestion"]}
    </p>
</div>
""", unsafe_allow_html=True)

# Detailed Score Breakdown
st.markdown("### 📊 Score Breakdown Analysis")

score_col1, score_col2 = st.columns([2, 1])

with score_col1:
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
        bar_direction = "right" if score_val >= 0 else "left"
        
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

with score_col2:
    st.markdown("#### Score Scale")
    st.markdown("""
    <div style="padding: 15px; border-radius: 10px; background-color: #f8f9fa;">
        <div style="margin: 5px 0; color: #2e7d32; font-weight: 600;">+60 to +100: Strong Bullish</div>
        <div style="margin: 5px 0; color: #558b2f; font-weight: 600;">+30 to +60: Bullish</div>
        <div style="margin: 5px 0; color: #8bc34a; font-weight: 600;">+15 to +30: Bullish Bias</div>
        <div style="margin: 5px 0; color: #424242; font-weight: 600;">-15 to +15: Neutral</div>
        <div style="margin: 5px 0; color: #d84315; font-weight: 600;">-30 to -15: Bearish Bias</div>
        <div style="margin: 5px 0; color: #c62828; font-weight: 600;">-60 to -30: Bearish</div>
        <div style="margin: 5px 0; color: #b71c1c; font-weight: 600;">-100 to -60: Strong Bearish</div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# Table view
# ----------------------------
st.subheader(f"Option Chain Table ({itm_count} ITM each side)")

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

# Reorder table columns for better readability
table_display = table[[
    "CE_OI", "CE_LTP", "CE_Change", "CE_Volume", "CE_ChgOI", "CE_IV", "CE_Position",
    "Strike",
    "PE_Position", "PE_IV", "PE_ChgOI", "PE_Volume", "PE_Change", "PE_LTP", "PE_OI",
    "PCR_Strike_OI", "PCR_Volume", "PCR_ChgOI"
]]

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

styled = (
    table_display.style
    .apply(row_highlight, axis=1)
    .applymap(lambda v: cell_green(v, ce_oi_max), subset=["CE_OI"])
    .applymap(lambda v: cell_green(v, ce_vol_max), subset=["CE_Volume"])
    .applymap(lambda v: cell_green(v, pe_oi_max), subset=["PE_OI"])
    .applymap(lambda v: cell_green(v, pe_vol_max), subset=["PE_Volume"])
    .applymap(position_color_style, subset=["CE_Position", "PE_Position"])
    .format({
        "CE_OI": format_table_number, "CE_LTP": "{:,.2f}", "CE_Change": "{:+.2f}",
        "CE_Volume": format_table_number, "CE_ChgOI": format_table_number, "CE_IV": "{:.2f}",
        "Strike": "{:,.0f}",
        "PE_OI": format_table_number, "PE_LTP": "{:,.2f}", "PE_Change": "{:+.2f}",
        "PE_Volume": format_table_number, "PE_ChgOI": format_table_number, "PE_IV": "{:.2f}",
        "PCR_Strike_OI": "{:.3f}", "PCR_Volume": "{:.3f}", "PCR_ChgOI": "{:.3f}"
    })
)

st.dataframe(styled, use_container_width=True)

# Summary statistics
st.markdown("---")
st.subheader("📈 Quick Stats")

stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

with stats_col1:
    st.metric("Total CE OI", format_number(table["CE_OI"].sum()))
    
with stats_col2:
    st.metric("Total PE OI", format_number(table["PE_OI"].sum()))
    
with stats_col3:
    st.metric("Max Pain Strike", f"{table.loc[table['CE_OI'].idxmax(), 'Strike']:,.0f}")
    
with stats_col4:
    total_ce_vol = table["CE_Volume"].sum()
    total_pe_vol = table["PE_Volume"].sum()
    volume_pcr = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0
    st.metric("Total Volume PCR", f"{volume_pcr:.3f}")

st.markdown("---")
st.caption(f"Data refreshed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Snapshot saved: {save_path}")
