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

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="NSE Option Chain", layout="wide")
st.title("📊 NSE Option Chain Dashboard")

with st.sidebar:
    from nsepython import fnolist
    fno_list = [x for x in fnolist()]
    symbol = st.selectbox("Symbol", fno_list, index=0)
    itm_count = st.radio("ITM Strikes", [3, 5], index=1)
    refresh_sec = st.slider("Auto-Refresh (sec)", 10, 60, 30)
    st.caption("Install `streamlit-autorefresh` for auto refresh.")

if AUTORFR and is_market_open():
    st_autorefresh(interval=refresh_sec * 1000, key="oc_refresh")

if "prev_buckets" not in st.session_state:
    st.session_state.prev_buckets = None

# ----------------------------
# Fetch data
# ----------------------------
@st.cache_data(ttl=15)
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
    flat.append({
        "CE_OI": float(g(ce, "openInterest")),
        "CE_LTP": float(g(ce, "lastPrice")),
        "CE_Volume": float(g(ce, "totalTradedVolume")),
        "CE_ChgOI": float(g(ce, "changeinOpenInterest")),
        "CE_IV": float(g(ce, "impliedVolatility")),
        "Strike": float(r["strikePrice"]),
        "PE_OI": float(g(pe, "openInterest")),
        "PE_LTP": float(g(pe, "lastPrice")),
        "PE_Volume": float(g(pe, "totalTradedVolume")),
        "PE_ChgOI": float(g(pe, "changeinOpenInterest")),
        "PE_IV": float(g(pe, "impliedVolatility")),
    })

table = pd.DataFrame(flat)

# ----------------------------
# Grouped Chart: OI, ChgOI, Volume
# ----------------------------
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

# ----------------------------
# Show summaries
# ----------------------------
st.subheader(f"Bucket Summaries ({itm_count} ITM each side)")
left, right = st.columns(2)

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

with right:
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

st.session_state.prev_buckets = bucket_summary

# ----------------------------
# Table view
# ----------------------------
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

table = table[[
    "CE_OI", "CE_LTP", "CE_Volume", "CE_ChgOI", "CE_IV",
    "Strike",
    "PE_OI", "PE_LTP", "PE_Volume", "PE_ChgOI", "PE_IV"
]]

st.subheader(f"Table ({itm_count} ITM each side)")
styled = (
    table.style
    .apply(row_highlight, axis=1)
    .applymap(lambda v: cell_green(v, ce_oi_max), subset=["CE_OI"])
    .applymap(lambda v: cell_green(v, ce_vol_max), subset=["CE_Volume"])
    .applymap(lambda v: cell_green(v, pe_oi_max), subset=["PE_OI"])
    .applymap(lambda v: cell_green(v, pe_vol_max), subset=["PE_Volume"])
    .format({
        "CE_OI": "{:,.0f}", "CE_LTP": "{:,.2f}", "CE_Volume": "{:,.0f}",
        "CE_ChgOI": "{:,.0f}", "CE_IV": "{:.2f}",
        "Strike": "{:,.0f}",
        "PE_OI": "{:,.0f}", "PE_LTP": "{:,.2f}", "PE_Volume": "{:,.0f}",
        "PE_ChgOI": "{:,.0f}", "PE_IV": "{:.2f}",
    })
)

st.dataframe(styled, use_container_width=True)
