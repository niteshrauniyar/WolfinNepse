"""
NEPSE Institutional Sentinel
==============================
A high-performance Streamlit dashboard for detecting Smart Money
and institutional footprints in the Nepal Stock Exchange (NEPSE).

Run with:  streamlit run app.py
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# Ensure modules on path
sys.path.insert(0, str(Path(__file__).parent))

from modules.data_ingestion import get_floorsheet, get_market_depth
from modules.order_flow     import run_order_flow_engine, detect_changepoints
from modules.liquidity       import run_liquidity_engine
from modules.network         import run_network_engine
from modules.clustering      import run_clustering_engine
from modules.visualisations  import (
    plot_price_volume,
    plot_ofi,
    plot_broker_acf,
    plot_amihud,
    plot_kyle_lambda,
    plot_market_impact_curve,
    plot_network,
    plot_cluster_scatter,
    plot_liquidity_heatmap,
    plot_centrality_bars,
    COLORS,
)

logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
#  Page config & global CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NEPSE Institutional Sentinel",
    page_icon="🏔",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@300;400;600;700&display=swap');

:root {
    --bg:       #0f172a;
    --surface:  #1e293b;
    --border:   #334155;
    --text:     #f1f5f9;
    --muted:    #94a3b8;
    --orange:   #f97316;
    --blue:     #60a5fa;
    --green:    #34d399;
    --yellow:   #facc15;
    --red:      #ef4444;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'Space Grotesk', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.75rem; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'JetBrains Mono', monospace; }
[data-testid="stMetricDelta"] svg { display: none; }

.sentinel-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid var(--border);
    border-bottom: 2px solid var(--orange);
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 20px;
}

.sentinel-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--orange) 0%, var(--yellow) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}

.sentinel-subtitle {
    color: var(--muted);
    font-size: 0.85rem;
    margin-top: 4px;
}

.flag-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.flag-inst  { background: rgba(249,115,22,0.2); color: var(--orange); border: 1px solid var(--orange); }
.flag-alert { background: rgba(239,68,68,0.2);  color: var(--red);    border: 1px solid var(--red); }
.flag-info  { background: rgba(96,165,250,0.2); color: var(--blue);   border: 1px solid var(--blue); }

.module-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 16px;
}

.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--orange);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

div[data-testid="stTabs"] button {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}

.stDataFrame { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; }

.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 8px var(--green);
    animation: pulse 1.5s infinite;
    margin-right: 6px;
    vertical-align: middle;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

hr { border-color: var(--border); }

[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "floorsheet_df": None,
        "market_depth_df": None,
        "of_results": None,
        "liq_results": None,
        "net_results": None,
        "clust_results": None,
        "last_refresh": None,
        "use_demo": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar controls
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 20px 0;'>
        <div style='font-family: JetBrains Mono; font-size: 1.1rem; color: #f97316; font-weight:700;'>
            🏔 SENTINEL
        </div>
        <div style='color: #94a3b8; font-size: 0.7rem; margin-top:4px;'>NEPSE Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Configuration")

    use_demo = st.toggle(
        "Use Demo Data",
        value=True,
        help="Toggle OFF to fetch live data from NepseAlpha/ShareSansar (requires internet & may be slow)."
    )
    st.session_state["use_demo"] = use_demo

    if use_demo:
        st.info("🎲 Demo mode: synthetic data with realistic institutional patterns embedded.")
    else:
        st.warning("🌐 Live mode: scraping NEPSE portals. May fail if site structure changed.")

    st.markdown("---")

    SYMBOLS = ["NABIL", "NICA", "SCB", "ADBL", "EBL", "PCBL", "KBL",
                "HIDCL", "NHPC", "BPCL", "CHCL", "NTC", "SBL", "LBL"]

    selected_symbol = st.selectbox("Focus Symbol", SYMBOLS, index=0)

    n_clusters = st.slider("ML Clusters (Module 5)", min_value=2, max_value=8, value=4)

    ofi_window = st.slider("OFI Window (trades)", 5, 100, 20)

    st.markdown("---")

    # Refresh button
    if st.button("🔄  Refresh & Analyse", use_container_width=True, type="primary"):
        with st.spinner("Fetching data…"):
            df = get_floorsheet(symbol=selected_symbol if not use_demo else "",
                                use_demo=use_demo)
            md = get_market_depth(symbol=selected_symbol, use_demo=use_demo)
            st.session_state["floorsheet_df"]  = df
            st.session_state["market_depth_df"] = md

        with st.spinner("Running Order Flow Engine (Module 2)…"):
            of_res = run_order_flow_engine(df)
            st.session_state["of_results"] = of_res

        with st.spinner("Running Liquidity Engine (Module 3)…"):
            liq_res = run_liquidity_engine(of_res["enriched_df"], symbol=selected_symbol)
            st.session_state["liq_results"] = liq_res

        with st.spinner("Running Network Engine (Module 4)…"):
            net_res = run_network_engine(df)
            st.session_state["net_results"] = net_res

        with st.spinner("Running Clustering Engine (Module 5)…"):
            clust_res = run_clustering_engine(
                liq_res["enriched_df"], n_clusters=n_clusters
            )
            st.session_state["clust_results"] = clust_res

        st.session_state["last_refresh"] = datetime.now().strftime("%H:%M:%S")
        st.success("✅ Analysis complete!")

    if st.session_state["last_refresh"]:
        st.markdown(
            f'<span class="live-dot"></span>'
            f'<span style="color:#94a3b8;font-size:0.75rem;">Last update: {st.session_state["last_refresh"]}</span>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("""
    <div style='color:#475569;font-size:0.68rem;line-height:1.6;'>
    <b>Data Sources</b><br>
    NepseAlpha · ShareSansar · NEPSE Official<br><br>
    <b>References</b><br>
    Tsaknaki et al. (2023)<br>
    Boehmer et al. (2020)<br>
    Collin-Dufresne &amp; Fos (2012)<br>
    Qu et al. (2022)<br>
    Tumminello et al. (2011)<br>
    Cont et al. (2023)<br>
    Balcau et al. (2024)
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="sentinel-header">
    <div>
        <div class="sentinel-title">🏔 NEPSE Institutional Sentinel</div>
        <div class="sentinel-subtitle">
            Smart Money &amp; Institutional Footprint Detection · Order Flow · Network · ML Clustering
        </div>
    </div>
    <div style='margin-left:auto;'>
        <span class="flag-badge flag-inst">LIVE ANALYSIS</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Auto-load demo data on first visit
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state["floorsheet_df"] is None:
    with st.spinner("Loading initial demo data…"):
        df = get_floorsheet(use_demo=True)
        md = get_market_depth(symbol=selected_symbol, use_demo=True)
        st.session_state["floorsheet_df"]  = df
        st.session_state["market_depth_df"] = md
        of_res    = run_order_flow_engine(df)
        liq_res   = run_liquidity_engine(of_res["enriched_df"], symbol=selected_symbol)
        net_res   = run_network_engine(df)
        clust_res = run_clustering_engine(liq_res["enriched_df"], n_clusters=n_clusters)
        st.session_state.update({
            "of_results": of_res, "liq_results": liq_res,
            "net_results": net_res, "clust_results": clust_res,
            "last_refresh": datetime.now().strftime("%H:%M:%S"),
        })


# ─────────────────────────────────────────────────────────────────────────────
#  Pull results from session state
# ─────────────────────────────────────────────────────────────────────────────

raw_df     = st.session_state["floorsheet_df"]
of_res     = st.session_state["of_results"]
liq_res    = st.session_state["liq_results"]
net_res    = st.session_state["net_results"]
clust_res  = st.session_state["clust_results"]

enriched   = of_res["enriched_df"]
broker_acf = of_res["broker_acf"]
broker_sum = of_res["broker_summary"]
liq_df     = liq_res["enriched_df"]
liq_sum    = liq_res["liquidity_summary"]


# ─────────────────────────────────────────────────────────────────────────────
#  KPI row
# ─────────────────────────────────────────────────────────────────────────────

n_inst   = int(enriched.get("is_institutional", pd.Series(dtype=bool)).sum()) if "is_institutional" in enriched else 0
n_meta   = int(broker_acf["is_metaorder"].sum()) if not broker_acf.empty else 0
total_tv = enriched["amount"].sum() / 1e6 if "amount" in enriched.columns else 0
n_trades = len(enriched)
top_broker = broker_sum.iloc[0]["broker"] if not broker_sum.empty else "—"

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Trades",       f"{n_trades:,}")
col2.metric("Turnover (NPR M)",   f"{total_tv:,.1f}")
col3.metric("Institutional Trades", f"{n_inst:,}", delta=f"{n_inst/max(n_trades,1)*100:.1f}% of flow")
col4.metric("Metaorder Brokers",   f"{n_meta}")
col5.metric("Top Volume Broker",   str(top_broker))

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Main tabs
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🌊 Order Flow",
    "💧 Liquidity",
    "🕸 Network",
    "🤖 ML Clustering",
    "📋 Raw Data",
])


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 – Overview                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

with tab1:
    st.markdown('<div class="section-label">Market Overview · ' + selected_symbol + '</div>',
                unsafe_allow_html=True)

    col_l, col_r = st.columns([2, 1])

    with col_l:
        fig = plot_price_volume(enriched, selected_symbol)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### 🔍 Liquidity Snapshot")
        sym_liq = liq_sum[liq_sum["symbol"] == selected_symbol]
        if not sym_liq.empty:
            row = sym_liq.iloc[0]
            st.metric("Liquidity Score",  f"{row.get('liquidity_score', 0):.4f}")
            st.metric("Avg Amihud ILLIQ", f"{row.get('avg_amihud', 0):.6f}")
            st.metric("VWAP Deviation",   f"{row.get('avg_vwap_dev', 0)*100:.2f}%")
            st.metric("Total Volume",     f"{row.get('total_volume', 0):,.0f}")
        else:
            st.info("Select a symbol with trades to see stats.")

    st.markdown("---")
    st.markdown("#### 🏆 Top Brokers by Turnover")
    top10 = broker_sum.head(10)[["broker","total_volume","net_volume","ofi_score","avg_trade_size"]].copy()
    top10.columns = ["Broker","Total Vol","Net Vol","OFI Score","Avg Trade Size"]

    def style_row(row):
        if abs(row["OFI Score"]) > 0.5:
            return [f"color: {COLORS['institutional']};"] * len(row)
        return [""] * len(row)

    st.dataframe(
        top10.style.apply(style_row, axis=1).format({
            "Total Vol": "{:,.0f}", "Net Vol": "{:,.0f}",
            "OFI Score": "{:.3f}", "Avg Trade Size": "{:,.0f}",
        }),
        use_container_width=True, height=280,
    )

    st.markdown("---")
    st.markdown("#### 📉 Liquidity Heatmap (All Symbols)")
    st.plotly_chart(plot_liquidity_heatmap(liq_sum), use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 – Order Flow Engine (Module 2)                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

with tab2:
    st.markdown('<div class="section-label">Module 2 · Order Flow & Trade Size Engine</div>',
                unsafe_allow_html=True)

    # Change-point detection
    bkps, signal = detect_changepoints(enriched, symbol=selected_symbol)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🌊 Order Flow Imbalance + Change-Points")
        st.caption(f"Ruptures PELT detected **{len(bkps)}** structural break(s) in OFI signal")
        fig_ofi = plot_ofi(enriched, selected_symbol, breakpoints=bkps)
        st.plotly_chart(fig_ofi, use_container_width=True)

    with c2:
        st.markdown("#### 🔁 Broker Autocorrelation (Metaorder Detection)")
        st.caption("Brokers with ACF > 0.15 are likely splitting large orders (Tsaknaki et al. 2023)")
        fig_acf = plot_broker_acf(broker_acf, top_n=15)
        st.plotly_chart(fig_acf, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🏦 ADV Institutional Trade Filter (Top 5% Flagged)")
    if "is_institutional" in enriched.columns:
        inst_trades = enriched[enriched["is_institutional"]].sort_values("quantity", ascending=False)
        display_cols = ["date","symbol","buyer_broker","seller_broker","quantity","price","amount","qty_pct_rank"]
        display_cols = [c for c in display_cols if c in inst_trades.columns]
        st.dataframe(
            inst_trades[display_cols].head(50).style.format({
                "quantity": "{:,.0f}", "price": "{:,.2f}",
                "amount": "{:,.0f}", "qty_pct_rank": "{:.3f}",
            }),
            use_container_width=True, height=300,
        )
    else:
        st.info("Run analysis first.")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 📋 Metaorder Brokers")
        if not broker_acf.empty:
            meta = broker_acf[broker_acf["is_metaorder"]].copy()
            st.dataframe(
                meta[["broker","n_trades","mean_autocorr","max_autocorr"]].style.format({
                    "mean_autocorr": "{:.4f}", "max_autocorr": "{:.4f}", "n_trades": "{:,}",
                }),
                use_container_width=True,
            )
        else:
            st.info("No metaorder data.")

    with col_b:
        st.markdown("#### 📊 Broker OFI Scores")
        fig_ofi_bar = plot_centrality_bars(
            broker_sum.rename(columns={"broker":"label", "ofi_score":"scoreness"}),
            metric="scoreness", title="Broker OFI Score (Net Directional Bias)",
        )
        st.plotly_chart(fig_ofi_bar, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 – Liquidity Engine (Module 3)                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

with tab3:
    st.markdown('<div class="section-label">Module 3 · Liquidity & Market Impact Engine</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📈 Amihud Illiquidity Ratio")
        st.caption("High ILLIQ = each unit of volume has larger price impact")
        st.plotly_chart(plot_amihud(liq_df, selected_symbol), use_container_width=True)

    with c2:
        st.markdown("#### ⚡ Kyle's Lambda (Price Impact Coefficient)")
        st.caption("λ > 0 = buying pressure moves prices up; high λ = informed market")
        kyle_df = liq_res.get("kyle_df", pd.DataFrame())
        st.plotly_chart(plot_kyle_lambda(kyle_df, selected_symbol), use_container_width=True)

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### 🎯 Market Impact Curve")
        st.caption("Concave shape = institutional TWAP/VWAP execution hiding footprint")
        impact_crv = liq_res.get("impact_curve", pd.DataFrame())
        st.plotly_chart(plot_market_impact_curve(impact_crv, selected_symbol), use_container_width=True)

    with c4:
        st.markdown("#### ⚠️ Adverse Selection Risk")
        st.caption("High score = large trades + widening spread = informed flow (CDF 2012)")
        if "adverse_selection_score" in liq_df.columns:
            sub = liq_df[liq_df["symbol"] == selected_symbol].sort_values("date")
            import plotly.graph_objects as _go
            fig_as = _go.Figure(_go.Scatter(
                x=sub["date"], y=sub["adverse_selection_score"],
                mode="lines", fill="tozeroy",
                line=dict(color=COLORS["danger"], width=1.5),
                fillcolor="rgba(239,68,68,0.08)",
            ))
            fig_as.update_layout(
                title="Adverse Selection Score",
                paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["surface"],
                font=dict(color=COLORS["text"]),
                xaxis=dict(gridcolor=COLORS["grid"]),
                yaxis=dict(gridcolor=COLORS["grid"], range=[0,1]),
                margin=dict(l=40, r=20, t=50, b=40),
            )
            fig_as.add_hline(y=0.7, line_dash="dash",
                             line_color=COLORS["accent"], annotation_text="Alert threshold")
            st.plotly_chart(fig_as, use_container_width=True)
        else:
            st.info("Adverse selection data not available.")

    st.markdown("---")
    st.markdown("#### 📊 Full Liquidity Summary Table")
    st.dataframe(
        liq_sum.style.format({
            "avg_amihud": "{:.6f}", "avg_price": "{:,.2f}",
            "total_volume": "{:,.0f}", "avg_vwap_dev": "{:.4f}",
            "liquidity_score": "{:.4f}",
        }).background_gradient(subset=["liquidity_score"], cmap="RdYlGn"),
        use_container_width=True,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 4 – Network Engine (Module 4)                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

with tab4:
    st.markdown('<div class="section-label">Module 4 · Network & Centrality Engine</div>',
                unsafe_allow_html=True)

    net_stats = net_res.get("net_stats", {})
    if net_stats:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Broker Nodes",  net_stats.get("nodes", "—"))
        m2.metric("Trading Pairs", net_stats.get("edges", "—"))
        m3.metric("Network Density", f"{net_stats.get('density', 0):.4f}")
        m4.metric("Avg Clustering", f"{net_stats.get('avg_clustering', 0):.4f}")

    st.markdown("---")

    view_choice = st.radio(
        "Graph View", ["Broker–Broker", "Broker–Stock (Bipartite)"], horizontal=True
    )

    if view_choice == "Broker–Broker":
        G   = net_res["G_broker"]
        pos = net_res["pos_broker"]
        title = "Broker Co-Trading Network"
    else:
        G   = net_res["G_bipartite"]
        pos = net_res["pos_bipartite"]
        title = "Bipartite Broker–Stock Network"

    centrality_df = net_res.get("centrality_df", pd.DataFrame())
    fig_net = plot_network(G, pos, centrality_df, title=title)
    st.plotly_chart(fig_net, use_container_width=True)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 💪 Node Strength Centrality (Top 20)")
        st.caption("Brokers with highest total trade volume in the network")
        strength_df = net_res.get("strength_df", pd.DataFrame())
        broker_strength = strength_df[strength_df["node_type"] == "broker"]
        st.plotly_chart(
            plot_centrality_bars(broker_strength, metric="strength_centrality",
                                 title="Strength Centrality"),
            use_container_width=True,
        )

    with c2:
        st.markdown("#### 🧅 S-Coreness Centrality")
        st.caption("Core–periphery structure; high scoreness = systemically important broker")
        scoreness_df = net_res.get("scoreness_df", pd.DataFrame())
        broker_score = scoreness_df[scoreness_df["node_type"].isin(["broker", "unknown"])]
        st.plotly_chart(
            plot_centrality_bars(broker_score, metric="scoreness",
                                 title="S-Coreness (Tumminello et al. 2011)"),
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("#### 📐 PageRank & Betweenness Centrality")
    if not centrality_df.empty:
        st.dataframe(
            centrality_df.head(30).style.format({
                "pagerank": "{:.6f}", "betweenness": "{:.6f}",
            }).background_gradient(subset=["pagerank"], cmap="YlOrRd"),
            use_container_width=True,
        )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 5 – ML Clustering (Module 5)                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

with tab5:
    st.markdown('<div class="section-label">Module 5 · Clustering & ML Segmentation</div>',
                unsafe_allow_html=True)

    clustered_df    = clust_res.get("clustered_df", pd.DataFrame())
    cluster_profile = clust_res.get("cluster_profile", pd.DataFrame())
    inst_brokers    = clust_res.get("inst_brokers", pd.DataFrame())

    # Silhouette score
    sil = clustered_df.get("silhouette", pd.Series([0])).iloc[0] if not clustered_df.empty else 0
    method = clustered_df.get("cluster_method", pd.Series(["—"])).iloc[0] if not clustered_df.empty else "—"
    var    = clustered_df.get("pca_var_explained", pd.Series([0])).iloc[0] if not clustered_df.empty else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Clustering Method",     str(method).upper())
    c2.metric("Silhouette Score",       f"{sil:.4f}")
    c3.metric("PCA Variance Explained", f"{float(var)*100:.1f}%")

    st.markdown("---")
    st.markdown("#### 🌐 Broker Segmentation (PCA Space)")
    st.caption("Each point = a broker. Colour = cluster. Institutional cluster has largest trade sizes & highest VWAP alpha.")
    st.plotly_chart(plot_cluster_scatter(clustered_df), use_container_width=True)

    st.markdown("---")
    c_l, c_r = st.columns([1, 1])

    with c_l:
        st.markdown("#### 🏛️ Institutional Broker Roster")
        st.caption("Brokers identified by ML as Informed/Large Actors")
        if not inst_brokers.empty:
            disp = inst_brokers.reset_index()[
                ["broker", "avg_trade_size", "vwap_alpha", "net_direction", "trades_per_hour"]
            ].head(20)
            st.dataframe(
                disp.style.format({
                    "avg_trade_size": "{:,.0f}",
                    "vwap_alpha": "{:.4f}",
                    "net_direction": "{:.3f}",
                    "trades_per_hour": "{:.2f}",
                }).background_gradient(subset=["avg_trade_size"], cmap="Oranges"),
                use_container_width=True,
            )
        else:
            st.info("No institutional brokers identified.")

    with c_r:
        st.markdown("#### 📊 Cluster Profile Comparison")
        if not cluster_profile.empty:
            st.dataframe(
                cluster_profile.style.format(
                    {c: "{:.3f}" for c in cluster_profile.select_dtypes("float").columns}
                ),
                use_container_width=True,
            )

    st.markdown("---")
    st.markdown("#### 🔬 Full Broker Feature Matrix")
    feat_df = clust_res.get("feature_df", pd.DataFrame())
    if not feat_df.empty:
        full_display = clustered_df.reset_index()[
            ["broker", "cluster_label", "avg_trade_size", "std_trade_size",
             "large_trade_ratio", "timing_gini", "trades_per_hour",
             "net_direction", "ofi_mean", "vwap_alpha", "n_trades", "total_volume"]
        ].sort_values("avg_trade_size", ascending=False)

        st.dataframe(
            full_display.style.format({
                "avg_trade_size": "{:,.0f}", "std_trade_size": "{:,.0f}",
                "large_trade_ratio": "{:.3f}", "timing_gini": "{:.3f}",
                "trades_per_hour": "{:.2f}", "net_direction": "{:.3f}",
                "ofi_mean": "{:.4f}", "vwap_alpha": "{:.4f}",
                "n_trades": "{:,}", "total_volume": "{:,.0f}",
            }).background_gradient(subset=["avg_trade_size"], cmap="YlOrRd"),
            use_container_width=True, height=400,
        )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 6 – Raw Data                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

with tab6:
    st.markdown('<div class="section-label">Raw Floorsheet Data</div>', unsafe_allow_html=True)

    col_filter, col_info = st.columns([3, 1])
    with col_filter:
        sym_filter = st.multiselect(
            "Filter by Symbol",
            options=sorted(raw_df["symbol"].unique().tolist()),
            default=[],
        )
    with col_info:
        st.metric("Total Records", f"{len(raw_df):,}")

    display_df = raw_df[raw_df["symbol"].isin(sym_filter)] if sym_filter else raw_df

    show_cols = ["date","symbol","buyer_broker","seller_broker","quantity","price","amount"]
    show_cols = [c for c in show_cols if c in display_df.columns]
    st.dataframe(
        display_df[show_cols].head(1000).style.format({
            "quantity": "{:,.0f}", "price": "{:,.2f}", "amount": "{:,.0f}",
        }),
        use_container_width=True, height=500,
    )

    st.download_button(
        "⬇️  Download CSV",
        data=display_df.to_csv(index=False),
        file_name=f"nepse_floorsheet_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<hr>
<div style='text-align:center; color:#475569; font-size:0.7rem; padding:12px 0;'>
    NEPSE Institutional Sentinel · Built on Streamlit + Plotly + NetworkX + scikit-learn<br>
    Academic references: Tsaknaki et al. (2023) · Boehmer et al. (2020) · Collin-Dufresne & Fos (2012)
    · Qu et al. (2022) · Tumminello et al. (2011) · Cont et al. (2023) · Balcau et al. (2024)
</div>
""", unsafe_allow_html=True)
