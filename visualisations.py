"""
Visualisation helpers for the NEPSE Institutional Sentinel dashboard.
All functions return Plotly figures ready for st.plotly_chart().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx

# ── colour palette ──────────────────────────────────────────────────────────
COLORS = {
    "institutional": "#f97316",   # vivid orange
    "retail":        "#60a5fa",   # sky blue
    "maker":         "#34d399",   # emerald
    "noise":         "#9ca3af",   # slate
    "accent":        "#facc15",   # yellow
    "danger":        "#ef4444",
    "bg":            "#0f172a",   # navy
    "surface":       "#1e293b",
    "text":          "#f1f5f9",
    "grid":          "#334155",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor =COLORS["surface"],
    font         =dict(color=COLORS["text"], family="JetBrains Mono, monospace"),
    xaxis        =dict(gridcolor=COLORS["grid"], zeroline=False),
    yaxis        =dict(gridcolor=COLORS["grid"], zeroline=False),
    legend       =dict(bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["grid"]),
    margin       =dict(l=40, r=20, t=50, b=40),
)

_CLUSTER_COLORS = {
    "🏛️ Institutional / Informed": COLORS["institutional"],
    "⚡ Active Retail":              COLORS["retail"],
    "🔄 Market Maker / Passive":    COLORS["maker"],
    "🌫️ Noise / Low-Activity":      COLORS["noise"],
}


# ─────────────────────────────────────────────────────────────────────────────
#  1. Price & Volume chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_price_volume(df: pd.DataFrame, symbol: str) -> go.Figure:
    sub = df[df["symbol"] == symbol].sort_values("date")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.04)

    fig.add_trace(go.Scatter(
        x=sub["date"], y=sub["price"],
        mode="lines", name="Price",
        line=dict(color=COLORS["accent"], width=1.2),
        fill="tozeroy", fillcolor="rgba(250,204,21,0.05)",
    ), row=1, col=1)

    # Colour bars by institutional flag
    if "is_institutional" in sub.columns:
        colors = [COLORS["institutional"] if v else COLORS["retail"]
                  for v in sub["is_institutional"]]
    else:
        colors = COLORS["retail"]

    fig.add_trace(go.Bar(
        x=sub["date"], y=sub["quantity"],
        name="Volume", marker_color=colors, opacity=0.7,
    ), row=2, col=1)

    fig.update_layout(title=f"Price & Volume — {symbol}", **PLOTLY_LAYOUT)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  2. Order Flow Imbalance
# ─────────────────────────────────────────────────────────────────────────────

def plot_ofi(df: pd.DataFrame, symbol: str, breakpoints: list[int] = None) -> go.Figure:
    sub = df[df["symbol"] == symbol].sort_values("date")
    if "ofi" not in sub.columns:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["date"], y=sub["ofi"],
        mode="lines", name="OFI",
        line=dict(color=COLORS["retail"], width=1),
        fill="tozeroy",
        fillcolor="rgba(96,165,250,0.1)",
    ))

    # Add change-point lines
    if breakpoints:
        sub_reset = sub.reset_index(drop=True)
        for bp in breakpoints:
            if bp < len(sub_reset):
                x_val = sub_reset.loc[bp, "date"]
                fig.add_vline(x=x_val, line_dash="dash",
                              line_color=COLORS["danger"], opacity=0.8)

    fig.add_hline(y=0, line_color=COLORS["grid"], line_dash="dot")
    fig.update_layout(title=f"Order Flow Imbalance with Change-Points — {symbol}", **PLOTLY_LAYOUT)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  3. Broker Autocorrelation (metaorder)
# ─────────────────────────────────────────────────────────────────────────────

def plot_broker_acf(acf_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    if acf_df.empty:
        return go.Figure()

    top = acf_df.head(top_n).copy()
    colors = [COLORS["institutional"] if v else COLORS["retail"]
              for v in top["is_metaorder"]]

    fig = go.Figure(go.Bar(
        y=top["broker"], x=top["mean_autocorr"],
        orientation="h", marker_color=colors,
        text=[f"{v:.3f}" for v in top["mean_autocorr"]],
        textposition="outside",
    ))
    fig.add_vline(x=0.15, line_dash="dash",
                  line_color=COLORS["accent"], annotation_text="Metaorder threshold (0.15)")
    fig.update_layout(
        title="Broker Order-Flow Autocorrelation (Metaorder Detection)",
        xaxis_title="Mean ACF", yaxis_title="Broker",
        **PLOTLY_LAYOUT,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  4. Amihud Illiquidity
# ─────────────────────────────────────────────────────────────────────────────

def plot_amihud(df: pd.DataFrame, symbol: str) -> go.Figure:
    sub = df[df["symbol"] == symbol].sort_values("date")
    if "amihud" not in sub.columns:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["date"], y=sub["amihud"],
        mode="lines", fill="tozeroy",
        line=dict(color=COLORS["danger"], width=1.5),
        fillcolor="rgba(239,68,68,0.08)",
        name="Amihud ILLIQ",
    ))
    fig.update_layout(
        title=f"Amihud Illiquidity Ratio — {symbol}",
        yaxis_title="ILLIQ (|R| / Volume)",
        **PLOTLY_LAYOUT,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  5. Kyle's Lambda
# ─────────────────────────────────────────────────────────────────────────────

def plot_kyle_lambda(kyle_df: pd.DataFrame, symbol: str) -> go.Figure:
    if kyle_df.empty or "kyle_lambda" not in kyle_df.columns:
        return go.Figure()

    fig = go.Figure(go.Scatter(
        x=kyle_df["date"], y=kyle_df["kyle_lambda"].fillna(method="ffill"),
        mode="lines", line=dict(color=COLORS["maker"], width=1.5),
        name="Kyle's λ",
    ))
    fig.add_hline(y=0, line_color=COLORS["grid"], line_dash="dot")
    fig.update_layout(
        title=f"Kyle's Lambda (Price Impact per Volume) — {symbol}",
        yaxis_title="λ (ΔP / signed_vol)",
        **PLOTLY_LAYOUT,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  6. Market Impact Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_market_impact_curve(impact_df: pd.DataFrame, symbol: str) -> go.Figure:
    if impact_df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=impact_df["avg_qty"], y=impact_df["avg_impact"],
        mode="markers+lines", name="Empirical Impact",
        marker=dict(color=COLORS["accent"], size=8),
        line=dict(color=COLORS["accent"], width=1.5),
    ))
    if "fitted_impact" in impact_df.columns:
        fig.add_trace(go.Scatter(
            x=impact_df["avg_qty"], y=impact_df["fitted_impact"],
            mode="lines", name="Log fit (concave)",
            line=dict(color=COLORS["danger"], dash="dash", width=2),
        ))
    fig.update_layout(
        title=f"Market Impact Curve — {symbol}  (concave = institutional TWAP)",
        xaxis_title="Avg Trade Size (qty)",
        yaxis_title="Avg Absolute Price Impact",
        **PLOTLY_LAYOUT,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  7. Network graph (Plotly)
# ─────────────────────────────────────────────────────────────────────────────

def plot_network(G: nx.Graph, pos: dict, centrality_df: pd.DataFrame = None,
                 title: str = "Broker Network") -> go.Figure:
    if G.number_of_nodes() == 0:
        return go.Figure()

    # Build centrality lookup
    cent_lookup: dict[str, float] = {}
    if centrality_df is not None and not centrality_df.empty:
        for _, row in centrality_df.iterrows():
            cent_lookup[str(row.get("label", row.get("node_id", "")))] = row.get("pagerank", 0)

    # Edge traces
    edge_x, edge_y = [], []
    for u, v, d in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.5, color=COLORS["grid"]),
        hoverinfo="none",
        name="Connections",
    )

    # Node traces
    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
    for node in G.nodes():
        if node not in pos:
            continue
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        label  = G.nodes[node].get("label", node)
        ntype  = G.nodes[node].get("node_type", "broker")
        pr     = cent_lookup.get(str(label), 0.0)
        degree = G.degree(node)

        node_text.append(f"<b>{label}</b><br>Type: {ntype}<br>Degree: {degree}<br>PageRank: {pr:.5f}")
        node_size.append(max(6, min(40, degree * 3)))
        node_color.append(COLORS["institutional"] if ntype == "broker" else COLORS["maker"])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            size=node_size, color=node_color,
            line=dict(width=1, color=COLORS["bg"]),
        ),
        name="Nodes",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  8. Cluster scatter (PCA)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_scatter(clustered_df: pd.DataFrame) -> go.Figure:
    if clustered_df.empty or "pca_x" not in clustered_df.columns:
        return go.Figure()

    df = clustered_df.reset_index()
    label_col = "cluster_label" if "cluster_label" in df.columns else "cluster"

    fig = go.Figure()
    for label in df[label_col].unique():
        sub = df[df[label_col] == label]
        c = _CLUSTER_COLORS.get(str(label), COLORS["noise"])
        fig.add_trace(go.Scatter(
            x=sub["pca_x"], y=sub["pca_y"],
            mode="markers+text",
            name=str(label),
            text=sub.get("broker", sub.index),
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(color=c, size=10, line=dict(width=1, color=COLORS["bg"])),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Avg Trade Size: %{customdata[0]:,.0f}<br>"
                "OFI Mean: %{customdata[1]:.3f}<br>"
                "Trades/hr: %{customdata[2]:.2f}"
            ),
            customdata=sub[["avg_trade_size", "ofi_mean", "trades_per_hour"]].values
            if all(c in sub.columns for c in ["avg_trade_size", "ofi_mean", "trades_per_hour"])
            else None,
        ))

    var = clustered_df.get("pca_var_explained", pd.Series([0])).iloc[0] if not clustered_df.empty else 0
    fig.update_layout(
        title=f"Broker Segmentation (PCA — {var * 100:.1f}% variance explained)",
        xaxis_title="PC1", yaxis_title="PC2",
        **PLOTLY_LAYOUT,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  9. Liquidity heatmap (symbol × metric)
# ─────────────────────────────────────────────────────────────────────────────

def plot_liquidity_heatmap(liq_df: pd.DataFrame) -> go.Figure:
    if liq_df.empty:
        return go.Figure()

    metrics = ["avg_amihud", "avg_price", "avg_vwap_dev", "liquidity_score"]
    available = [m for m in metrics if m in liq_df.columns]
    sub = liq_df.set_index("symbol")[available]
    # Normalise
    sub_norm = (sub - sub.min()) / (sub.max() - sub.min() + 1e-9)

    fig = go.Figure(go.Heatmap(
        z=sub_norm.values,
        x=available,
        y=sub_norm.index.tolist(),
        colorscale="RdYlGn",
        hoverongaps=False,
        text=sub.round(4).values,
        texttemplate="%{text}",
    ))
    fig.update_layout(
        title="Liquidity Metrics Heatmap (Normalised)",
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
        xaxis=dict(tickangle=-20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  10. Scoreness / Centrality bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_centrality_bars(df: pd.DataFrame, metric: str = "scoreness",
                         top_n: int = 20, title: str = "Broker Centrality") -> go.Figure:
    if df.empty:
        return go.Figure()

    top = df.head(top_n).copy()
    label_col = "label" if "label" in top.columns else top.columns[0]
    fig = go.Figure(go.Bar(
        x=top[metric], y=top[label_col].astype(str),
        orientation="h",
        marker=dict(
            color=top[metric],
            colorscale="Oranges",
            showscale=True,
        ),
    ))
    fig.update_layout(
        title=title,
        xaxis_title=metric,
        yaxis_title="Broker",
        yaxis=dict(autorange="reversed"),
        **PLOTLY_LAYOUT,
    )
    return fig
