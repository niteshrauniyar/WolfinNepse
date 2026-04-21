"""
Module 4: Network & Centrality Engine
=======================================
Implements:
  - Bipartite broker-stock graph (Qu et al. 2022)
  - Node Strength Centrality
  - S-Coreness Centrality (Tumminello et al. 2011)
  - Community detection for broker clusters
"""

from __future__ import annotations

import logging
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Build bipartite broker-stock graph
# ─────────────────────────────────────────────────────────────────────────────

def build_broker_stock_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Bipartite graph:
      - Left nodes  : Broker IDs (buyer or seller)
      - Right nodes : Stock symbols
      - Edges       : Weighted by total traded amount (buy + sell)

    Attributes added to each node:
      - node_type : 'broker' or 'stock'
      - volume    : total quantity
      - amount    : total turnover
    """
    G = nx.Graph()

    # Add broker ↔ stock edges (buyer side)
    buy_agg = (
        df.groupby(["buyer_broker", "symbol"])
          .agg(volume=("quantity", "sum"), amount=("amount", "sum"))
          .reset_index()
    )
    for _, row in buy_agg.iterrows():
        broker = f"B_{row['buyer_broker']}"
        stock  = f"S_{row['symbol']}"

        if not G.has_node(broker):
            G.add_node(broker, node_type="broker", label=row["buyer_broker"])
        if not G.has_node(stock):
            G.add_node(stock, node_type="stock", label=row["symbol"])

        if G.has_edge(broker, stock):
            G[broker][stock]["weight"] += row["amount"]
            G[broker][stock]["volume"] += row["volume"]
        else:
            G.add_edge(broker, stock, weight=row["amount"], volume=row["volume"])

    # Add seller side
    sell_agg = (
        df.groupby(["seller_broker", "symbol"])
          .agg(volume=("quantity", "sum"), amount=("amount", "sum"))
          .reset_index()
    )
    for _, row in sell_agg.iterrows():
        broker = f"B_{row['seller_broker']}"
        stock  = f"S_{row['symbol']}"

        if not G.has_node(broker):
            G.add_node(broker, node_type="broker", label=row["seller_broker"])
        if not G.has_node(stock):
            G.add_node(stock, node_type="stock", label=row["symbol"])

        if G.has_edge(broker, stock):
            G[broker][stock]["weight"] += row["amount"]
            G[broker][stock]["volume"] += row["volume"]
        else:
            G.add_edge(broker, stock, weight=row["amount"], volume=row["volume"])

    logger.info(
        "Bipartite graph: %d nodes, %d edges",
        G.number_of_nodes(), G.number_of_edges()
    )
    return G


# ─────────────────────────────────────────────────────────────────────────────
#  Build broker-broker co-trading graph
# ─────────────────────────────────────────────────────────────────────────────

def build_broker_broker_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Direct broker-to-broker graph:
      - Edge between buyer_broker and seller_broker
      - Weight = total traded amount between the pair

    Reveals dominant trading pairs and potential coordination.
    """
    G = nx.Graph()
    pair_agg = (
        df.groupby(["buyer_broker", "seller_broker"])
          .agg(
              trades=("quantity", "count"),
              volume=("quantity", "sum"),
              amount=("amount", "sum"),
          )
          .reset_index()
    )

    for _, row in pair_agg.iterrows():
        b = str(row["buyer_broker"])
        s = str(row["seller_broker"])
        if b == s:
            continue
        if G.has_edge(b, s):
            G[b][s]["weight"] += row["amount"]
            G[b][s]["trades"] += row["trades"]
        else:
            G.add_edge(b, s,
                       weight=float(row["amount"]),
                       trades=int(row["trades"]),
                       volume=float(row["volume"]))

    logger.info("Broker-broker graph: %d brokers, %d pairs", G.number_of_nodes(), G.number_of_edges())
    return G


# ─────────────────────────────────────────────────────────────────────────────
#  Node Strength Centrality
# ─────────────────────────────────────────────────────────────────────────────

def compute_strength_centrality(G: nx.Graph) -> pd.DataFrame:
    """
    Strength centrality = sum of edge weights incident on each node.
    Normalised by max weight in graph.

    High strength → broker dominates market by volume/turnover.
    """
    strengths = {}
    max_w = max((d.get("weight", 1) for _, _, d in G.edges(data=True)), default=1)

    for node in G.nodes():
        s = sum(G[node][nbr].get("weight", 0) for nbr in G.neighbors(node))
        strengths[node] = s / max_w if max_w > 0 else 0.0

    df = pd.DataFrame(
        [(n, strengths[n], G.nodes[n].get("node_type", "unknown"),
          G.nodes[n].get("label", n), G.degree(n))
         for n in G.nodes()],
        columns=["node_id", "strength_centrality", "node_type", "label", "degree"],
    )
    df = df.sort_values("strength_centrality", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  S-Coreness Centrality (weighted k-core)
# ─────────────────────────────────────────────────────────────────────────────

def compute_scoreness(G: nx.Graph) -> pd.DataFrame:
    """
    Tumminello et al. (2011): S-coreness extends k-coreness to weighted graphs.
    A node belongs to s-core k if it has at least k neighbours each with
    edge weight ≥ threshold.

    Approximated here via iterative pruning: nodes are assigned a coreness
    equal to the weighted-degree threshold at which they are removed.

    Returns DataFrame with columns: broker, scoreness.
    """
    # Work on a copy with only broker nodes
    broker_nodes = [n for n in G.nodes() if G.nodes[n].get("node_type") == "broker"]
    sub = G.subgraph(broker_nodes).copy()

    if sub.number_of_nodes() == 0:
        # Fall back to full graph
        sub = G.copy()

    coreness: dict[str, int] = {}
    max_iter = 100

    for k in range(1, max_iter):
        remove = []
        for node in list(sub.nodes()):
            weighted_deg = sum(sub[node][nbr].get("weight", 0) for nbr in sub.neighbors(node))
            if weighted_deg < k:
                remove.append(node)

        for node in remove:
            if node not in coreness:
                coreness[node] = max(k - 1, 0)

        sub.remove_nodes_from(remove)
        if sub.number_of_nodes() == 0:
            break

    # Remaining nodes get the highest coreness
    for node in sub.nodes():
        if node not in coreness:
            coreness[node] = max_iter

    records = [
        {"node_id": n, "scoreness": coreness.get(n, 0),
         "label": G.nodes[n].get("label", n),
         "node_type": G.nodes[n].get("node_type", "unknown")}
        for n in G.nodes()
    ]
    df = pd.DataFrame(records).sort_values("scoreness", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Betweenness & PageRank centrality
# ─────────────────────────────────────────────────────────────────────────────

def compute_additional_centralities(G: nx.Graph) -> pd.DataFrame:
    """
    Compute betweenness centrality and PageRank (weighted).
    Betweenness high → broker is a "bridge" in the market network.
    PageRank high    → broker trades with other influential brokers.
    """
    try:
        pagerank = nx.pagerank(G, weight="weight", max_iter=500)
    except Exception:
        pagerank = {n: 0.0 for n in G.nodes()}

    try:
        betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    except Exception:
        betweenness = {n: 0.0 for n in G.nodes()}

    records = [
        {
            "node_id":     n,
            "label":       G.nodes[n].get("label", n),
            "node_type":   G.nodes[n].get("node_type", "unknown"),
            "pagerank":    round(pagerank.get(n, 0), 6),
            "betweenness": round(betweenness.get(n, 0), 6),
        }
        for n in G.nodes()
    ]
    return pd.DataFrame(records).sort_values("pagerank", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Network summary for dashboard
# ─────────────────────────────────────────────────────────────────────────────

def get_graph_layout(G: nx.Graph, layout: str = "spring") -> dict:
    """
    Compute 2D node positions for Plotly visualisation.
    Returns {node_id: (x, y)} dict.
    """
    if G.number_of_nodes() == 0:
        return {}
    try:
        if layout == "spring":
            pos = nx.spring_layout(G, weight="weight", seed=42, k=0.5)
        elif layout == "kamada":
            pos = nx.kamada_kawai_layout(G, weight="weight")
        else:
            pos = nx.circular_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=42)
    return pos


def network_stats_summary(G: nx.Graph) -> dict:
    """High-level network statistics."""
    if G.number_of_nodes() == 0:
        return {}
    stats = {
        "nodes":           G.number_of_nodes(),
        "edges":           G.number_of_edges(),
        "density":         round(nx.density(G), 4),
        "avg_clustering":  round(nx.average_clustering(G), 4) if not nx.is_directed(G) else 0,
    }
    try:
        if nx.is_connected(G):
            stats["avg_shortest_path"] = round(nx.average_shortest_path_length(G), 4)
        else:
            # Largest connected component
            lcc = max(nx.connected_components(G), key=len)
            sub = G.subgraph(lcc)
            stats["avg_shortest_path_lcc"] = round(nx.average_shortest_path_length(sub), 4)
            stats["n_components"] = nx.number_connected_components(G)
    except Exception:
        pass
    return stats


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience: run full Module 4 pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_network_engine(df: pd.DataFrame) -> dict:
    """
    Build graphs and compute all centrality metrics.
    """
    G_bipartite    = build_broker_stock_graph(df)
    G_broker       = build_broker_broker_graph(df)

    strength_df    = compute_strength_centrality(G_bipartite)
    scoreness_df   = compute_scoreness(G_broker)
    centrality_df  = compute_additional_centralities(G_broker)
    pos_bipartite  = get_graph_layout(G_bipartite, "spring")
    pos_broker     = get_graph_layout(G_broker, "spring")
    net_stats      = network_stats_summary(G_broker)

    return {
        "G_bipartite":   G_bipartite,
        "G_broker":      G_broker,
        "strength_df":   strength_df,
        "scoreness_df":  scoreness_df,
        "centrality_df": centrality_df,
        "pos_bipartite": pos_bipartite,
        "pos_broker":    pos_broker,
        "net_stats":     net_stats,
    }
