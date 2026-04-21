"""
Module 5: Clustering & ML Segmentation Engine
===============================================
Implements:
  - Broker feature extraction (Cont et al. 2023 / Balcau et al. 2024)
  - Spectral clustering to identify Informed vs Retail actors
  - Cluster profiling and labelling
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_broker_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a feature matrix for each broker based on:

    Trading volume / size features:
      - avg_trade_size          : mean quantity per trade
      - std_trade_size          : std of quantity per trade
      - max_trade_size          : 95th percentile quantity
      - large_trade_ratio       : fraction of trades in top-10% by size

    Timing / rhythm features:
      - trade_concentration     : Gini coefficient of intraday trade timing
      - avg_trades_per_hour     : activity rate

    Directional / inventory features:
      - net_direction           : |buy_vol - sell_vol| / total_vol (accumulation)
      - ofi_mean                : mean OFI score (from Module 2 if available)

    Profitability proxy:
      - vwap_alpha              : systematic deviation from VWAP (buy low, sell high)
    """
    records = []

    for broker in set(df["buyer_broker"].tolist() + df["seller_broker"].tolist()):
        buys   = df[df["buyer_broker"]  == broker]
        sells  = df[df["seller_broker"] == broker]
        all_trades = pd.concat([buys, sells]).drop_duplicates(subset=["transaction_id"] if "transaction_id" in df.columns else None)

        if len(all_trades) < 5:
            continue

        quantities  = all_trades["quantity"].values
        buy_vol     = buys["quantity"].sum()
        sell_vol    = sells["quantity"].sum()
        total_vol   = buy_vol + sell_vol

        # Trade size features
        avg_sz   = float(np.mean(quantities))
        std_sz   = float(np.std(quantities))
        p95_sz   = float(np.percentile(quantities, 95))
        large_rt = float(np.mean(quantities >= np.percentile(quantities, 90)))

        # Timing concentration (Gini)
        try:
            if "date" in all_trades.columns:
                hours = pd.to_datetime(all_trades["date"]).dt.hour.values
                hour_counts = np.bincount(hours, minlength=24).astype(float)
                hour_counts /= hour_counts.sum() + 1e-9
                # Gini index
                sorted_counts = np.sort(hour_counts)
                n = len(sorted_counts)
                cumsum = np.cumsum(sorted_counts)
                gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_counts) - (n + 1)) / n
                gini = float(np.clip(gini, 0, 1))
            else:
                gini = 0.5
        except Exception:
            gini = 0.5

        # Activity rate
        if "date" in all_trades.columns:
            time_span = (
                pd.to_datetime(all_trades["date"]).max()
                - pd.to_datetime(all_trades["date"]).min()
            ).total_seconds() / 3600 + 1e-9
            trades_per_hr = len(all_trades) / time_span
        else:
            trades_per_hr = len(all_trades) / 6.5   # NEPSE session ~6.5 hrs

        # Directional / inventory
        net_dir = abs(buy_vol - sell_vol) / (total_vol + 1e-9)

        # OFI proxy if available
        ofi_mean = float(all_trades["ofi"].mean()) if "ofi" in all_trades.columns else 0.0

        # VWAP alpha proxy
        if "vwap_deviation" in all_trades.columns:
            buy_vwap_dev  = buys["vwap_deviation"].mean()  if len(buys)  > 0 else 0.0
            sell_vwap_dev = sells["vwap_deviation"].mean() if len(sells) > 0 else 0.0
            # Skilled traders buy below VWAP (negative dev) and sell above (positive dev)
            vwap_alpha = float(-buy_vwap_dev + sell_vwap_dev)
        else:
            vwap_alpha = 0.0

        # Metaorder flag from Module 2 (if available)
        is_metaorder = int(all_trades.get("is_metaorder", pd.Series([False])).any() if "is_metaorder" in all_trades.columns else False)

        records.append({
            "broker":              str(broker),
            "n_trades":            len(all_trades),
            "total_volume":        float(total_vol),
            "avg_trade_size":      avg_sz,
            "std_trade_size":      std_sz,
            "p95_trade_size":      p95_sz,
            "large_trade_ratio":   large_rt,
            "timing_gini":         gini,
            "trades_per_hour":     float(trades_per_hr),
            "net_direction":       float(net_dir),
            "ofi_mean":            ofi_mean,
            "vwap_alpha":          vwap_alpha,
            "is_metaorder_flag":   is_metaorder,
        })

    feat_df = pd.DataFrame(records).set_index("broker")
    logger.info("Feature matrix: %d brokers × %d features", *feat_df.shape)
    return feat_df


# ─────────────────────────────────────────────────────────────────────────────
#  Spectral Clustering
# ─────────────────────────────────────────────────────────────────────────────

_FEATURE_COLS = [
    "avg_trade_size", "std_trade_size", "p95_trade_size",
    "large_trade_ratio", "timing_gini", "trades_per_hour",
    "net_direction", "ofi_mean", "vwap_alpha",
]


def run_spectral_clustering(
    feat_df: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Cont et al. (2023) / Balcau et al. (2024):
    Segment brokers into clusters using Spectral Clustering on the
    similarity (affinity) matrix of standardised broker features.

    Cluster labels are then heuristically mapped to:
      0 - Institutional / Informed (large size, metaorder, high alpha)
      1 - Active Retail
      2 - Passive/Market-Maker
      3 - Noise/Low-Activity

    Returns feat_df with added columns: cluster, cluster_label, pca_x, pca_y.
    """
    available = [c for c in _FEATURE_COLS if c in feat_df.columns]
    X_raw = feat_df[available].fillna(0).values

    if len(X_raw) < n_clusters + 1:
        n_clusters = max(2, len(X_raw) - 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Try Spectral; fall back to KMeans if graph connectivity issues
    try:
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="rbf",
            gamma=0.5,
            random_state=random_state,
            n_init=10,
        )
        labels = model.fit_predict(X_scaled)
        method = "spectral"
    except Exception as exc:
        logger.warning("SpectralClustering failed (%s); using KMeans.", exc)
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = model.fit_predict(X_scaled)
        method = "kmeans"

    feat_df = feat_df.copy()
    feat_df["cluster"] = labels
    feat_df["cluster_method"] = method

    # Silhouette score
    if len(set(labels)) > 1:
        sil = silhouette_score(X_scaled, labels)
        feat_df["silhouette"] = round(sil, 4)
        logger.info("Silhouette score: %.4f (method=%s)", sil, method)

    # PCA for 2D visualisation
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_scaled)
    feat_df["pca_x"] = coords[:, 0]
    feat_df["pca_y"] = coords[:, 1]
    feat_df["pca_var_explained"] = round(pca.explained_variance_ratio_.sum(), 4)

    # Label clusters heuristically
    feat_df = _label_clusters(feat_df)

    return feat_df


def _label_clusters(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign human-readable labels based on cluster-mean feature values.
    The 'Institutional' cluster has the highest average trade size and
    highest VWAP alpha. The 'Noise' cluster has lowest activity.
    """
    if "cluster" not in feat_df.columns:
        return feat_df

    cluster_means = feat_df.groupby("cluster")[
        ["avg_trade_size", "vwap_alpha", "trades_per_hour", "n_trades"]
    ].mean()

    # Score each cluster
    cluster_means["inst_score"] = (
        cluster_means["avg_trade_size"].rank()
        + cluster_means["vwap_alpha"].rank()
    )
    cluster_means["noise_score"] = (
        (cluster_means["n_trades"].rank(ascending=False))
        + (cluster_means["trades_per_hour"].rank(ascending=False))
    )

    ranked_inst  = cluster_means["inst_score"].sort_values(ascending=False).index
    ranked_noise = cluster_means["noise_score"].sort_values(ascending=False).index

    label_map: dict[int, str] = {}
    n = len(cluster_means)
    labels_pool = [
        "🏛️ Institutional / Informed",
        "⚡ Active Retail",
        "🔄 Market Maker / Passive",
        "🌫️ Noise / Low-Activity",
    ]

    for i, cid in enumerate(ranked_inst):
        label_map[int(cid)] = labels_pool[min(i, len(labels_pool) - 1)]

    # Any unlabelled cluster → noise
    for cid in cluster_means.index:
        if int(cid) not in label_map:
            label_map[int(cid)] = labels_pool[-1]

    feat_df["cluster_label"] = feat_df["cluster"].map(label_map)
    return feat_df


# ─────────────────────────────────────────────────────────────────────────────
#  Cluster profiling
# ─────────────────────────────────────────────────────────────────────────────

def profile_clusters(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary table comparing clusters across all features.
    """
    if "cluster_label" not in feat_df.columns:
        return pd.DataFrame()

    profile_cols = [c for c in _FEATURE_COLS if c in feat_df.columns] + ["n_trades", "total_volume"]
    profile = (
        feat_df.groupby("cluster_label")[profile_cols]
               .mean()
               .round(3)
               .reset_index()
    )
    profile["n_brokers"] = feat_df.groupby("cluster_label").size().values
    return profile


# ─────────────────────────────────────────────────────────────────────────────
#  Top institutional brokers
# ─────────────────────────────────────────────────────────────────────────────

def get_institutional_brokers(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return brokers labelled Institutional, sorted by avg_trade_size.
    """
    mask = feat_df.get("cluster_label", pd.Series(dtype=str)).str.contains(
        "Institutional", na=False
    )
    inst = feat_df[mask].reset_index()
    if inst.empty:
        # Fallback: top 10% by avg_trade_size
        thresh = feat_df["avg_trade_size"].quantile(0.90)
        inst = feat_df[feat_df["avg_trade_size"] >= thresh].reset_index()

    return inst.sort_values("avg_trade_size", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience: run full Module 5 pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_clustering_engine(df: pd.DataFrame, n_clusters: int = 4) -> dict:
    """
    Extract features, cluster, and profile.
    """
    feat_df      = extract_broker_features(df)
    clustered_df = run_spectral_clustering(feat_df, n_clusters=n_clusters)
    cluster_profile = profile_clusters(clustered_df)
    inst_brokers    = get_institutional_brokers(clustered_df)

    return {
        "feature_df":       feat_df,
        "clustered_df":     clustered_df,
        "cluster_profile":  cluster_profile,
        "inst_brokers":     inst_brokers,
    }
