"""
Data-centric helpers: build feature matrices, graphs and centrality-based
scores for the ranking pipeline.  Visualisation lives in :pymod:`py.plots`;
CLI parsing in :pymod:`py.cli`.
"""

from __future__ import annotations

import math
import logging
import numpy as np  # for LinAlgError catch
from collections import Counter
from typing import Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

# Project‑level utilities ---------------------------------------------------
from .config import (
    TAG_SEP,
    DEFAULT_LAYER_WEIGHTS,
)
from .utils import (
    parse_tag_scores,
    weighted_project_bipartite,
    to_simple,
    _is_placeholder,
)

LOGGER = logging.getLogger(__name__)

# Public API surface
__all__ = [
    "build_feature_matrix",
    "build_multilayer_graph",
    "centrality_metrics",          #  ← NEW
    "centrality_ranking",
    "composite_centrality_scores",
    "composite_ranking",
    "monte_carlo_sim",
]

# ── Centrality helpers ─────────────────────────────────────────────────────

def centrality_metrics(
    G: nx.Graph,
    *,
    force_power: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Return ``(eigenvector, betweenness)`` centralities on a **simple** graph.

    *Best-effort* strategy:
    1. Try the fast NumPy solver on the whole graph.
    2. If the graph is **disconnected** (`AmbiguousSolution`) **or**
       `force_power` is *True*, fall back to the power-iteration variant that
       gracefully handles multiple components.
    """
    Gs = to_simple(G) if isinstance(G, nx.MultiGraph) else G

    if not force_power:
        try:
            eig = nx.eigenvector_centrality_numpy(Gs, weight="weight")
        except (nx.AmbiguousSolution,
            nx.PowerIterationFailedConvergence,
            np.linalg.LinAlgError):        # ← disconnected or ill-conditioned
            # Fallback: always succeed, even on disconnected graphs
            eig = nx.eigenvector_centrality(Gs, weight="weight", max_iter=1000)
    else:
        eig = nx.eigenvector_centrality(Gs, weight="weight", max_iter=1000)

    betw = nx.betweenness_centrality(Gs, weight="weight", normalized=True)
    return eig, betw

# ── Feature‑matrix & graph construction ───────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    layer_weights: dict[str, float],
    *,
    extra_tag_columns: Sequence[str] | None = None,
) -> Tuple[np.ndarray, list[str]]:
    """Return a *(row‑normalised)* tag feature matrix and its vocabulary.

    Parameters
    ----------
    df
        Raw cardinal data frame.
    layer_weights
        Mapping of ``col_name → multiplier``; usually comes from
        :pyattr:`config.DEFAULT_LAYER_WEIGHTS` (optionally customised at run‑time).
    extra_tag_columns
        Any additional ``*_Tags`` columns to fold into the matrix – used for
        alternative L3 thematic axes.
    """
    layer_weights = dict(layer_weights)          # ← avoid in-place mutation
    # keep only tag-columns that actually exist in the CSV -----------------
    tag_cols   = [c for c in layer_weights if c in df.columns]
    score_cols = [c.replace("_Tags", "_Scores") for c in tag_cols]

    if extra_tag_columns:
        for col in extra_tag_columns:
            if col not in df.columns:
                continue
            tag_cols.append(col)
            score_cols.append(col.replace("_Tags", "_Scores"))
            # inherit the L3 weight unless explicitly provided
            layer_weights[col] = layer_weights.get("L3_Tags", 1.0)

    # ---------------------------------------------------------------- vocab
    vocab: set[str] = set()
    for col in tag_cols:
        for cell in df[col].fillna(""):
            if cell:
                vocab.update(t.strip() for t in str(cell).split(TAG_SEP) if t.strip())

    ordered_tags = sorted(vocab)
    tag_idx = {t: i for i, t in enumerate(ordered_tags)}

    # -------------------------------------------------------------- raw matrix
    M = np.zeros((len(df), len(ordered_tags)), dtype=float)
    for row_i, (_, row) in enumerate(df.iterrows()):   # index-safe
        for tag_col, score_col in zip(tag_cols, score_cols):
            w_layer = layer_weights.get(tag_col, 1.0)
            for tag, s in parse_tag_scores(row.get(tag_col, ""), row.get(score_col)):
                M[row_i, tag_idx[tag]] += w_layer * s

    # ① Row‑normalise so cosine similarity emphasises orientation ---------
    M = normalize(M, norm="l2")

    # ② Bloc‑size lift *after* normalisation (otherwise re‑scaled away) ---
    df_counts = (M > 0).sum(axis=0)
    M *= np.log(df_counts + 1)
    return M, ordered_tags


def build_multilayer_graph(
    df: pd.DataFrame,
    layers: Sequence[str],
    *,
    min_weights: dict[str, float],
    layer_weights: dict[str, float] | None = None,
) -> nx.MultiGraph:
    """Return a **multi‑layer** projection of the cardinal‑↔‑tag bipartite graph."""
    persons = set(df["Name"])
    MG = nx.MultiGraph()
    MG.add_nodes_from(persons)

    for layer in layers:
        if layer not in df.columns:
            LOGGER.warning("Requested layer '%s' missing in CSV – skipped.", layer)
            continue
        score_col = layer.replace("_Tags", "_Scores")

        # Build bipartite graph for *this* layer then project onto persons
        B = nx.Graph()
        B.add_nodes_from((p, {"bipartite": 0}) for p in persons)

        for _, row in df.iterrows():
            p = row["Name"]
            w_layer = (layer_weights or {}).get(layer, 1.0)
            for tag, score in parse_tag_scores(row.get(layer, ""), row.get(score_col)):
                score *= w_layer
                if score < min_weights.get(layer, 0):
                    continue
                B.add_node(tag, bipartite=1)
                B.add_edge(p, tag, weight=score)

        G_layer = weighted_project_bipartite(B, persons)
        for u, v, data in G_layer.edges(data=True):
            MG.add_edge(u, v, layer=layer, weight=data["weight"])

    return MG

# ── Ranking helpers ───────────────────────────────────────────────────────

def _bloc_weights(df: pd.DataFrame, *, layer: str = "L5_Tags", gamma: float = 1.0) -> pd.Series:
    """Cardinal‑specific bloc lift weights (log‑scaled)."""
    if layer not in df.columns:
        raise KeyError(f"Bloc layer '{layer}' missing in dataframe")

    bloc_sizes = Counter(
        tag
        for cell in df[layer].dropna()
        for tag in cell.split(TAG_SEP)
        if tag and not _is_placeholder(tag)
    )

    def biggest(cell: str) -> int | None:
        tags = [
            t for t in str(cell).split(TAG_SEP)
            if t and not _is_placeholder(t)
        ]
        if not tags:           # completely blank → weight 0
            return 0
        return max((bloc_sizes.get(t, 1) for t in tags), default=1)

    if not df["Name"].is_unique:
        raise ValueError("Duplicate cardinal names – uniqueness required.")
    sizes = df[layer].fillna("").apply(biggest).astype(float)
    sizes.index = df["Name"]

    # natural log growth per v2‑spec
    return sizes.map(lambda s: 0.0 if s == 0 else (math.log(s) + 1.0) ** gamma)


def composite_centrality_scores(
    df: pd.DataFrame,
    G: nx.Graph,
    *,
    alpha: float = 0.7,
    eig: dict[str, float] | None = None,
    betw: dict[str, float] | None = None,
    bloc_layer: str = "L5_Tags",
    bloc_gamma: float = 1.0,
) -> pd.Series:
    """(α·eig + (1‑α)·betweenness) × bloc_weight."""
    if G.number_of_nodes() == 0:
        raise ValueError("Graph has no nodes – cannot rank.")

    if eig is None or betw is None:
        eig, betw = centrality_metrics(G)

    blend = {n: alpha * eig.get(n, 0.0) + (1 - alpha) * betw.get(n, 0.0) for n in G.nodes()}
    bloc_w = _bloc_weights(df, layer=bloc_layer, gamma=bloc_gamma).reindex(G.nodes()).fillna(1.0)

    return (pd.Series(blend) * bloc_w).sort_values(ascending=False)


def centrality_ranking(
    G: nx.Graph,
    *,
    alpha: float = 0.7,
    eig: dict[str, float] | None = None,
    betw: dict[str, float] | None = None,
) -> pd.Series:
    """Pure centrality blend without bloc weighting."""
    if G.number_of_nodes() == 0:
        raise ValueError("Graph empty – cannot rank.")

    if eig is None or betw is None:
        eig, betw = centrality_metrics(G)

    blended = {
        n: alpha * eig.get(n, 0.0) + (1.0 - alpha) * betw.get(n, 0.0)
        for n in G.nodes()
    }
    return pd.Series(blended).sort_values(ascending=False)


def composite_ranking(
    df: pd.DataFrame,
    G: nx.Graph,
    *,
    alpha: float = 0.7,
) -> pd.Series:
    """Alias for :pyfunc:`composite_centrality_scores` (kept for API parity)."""
    return composite_centrality_scores(df, G, alpha=alpha)


# ── Monte‑Carlo ballot simulation (non‑visual) ────────────────────────────

def monte_carlo_sim(
    scores: pd.Series,
    ballots: int,
    seed: int | None,
    include_ci: bool = True,
) -> pd.DataFrame:
    """
    Vectorized multinomial sampling over the score-based probability vector.
    
    Uses numpy.random.multinomial for efficient memory usage and performance.
    
    Parameters
    ----------
    scores : pd.Series
        Cardinal scores used to determine ballot probabilities
    ballots : int
        Number of ballots to simulate
    seed : int or None
        Random number generator seed for reproducibility
    include_ci : bool, default True
        Whether to include 95% credible intervals for the win rates
        
    Returns
    -------
    pd.DataFrame
        DataFrame with WinCount, WinRate, and optional CI bounds columns,
        sorted by WinCount descending
    """
    import scipy.stats

    rng = np.random.default_rng(seed)
    probs = scores / scores.sum()                          # normalize once
    counts = rng.multinomial(ballots, probs.values)        # single vectorized draw
    
    # Create the base result DataFrame
    df = pd.DataFrame({
        "WinCount": counts,
        "WinRate": counts / ballots,
    }, index=probs.index)
    
    # Add 95% credible intervals using beta distribution (conjugate prior)
    if include_ci:
        alpha_post = 1 + counts  # adding 1 for Bayesian update (Beta(1,1) prior)
        beta_post = 1 + ballots - counts
        
        # Calculate 95 % credible intervals.
        # NOTE: beta.ppf is vectorised over α and β, but when q has shape (2,)
        # it tries to broadcast (2,) against (n,) and fails on NumPy < 2.0.
        # Calling it twice with scalar q values avoids that pitfall and is
        # still fully vectorised over the posterior parameters.
        ci_low  = scipy.stats.beta.ppf(0.025, alpha_post, beta_post)
        ci_high = scipy.stats.beta.ppf(0.975, alpha_post, beta_post)
        
        df["CI_Low"] = ci_low
        df["CI_High"] = ci_high
    
    return df.sort_values("WinCount", ascending=False)
