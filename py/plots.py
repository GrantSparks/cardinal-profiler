"""Pure-visualisation helpers (no I/O side-effects)."""
from __future__ import annotations

import logging
import textwrap
from itertools import compress
from collections import Counter
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from py.utils import (
    _cmap,
    parse_tag_scores,
    weighted_project_bipartite,
    to_simple,
)
from py.analysis import (
    _bloc_weights,
    centrality_metrics,
)
# MC helper
from py.plots_mc import make_mc_win_probability_bars
from py.config import (
    Config,
    TAG_SEP,
    DEFAULT_LAYER_WEIGHTS,
    PALETTE,
)

# Optional – used for label overlap‑avoidance in scatter
try:
    from adjustText import adjust_text  # type: ignore
except ImportError:  # pragma: no cover
    adjust_text = None  # type: ignore

LOGGER = logging.getLogger(__name__)

__all__ = [
    "render_network",
    "make_bloc_bar",
    "make_anchor_bridge_scatter",
    "render_dendrogram",
    "make_cardinal_profile_figure",
]

# ── Network quick‑view ─────────────────────────────────────────────────────

def render_network(
    df: pd.DataFrame,
    min_weight: float = 0.3,
    *,
    html_path: str | "os.PathLike[str]" | None = None,
) -> tuple[Figure, nx.Graph]:
    """Return *(fig, G)* for the L5 bipartite network quick‑view.

    If *html_path* is supplied **and** :pymod:`pyvis` is installed we also
    create an interactive HTML file on disk, otherwise that step is skipped
    silently.  The caller is responsible for saving the PNG if desired.
    """
    if "Name" not in df.columns or "L5_Tags" not in df.columns:
        raise ValueError("CSV must contain Name and L5_Tags columns.")

    # 1️⃣ Build bipartite graph ------------------------------------------------
    B = nx.Graph()
    persons = df["Name"].tolist()
    B.add_nodes_from((p, {"bipartite": 0}) for p in persons)
    for _, row in df.iterrows():
        p = row["Name"]
        for tag, score in parse_tag_scores(row["L5_Tags"], row.get("L5_Scores")):
            if score < min_weight:
                continue
            B.add_node(tag, bipartite=1)
            B.add_edge(p, tag, weight=score)

    # 2️⃣ Project onto persons‑layer ------------------------------------------
    G = weighted_project_bipartite(B, persons)

    # ---- colour persons by largest L5 tag family -----------------
    def largest_tag(p):
        l5 = df.set_index("Name").at[p, "L5_Tags"]
        if pd.isna(l5) or not l5:
            return "_NONE"
        return max(l5.split(TAG_SEP), key=len)

    person_tags = {p: largest_tag(p) for p in persons}
    families = sorted({t.split("_")[0] for t in person_tags.values()})
    cmap = _cmap("tab10", len(families))
    fam_idx = {f: i for i, f in enumerate(families)}
    node_colors = [cmap(fam_idx[person_tags[p].split("_")[0]]) for p in persons]

    # 3️⃣ Matplotlib static view ----------------------------------------------
    fig = plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(B, seed=42, k=1/np.sqrt(len(B)), iterations=60)
    nx.draw_networkx_edges(B, pos, alpha=0.1, width=.4)
    nx.draw_networkx_nodes(B, pos, nodelist=persons,
                           node_color=node_colors, node_size=320,
                           edgecolors="black")
    # draw tag nodes faintly
    nx.draw_networkx_nodes(B, pos,
                           nodelist=[n for n in B if n not in persons],
                           node_color="lightgrey", node_shape="s", node_size=200)
    nx.draw_networkx_labels(B, pos, font_size=6)
    plt.title(f"Cardinals ↔ L5 factions (≥ {min_weight:.1f})")
    # categorical legend
    proxies = [Line2D([], [], marker='o', ls='',
                      color=cmap(fam_idx[f])) for f in families]
    plt.legend(proxies, families, title="Bloc family",
               frameon=False, loc="upper right")
    plt.axis("off")

    # 4️⃣ Optional interactive HTML using *pyvis* -----------------------------
    if html_path is not None:
        try:
            from pyvis.network import Network  # type: ignore
        except ImportError:
            LOGGER.debug("pyvis not installed – skipping HTML export.")
        else:
            nt = Network(height="850px", width="100%")
            nt.from_nx(B)
            nt.show_buttons(filter_=["physics"])
            nt.save_graph(str(html_path))
            LOGGER.info("[html] network → %s", html_path)

    return fig, G


# ── Bloc‑strength bar ──────────────────────────────────────────────────────

def make_bloc_bar(bloc_counts: pd.Series) -> Figure:
    """Horizontal bar chart of bloc head‑counts (long names read nicely)."""
    # Re-order: group by family then by head-count
    families = [t.split("_")[0] for t in bloc_counts.index]
    order = (
        pd.DataFrame({"fam": families, "cnt": bloc_counts})
        .sort_values(["fam", "cnt"], ascending=[True, False])
        .index
    )
    bloc_counts = bloc_counts.loc[order]
    families = [t.split("_")[0] for t in bloc_counts.index]

    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = _cmap("tab10", len(set(families)))
    fam_idx = {f: i for i, f in enumerate(sorted(set(families)))}
    colors = [cmap(fam_idx[f]) for f in families]

    ax.barh(bloc_counts.index, bloc_counts.values, color=colors)
    ax.set_title("Bloc Strength (Head‑counts)")
    ax.set_xlabel("Cardinal count")

    # annotate counts
    for y, v in enumerate(bloc_counts.values):
        ax.text(v + 0.6, y, f"{v}", va="center", fontsize=8)

    # compact legend (family colours)
    handles = [Line2D([], [], marker='s', ls='',
                      color=cmap(fam_idx[f])) for f in sorted(set(families))]
    ax.legend(handles, sorted(set(families)), title="Bloc Family",
              frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    return fig


# ── Anchor‑vs‑Bridge scatter ───────────────────────────────────────────────

def make_anchor_bridge_scatter(
    df: pd.DataFrame,
    G: nx.Graph,
    scores: pd.Series,
    *,
    cfg: Config,
    eig: dict[str, float] | None = None,
    betw: dict[str, float] | None = None,
) -> Figure:
    """Composite‑score vs bridge centrality scatter with de‑overlapped labels."""
    if eig is None or betw is None:
        eig, betw = centrality_metrics(G)
    tag_counts = _bloc_weights(df)

    # Prefer canonical L1_Tags but fall back to Country_Tags so the scatter
    # survives even when layer 1 is absent.
    region_src = (
        df.set_index("Name")
          .apply(lambda r: r.get("L1_Tags") or r.get("Country_Tags") or "", axis=1)
    )
    regions = region_src.apply(lambda s: s.split(TAG_SEP)[0] or "Other")
    cmap = _cmap("Set3", len(set(regions)))
    reg_idx = {r: i for i, r in enumerate(sorted(set(regions)))}

    fig, ax = plt.subplots(figsize=(10, 8))
    xs    = np.array([eig.get(n, 0.0)  for n in scores.index])
    ys    = np.array([betw.get(n, 0.0) for n in scores.index])
    sizes = np.array([tag_counts.loc[n] * 50 for n in scores.index])
    cols  = np.array([cmap(reg_idx[regions.loc[n]]) for n in scores.index])

    # split "signal" vs "noise" – keeps clutter down
    sig = (scores.rank(ascending=False) <= 50) | \
          (pd.Series(betw).rank(ascending=False) <= 10)
    ax.scatter(xs[~sig], ys[~sig], s=sizes[~sig],
               c="lightgrey", alpha=.4, zorder=1)
    ax.scatter(xs[sig], ys[sig], s=sizes[sig],
               c=cols[sig], alpha=.8, zorder=2)

    ax.set_yscale("log")

    # label only top-15 composite **or** top-5 betweenness
    label_mask = (scores.rank(ascending=False) <= 15) | \
                 (pd.Series(betw).rank(ascending=False) <= 5)
    texts = [ax.text(x, y, n, fontsize=7)
             for n, x, y, m in zip(scores.index, xs, ys, label_mask) if m]
    if adjust_text:
        adjust_text(texts, ax=ax)

    ax.set_xlabel("Eigenvector centrality (Anchor strength)")
    ax.set_ylabel("Betweenness centrality (Bridge score)")
    ax.set_title(f"Anchor vs Bridge scatter (α = {cfg.centrality_mix:.2g})")
    
    # region legend
    handles = [Line2D([], [], marker='o', ls='',
                      color=cmap(reg_idx[r])) for r in sorted(reg_idx)]
    ax.legend(handles, sorted(reg_idx), title="Region",
              frameon=False, loc="upper right")
    return fig


# ── Dendrogram ─────────────────────────────────────────────────────────────

def render_dendrogram(
    X: np.ndarray,
    labels: Sequence[str],
    regions: Sequence[str] | None,
    title: str,
    metric: str,
    *,
    linkage_method: str = "average",
    max_d: float | None = None,
) -> Figure:
    """Hierarchical‑clustering dendrogram coloured by L1 *regions*."""
    dist_vec = pdist(X, metric=metric)
    Z = linkage(dist_vec, method=linkage_method)

    h = 0.4 * len(labels) + 4
    if len(labels) > 80:          # shrink big ones so PNG stays < 3 kpx
        h = 0.18 * len(labels) + 3
    fig = plt.figure(figsize=(12, h))
    wrapped = [textwrap.fill(lbl, 25) for lbl in labels]

    dn = dendrogram(
        Z,
        labels=wrapped,
        orientation="right",
        leaf_font_size=9,
        link_color_func=lambda _: "lightgrey",
    )
    plt.title(title, fontsize=14)
    plt.xlabel(f"{metric.capitalize()} distance")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # optional cut-line for quick cluster eyeballing
    if max_d is not None:
        plt.axvline(max_d, c="grey", ls="--", lw=.8)

    if regions:
        # Colour-map must follow *leaf* order, not original row order
        mapping = {lbl: reg for lbl, reg in zip(labels, regions)}
        ordered_regions = [
            mapping.get(lbl.replace("\n", " "), "Other") for lbl in dn["ivl"]
        ]
        cmap = _cmap("Set3", len(set(ordered_regions)))
        idx = {r: i for i, r in enumerate(sorted(set(ordered_regions)))}
        ax = plt.gca()
        for tick, r in zip(ax.get_ymajorticklabels(), ordered_regions):
            tick.set_color(cmap(idx.get(r, 0)))

    return fig


# ── Cardinal cheatsheet profile ────────────────────────────────────────────

def _top_tags(
    row: pd.Series,
    layer: str,
    k: int = 5,
    *,
    layer_weights: dict[str, float] | None = None,
) -> list[str]:
    """Return the *k* strongest tags in *layer* after weight adjustment."""
    score_col = layer.replace("_Tags", "_Scores")
    tag_cell = row.get(layer)
    if tag_cell is None or pd.isna(tag_cell):
        return []
    pairs = parse_tag_scores(tag_cell, row.get(score_col))
    if not pairs:
        return []
    weights = layer_weights or DEFAULT_LAYER_WEIGHTS
    w_layer = weights.get(layer, 1.0)
    pairs = [(t, s * w_layer) for t, s in pairs]
    return [t for t, _ in sorted(pairs, key=lambda p: p[1], reverse=True)[:k]]


def make_cardinal_profile_figure(
    name: str,
    row: pd.Series,
    G: nx.Graph,
    scores: pd.Series,
    bloc_sizes: pd.Series,
    eig: dict[str, float],
    betw: dict[str, float],
    *,
    layers: Sequence[str] | None = None,
    layer_weights: dict[str, float] | None = None,
    mc_results: pd.DataFrame | None = None,
    **kwargs,
) -> Figure:
    """Single‑page A4 portrait explaining why *name* ranks where it does."""
    if layers is None:
        layers = ["L1_Tags", "L3_Tags", "L5_Tags"]

    # Add one extra row for the paragraph
    height = [1.3, 1.0] + [0.9] * len(layers) + [1.15]
    mosaic = [["title", "network"], 
              ["metrics", "network"]] + \
             [[f"tags_{layer}", "network"] for layer in layers] + \
             [["paragraph", "paragraph"]]

    fig, axd = plt.subplot_mosaic(
        mosaic,
        figsize=(8.27, 11.69),  # A4 portrait
        height_ratios=height,
        gridspec_kw={"hspace": 0.35, "wspace": 0.08},
        constrained_layout=True,
    )

    # Title -----------------------------------------------------------------
    ax_title = axd["title"]
    ax_title.axis("off")
    ax_title.text(0.5, 0.5, name, ha="center", va="center", fontsize=20, fontweight="bold")
    if name in scores.index:
        rank_pos = scores.index.get_loc(name) + 1
        score_val = scores.loc[name]
        subtitle = f"Composite score rank #{rank_pos} (score {score_val:.3f})"
    else:
        subtitle = "Composite score unavailable"
    ax_title.text(
        0.5,
        0.1,
        subtitle,
        ha="center",
        va="center",
        fontsize=12,
    )

    # Metrics bar -----------------------------------------------------------
    ax_bar = axd["metrics"]
    metrics = ["Composite", "Eig.", "Betw.", "Bloc-wt"]
    values   = [
        scores.loc[name],
        eig.get(name, 0.0),
        betw.get(name, 0.0),
        bloc_sizes.loc[name],
    ]
    ax_bar.bar(metrics, values, color=PALETTE["blue"])
    ax_bar.set_ylim(0, max(values) * 1.15)
    ax_bar.set_title("Ranking drivers")
    # dashed field median
    med = pd.Series(values).median()
    ax_bar.axhline(med, ls="--", lw=.8, c="grey")
    for i, v in enumerate(values):
        ax_bar.text(i, v + (ax_bar.get_ylim()[1] * 0.02), f"{v:.3f}", ha="center", fontsize=8)

    # ── tiny MC win-probability bar inside an inset ────────────────
    if mc_results is not None and name in mc_results.index:
        inset = ax_bar.inset_axes([0.55, 0.45, 0.4, 0.45])  # x, y, w, h in axes fraction
        make_mc_win_probability_bars(name, mc_results, ax=inset, show_stats=False)
        inset.set_title("MC p(win)", fontsize=7, pad=2)

    # Per‑layer tag tables ---------------------------------------------------
    for layer in layers:
        ax = axd[f"tags_{layer}"]
        ax.axis("off")
        tags = _top_tags(row, layer, layer_weights=layer_weights)
        clean = layer.replace("_Tags", "")
        ax.set_title(f"{clean} – top {len(tags) or 0}", fontsize=10, loc="left")
        if tags:
            cell_text = [[t] for t in tags]
            table = ax.table(cellText=cell_text,
                             loc="upper left",
                             colWidths=[0.9])
            table.scale(1, 1.2)
            for i, (k, cell) in enumerate(table.get_celld().items()):
                if k[0] == 0:          # header row is hidden
                    cell.set_edgecolor("w")
                else:
                    cell.set_facecolor("#f5f5f5" if i%2 else "#ffffff")
            ax.set_xlim(0,1); ax.set_ylim(0,1)
        else:
            ax.text(0.01, 0.5, "– none –", va="center", fontsize=9)

    # Ego network -----------------------------------------------------------
    ax_net = axd["network"]
    ego = nx.ego_graph(G, name, radius=1)
    pos = nx.spring_layout(ego, seed=42)
    nx.draw_networkx_edges(ego, pos, ax=ax_net, alpha=0.3)
    nx.draw_networkx_nodes(
        ego,
        pos,
        nodelist=[name],
        node_color=PALETTE["gold"],
        node_size=500,
        edgecolors="black",
        ax=ax_net,
    )
    others = [n for n in ego if n != name]
    nx.draw_networkx_nodes(
        ego,
        pos,
        nodelist=others,
        node_color=PALETTE["blue"],
        node_size=200,
        ax=ax_net,
    )
    nx.draw_networkx_labels(ego, pos, font_size=6, ax=ax_net)
    ax_net.set_title("Immediate network neighbourhood", fontsize=10)
    ax_net.axis("off")
    
    # Paragraph panel -------------------------------------------------------
    ax_par = axd["paragraph"]
    ax_par.axis("off")
    
    para = (row.get("Paragraph") or "").strip()
    if not para:
        para = "— no summary paragraph available —"
        
    wrapped = textwrap.fill(para, width=105)
    ax_par.text(
        0, 1, wrapped,
        va="top", ha="left",
        fontsize=9,
        linespacing=1.25,
        wrap=True,
    )

    return fig
