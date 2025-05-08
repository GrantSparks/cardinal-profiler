from __future__ import annotations
"""Report assemblers: collect matplotlib figures, optionally bundle into PDF."""

# stdlib / typing -----------------------------------------------------------
from pathlib import Path
import logging
from typing import Sequence, List, Optional

# third-party ---------------------------------------------------------------
import networkx as nx
import pandas as pd
from matplotlib.figure import Figure

# project-local -------------------------------------------------------------
# keep the pretty header/footer from the original script ✨
from py.utils import PDFReportBuilder
from py.config import Config
from py.plots import (
    render_network,
    make_bloc_bar,
    make_anchor_bridge_scatter,
    make_cardinal_profile_figure,
)
from collections import Counter
import pandas as pd
from py.utils import parse_tag_scores
from py.analysis import (
    composite_centrality_scores,
    centrality_metrics,
    _bloc_weights,
)
from py.utils import to_simple

LOGGER = logging.getLogger(__name__)

__all__ = [
    "make_overview_report",
    "make_cardinal_profiles_report",
]

###############################################################################
# Executive‑summary multi‑page PDF                                           #
###############################################################################

# local alias for readability
_centrality_metrics = centrality_metrics


def make_overview_report(
    df: pd.DataFrame,
    cfg: Config,
    *,
    G: Optional[nx.Graph] = None,
    fig_net: Optional[Figure] = None,
    scores: Optional[pd.Series] = None,
    eig: Optional[dict[str, float]] = None,
    betw: Optional[dict[str, float]] = None,
    bloc_series: Optional[pd.Series] = None,
    fig_den: Optional[Figure] = None,
    min_weight: float = 0.3,
    output_path: str | Path | None = None,
    mc_results: Optional[pd.DataFrame] = None,
) -> List[Figure]:
    """Generate and return a *three‑page* executive‑summary report.

    **Pages**
        1. *Network quick‑view*       – Cardinal ↔ L5 bipartite graph
        2. *Bloc strength bar chart* – head‑counts per bloc
        3. *Anchor/Bridge scatter*   – composite vs bridge centrality

    Parameters
    ----------
    df
        Source data frame exactly as ingested by :pymod:`py.ingest` – must
        contain canonical *Name* and tag score columns.
    cfg
        Runtime configuration – the same object that steers analysis.
    G
        Optional pre-computed network graph to use (prevents recalculation).
    scores
        Optional pre-computed composite scores to use (prevents recalculation).
    eig
        Optional pre-computed eigenvector centrality metrics (prevents recalculation).
    betw
        Optional pre-computed betweenness centrality metrics (prevents recalculation).
    bloc_series
        Optional pre-computed bloc head count series (prevents recalculation).
    min_weight
        Threshold for including a person‑tag edge in the network view.
    output_path
        If supplied, a single PDF is written to this path.  Any parent
        directories are created automatically.  When *None* (default) **no
        file is written** and the caller receives the figures only.

    Returns
    -------
    list[matplotlib.figure.Figure]
        The three generated figures in page order.
    """
    # ── Network quick-view ───────────────────────────────────────────────────
    if fig_net is None or G is None:
        fig_net, G_vis = render_network(df, min_weight=min_weight)
        G_use = G or G_vis
    else:
        G_use = G

    # ── Centralities (reuse if caller already solved them) ───────────────────
    if eig is None or betw is None:
        eig, betw = _centrality_metrics(G_use)

    # ── Bloc head-counts (can be injected to avoid recomputation) ────────────
    if bloc_series is None:
        bloc_counts = Counter()
        for _, row in df.iterrows():
            for tag, s in parse_tag_scores(row.get("L5_Tags", ""), row.get("L5_Scores", None)):
                if s >= min_weight:
                    bloc_counts[tag] += 1
        bloc_series = pd.Series(bloc_counts).sort_values(ascending=False)
    fig_bloc = make_bloc_bar(bloc_series)

    # ── Composite score calculation (only if caller didn't) ──────────────────
    if scores is None:
        scores = composite_centrality_scores(df, G_use, alpha=cfg.centrality_mix)
    fig_scatter = make_anchor_bridge_scatter(
        df, G_use, scores, cfg=cfg, eig=eig, betw=betw
    )
    
    # ── Add Monte Carlo win probability chart if results are available ─────
    figures: List[Figure] = [fig_net, fig_bloc, fig_scatter]
    if mc_results is not None:
        # Import the MC visualization functions 
        from py.plots_mc import make_mc_probability_chart
        fig_mc = make_mc_probability_chart(mc_results, limit=cfg.top_n)
        figures.append(fig_mc)
        
    if fig_den is not None:
        figures.insert(0, fig_den)

    # ── Pretty PDF with timestamp/footer ────────────
    if output_path is not None:
        builder = PDFReportBuilder(output_path, cfg)
        for fig in figures:
            builder.add_figure(fig)
        builder.build()
        LOGGER.info("[pdf] overview report → %s", output_path)

    return figures

###############################################################################
# Cheatsheet per‑cardinal PDF                                                #
###############################################################################

def make_cardinal_profiles_report(
    df: pd.DataFrame,
    cfg: Config,
    *,
    G: Optional[nx.Graph] = None,
    scores: Optional[pd.Series] = None,
    eig: Optional[dict[str, float]] = None,
    betw: Optional[dict[str, float]] = None,
    bloc_sizes: Optional[pd.Series] = None,
    output_path: str | Path | None = None,
    min_weight: float = 0.3,
    layers: Sequence[str] | None = None,
    limit: int | None = None,
    progress: bool = True,
    mc_results: Optional[pd.DataFrame] = None,
) -> List[Figure]:
    """Assemble *one A4 cheatsheet page per cardinal* and optionally PDF‑persist.

    The cheatsheet heavily re‑uses the building block in
    :func:`py.plots.make_cardinal_profile_figure` so that the visual language
    remains consistent across standalone dashboards and compiled reports.

    Parameters
    ----------
    df
        Input data frame with requisite tag columns.
    cfg
        Active run configuration.
    G
        Optional pre-computed network graph to use (prevents recalculation).
    scores
        Optional pre-computed composite scores to use (prevents recalculation).
    eig
        Optional pre-computed eigenvector centrality metrics (prevents recalculation).
    betw
        Optional pre-computed betweenness centrality metrics (prevents recalculation).
    bloc_sizes
        Optional pre-computed bloc weights (prevents recalculation).
    output_path
        Target PDF path.  When omitted the pages are merely returned.
    min_weight
        Edge‑weight threshold for the L5 network used inside profile pages.
    layers
        Optional iterable of tag column names (defaults to
        ``["L1_Tags", "L3_Tags", "L5_Tags"]``) that will be displayed.
    limit
        Upper bound on number of pages (useful for quick smoke‑tests).
    progress
        Emit a tqdm progress‑bar if ``tqdm`` is import‑able **and** this flag
        is *True* (default).
    """
    # ── Preparation ─────────────────────────────────────────────────────────
    from itertools import islice

    try:
        from tqdm import tqdm  # type: ignore
    except ImportError:  # pragma: no cover
        tqdm = None

    # ── Reuse caller-supplied artefacts where possible -----------------------
    if G is None:
        _, G_calc = render_network(df, min_weight=min_weight)
        G = G_calc
    if eig is None or betw is None:
        eig, betw = _centrality_metrics(G)
    if scores is None:
        scores = composite_centrality_scores(df, G, alpha=cfg.centrality_mix)
    if bloc_sizes is None:
        bloc_sizes = _bloc_weights(df)

    # -----------------------------------------------------------------------
    # Iterate **in ranking order** (scores desc), limited to *limit*
    # -----------------------------------------------------------------------
    if scores is not None:
        ordered_names: List[str] = list(scores.head(limit).index) if limit else list(scores.index)
    else:                                  # fallback – unlikely but safe
        ordered = df["Name"]
        ordered_names = list(ordered.head(limit)) if limit else list(ordered)

    if progress and tqdm is not None:
        ordered_names = tqdm(ordered_names, total=len(ordered_names), desc="Profiles")  # type: ignore

    # one-time reindex for fast row look-ups
    df_by_name = df.set_index("Name")

    figures: List[Figure] = []
    for name in ordered_names:
        row_ser = df_by_name.loc[name]
        fig = make_cardinal_profile_figure(
            name,
            row_ser,
            G,
            scores,
            bloc_sizes,
            eig,
            betw,
            layers=layers,
            layer_weights=cfg.layer_weights,
            mc_results=mc_results,  # Pass Monte Carlo results to the profile generator
        )
        figures.append(fig)

    # ── Pretty PDF with timestamp/footer ────────────
    if output_path is not None:
        builder = PDFReportBuilder(output_path, cfg)
        for fig in figures:
            builder.add_figure(fig)
        builder.build()
        LOGGER.info("[pdf] cardinal profile report → %s", output_path)

    return figures
