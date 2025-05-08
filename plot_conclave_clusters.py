#!/usr/bin/env python3
"""Load a profiled CSV, build the multi-layer graph, then delegate plotting and
report generation to :pymod:`py.plots` and :pymod:`py.reports`."""

from __future__ import annotations

import logging
import datetime as _dt
from pathlib import Path

import matplotlib as mpl
import importlib
importlib.import_module("matplotlib.style")  # ensure sub‑module import
mpl.style.use(str(Path(__file__).with_name("style.mplstyle")))

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from py.cli import parse_cli
from py.config import Config
from py.reports import (
    make_overview_report,
    make_cardinal_profiles_report,
)
from py.utils import PDFReportBuilder
from py.preprocess import augment_tag_layers

from py.plots import (
    render_network,
    make_bloc_bar,
    make_anchor_bridge_scatter,
    render_dendrogram,
    make_cardinal_profile_figure,
)
from collections import Counter
from py.utils import (
    to_simple,
    parse_tag_scores,
    TAG_SEP,
    _assert_no_warnings,
)

from py.analysis import (
    build_feature_matrix,
    build_multilayer_graph,
    _bloc_weights,
    centrality_metrics,
)
from py.ranking import (
    compute_scores,
    log_top,
    save_scores,
    run_monte_carlo,
)

LOGGER = logging.getLogger(__name__)


def main() -> None:
    cfg: Config = parse_cli()

    df = pd.read_csv(cfg.csv_path)
    df = augment_tag_layers(df)      # promote Country/Order/… to *_Tags

    # The CSV already contains only valid electors, so no eligibility filter.

    # Get tag columns for later use
    tag_cols   = [c for c in df.columns if c.endswith("_Tags")]
    score_cols = [c.replace("_Tags", "_Scores") for c in tag_cols]

    out_dir: Path = cfg.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # every *_Tags column that truly exists -------------------------------
    discovered = [c for c in df.columns if c.endswith("_Tags")]

    # keep CLI order but drop ones that are missing -----------------------
    base_layers = [c for c in cfg.layers if c in discovered]

    # append any previously unknown *_Tags columns ------------------------
    graph_layers = base_layers + [c for c in discovered if c not in base_layers]

    # guarantee a default weight for every layer --------------------------
    for col in graph_layers:
        cfg.layer_weights.setdefault(col, 1.0)

    # legacy "extra L3" path kept for back‑compat (remove in v3)
    min_weights = {layer: cfg.min_weight for layer in graph_layers}
    G = build_multilayer_graph(
        df, graph_layers,
        min_weights=min_weights,
        layer_weights=cfg.layer_weights,
    )
    
    # Collapse the *multi-layer* graph exactly once and re-use it everywhere
    G_simple = to_simple(G)
    
    # Calculate the eigenvector centrality and betweenness
    eig, betw = centrality_metrics(G_simple)
    
    # Composite/centrality scores
    scores = compute_scores(
        df,
        G_simple,
        rank_by=cfg.rank_by,
        alpha=cfg.centrality_mix,
        eig=eig,
        betw=betw,
        bloc_layer=cfg.bloc_layer,
        bloc_gamma=cfg.bloc_gamma,
    )
    
    # Log the top-ranked cardinals
    log_top(scores, top_n=cfg.top_n)
    
    # Generate timestamp for all output files
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Save scores to CSV and HTML
    csv_path, html_path = save_scores(
        scores,
        outdir=cfg.outdir,
        ts=ts,
        no_html=cfg.no_html,
    )
    
    # Create dendrogram
    X, _ = build_feature_matrix(
        df,
        cfg.layer_weights,
    )
    labels = df.get("Name", df.index).astype(str).tolist()
    # region labels for dendrogram colouring ------------------------------------
    region_src = (
        df["L1_Tags"]
        if "L1_Tags" in df.columns
        else df.get("Country_Tags", pd.Series(["Other"] * len(df)))
    )
    regions = region_src.fillna("").apply(lambda s: s.split(TAG_SEP)[0] or "Other").tolist()

        # Guard-rail: all-zero feature vectors explode cosine pdist → NaNs.
    # Drop those rows *only* for the dendrogram (they remain elsewhere).
    row_norms = np.linalg.norm(X, axis=1)
    keep = row_norms > 0
    if not keep.all():
        LOGGER.warning(
            "Omitting %d cardinal(s) with no tag signal from dendrogram.",
            int((~keep).sum()),
        )
        X = X[keep]
        labels = [lbl for lbl, k in zip(labels, keep) if k]
        regions = [reg for reg, k in zip(regions, keep) if k]

    fig_den = _assert_no_warnings(
        render_dendrogram,
        X,
        labels,
        regions,
        "Cardinals cluster map",
        cfg.metric,
    )
    
    # Save dendrogram PNG
    if not cfg.no_png:
        den_png = cfg.outdir / f"dendrogram_{ts}.png"
        fig_den.savefig(den_png, dpi=250, bbox_inches="tight")
        LOGGER.info("[png] dendrogram → %s", den_png.name)
    if not cfg.no_show:
        plt.show()
    plt.close(fig_den)
    
    net_html = None if cfg.no_html else cfg.outdir / f"L5_network_{ts}.html"
    fig_net, _ = render_network(df, cfg.min_weight, html_path=net_html)
    
    # Save network PNG
    if not cfg.no_png:
        net_png = cfg.outdir / f"network_{ts}.png"
        fig_net.savefig(net_png, dpi=300, bbox_inches="tight")
        LOGGER.info("[png] network → %s", net_png.name)
    if not cfg.no_show:
        plt.show()
    plt.close(fig_net)
    
    # Bloc head‑counts – used by the strength bar‑chart
    bloc_counts = Counter()
    for _, row in df.iterrows():
        for tag, s in parse_tag_scores(row.get("L5_Tags", ""), row.get("L5_Scores", None)):
            if s >= cfg.min_weight:
                bloc_counts[tag] += 1
    bloc_series = pd.Series(bloc_counts).sort_values(ascending=False)
    bloc_sizes  = _bloc_weights(df)                # ← needed by profile pages
    fig_bloc = _assert_no_warnings(make_bloc_bar, bloc_series)
    
    # Save bloc strength PNG
    if not cfg.no_png:
        bloc_png = cfg.outdir / f"bloc_strength_{ts}.png"
        fig_bloc.savefig(bloc_png, dpi=300, bbox_inches="tight")
        LOGGER.info("[png] bloc strength → %s", bloc_png.name)
    if not cfg.no_show:
        plt.show()
    plt.close(fig_bloc)
    
    # Create anchor-bridge scatter plot
    fig_scatter = _assert_no_warnings(
        make_anchor_bridge_scatter, df, G_simple, scores,
        cfg=cfg, eig=eig, betw=betw
    )
    
    # Save scatter PNG
    if not cfg.no_png:
        sc_png = cfg.outdir / f"scatter_{ts}.png"
        fig_scatter.savefig(sc_png, dpi=300, bbox_inches="tight")
        LOGGER.info("[png] scatter → %s", sc_png.name)
    if not cfg.no_show:
        plt.show()
    plt.close(fig_scatter)
    
    # Run Monte Carlo simulation if requested (auto or numeric value > 0)
    sim_df = None
    if cfg.mc != 0:  # Handle both 'auto' and positive integers
        sim_df = run_monte_carlo(
            scores,
            mc_iter=cfg.mc,
            seed=cfg.seed,
            top_n=cfg.top_n,
            outdir=cfg.outdir,
            ts=ts,
        )
        
        # Update the scores CSV/HTML to include Monte Carlo results
        csv_path, html_path = save_scores(
            scores,
            outdir=cfg.outdir,
            ts=ts,
            no_html=cfg.no_html,
            mc_results=sim_df,
        )

    # ───────────────────────────────────────────────────────────────────
        # Cheatsheet PDF
    overview_figs = make_overview_report(
        df,
        cfg,
        G=G_simple,
        fig_net=fig_net,
        scores=scores,
        eig=eig,
        betw=betw,
        bloc_series=bloc_series,
        fig_den=fig_den,
        min_weight=cfg.min_weight,
        mc_results=sim_df,
    )

    profile_figs = make_cardinal_profiles_report(
        df,
        cfg,
        G=G_simple,
        scores=scores,
        eig=eig,
        betw=betw,
        bloc_sizes=bloc_sizes,
        min_weight=cfg.min_weight,
        layers=cfg.layers,
        limit=cfg.top_n,
    )

    if cfg.cheatsheet:
        cheatsheet_pdf = out_dir / f"cheatsheet_{ts}.pdf"
        builder = PDFReportBuilder(cheatsheet_pdf, cfg)
        for fig in overview_figs + profile_figs:
            builder.add_figure(fig)
        builder.build()
        LOGGER.info("[pdf] cheatsheet → %s", cheatsheet_pdf.name)

if __name__ == "__main__":
    main()
