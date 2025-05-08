"""Project-wide, side-effect-free helper utilities."""
from __future__ import annotations

# Std-lib
import datetime as _dt
import inspect
import math
import sys
import warnings
from collections import Counter
from typing import Iterable, Sequence

# Third‑party deps (lightweight) -------------------------------------------
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
from packaging import version

# Project‑local imports (cycle‑safe) ---------------------------------------
from .config import (
    TAG_SEP,
    SCORE_SEP,
    PLACEHOLDER_PATTERNS,
    PALETTE,
)

# Version guards – bail early if minimum deps aren't met.
if version.parse(nx.__version__) < version.parse("3.3"):
    raise RuntimeError(
        f"Cardinal‑Profiler requires networkx ≥ 3.3 (detected {nx.__version__})."
    )
if version.parse(matplotlib.__version__) < version.parse("3.8"):
    raise RuntimeError(
        f"Requires matplotlib ≥ 3.8 for stable colormap API (detected {matplotlib.__version__})."
    )

# Silence very chatty libraries – caller can override via root logger level
for spammer in (
    "matplotlib",
    "matplotlib.font_manager",
    "PIL",
    "PIL.PngImagePlugin",
):
    warnings.filterwarnings("ignore", category=UserWarning, module=spammer)

# Matplotlib ≥ 3.7 convenience one‑liner
matplotlib.set_loglevel("warning")

# Public helpers
__all__ = [
    "_cmap",
    "_is_placeholder",
    "_assert_no_warnings",
    "parse_tag_scores",
    "weighted_project_bipartite",
    "to_simple",
    "PDFReportBuilder",
]

# Compatibility‑safe colormap accessor -------------------------------------
try:  # Matplotlib ≥ 3.7
    _safe_get_cmap = matplotlib.colormaps.get_cmap  # type: ignore[attr-defined]
except AttributeError:  # fallback (<3.7)
    _safe_get_cmap = plt.cm.get_cmap  # type: ignore[assignment]

def _cmap(name: str, n: int):
    """
    Return *n* discrete colours from **name** colormap, hiding «lut»/signature
    differences across Matplotlib versions.
    """
    try:                                    # Matplotlib ≥ 3.7
        return _safe_get_cmap(name, lut=n)  # type: ignore[arg-type]
    except TypeError:                       # older – manual resample
        base = _safe_get_cmap(name)
        if hasattr(base, "resampled"):
            return base.resampled(n)        # type: ignore[attr-defined]
        colours = base(np.linspace(0, 1, n))
        return matplotlib.colors.ListedColormap(colours, name=f"{name}_{n}")

# PDF cheatsheet builder ---------------------------------------------------
class PDFReportBuilder:
    """Collects :class:`matplotlib.figure.Figure` objects and writes a multi‑page PDF."""

    def __init__(self, out_path: "str | os.PathLike[str]", cfg) -> None:  # type: ignore[valid-type]
        self.out_path = out_path
        self.cfg = cfg
        self.figures: list[Figure] = []

    # ------------------------------------------------------------------ public API
    def add_figure(self, fig: Figure) -> None:
        """Queue a fully‑rendered **fig** for later PDF emission (background forced white)."""
        fig.patch.set_facecolor("white")
        self.figures.append(fig)

    def build(self, *, close: bool = True) -> None:
        """Write all queued figures to disk as a single PDF, adding timestamp and paging."""
        if not self.figures:
            raise RuntimeError("PDFReportBuilder.build() called with no figures queued")

        # Local style context – does NOT leak rcParams globally
        with plt.rc_context():
            try:
                plt.style.use("seaborn-v0_8-whitegrid")
            except (ValueError, OSError):
                plt.style.use("default")  # ensure rcParams are reset on failure

            # Ensure destination directory exists
            from pathlib import Path
            Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)

            with PdfPages(self.out_path) as pdf:
                total = len(self.figures)
                for i, fig in enumerate(self.figures, start=1):
                    # Header timestamp
                    hdr_ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
                    fig.text(
                        0.01,
                        0.985,
                        f"Cardinal Analysis Report – {hdr_ts}",
                        fontsize=9,
                        color="gray",
                        backgroundcolor=PALETTE["header_bg"],
                    )
                    # Separation line
                    line = Line2D(
                        [0, 1],
                        [0.97, 0.97],
                        transform=fig.transFigure,
                        linewidth=0.5,
                        color=PALETTE["grid_line"],
                    )
                    fig.add_artist(line)
                    # Footer page number
                    fig.text(
                        0.5,
                        0.01,
                        f"Page {i} of {total}",
                        ha="center",
                        fontsize=9,
                        color="gray",
                    )
                    pdf.savefig(fig)
                    # Free the canvas unless explicit reuse is requested
                    if close:
                        plt.close(fig)

# Generic CSV‑parsing helpers ----------------------------------------------

def _is_placeholder(tag: str) -> bool:
    """Return *True* if **tag** matches any configured placeholder pattern."""
    return any(str(tag).upper().endswith(p) for p in PLACEHOLDER_PATTERNS)


def _assert_no_warnings(builder, *args, **kwargs):
    """
    Run **builder** but abort only on *unexpected* warnings.
    Glyph-fallback and tight-layout warnings are now tolerated.
    """
    # Only tolerate benign Matplotlib deprecation chatter – surface everything else
    try:
        from scipy.cluster.hierarchy import ClusterWarning
        benign = (matplotlib.MatplotlibDeprecationWarning, ClusterWarning)
    except ImportError:        # SciPy absent – keep original tuple
        benign = (matplotlib.MatplotlibDeprecationWarning,)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fig = builder(*args, **kwargs)
        for warn in w:
            if not issubclass(warn.category, benign):
                print(f"Captured warning: {warn.message}", file=sys.stderr)
                raise RuntimeError("Unexpected warning captured – see stderr.")
    return fig


def parse_tag_scores(
    tag_str: str,
    score_str: str | float | int | None,
    *,
    skip_placeholders: bool = True,
) -> list[tuple[str, float]]:
    """Parse the semi‑colon‑delimited *tag* / *score* strings from the CSV dump.

    If *score_str* is empty or ``NaN`` the implicit weight defaults to ``1.0``.
    When *skip_placeholders* is true, tags matching the configured
    ``PLACEHOLDER_PATTERNS`` are silently dropped (placeholder lines add noise
    to similarity math and their omission is downstream‑safe).
    """
    if pd.isna(tag_str) or not str(tag_str).strip():
        return []

    tags = [
        t.strip()
        for t in str(tag_str).split(TAG_SEP)
        if t.strip() and (not skip_placeholders or not _is_placeholder(t))
    ]

    if score_str is None or (
        isinstance(score_str, float) and math.isnan(score_str)
    ):
        return [(t, 1.0) for t in tags]

    raw_scores = str(score_str).split(SCORE_SEP)
    scores = [float(s) if s.strip() else 1.0 for s in raw_scores]
    scores.extend([1.0] * (len(tags) - len(scores)))  # pad out if fewer scores

    return list(zip(tags, scores[: len(tags)]))

# Lightweight graph helpers ------------------------------------------------

def weighted_project_bipartite(B: nx.Graph, persons: Iterable[str]) -> nx.Graph:
    """
    Weighted 1-mode projection **O(t × k²)** using pre-grouping
    """
    from itertools import combinations

    persons = list(persons)
    G = nx.Graph()
    G.add_nodes_from(persons)

    edge_weights: dict[tuple[str, str], float] = {}

    # tag → [(person, weight), …]
    tag_map: dict[str, list[tuple[str, float]]] = {}
    for p in persons:
        for tag in B.neighbors(p):
            if tag in persons:
                continue
            tag_map.setdefault(tag, []).append((p, B[p][tag]["weight"]))

    for tag, pw in tag_map.items():
        for (u, w_u), (v, w_v) in combinations(pw, 2):
            # canonicalise orientation so (A,B) and (B,A) are treated identically
            key = tuple(sorted((u, v)))
            edge_weights[key] = edge_weights.get(key, 0.0) + w_u * w_v

    G.add_edges_from(((u, v, {"weight": w}) for (u, v), w in edge_weights.items()))
    return G


def to_simple(Gm: nx.MultiGraph) -> nx.Graph:
    """Collapse a :class:`~networkx.MultiGraph` to a simple weighted graph.

    Parallel edges are merged by summing their respective ``weight`` attributes.
    The resulting graph preserves node attributes and is safe for centrality
    algorithms that expect *simple* connectivity semantics.
    """
    G = nx.Graph()
    G.add_nodes_from(Gm.nodes(data=True))

    for u, v, data in Gm.edges(data=True):
        w = data.get("weight", 1.0)
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)
    return G
