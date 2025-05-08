"""Config dataclass, colour palette and constants."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Dict, List

###############################################################################
# Constants & colour grammar                                                  #
###############################################################################

# Tag / score grammar
TAG_SEP: str = ";"
SCORE_SEP: str = ";"
PLACEHOLDER_PATTERNS: tuple[str, ...] = ("_UNK", "_NONE")

# ── Palette -----------------------------------------------------------------
PALETTE: dict[str, str] = {
    "blue": "#4477AA",
    "gold": "#DDCC77",
    "green": "#228833",
    "table_odd": "#f5f5f5",
    "table_even": "#ffffff",
    "header_bg": "#e0e0e0",
    "grid_line": "gray",
}

RADAR_COLOUR: str = PALETTE["blue"]
TOP_TAGS_PER_LAYER: int = 3

DEFAULT_LAYER_WEIGHTS: dict[str, float] = {
    "L1_Tags": 1.0,
    # 2025-05-08 taxonomy audit (v4)
    # • Institutional roles matter more → 1.1
    # • Ideology, networks, factions unchanged
    "L2_Tags": 1.1,   # Curia / residential-see weight bump
    "L3_Tags": 1.2,   # edge-only ideological tags
    "L4_Tags": 0.9,   # soft-signal movements / networks
    "L5_Tags": 1.5,   # decisive conclave alignments
}

###############################################################################
# Dataclass                                                                   #
###############################################################################

@dataclass(slots=True)
class Config:
    """Runtime configuration collected from CLI flags or programmatic use."""

    # Core paths -------------------------------------------------------------
    csv_path: pathlib.Path
    outdir: pathlib.Path = field(default_factory=lambda: pathlib.Path("outputs"))

    # Analysis parameters ----------------------------------------------------
    metric: str = "cosine"
    layer_weights: dict[str, float] = field(
        default_factory=lambda: DEFAULT_LAYER_WEIGHTS.copy()
    )
    layers: List[str] = field(
        default_factory=lambda: ["L1_Tags", "L2_Tags", "L3_Tags", "L4_Tags", "L5_Tags"]
    )
    min_weight: float = 0.3  # cardinal↔tag edge‑weight floor

    # Ranking parameters -----------------------------------------------------
    centrality_mix: float = 0.7  # α blend between eigenvector & betweenness
    rank_by: str = "composite"  # {"composite", "centrality"}
    top_n: int = 15

    # ───────────────────────────────────────────────────────────────────
    # Back-compat: some legacy code still expects cfg.alpha.  Provide a
    # read-only alias so those call-sites don't raise AttributeError.
    # ───────────────────────────────────────────────────────────────────
    @property
    def alpha(self) -> float:            # noqa: D401 (simple alias)
        """Alias for :pyattr:`centrality_mix` (kept for backward-compat)."""
        return self.centrality_mix

    # I/O toggles ------------------------------------------------------------
    no_show: bool = False  # suppress GUI popups
    no_png: bool = False   # disable PNG export
    no_html: bool = False  # disable HTML outputs

    # Simulation -------------------------------------------------------------
    mc: int | str = 0      # Monte‑Carlo ballots (0 = skip, 'auto' = auto-scale)
    seed: int = 42         # RNG seed

    # Cheatsheet -------------------------------------------------------------
    cheatsheet: bool = True  # generate PDF report

    # Bloc‑lift parameters ---------------------------------------------
    bloc_layer: str = "L5_Tags"
    bloc_gamma: float = 1.0


__all__ = [
    "Config",
    "TAG_SEP",
    "SCORE_SEP",
    "PLACEHOLDER_PATTERNS",
    "PALETTE",
    "RADAR_COLOUR",
    "TOP_TAGS_PER_LAYER",
    "DEFAULT_LAYER_WEIGHTS",
]
