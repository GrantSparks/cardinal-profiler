"""High-level helpers that wrap analysis into user-friendly score vectors."""

from __future__ import annotations

import logging
import pathlib
from typing import Tuple

import pandas as pd
import networkx as nx

from .analysis import (
    centrality_ranking,
    composite_centrality_scores,
    monte_carlo_sim,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "compute_scores",
    "log_top",
    "save_scores",
    "run_monte_carlo",
]


def compute_scores(
    df: pd.DataFrame,
    G: nx.Graph,
    *,
    rank_by: str = "composite",
    alpha: float = 0.7,
    eig: dict[str, float] | None = None,
    betw: dict[str, float] | None = None,
    bloc_layer: str = "L5_Tags",
    bloc_gamma: float = 1.0,
) -> pd.Series:
    """
    Compute a 1‑D score vector for each cardinal.

    Parameters
    ----------
    df : DataFrame
        Raw CSV loaded into a dataframe.
    G : Graph
        Multi‑layer projection graph.
    rank_by : {'composite', 'centrality'}, default 'composite'
        Which algorithm to apply.
    alpha : float, default 0.5
        Blend factor passed through to the analysis helpers.
    eig : dict, optional
        Pre-computed eigenvector centrality metrics.
    betw : dict, optional
        Pre-computed betweenness centrality metrics.

    Returns
    -------
    Series
        Index = cardinal names; values = scores sorted descending.
    """
    if rank_by == "centrality":
        scores = centrality_ranking(G, alpha=alpha, eig=eig, betw=betw)
    else:
        scores = composite_centrality_scores(
            df, G, alpha=alpha, eig=eig, betw=betw,
            bloc_layer=bloc_layer, bloc_gamma=bloc_gamma,
        )

    # ensure the result is sorted descending
    return scores.sort_values(ascending=False)


def log_top(scores: pd.Series, *, top_n: int = 10, logger: logging.Logger | None = None) -> None:
    """Pretty‑print the first *top_n* items of *scores* to the log."""
    logger = logger or LOGGER
    logger.info("Ranking computed – top %d:", top_n)
    for i, (name, sc) in enumerate(scores.head(top_n).items(), 1):
        logger.info("%2d. %-25s %.3f", i, name, sc)


def save_scores(
    scores: pd.Series,
    outdir: pathlib.Path,
    ts: str,
    *,
    no_html: bool = False,
    mc_results: pd.DataFrame = None,
) -> Tuple[pathlib.Path, pathlib.Path | None]:
    """
    Persist the *scores* to CSV (and optionally a pretty HTML table).
    
    If Monte Carlo results are provided, they will be integrated into the output.
    
    Parameters
    ----------
    scores : Series
        Cardinal scores to save
    outdir : Path
        Output directory
    ts : str
        Timestamp for filenames
    no_html : bool, default False
        Whether to skip HTML output
    mc_results : DataFrame, optional
        Monte Carlo simulation results to include
        
    Returns
    -------
    Tuple[Path, Optional[Path]]
        Paths to the CSV and HTML outputs (if HTML was generated)
    """
    # Start with the composite scores
    output_df = scores.rename("CompositeScore").to_frame()
    
    # Add Monte Carlo results if available
    if mc_results is not None and not mc_results.empty:
        # Add key Monte Carlo columns to the main output
        output_df["WinRate"] = mc_results.reindex(output_df.index).get("WinRate", pd.Series(dtype=float))
        
        if "CI_Low" in mc_results.columns and "CI_High" in mc_results.columns:
            output_df["CI_Low"] = mc_results.reindex(output_df.index).get("CI_Low", pd.Series(dtype=float))
            output_df["CI_High"] = mc_results.reindex(output_df.index).get("CI_High", pd.Series(dtype=float))
    
    # Save to CSV
    csv_path = outdir / f"ranking_{ts}.csv"
    output_df.to_csv(csv_path, header=True)
    LOGGER.info("[csv] ranking → %s", csv_path.name)

    html_path: pathlib.Path | None = None
    if not no_html:
        # Create a styled HTML version
        styled = output_df.style
        
        # Add gradient styling for the scores columns
        styled = styled.background_gradient(cmap="YlGn", vmin=0, subset=["CompositeScore"])
        
        # If we have Monte Carlo results, style those columns too
        if "WinRate" in output_df.columns:
            styled = styled.background_gradient(cmap="OrRd", vmin=0, vmax=1.0, subset=["WinRate"])
            
        # Format the numeric columns
        format_dict = {
            "CompositeScore": "{:.3f}",
            "WinRate": "{:.3f}",
            "CI_Low": "{:.3f}",
            "CI_High": "{:.3f}"
        }
        styled = styled.format(format_dict)
        
        html_path = outdir / f"ranking_{ts}.html"
        styled.to_html(html_path)
        LOGGER.info("[html] ranking table → %s", html_path.name)
    return csv_path, html_path


def run_monte_carlo(
    scores: pd.Series,
    *,
    mc_iter: int | str,
    seed: int | None,
    top_n: int,
    outdir: pathlib.Path,
    ts: str,
    logger: logging.Logger | None = None,
):
    """
    Execute Monte‑Carlo conclave simulations and dump results.
    
    Parameters
    ----------
    scores : pd.Series
        Cardinal scores used to determine ballot probabilities
    mc_iter : int or str
        Number of Monte Carlo ballots to simulate, or 'auto' to scale based on dataset size
    seed : int or None
        Random number generator seed for reproducibility
    top_n : int
        Number of top candidates to display in logs
    outdir : pathlib.Path
        Directory to save output files
    ts : str
        Timestamp string for output file names
    logger : Logger, optional
        Logger instance to use
        
    Returns
    -------
    DataFrame
        Simulation results with WinCount, WinRate, and CI bounds
    """
    logger = logger or LOGGER
    
    # Handle 'auto' mode - scale based on number of cardinals (1000 × len)
    if mc_iter == 'auto':
        n_cardinals = len(scores)
        mc_iter = 1_000 * n_cardinals
        logger.info("Auto-scaled Monte Carlo simulation: %d ballots (%d cardinals × 1000)", 
                    mc_iter, n_cardinals)
    
    # Skip if disabled
    if mc_iter <= 0:
        return None

    # Run the simulation with credible intervals
    sim_df = monte_carlo_sim(scores, mc_iter, seed)
    
    # Save to separate CSV
    csv_sim = outdir / f"simulation_{ts}.csv"
    sim_df.to_csv(csv_sim)
    logger.info("[csv] MC simulation → %s", csv_sim.name)

    # Log the top winners with their win rates
    logger.info("Top MC winners:")
    for i, (n, r) in enumerate(sim_df.head(top_n).iterrows(), 1):
        ci_str = f" (95% CI: {r.CI_Low:.3f}-{r.CI_High:.3f})" if "CI_Low" in sim_df.columns else ""
        logger.info("%2d. %-25s  wins=%d  rate=%.3f%s", 
                    i, n, r.WinCount, r.WinRate, ci_str)

    # Log the most likely winner    
    winner = sim_df.index[0]
    win_rate = sim_df.iloc[0]["WinRate"]
    logger.info("🏆 MC favourite: %s (p≈%.2f)", winner, win_rate)
    
    return sim_df
