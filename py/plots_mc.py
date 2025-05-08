"""Monte Carlo visualization components for the Cardinal Profiler."""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

from py.config import PALETTE
from typing import Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

__all__ = [
    "make_mc_probability_chart",
    "make_mc_win_probability_bars",
]


def make_mc_probability_chart(mc_results: pd.DataFrame, limit: int = 15) -> Figure:
    """
    Create a figure showing Monte Carlo win probabilities and confidence intervals.
    
    Parameters
    ----------
    mc_results : DataFrame
        DataFrame containing Monte Carlo simulation results with WinRate and CI columns
    limit : int, default 15
        Number of top candidates to display
        
    Returns
    -------
    Figure
        Matplotlib figure with win probability chart
    """
    # Get the top N candidates by win rate
    top_results = mc_results.head(limit)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create the main bar chart for win rates
    bars = ax.barh(
        y=range(len(top_results)),
        width=top_results["WinRate"],
        height=0.6,
        color=PALETTE["green"],
        alpha=0.7,
        label="Win Probability"
    )
    
    # Add confidence interval markers if available
    has_ci = "CI_Low" in top_results.columns and "CI_High" in top_results.columns
    if has_ci:
        # Plot error bars for the confidence intervals
        for i, (_, row) in enumerate(top_results.iterrows()):
            ax.plot(
                [row["CI_Low"], row["CI_High"]], 
                [i, i],
                color='black',
                linewidth=2.0
            )
            # Add small caps at the ends of the confidence interval
            ax.plot([row["CI_Low"], row["CI_Low"]], [i-0.1, i+0.1], color='black', linewidth=2.0)
            ax.plot([row["CI_High"], row["CI_High"]], [i-0.1, i+0.1], color='black', linewidth=2.0)
    
    # Set axis labels and titles
    ax.set_title("Monte Carlo Win Probabilities", fontsize=14)
    ax.set_xlabel("Probability", fontsize=12)
    
    # Set y-tick labels to cardinal names
    ax.set_yticks(range(len(top_results)))
    ax.set_yticklabels(top_results.index)
    
    # Add values as text to the right of each bar
    for i, (_, row) in enumerate(top_results.iterrows()):
        if has_ci:
            ax.text(
                row["WinRate"] + 0.01,
                i,
                f"{row['WinRate']:.3f} (95% CI: {row['CI_Low']:.3f}–{row['CI_High']:.3f})",
                va='center',
                fontsize=9
            )
        else:
            ax.text(
                row["WinRate"] + 0.01,
                i,
                f"{row['WinRate']:.3f}",
                va='center'
            )
    
    # Add grid lines for readability
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Make room for the text labels
    plt.tight_layout()
    plt.subplots_adjust(right=0.7)
    
    return fig


def make_mc_win_probability_bars(
    name: str,
    mc_results: pd.DataFrame,
    ax=None,
    show_stats: bool = True
) -> Optional[plt.Axes]:
    """
    Add Monte Carlo win probability visualization for a specific cardinal.
    
    Parameters
    ----------
    name : str
        Cardinal name to highlight
    mc_results : DataFrame
        DataFrame containing Monte Carlo simulation results
    ax : Axes, optional
        Matplotlib axes to plot on. If None, no visualization is created.
    show_stats : bool, default True
        Whether to display additional statistics
        
    Returns
    -------
    Axes or None
        The axes object with the visualization if ax was provided, None otherwise
    """
    if ax is None:
        return None
    
    # Clear any existing content
    ax.clear()
    
    # Check if the cardinal exists in the MC results
    if name not in mc_results.index:
        ax.text(0.5, 0.5, "No Monte Carlo data available", 
                ha='center', va='center', fontsize=10, style='italic')
        ax.axis('off')
        return ax
    
    # Get the cardinal's row
    row = mc_results.loc[name]
    win_rate = row["WinRate"]
    
    # Create a horizontal bar for the win probability
    ax.barh(y=0, width=win_rate, height=0.4, color=PALETTE["green"], alpha=0.7)
    
    # Add confidence interval if available
    has_ci = "CI_Low" in row and "CI_High" in row
    if has_ci:
        ax.plot([row["CI_Low"], row["CI_High"]], [0, 0], color='black', linewidth=2.0)
        ax.plot([row["CI_Low"], row["CI_Low"]], [-0.1, 0.1], color='black', linewidth=2.0)
        ax.plot([row["CI_High"], row["CI_High"]], [-0.1, 0.1], color='black', linewidth=2.0)
    
    # Set axis limits
    ax.set_xlim(0, min(1.0, win_rate * 1.5))  # Allow some space after the bar
    ax.set_ylim(-0.5, 0.5)
    
    # Remove y-axis ticks and labels
    ax.set_yticks([])
    
    # Title and labels
    ax.set_title("Monte Carlo Win Probability", fontsize=10)
    
    # Add stats text
    if show_stats:
        stats_text = f"Win Rate: {win_rate:.3f}"
        if has_ci:
            stats_text += f"\n95% CI: {row['CI_Low']:.3f}–{row['CI_High']:.3f}"
        
        if "WinCount" in row:
            mc_rank = mc_results["WinRate"].rank(ascending=False)[name]
            stats_text += f"\nMC Rank: #{int(mc_rank)}"
            stats_text += f"\nWins: {int(row['WinCount'])}"
        
        ax.text(
            win_rate + 0.02,
            0,
            stats_text,
            va="center",
            fontsize=8,
        )
    
    # Add a grid for readability
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    return ax
