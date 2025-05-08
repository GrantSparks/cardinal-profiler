"""Promote scalars to *_Tags and add any missing *_Scores columns."""

from __future__ import annotations
import pandas as pd

TAG_SUFFIX = "_Tags"
SCORE_SUFFIX = "_Scores"

_EXTRA_SCALAR_TO_TAG = {
    #  "CSV column name"    → "new *_Tags name"
    "Country":     "Country_Tags",
    "Order":       "Order_Tags",
    "Consistory":  "Consistory_Tags",
    "Office":      "Office_Tags",
}


def augment_tag_layers(df: pd.DataFrame) -> pd.DataFrame:
    """Promote scalars to *_Tags and add missing *_Scores companions."""
    # (1) promote scalars ----------------------------------------------------
    for src, dst in _EXTRA_SCALAR_TO_TAG.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = (
                df[src]
                .fillna("")
                .astype(str)
                .str.strip()        # one tag per row, weight = 1.0
            )

    # (2) guarantee *_Scores companions -------------------------------------
    for col in [c for c in df.columns if c.endswith(TAG_SUFFIX)]:
        score_col = col.replace(TAG_SUFFIX, SCORE_SUFFIX)
        if score_col not in df.columns:
            df[score_col] = 1.0     # default uniform weight

    return df
