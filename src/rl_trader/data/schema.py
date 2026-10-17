"""OHLCV column schema and adjustment documentation.

Adjusted prices
---------------
Providers such as Yahoo Finance expose both unadjusted ``Close`` and
``Adj Close`` (split/dividend adjusted). This pipeline:

1. **Raw artifacts** store provider columns unchanged, including both
   ``Close`` and ``Adj Close`` when present.
2. **Cleaned artifacts** add adjusted OHLC columns by scaling with
   ``adjustment_factor = Adj Close / Close`` on each row:

   - ``open_adj  = Open  * adjustment_factor``
   - ``high_adj  = High  * adjustment_factor``
   - ``low_adj   = Low   * adjustment_factor``
   - ``close_adj = Adj Close`` (equals ``Close * adjustment_factor``)

   ``Volume`` is left unadjusted (Yahoo volume is not split-adjusted in the
   same way). Missing ``Adj Close`` leaves adjusted columns equal to raw OHLC
   with ``adjustment_factor = 1.0`` and a quality-report warning.

Timestamps are stored timezone-aware in **UTC**. For daily US equity bars the
index timestamp is midnight UTC on the trading session calendar date (the
session date itself — not a converted NYSE close instant).
"""

from __future__ import annotations

from typing import Final

RAW_OHLCV_COLUMNS: Final[tuple[str, ...]] = (
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
)

CLEANED_OHLCV_COLUMNS: Final[tuple[str, ...]] = (
    *RAW_OHLCV_COLUMNS,
    "adjustment_factor",
    "open_adj",
    "high_adj",
    "low_adj",
    "close_adj",
)

ADJUSTMENT_METHOD: Final[str] = (
    "scale_ohlc_by_adj_close_over_close; volume_unadjusted; raw_preserved"
)
