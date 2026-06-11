from __future__ import annotations

import pandas as pd

from src.data_provider import AkShareAStockProvider


def filter_stocks(
    pe_max: float,
    profit_growth_min: float,
    funding_min: float,
) -> pd.DataFrame:
    """
    Fetch all current A-share data and apply hard filters.

    Args:
        pe_max: 最大动态市盈率 PE，例如 30。
        profit_growth_min: 最低净利润同比增长率，单位为百分比，例如 20 表示 20%。
        funding_min: 今日主力资金最低净流入额，单位为元，例如 10000000 表示 1000 万元。

    Returns:
        A Pandas DataFrame with columns:
        code, name, price, pct_change, pe, profit_growth, funding_net_inflow, report_date.
    """
    provider = AkShareAStockProvider()
    df = provider.fetch_all()

    required = ["pe", "profit_growth", "funding_net_inflow"]
    clean = df.dropna(subset=required).copy()

    mask = (
        (clean["pe"] > 0)
        & (clean["pe"] <= pe_max)
        & (clean["profit_growth"] >= profit_growth_min)
        & (clean["funding_net_inflow"] >= funding_min)
    )

    return (
        clean.loc[mask]
        .sort_values(["funding_net_inflow", "profit_growth"], ascending=[False, False])
        .reset_index(drop=True)
    )
