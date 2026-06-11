from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import pandas as pd


CANONICAL_COLUMNS = [
    "code",
    "name",
    "price",
    "pct_change",
    "pe",
    "profit_growth",
    "funding_net_inflow",
    "report_date",
]


def _require_akshare():
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError(
            "缺少 akshare，请先安装依赖：pip install -r requirements.txt"
        ) from exc
    return ak


def _normalize_code(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().split(".")[0].zfill(6)


def _to_number(series: pd.Series) -> pd.Series:
    text = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .replace({"": None, "-": None, "--": None, "None": None, "nan": None})
    )
    multiplier = pd.Series(1.0, index=text.index)
    multiplier = multiplier.mask(text.str.endswith("亿", na=False), 100_000_000)
    multiplier = multiplier.mask(text.str.endswith("万", na=False), 10_000)
    text = text.str.replace("亿", "", regex=False).str.replace("万", "", regex=False)
    return pd.to_numeric(text, errors="coerce") * multiplier


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError(f"数据缺少必要字段，候选字段={list(candidates)}，实际字段={list(df.columns)}")


def _latest_report_dates(today: date | None = None) -> list[str]:
    current = today or date.today()
    quarters = [(3, 31), (6, 30), (9, 30), (12, 31)]
    dates: list[str] = []
    for year in range(current.year, current.year - 3, -1):
        for month, day in reversed(quarters):
            quarter_date = date(year, month, day)
            if quarter_date <= current:
                dates.append(quarter_date.strftime("%Y%m%d"))
    return dates


@dataclass(slots=True)
class AkShareAStockProvider:
    """Fetch and normalize A-share realtime, profit-growth, and fund-flow data."""

    max_report_date_attempts: int = 8

    def fetch_all(self) -> pd.DataFrame:
        spot = self.fetch_spot()
        profit = self.fetch_profit_growth()
        funding = self.fetch_funding_flow()

        df = spot.merge(profit, on="code", how="left")
        df = df.merge(funding, on="code", how="left", suffixes=("", "_funding"))

        if "name_funding" in df.columns:
            df["name"] = df["name"].fillna(df["name_funding"])
            df = df.drop(columns=["name_funding"])

        return df[CANONICAL_COLUMNS].sort_values("code").reset_index(drop=True)

    def fetch_spot(self) -> pd.DataFrame:
        ak = _require_akshare()
        raw = ak.stock_zh_a_spot_em()

        code_col = _first_existing_column(raw, ["代码", "股票代码"])
        name_col = _first_existing_column(raw, ["名称", "股票简称"])
        price_col = _first_existing_column(raw, ["最新价", "现价"])
        pct_col = _first_existing_column(raw, ["涨跌幅", "今日涨跌幅"])
        pe_col = _first_existing_column(raw, ["市盈率-动态", "市盈率", "市盈率(TTM)"])

        return pd.DataFrame(
            {
                "code": raw[code_col].map(_normalize_code),
                "name": raw[name_col].astype(str).str.strip(),
                "price": _to_number(raw[price_col]),
                "pct_change": _to_number(raw[pct_col]),
                "pe": _to_number(raw[pe_col]),
            }
        )

    def fetch_profit_growth(self) -> pd.DataFrame:
        ak = _require_akshare()
        last_error: Exception | None = None

        for report_date in _latest_report_dates()[: self.max_report_date_attempts]:
            try:
                raw = ak.stock_em_lrb(date=report_date)
            except Exception as exc:  # AkShare remote endpoints can be temporarily unstable.
                last_error = exc
                continue

            if raw is None or raw.empty:
                continue

            code_col = _first_existing_column(raw, ["股票代码", "代码"])
            growth_col = _first_existing_column(raw, ["净利润同比", "净利润增长率", "净利润同比增长"])

            return pd.DataFrame(
                {
                    "code": raw[code_col].map(_normalize_code),
                    "profit_growth": _to_number(raw[growth_col]),
                    "report_date": report_date,
                }
            )

        raise RuntimeError("未能获取最近财报利润表数据") from last_error

    def fetch_funding_flow(self) -> pd.DataFrame:
        ak = _require_akshare()
        raw = ak.stock_individual_fund_flow_rank(indicator="今日")

        code_col = _first_existing_column(raw, ["代码", "股票代码"])
        name_col = _first_existing_column(raw, ["名称", "股票简称"])
        funding_col = _first_existing_column(
            raw,
            ["今日主力净流入-净额", "主力净流入-净额", "今日主力净流入"],
        )

        return pd.DataFrame(
            {
                "code": raw[code_col].map(_normalize_code),
                "name": raw[name_col].astype(str).str.strip(),
                "funding_net_inflow": _to_number(raw[funding_col]),
            }
        )
