from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.agent import consult_llm
from src.screening import filter_stocks


DEFAULT_OUTPUT = "final_recommendations.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A 股自动筛选与智能体二次筛选")
    parser.add_argument("--config", type=str, default="", help="可选：JSON 配置文件路径")
    parser.add_argument("--pe-max", type=float, default=None, help="最大动态市盈率 PE")
    parser.add_argument(
        "--profit-growth-min",
        type=float,
        default=None,
        help="最低净利润同比增长率，单位：%%",
    )
    parser.add_argument(
        "--funding-min",
        type=float,
        default=None,
        help="今日主力资金最低净流入额，单位：元",
    )
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Excel 输出路径")
    parser.add_argument("--api-key", type=str, default=None, help="可选：LLM API Key")
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="可选：OpenAI 兼容 API 地址，例如 https://api.deepseek.com/v1",
    )
    parser.add_argument("--model", type=str, default=None, help="可选：模型名称")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    if not path:
        return {}

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def merge_settings(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    screening_config = config.get("screening", {})
    llm_config = config.get("llm", {})

    settings = {
        "pe_max": args.pe_max if args.pe_max is not None else screening_config.get("pe_max"),
        "profit_growth_min": (
            args.profit_growth_min
            if args.profit_growth_min is not None
            else screening_config.get("profit_growth_min")
        ),
        "funding_min": (
            args.funding_min if args.funding_min is not None else screening_config.get("funding_min")
        ),
        "output": args.output or config.get("output", DEFAULT_OUTPUT),
        "api_key": args.api_key if args.api_key is not None else llm_config.get("api_key"),
        "base_url": args.base_url if args.base_url is not None else llm_config.get("base_url"),
        "model": args.model if args.model is not None else llm_config.get("model"),
    }

    missing = [
        key
        for key in ["pe_max", "profit_growth_min", "funding_min"]
        if settings.get(key) is None
    ]
    if missing:
        raise ValueError(f"缺少必要筛选参数: {', '.join(missing)}")

    return settings


def _records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(records) if records else pd.DataFrame()


def _safe_params_for_export(settings: dict[str, Any], screened_count: int) -> pd.DataFrame:
    exported = {
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pe_max": settings["pe_max"],
        "profit_growth_min": settings["profit_growth_min"],
        "funding_min": settings["funding_min"],
        "model": settings.get("model") or "",
        "base_url": settings.get("base_url") or "",
        "screened_count": screened_count,
    }
    return pd.DataFrame([exported])


def save_results(
    output_path: str,
    screened_stocks: pd.DataFrame,
    llm_result: dict[str, Any],
    settings: dict[str, Any],
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True) if output.parent != Path(".") else None

    winners = _records_to_frame(llm_result.get("winners", []))
    rejected = _records_to_frame(llm_result.get("rejected", []))
    summary = pd.DataFrame([{"summary": llm_result.get("summary", "")}])
    params = _safe_params_for_export(settings, screened_count=len(screened_stocks))

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        winners.to_excel(writer, sheet_name="Final Picks", index=False)
        rejected.to_excel(writer, sheet_name="Rejected", index=False)
        screened_stocks.to_excel(writer, sheet_name="Screened Stocks", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        params.to_excel(writer, sheet_name="Run Params", index=False)

    return output


def run_pipeline(settings: dict[str, Any]) -> Path:
    print("1/4 正在抓取全量 A 股数据并进行基础筛选...")
    screened_stocks = filter_stocks(
        pe_max=float(settings["pe_max"]),
        profit_growth_min=float(settings["profit_growth_min"]),
        funding_min=float(settings["funding_min"]),
    )
    print(f"基础筛选完成，候选股票数量: {len(screened_stocks)}")

    if screened_stocks.empty:
        llm_result = {
            "winners": [],
            "rejected": [],
            "summary": "基础筛选后无候选股票，未调用智能体。",
        }
    else:
        print("2/4 正在发送候选股票给智能体进行二次筛选...")
        llm_result = consult_llm(
            screened_stocks,
            api_key=settings.get("api_key"),
            base_url=settings.get("base_url"),
            model=settings.get("model"),
        )
        print(f"智能体返回完成，最终胜出股票数量: {len(llm_result.get('winners', []))}")

    print("3/4 正在写入 Excel...")
    output_path = save_results(
        output_path=settings["output"],
        screened_stocks=screened_stocks,
        llm_result=llm_result,
        settings=settings,
    )
    print(f"4/4 完成，结果已保存: {output_path.resolve()}")
    return output_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    settings = merge_settings(args, config)
    run_pipeline(settings)


if __name__ == "__main__":
    main()
