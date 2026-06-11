from __future__ import annotations

import json
import os
import re
from typing import Any

import pandas as pd
import requests


SYSTEM_PROMPT = """你是一名资深投研专家，擅长 A 股基本面、行业趋势、技术面和资金面交叉分析。

你的任务：
1. 基于用户提供的初筛股票列表，进行二次筛选。
2. 重点分析行业景气度、估值合理性、净利润增长质量、近期涨跌幅、今日主力资金净流入等因素。
3. 不要编造用户没有提供的数据；如果信息不足，请在 reason 或 risks 中明确说明。
4. 输出必须是标准 JSON，不能包含 Markdown、解释性前后缀或代码块。

JSON 输出格式必须严格如下：
{
  "winners": [
    {
      "code": "股票代码",
      "name": "股票名称",
      "score": 0,
      "reason": "推荐理由，说明行业趋势、资金面、估值与业绩增长逻辑",
      "risks": ["主要风险1", "主要风险2"]
    }
  ],
  "rejected": [
    {
      "code": "股票代码",
      "name": "股票名称",
      "reason": "淘汰原因"
    }
  ],
  "summary": "整体判断摘要"
}

约束：
- winners 建议保留 3-5 只股票；如果候选数量很少，可以少于 3 只。
- score 使用 0-100 的整数，越高代表越值得进入后续人工复核。
- 所有股票代码必须来自用户给出的候选列表。
- 这不是投资建议，只是用于量化筛选后的投研辅助排序。"""


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4.1-mini"


def _records_from_stock_data(stock_data: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(stock_data, pd.DataFrame):
        records = stock_data.to_dict(orient="records")
    elif isinstance(stock_data, dict):
        records = [stock_data]
    else:
        records = list(stock_data)

    if not records:
        raise ValueError("stock_data 不能为空")

    return records


def _clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, float):
        return round(value, 4)
    return value


def format_stock_data_as_text(
    stock_data: pd.DataFrame | list[dict[str, Any]] | dict[str, Any],
    max_stocks: int = 25,
) -> str:
    """Convert screened stock rows into compact text for an LLM prompt."""
    records = _records_from_stock_data(stock_data)[:max_stocks]
    normalized: list[dict[str, Any]] = []

    field_map = {
        "code": "股票代码",
        "name": "名称",
        "price": "现价",
        "pct_change": "涨跌幅%",
        "pe": "PE",
        "profit_growth": "净利润增长率%",
        "funding_net_inflow": "今日主力资金净流入额",
        "report_date": "财报期",
    }

    for item in records:
        row: dict[str, Any] = {}
        for source_key, label in field_map.items():
            if source_key in item:
                row[label] = _clean_value(item[source_key])
        normalized.append(row)

    return json.dumps(normalized, ensure_ascii=False, indent=2)


def build_user_prompt(stock_data: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> str:
    stock_text = format_stock_data_as_text(stock_data)
    return f"""以下是经过硬条件初筛后的 A 股候选列表。

请你从行业趋势、估值合理性、净利润增长质量、近期涨跌幅、今日主力资金净流入额等维度进行二次筛选。

候选股票数据：
{stock_text}

请只返回标准 JSON。"""


def parse_llm_json(content: str) -> dict[str, Any]:
    """Parse model output into JSON, with a small fallback for accidental fences."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.S)
        if match:
            data = json.loads(match.group(1))
        else:
            start = content.find("{")
            end = content.rfind("}")
            if start < 0 or end <= start:
                raise
            data = json.loads(content[start : end + 1])

    if not isinstance(data, dict):
        raise ValueError("LLM 返回内容不是 JSON object")
    if "winners" not in data or not isinstance(data["winners"], list):
        raise ValueError("LLM JSON 缺少 winners 列表")

    for winner in data["winners"]:
        if not isinstance(winner, dict):
            raise ValueError("winners 中存在非 object 项")
        if not winner.get("code") or not winner.get("reason"):
            raise ValueError("winner 必须包含 code 和 reason")

    return data


def _chat_completions(
    messages: list[dict[str, str]],
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    timeout: int,
    max_tokens: int,
    json_mode: bool,
) -> requests.Response:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    return requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )


def consult_llm(
    stock_data: pd.DataFrame | list[dict[str, Any]] | dict[str, Any],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    timeout: int = 60,
    max_tokens: int = 2000,
    strict_json_mode: bool = True,
) -> dict[str, Any]:
    """
    Consult an OpenAI-compatible LLM and return parsed JSON recommendations.

    Environment variables:
        OPENAI_API_KEY: API key.
        OPENAI_BASE_URL: Optional compatible endpoint, e.g. https://api.deepseek.com/v1.
        OPENAI_MODEL: Optional model name.
    """
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError("缺少 API Key，请设置 OPENAI_API_KEY 或传入 api_key")

    resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    resolved_model = model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(stock_data)},
    ]

    response = _chat_completions(
        messages=messages,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        model=resolved_model,
        temperature=temperature,
        timeout=timeout,
        max_tokens=max_tokens,
        json_mode=strict_json_mode,
    )

    if response.status_code >= 400 and strict_json_mode:
        response = _chat_completions(
            messages=messages,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            model=resolved_model,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
            json_mode=False,
        )

    if response.status_code >= 400:
        raise RuntimeError(f"LLM API 调用失败: HTTP {response.status_code}, {response.text}")

    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    result = parse_llm_json(content)
    result["_meta"] = {
        "model": resolved_model,
        "base_url": resolved_base_url,
        "usage": payload.get("usage"),
    }
    return result
