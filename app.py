from __future__ import annotations

import os
from io import BytesIO
from typing import Any

import pandas as pd
import streamlit as st

from src.agent import consult_llm
from src.screening import filter_stocks


AGENT_OPTIONS = {
    "DeepSeek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "secret_key": "DEEPSEEK_API_KEY",
    },
    "ChatGPT": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4.1-mini",
        "secret_key": "OPENAI_API_KEY",
    },
    "Kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "model": "kimi-latest",
        "secret_key": "KIMI_API_KEY",
    },
}

DEFAULT_APP_PASSWORD = "123456"


def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets[name]
    except Exception:
        value = os.getenv(name, default)
    return str(value).strip() if value else default


def require_password() -> None:
    app_password = get_secret("APP_PASSWORD")
    valid_passwords = {DEFAULT_APP_PASSWORD}
    if app_password:
        valid_passwords.add(app_password)

    if st.session_state.get("authenticated"):
        return

    with st.sidebar:
        password = st.text_input("访问密码", type="password")
        login = st.button("进入系统", use_container_width=True)

    if not login:
        st.warning("请输入访问密码，然后点击“进入系统”。")
        st.stop()

    if password in valid_passwords:
        st.session_state["authenticated"] = True
        st.rerun()

    st.error("访问密码错误，请重新输入。")
    st.stop()


def to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(records) if records else pd.DataFrame()


def build_excel_download(final_df: pd.DataFrame, screened_df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        final_df.to_excel(writer, sheet_name="Final Picks", index=False)
        screened_df.to_excel(writer, sheet_name="Screened Stocks", index=False)
    return buffer.getvalue()


def render_research_cards(final_df: pd.DataFrame) -> None:
    if final_df.empty:
        return

    st.subheader("大模型投研意见")
    for _, row in final_df.iterrows():
        code = row.get("code", "")
        name = row.get("name", "")
        score = row.get("score", "")
        reason = row.get("reason", "")
        risks = row.get("risks", [])

        title = f"{code} {name}"
        if score != "":
            title = f"{title} | 评分：{score}"

        with st.expander(title, expanded=True):
            st.info(reason or "智能体未返回推荐理由。")
            if isinstance(risks, list) and risks:
                st.write("主要风险：")
                for risk in risks:
                    st.write(f"- {risk}")


def main() -> None:
    st.set_page_config(page_title="AI 智能联动选股系统", layout="wide")

    st.title("AI 智能联动选股系统")
    st.caption("输入硬筛条件后，系统会抓取 A 股数据，完成基础筛选，并调用智能体进行二次投研分析。")
    require_password()

    with st.sidebar:
        st.header("选股参数")
        pe_max = st.slider("最大市盈率 PE", min_value=1.0, max_value=200.0, value=30.0, step=1.0)
        profit_growth_min = st.slider(
            "最低净利润增长率 %",
            min_value=-100.0,
            max_value=300.0,
            value=20.0,
            step=5.0,
        )
        funding_min_wan = st.number_input(
            "主力资金净流入阈值（万元）",
            min_value=-100000.0,
            max_value=1000000.0,
            value=1000.0,
            step=100.0,
        )

        st.header("智能体配置")
        agent_name = st.selectbox("选择智能体", options=list(AGENT_OPTIONS.keys()))
        default_agent = AGENT_OPTIONS[agent_name]
        base_url = st.text_input("API Base URL", value=default_agent["base_url"])
        model = st.text_input("模型名称", value=default_agent["model"])
        st.caption(f"API Key 将从 Streamlit Secrets 的 {default_agent['secret_key']} 读取。")

    col1, col2, col3 = st.columns(3)
    col1.metric("最大 PE", f"{pe_max:.0f}")
    col2.metric("最低净利润增长率", f"{profit_growth_min:.0f}%")
    col3.metric("资金净流入阈值", f"{funding_min_wan:.0f} 万元")

    run = st.button("开始智能选股", type="primary", use_container_width=True)

    if not run:
        st.info("请在左侧设置参数，点击“开始智能选股”运行流程。")
        return

    api_key = get_secret(default_agent["secret_key"])
    if not api_key:
        st.error(f"后台 Secrets 缺少 {default_agent['secret_key']}，请先在 Streamlit Cloud 里配置。")
        return

    funding_min = funding_min_wan * 10000

    with st.spinner("正在分析中..."):
        st.write("正在抓取数据并执行基础筛选...")
        screened_df = filter_stocks(
            pe_max=pe_max,
            profit_growth_min=profit_growth_min,
            funding_min=funding_min,
        )

        if screened_df.empty:
            st.warning("基础筛选后没有符合条件的股票，请放宽筛选条件后重试。")
            return

        st.write(f"基础筛选完成，共 {len(screened_df)} 只候选股票，正在同步给大模型分析...")
        llm_result = consult_llm(
            screened_df,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

    final_df = to_dataframe(llm_result.get("winners", []))
    excel_bytes = build_excel_download(final_df=final_df, screened_df=screened_df)

    st.success("智能选股完成。")

    st.subheader("最终推荐结果")
    st.dataframe(final_df, use_container_width=True, hide_index=True)

    render_research_cards(final_df)

    with st.expander("查看基础筛选候选股票", expanded=False):
        st.dataframe(screened_df, use_container_width=True, hide_index=True)

    summary = llm_result.get("summary")
    if summary:
        st.subheader("整体判断摘要")
        st.info(summary)

    st.download_button(
        label="下载 final_recommendations.xlsx",
        data=excel_bytes,
        file_name="final_recommendations.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
