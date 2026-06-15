from __future__ import annotations

import html

import requests
import streamlit as st


st.set_page_config(page_title="金融舆情风险预警", layout="wide")

API_URL = st.sidebar.text_input("API URL", value="http://127.0.0.1:8000")
HTTP = requests.Session()
HTTP.trust_env = False

STATUS_LABELS = {
    "pending": "待处理",
    "reviewed": "已复核",
    "ignored": "已忽略",
}
REASON_LABELS = {
    "entity_missing_or_filtered": "主体缺失或被过滤",
    "high_risk": "高风险样本",
    "manual_confirmed": "人工确认",
    "人工复核": "人工复核",
}


st.markdown(
    """
    <style>
    #MainMenu, footer, header, .stDeployButton {visibility: hidden;}
    .block-container {
        max-width: 1180px;
        padding-top: 2.0rem;
        padding-bottom: 2.0rem;
    }
    [data-testid="stSidebar"] {
        background: #eef3f8;
        border-right: 1px solid #dbe3ee;
    }
    .app-header {
        border: 1px solid #d8e1ec;
        border-left: 6px solid #2f6fdb;
        border-radius: 10px;
        padding: 22px 24px 20px;
        background: #ffffff;
        margin-bottom: 18px;
        box-shadow: 0 10px 24px rgba(26, 43, 72, 0.06);
    }
    .app-title {
        font-size: 34px;
        line-height: 1.15;
        font-weight: 800;
        color: #182235;
        margin: 0 0 8px 0;
    }
    .app-subtitle {
        color: #52617a;
        font-size: 15px;
        margin: 0 0 14px 0;
    }
    .chip {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: #eef5ff;
        color: #2f6fdb;
        font-size: 13px;
        font-weight: 600;
        margin-right: 8px;
    }
    .chip-green {
        background: #ebf8f2;
        color: #08794f;
    }
    .metric-card {
        border: 1px solid #d9e2ee;
        border-radius: 9px;
        background: #ffffff;
        padding: 14px 16px 13px;
        min-height: 92px;
        box-shadow: 0 8px 18px rgba(26, 43, 72, 0.045);
    }
    .metric-card .label {
        color: #59677f;
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .metric-card .value {
        color: #182235;
        font-size: 31px;
        font-weight: 760;
        letter-spacing: 0;
    }
    .section-label {
        color: #182235;
        font-size: 20px;
        font-weight: 760;
        margin: 16px 0 10px;
    }
    .summary-card {
        border: 1px solid #d9e2ee;
        border-radius: 9px;
        padding: 13px 15px;
        background: #ffffff;
    }
    .summary-card .label {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
    }
    .summary-card .value {
        color: #182235;
        font-size: 25px;
        font-weight: 760;
        margin-top: 5px;
    }
    .risk-card {
        border: 1px solid #d8e1ec;
        border-radius: 10px;
        background: #ffffff;
        padding: 16px 18px;
        margin-top: 8px;
        box-shadow: 0 8px 20px rgba(26, 43, 72, 0.05);
    }
    .risk-card-title {
        color: #182235;
        font-size: 20px;
        font-weight: 800;
        margin-bottom: 10px;
    }
    .risk-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin-bottom: 12px;
    }
    .risk-kv {
        border-radius: 8px;
        background: #f6f8fb;
        padding: 10px 12px;
    }
    .risk-kv span {
        display: block;
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .risk-kv strong {
        color: #182235;
        font-size: 18px;
    }
    .evidence {
        border-left: 4px solid #d95f59;
        background: #fff7f6;
        color: #334155;
        padding: 9px 12px;
        border-radius: 7px;
        font-size: 16px;
    }
    textarea {
        font-size: 17px !important;
        line-height: 1.65 !important;
        color: #182235 !important;
    }
    div[data-testid="stTextArea"] textarea {
        font-size: 17px !important;
        line-height: 1.65 !important;
    }
    .profile-card {
        border: 1px solid #d9e2ee;
        border-radius: 10px;
        background: #ffffff;
        padding: 15px 17px;
        min-height: 210px;
    }
    .profile-card h3 {
        margin: 0 0 10px 0;
        color: #182235;
        font-size: 19px;
    }
    .profile-card p {
        margin: 7px 0;
        color: #334155;
        font-size: 14px;
    }
    div.stButton > button:first-child {
        background: #d9363e;
        color: white;
        border: 0;
        border-radius: 8px;
        padding: 0.55rem 1.1rem;
        font-weight: 700;
    }
    div.stButton > button:first-child:hover {
        background: #bd2630;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    f"""
    <div class="app-header">
      <div class="app-title">中文金融舆情风险智能预警系统</div>
      <p class="app-subtitle">Encoder 初筛、Qwen3-8B QLoRA 主体级复核、风险事件入库与人工审核队列。</p>
      <span class="chip chip-green">真实 GPU 推理</span>
      <span class="chip">FastAPI: {html.escape(API_URL)}</span>
      <span class="chip">Streamlit 审核台</span>
    </div>
    """,
    unsafe_allow_html=True,
)


def format_review_queue(rows: list[dict]) -> list[dict]:
    formatted = []
    for row in rows:
        formatted.append(
            {
                "状态": STATUS_LABELS.get(row.get("status", ""), row.get("status", "")),
                "原因": REASON_LABELS.get(row.get("reason", ""), row.get("reason", "")),
                "主体": row.get("entity") or "待确认",
                "风险分": row.get("risk_score"),
                "风险等级": row.get("risk_level"),
                "文本ID": row.get("text_id"),
            }
        )
    return formatted


def format_risk_events(events: list[dict]) -> list[dict]:
    return [
        {
            "主体": event.get("entity"),
            "风险类型": event.get("risk_type"),
            "等级": event.get("risk_level"),
            "风险分": event.get("risk_score"),
            "置信度": event.get("confidence"),
            "证据": event.get("evidence"),
            "处置": event.get("action"),
        }
        for event in events
    ]


def metric_card(label: str, value: str) -> str:
    return f'<div class="metric-card"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div></div>'


def summary_card(label: str, value: str) -> str:
    return f'<div class="summary-card"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div></div>'


def risk_event_card(event: dict) -> str:
    evidence = html.escape(str(event.get("evidence", "")))
    return f"""
    <div class="risk-card">
      <div class="risk-card-title">{html.escape(str(event.get("entity", "待确认主体")))}</div>
      <div class="risk-grid">
        <div class="risk-kv"><span>风险类型</span><strong>{html.escape(str(event.get("risk_type", "-")))}</strong></div>
        <div class="risk-kv"><span>风险等级</span><strong>{html.escape(str(event.get("risk_level", "-")))}</strong></div>
        <div class="risk-kv"><span>风险分</span><strong>{html.escape(str(event.get("risk_score", "-")))}</strong></div>
        <div class="risk-kv"><span>处置动作</span><strong>{html.escape(str(event.get("action", "-")))}</strong></div>
      </div>
      <div class="evidence">{evidence}</div>
    </div>
    """


def profile_card(profile: dict | None) -> str:
    if not profile:
        return '<div class="profile-card"><h3>主体画像</h3><p>暂无主体画像。</p></div>'
    risk_dist = profile.get("risk_type_distribution", {})
    dist_text = "；".join(f"{key}: {value}" for key, value in risk_dist.items()) or "暂无"
    return f"""
    <div class="profile-card">
      <h3>主体画像：{html.escape(str(profile.get("entity", "-")))}</h3>
      <p><b>事件次数：</b>{html.escape(str(profile.get("event_count", "-")))}</p>
      <p><b>最高风险等级：</b>{html.escape(str(profile.get("max_risk_level", "-")))}</p>
      <p><b>平均风险分：</b>{html.escape(str(round(float(profile.get("avg_risk_score", 0)), 2)))}</p>
      <p><b>风险类型分布：</b>{html.escape(dist_text)}</p>
      <p><b>最近证据：</b>{html.escape(str(profile.get("latest_evidence", "-")))}</p>
    </div>
    """


metric_cols = st.columns(5)
for column, (label, value) in zip(
    metric_cols,
    [
        ("Qwen3 F1", "0.9732"),
        ("Entity-F1", "0.8331"),
        ("实体在文率", "98.64%"),
        ("幻觉率", "1.36%"),
        ("LLM 调用率", "53.2%"),
    ],
):
    column.markdown(metric_card(label, value), unsafe_allow_html=True)

st.markdown('<div class="section-label">样本输入</div>', unsafe_allow_html=True)
input_left, input_right = st.columns([4, 1])
with input_left:
    text = st.text_area(
        "金融新闻/社媒文本",
        label_visibility="collapsed",
        height=150,
        value="小资钱包平台自2018年9月起几乎全部逾期，出借人维权称本金无法兑付。",
    )
with input_right:
    force_llm = st.checkbox("强制 LLM 复核", value=True)
    score_clicked = st.button("风险识别", type="primary", use_container_width=True)

if score_clicked:
    try:
        response = HTTP.post(f"{API_URL}/score", json={"text": text, "force_llm": force_llm}, timeout=600)
        response.raise_for_status()
        st.session_state["last_result"] = response.json()
        st.session_state["last_profile"] = None
        events = st.session_state["last_result"].get("risk_events", [])
        if events:
            profile_response = HTTP.get(f"{API_URL}/entity/{events[0]['entity']}", timeout=30)
            if profile_response.status_code == 200:
                st.session_state["last_profile"] = profile_response.json()
    except Exception as exc:
        st.error(f"风险识别请求失败：{exc}")

result = st.session_state.get("last_result")
if result:
    st.markdown('<div class="section-label">识别结果</div>', unsafe_allow_html=True)
    status_cols = st.columns(4)
    status_cols[0].markdown(summary_card("模型阶段", str(result.get("stage", "-"))), unsafe_allow_html=True)
    status_cols[1].markdown(summary_card("是否风险", "是" if result.get("has_negative") else "否"), unsafe_allow_html=True)
    status_cols[2].markdown(summary_card("编码器置信度", f"{result.get('encoder_confidence', 0):.4f}"), unsafe_allow_html=True)
    status_cols[3].markdown(summary_card("耗时", f"{result.get('latency_sec', 0):.2f}s"), unsafe_allow_html=True)

    events = result.get("risk_events", [])
    if events:
        st.markdown('<div class="section-label">风险事件</div>', unsafe_allow_html=True)
        for event in events:
            st.markdown(risk_event_card(event), unsafe_allow_html=True)
    else:
        st.info("当前样本未生成可入库风险事件。")

    lower_left, lower_right = st.columns([1, 1])
    with lower_left:
        st.markdown(profile_card(st.session_state.get("last_profile")), unsafe_allow_html=True)
    with lower_right:
        st.markdown('<div class="section-label">待审核队列</div>', unsafe_allow_html=True)
        try:
            queue = HTTP.get(f"{API_URL}/review_queue", timeout=30).json()
            st.dataframe(format_review_queue(queue), width="stretch", hide_index=True, height=230)
        except Exception as exc:
            st.info(f"审核队列暂不可用：{exc}")

    with st.expander("原始 JSON 输出", expanded=False):
        st.json(result)
else:
    st.markdown('<div class="section-label">待审核队列</div>', unsafe_allow_html=True)
    try:
        queue = HTTP.get(f"{API_URL}/review_queue", timeout=30).json()
        st.dataframe(format_review_queue(queue), width="stretch", hide_index=True, height=250)
    except Exception as exc:
        st.info(f"审核队列暂不可用：{exc}")

st.markdown('<div class="section-label">服务指标</div>', unsafe_allow_html=True)
try:
    metrics = HTTP.get(f"{API_URL}/metrics", timeout=30).json()
    metric_cols = st.columns(4)
    metric_cols[0].markdown(summary_card("风险事件数", str(metrics.get("risk_event_count", "-"))), unsafe_allow_html=True)
    metric_cols[1].markdown(summary_card("主体画像数", str(metrics.get("entity_profile_count", "-"))), unsafe_allow_html=True)
    metric_cols[2].markdown(summary_card("待审核数", str(metrics.get("pending_review_count", "-"))), unsafe_allow_html=True)
    metric_cols[3].markdown(summary_card("平均风险分", f"{float(metrics.get('avg_risk_score', 0)):.2f}"), unsafe_allow_html=True)
except Exception as exc:
    st.info(f"服务指标暂不可用：{exc}")
