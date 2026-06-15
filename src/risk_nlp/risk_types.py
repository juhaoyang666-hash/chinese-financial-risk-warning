from __future__ import annotations

import re
from dataclasses import dataclass


RISK_TYPES = [
    "诈骗/非法集资",
    "兑付/逾期风险",
    "监管/处罚风险",
    "经营异常",
    "财务恶化",
    "投诉纠纷",
    "洗钱/涉黑等严重违法",
]

RISK_KEYWORDS = {
    "诈骗/非法集资": ["诈骗", "非法集资", "非法吸收公众存款", "集资诈骗", "欺诈发行", "套路贷"],
    "兑付/逾期风险": ["逾期", "兑付", "提现困难", "无法提现", "不回本金", "爆雷", "资金链断裂", "清盘", "良性退出"],
    "监管/处罚风险": ["监管", "处罚", "通报", "立案", "证监会", "银监", "经侦", "公安", "风险提示", "备案"],
    "经营异常": ["失联", "跑路", "停业", "清退", "破产", "退出", "关停", "法人变更"],
    "财务恶化": ["亏损", "净利润下降", "债务危机", "减值", "资不抵债", "暴跌", "下滑"],
    "投诉纠纷": ["投诉", "维权", "纠纷", "受害人", "出借人", "拒绝", "讨债"],
    "洗钱/涉黑等严重违法": ["洗钱", "涉黑", "黑恶", "犯罪", "刑事", "高利贷", "夜总会"],
}

RISK_TYPE_SEVERITY = {
    "诈骗/非法集资": "critical",
    "洗钱/涉黑等严重违法": "critical",
    "兑付/逾期风险": "high",
    "监管/处罚风险": "high",
    "经营异常": "high",
    "财务恶化": "medium",
    "投诉纠纷": "medium",
}

SEVERITY_SCORES = {
    "low": 0.25,
    "medium": 0.55,
    "high": 0.78,
    "critical": 1.0,
}

SOURCE_RELIABILITY = {
    "regulator": 1.0,
    "news": 0.8,
    "social": 0.55,
    "unknown": 0.65,
}


@dataclass(frozen=True)
class RiskTypeMatch:
    risk_type: str
    severity: str
    keyword: str


def infer_source_type(text: str) -> str:
    text = text or ""
    if any(word in text for word in ["证监会", "银监", "公安", "法院", "检察", "监管", "处非办", "经侦"]):
        return "regulator"
    if any(word in text for word in ["记者", "报道", "公告", "中新经纬", "金融界", "基金报", "新华社"]):
        return "news"
    if any(word in text for word in ["微博", "超话", "@", "全文：", "http://m.weibo"]):
        return "social"
    return "unknown"


def infer_risk_type(text: str) -> RiskTypeMatch:
    text = text or ""
    best: RiskTypeMatch | None = None
    for risk_type in RISK_TYPES:
        for keyword in RISK_KEYWORDS[risk_type]:
            if keyword in text:
                return RiskTypeMatch(risk_type, RISK_TYPE_SEVERITY[risk_type], keyword)
            if re.search(re.escape(keyword), text, flags=re.IGNORECASE):
                best = RiskTypeMatch(risk_type, RISK_TYPE_SEVERITY[risk_type], keyword)
    return best or RiskTypeMatch("投诉纠纷", "medium", "")


def extract_evidence(text: str, entity: str = "", keyword: str = "", max_chars: int = 96) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    pieces = re.split(r"(?<=[。！？!?])|[\n\r]+", text)
    for piece in pieces:
        if entity and entity not in piece:
            continue
        if keyword and keyword not in piece:
            continue
        return piece.strip()[:max_chars]
    for piece in pieces:
        if entity and entity in piece:
            return piece.strip()[:max_chars]
    for piece in pieces:
        if keyword and keyword in piece:
            return piece.strip()[:max_chars]
    return text[:max_chars]


def severity_score(severity: str) -> float:
    return SEVERITY_SCORES.get(severity, SEVERITY_SCORES["medium"])


def source_reliability_score(source_type: str) -> float:
    return SOURCE_RELIABILITY.get(source_type, SOURCE_RELIABILITY["unknown"])
