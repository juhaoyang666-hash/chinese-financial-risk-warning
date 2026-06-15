from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "你是金融风控文本分析助手。你的任务是识别金融新闻或社交媒体文本中"
    "是否存在针对金融主体的负面风险信息，并抽取对应负面主体。"
    "你必须只输出一个 JSON 对象，不要输出思考过程、解释、Markdown 或代码块。"
)

NO_NEGATIVE_PATTERNS = [
    r"不包含.*负面",
    r"不存在.*负面",
    r"没有.*负面",
    r"未包含.*负面",
    r"未发现.*负面",
    r"无.*负面",
    r"没有发现",
    r"分析上述文本，?发现没有",
]

POSITIVE_WORD_PATTERNS = [
    r"包含.*负面",
    r"存在.*负面",
    r"负面金融主体",
    r"负面金融实体",
    r"负面主体",
    r"负面实体",
]

ENTITY_PREFIX_PATTERNS = [
    r"^负面金融实体有以下几个",
    r"^负面金融主体包含以下几个",
    r"^文中包含的负面主体",
    r"^负面金融实体",
    r"^负面金融主体",
    r"^负面主体",
    r"^负面实体",
    r"^包含的负面主体",
]


def normalize_spaces(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def is_no_negative(text: str) -> bool:
    compact = normalize_spaces(text)
    return any(re.search(pattern, compact) for pattern in NO_NEGATIVE_PATTERNS)


def split_entities(raw: str) -> list[str]:
    text = normalize_spaces(raw)
    if not text:
        return []

    text = re.split(r"[。.!！?？\n]", text, maxsplit=1)[0]
    text = re.sub(r"^(有|包括|分别为|如下)\s*", "", text)
    pieces = re.split(r"[;；、,，/|]+", text)
    entities: list[str] = []
    for piece in pieces:
        ent = piece.strip()
        ent = re.sub(r"^(和|及|以及|等)\s*", "", ent)
        ent = ent.strip(" \t\r\n:：.。;；,，、[]【】()（）<>《》\"'“”‘’")
        ent = re.sub(r"(等|等等)$", "", ent).strip()
        if not ent:
            continue
        if ent in {"有", "无", "没有", "不存在", "不包含"}:
            continue
        if len(ent) > 80:
            continue
        entities.append(ent)
    return dedupe_preserve_order(entities)


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def extract_entities_from_output(output: str) -> list[str]:
    text = normalize_spaces(output)
    if not text or is_no_negative(text):
        return []

    if "：" in text or ":" in text:
        tail = re.split(r"[:：]", text, maxsplit=1)[1]
    else:
        tail = text
        for pattern in ENTITY_PREFIX_PATTERNS:
            tail = re.sub(pattern, "", tail).strip()

    tail = re.sub(r"^有以下几个", "", tail).strip()
    return split_entities(tail)


def normalize_finnsp_output(output: str) -> dict[str, Any]:
    text = normalize_spaces(output)
    if not text:
        return {"has_negative": False, "entities": []}
    if is_no_negative(text) or text in {"无", "没有", "否"}:
        return {"has_negative": False, "entities": []}
    entities = extract_entities_from_output(text)
    has_negative = bool(entities)
    if not has_negative:
        if text in {"有", "是"} or any(re.search(pattern, text) for pattern in POSITIVE_WORD_PATTERNS):
            has_negative = True
    return {"has_negative": has_negative, "entities": entities}


def build_user_prompt(text: str, instruction: str | None = None) -> str:
    task = instruction or "判断以下文本是否包含金融主体的负面风险信息，并抽取负面主体。"
    return (
        f"{task}\n\n"
        "请严格按 JSON 输出，字段如下：\n"
        "- has_negative: 布尔值，表示文本是否包含负面金融主体。\n"
        "- entities: 字符串数组，列出负面金融主体名称；若不存在则为空数组。\n\n"
        "硬性要求：只输出一个 JSON 对象；不要输出 <think>、分析过程、解释文字、Markdown 代码块或多余前后缀。\n\n"
        "文本：\n"
        f"{text}\n\n"
        '输出示例：{"has_negative": true, "entities": ["主体A", "主体B"]}'
    )


def format_target_json(has_negative: bool, entities: list[str]) -> str:
    return json.dumps(
        {"has_negative": bool(has_negative), "entities": dedupe_preserve_order(entities)},
        ensure_ascii=False,
        sort_keys=True,
    )


def _find_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def clean_model_output(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    return cleaned


def parse_model_output(text: str) -> tuple[dict[str, Any], bool]:
    raw = normalize_spaces(clean_model_output(text))
    json_text = _find_json_object(raw)
    if json_text:
        try:
            parsed = json.loads(json_text)
            has_negative = parsed.get("has_negative", False)
            if isinstance(has_negative, str):
                has_negative = has_negative.strip().lower() in {"true", "1", "yes", "有", "是"}
            entities = parsed.get("entities", [])
            if isinstance(entities, str):
                entities = split_entities(entities)
            elif isinstance(entities, list):
                entities = split_entities(";".join(str(item) for item in entities))
            else:
                entities = []
            risk_events = parsed.get("risk_events", [])
            if isinstance(risk_events, list) and risk_events:
                event_entities = []
                for event in risk_events:
                    if isinstance(event, dict) and event.get("entity"):
                        event_entities.append(str(event["entity"]))
                entities = dedupe_preserve_order(entities + event_entities)
            result = {"has_negative": bool(has_negative), "entities": entities}
            if isinstance(risk_events, list):
                result["risk_events"] = risk_events
            return result, False
        except json.JSONDecodeError:
            pass

    normalized = normalize_finnsp_output(raw)
    return normalized, True


def redact_text(text: str) -> str:
    redacted = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    redacted = re.sub(r"@\w+", "@USER", redacted)
    redacted = re.sub(r"\b\d{7,}\b", "[NUMBER]", redacted)
    return redacted


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
