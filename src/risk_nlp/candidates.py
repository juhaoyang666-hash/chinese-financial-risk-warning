from __future__ import annotations

import re
from typing import Iterable

from .schema import dedupe_preserve_order, split_entities


ENTITY_HINT_PATTERN = re.compile(r"实体[:：]\s*([^。！？?\n]+)")
ORG_SUFFIX_PATTERN = re.compile(
    r"[\u4e00-\u9fa5A-Za-z0-9（）()·]{2,40}"
    r"(?:银行|信托|证券|保险|基金|期货|租赁|金融|金服|钱包|网贷|平台|贷|财富|投资|"
    r"资本|创投|资管|集团|公司|控股|股份|支付|交易所|钱庄|理财|消费金融)"
)


def normalize_entity(entity: str) -> str:
    return "".join(str(entity).casefold().split())


def entity_in_text(entity: str, text: str) -> bool:
    key = normalize_entity(entity)
    return bool(key) and key in normalize_entity(text)


def has_explicit_entity_hint(text: str) -> bool:
    return bool(ENTITY_HINT_PATTERN.search(text or ""))


def extract_explicit_entity_hints(text: str) -> list[str]:
    candidates: list[str] = []
    for match in ENTITY_HINT_PATTERN.finditer(text or ""):
        candidates.extend(split_entities(match.group(1)))
    return dedupe_preserve_order([candidate for candidate in candidates if candidate])


def extract_candidate_entities(text: str, known_entities: Iterable[str] | None = None) -> list[str]:
    candidates: list[str] = []
    candidates.extend(extract_explicit_entity_hints(text))
    candidates.extend(match.group(0).strip() for match in ORG_SUFFIX_PATTERN.finditer(text or ""))
    if known_entities:
        candidates.extend(str(entity).strip() for entity in known_entities if str(entity).strip())
    return dedupe_preserve_order([candidate for candidate in candidates if candidate])


def filter_entities_by_candidates(
    entities: Iterable[str],
    text: str,
    candidate_entities: Iterable[str] | None = None,
) -> list[str]:
    candidates = list(candidate_entities or [])
    candidate_keys = {normalize_entity(candidate) for candidate in candidates}
    strict_candidates = has_explicit_entity_hint(text)
    if strict_candidates:
        candidate_keys = {normalize_entity(candidate) for candidate in extract_explicit_entity_hints(text)}
    filtered: list[str] = []
    for entity in entities:
        entity = str(entity).strip()
        if not entity:
            continue
        key = normalize_entity(entity)
        if strict_candidates and candidates and key not in candidate_keys:
            continue
        if not entity_in_text(entity, text):
            continue
        filtered.append(entity)
    return dedupe_preserve_order(filtered)
