"""Constrained natural-language interpretation for inventory questions.

This module converts ordinary business phrasing into a small, approved query
vocabulary. It identifies analytical meaning only; entity values are resolved
and validated against the current item master by ``inventory_assistant``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re


NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15,
}

# Centralized business vocabulary. More specific concepts are checked first.
METRIC_ALIASES = {
    "lead_time_variability": (
        "lead time variability", "lead time variation", "variation in lead time", "variable lead time",
        "inconsistent lead time", "unstable lead time", "lead time inconsistency",
        "lead time consistency", "receipt variability", "inconsistent receipt",
    ),
    "demand_variability": (
        "demand variability", "demand variation", "demand volatility",
        "volatile demand", "unstable demand", "variable demand", "stable demand",
    ),
    "safety_stock_gap": (
        "safety stock gap", "ss gap", "difference in safety stock",
        "increase in safety stock", "more safety stock", "recommended ss exceeds",
        "recommended safety stock than current", "insufficient safety stock",
        "under protected", "underprotected",
    ),
    "rop_gap": (
        "rop gap", "reorder point gap", "difference in reorder point",
        "recommended rop", "reorder point increase",
    ),
    "stockout_risk": (
        "stockout risk", "stock out risk", "riskiest", "highest risk",
        "most exposed", "inventory risk", "risky",
    ),
    "demand_trend": (
        "demand trend", "demand changed", "demand change", "increasing demand",
        "rising demand", "growing demand", "declining demand", "recent demand",
    ),
    "lead_time_mean": (
        "average lead time", "mean lead time", "highest lead time",
        "lowest lead time", "longest lead time", "shortest lead time",
        "high lead time", "long lead time", "lead time",
    ),
    "demand_level": (
        "average demand", "mean demand", "highest demand", "lowest demand",
        "most demand", "demand",
    ),
}

ALLOWED_QUERY_TYPES = {
    "ranking", "comparison", "filter", "metric_lookup", "time_series",
    "explanation", "forecast", "profile", "clarification",
}
ALLOWED_ENTITY_TYPES = {"sku", "supplier"}
ALLOWED_CONDITIONS = {
    "high_demand_variability", "high_lead_time_variability", "increasing_demand",
    "stable_demand", "stable_lead_time", "long_lead_time",
    "positive_safety_stock_gap", "positive_rop_gap", "low_current_safety_stock",
}

RANKING_LANGUAGE = (
    "highest", "lowest", "most", "least", "largest", "smallest",
    "greatest", "biggest", "top", "bottom", "best", "worst", "rank",
)


@dataclass(frozen=True)
class LanguageInterpretation:
    query_type: str
    entity_type: str
    metric: str | None
    metrics: tuple[str, ...] = ()
    direction: str | None = None
    limit: int = 10
    conditions: tuple[str, ...] = ()
    confidence: str = "high"
    clarification: str | None = None


def validate_interpretation_payload(
    payload: dict, *, maximum_rows: int = 15
) -> LanguageInterpretation | None:
    """Validate an optional LLM interpretation against the approved schema."""
    if not isinstance(payload, dict):
        return None
    query_type = payload.get("query_type") or payload.get("intent")
    entity_type = payload.get("entity_type", "sku")
    metric = payload.get("metric")
    if query_type not in ALLOWED_QUERY_TYPES or entity_type not in ALLOWED_ENTITY_TYPES:
        return None
    if metric is not None and metric not in METRIC_ALIASES:
        return None
    raw_metrics = payload.get("metrics") or ([metric] if metric else [])
    if not isinstance(raw_metrics, list) or any(value not in METRIC_ALIASES for value in raw_metrics):
        return None
    direction = payload.get("direction")
    if direction not in {None, "ascending", "descending"}:
        return None
    raw_conditions = payload.get("conditions") or []
    if not isinstance(raw_conditions, list) or any(
        value not in ALLOWED_CONDITIONS for value in raw_conditions
    ):
        return None
    try:
        limit = min(max(int(payload.get("limit", 10)), 1), maximum_rows)
    except (TypeError, ValueError):
        return None
    clarification = payload.get("clarification")
    return LanguageInterpretation(
        query_type=query_type,
        entity_type=entity_type,
        metric=metric,
        metrics=tuple(dict.fromkeys(raw_metrics)),
        direction=direction,
        limit=limit,
        conditions=tuple(dict.fromkeys(raw_conditions)),
        confidence="llm_validated",
        clarification=str(clarification)[:300] if clarification else None,
    )


def normalize_language(text: str) -> str:
    normalized = str(text).casefold().replace("-", " ")
    normalized = re.sub(r"[^a-z0-9%]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _phrase_present(text: str, phrase: str) -> bool:
    return normalize_language(phrase) in text


def _metrics(text: str) -> list[str]:
    found: list[str] = []
    for metric, aliases in METRIC_ALIASES.items():
        if any(_phrase_present(text, alias) for alias in aliases):
            found.append(metric)

    variability_words = {"variable", "variability", "variation", "inconsistent", "inconsistency", "unstable"}
    tokens = set(text.split())
    if tokens & {"receipt", "receipts"} and tokens & variability_words:
        if "lead_time_variability" not in found:
            found.append("lead_time_variability")
    if "demand" in tokens and tokens & (variability_words | {"volatile", "volatility"}):
        if "demand_variability" not in found:
            found.append("demand_variability")
    if (
        ("safety stock" in text or " ss " in f" {text} ")
        and tokens & {"gap", "gaps", "increase", "increases", "difference", "recommended", "current", "need", "needs", "insufficient"}
    ):
        if "safety_stock_gap" not in found:
            found.append("safety_stock_gap")
    if (
        ("reorder point" in text or " rop " in f" {text} ")
        and tokens & {"gap", "gaps", "increase", "increases", "difference", "recommended", "current"}
    ):
        if "rop_gap" not in found:
            found.append("rop_gap")

    # A specific variability concept supersedes its generic base metric.
    if "lead_time_variability" in found and "lead_time_mean" in found:
        found.remove("lead_time_mean")
    if "demand_variability" in found:
        found = [metric for metric in found if metric not in {"demand_level", "demand_trend"}]
    if (
        "demand" in text
        and any(phrase in text for phrase in ("supply side", "lead time risk", "receipt risk"))
        and any(phrase in text for phrase in ("both", "risk from", "risk"))
    ):
        for metric in ("demand_variability", "lead_time_variability"):
            if metric not in found:
                found.append(metric)
    return found


def _requested_limit(text: str, singular_superlative: bool, maximum: int) -> int:
    number_pattern = "|".join(NUMBER_WORDS)
    match = re.search(
        rf"\b(?:top|bottom|worst|best|show|give|which|what)\s+(?:me\s+|are\s+the\s+)?(\d{{1,2}}|{number_pattern})\b",
        text,
    )
    if match:
        raw = match.group(1)
        value = int(raw) if raw.isdigit() else NUMBER_WORDS[raw]
        return min(max(value, 1), maximum)
    return 1 if singular_superlative else min(10, maximum)


def _direction(text: str, metric: str | None) -> str | None:
    if metric in {"lead_time_variability", "demand_variability"}:
        if any(phrase in text for phrase in (
            "most consistent", "best consistency", "highest consistency",
            "lowest variability", "least variable", "least volatile",
        )):
            return "ascending"
        if "bottom" in text and "consistency" in text:
            return "descending"
        if any(phrase in text for phrase in (
            "worst consistency", "most inconsistent", "most unstable",
            "highest variability", "greatest variation", "most variable",
            "most volatile",
        )):
            return "descending"
    if any(word in text.split() for word in ("lowest", "least", "smallest")):
        return "ascending"
    if "bottom" in text:
        return "ascending"
    if any(word in text.split() for word in (
        "highest", "most", "largest", "greatest", "biggest", "top", "worst",
    )):
        return "descending"
    return "descending" if metric else None


def _conditions(text: str, metrics: list[str]) -> list[str]:
    conditions: list[str] = []
    if "demand_variability" in metrics and any(
        phrase in text for phrase in ("high demand", "volatile demand", "unstable demand", "variable demand")
    ):
        conditions.append("high_demand_variability")
    if "lead_time_variability" in metrics and any(
        phrase in text for phrase in (
            "unstable lead", "inconsistent lead", "inconsistent receipt",
            "high lead time variability", "most inconsistent",
        )
    ):
        conditions.append("high_lead_time_variability")
    if any(phrase in text for phrase in ("increasing demand", "rising demand", "growing demand")):
        conditions.append("increasing_demand")
    if "stable demand" in text:
        conditions.append("stable_demand")
    if any(phrase in text for phrase in ("stable lead", "consistent receipt", "consistent lead")):
        conditions.append("stable_lead_time")
    if any(phrase in text for phrase in ("high lead time", "long lead time", "longer than average")):
        conditions.append("long_lead_time")
    if "safety_stock_gap" in metrics and any(phrase in text for phrase in (
        "need", "increase", "exceeds current", "insufficient", "under protected", "underprotected",
    )):
        conditions.append("positive_safety_stock_gap")
    if "rop_gap" in metrics and any(phrase in text for phrase in ("large", "high", "increase", "gap")):
        conditions.append("positive_rop_gap")
    if any(phrase in text for phrase in ("low current safety stock", "low safety stock", "low current ss")):
        conditions.append("low_current_safety_stock")
    return list(dict.fromkeys(conditions))


def interpret_inventory_question(
    question: str,
    *,
    sku_count: int = 0,
    supplier_count: int = 0,
    maximum_rows: int = 15,
) -> LanguageInterpretation:
    """Interpret natural language into a constrained analytical vocabulary."""
    text = normalize_language(question)
    metrics = _metrics(text)
    metric = metrics[0] if metrics else None
    if "safety_stock_gap" in metrics and (
        re.search(r"\b(largest|biggest|highest|large)\b.{0,30}\bsafety stock\b", text)
        or re.search(r"\bsafety stock increases?\b", text)
    ):
        metric = "safety_stock_gap"
    elif "rop_gap" in metrics and re.search(
        r"\b(largest|biggest|highest|large)\b.{0,30}\b(rop|reorder point)\b", text
    ):
        metric = "rop_gap"

    item_words = bool(re.search(r"\b(sku|skus|item|items|product|products)\b", text))
    supplier_words = bool(re.search(r"\b(supplier|suppliers|vendor|vendors)\b", text))
    supplier_target = bool(re.match(r"^(which|what)\s+(supplier|vendor)\b", text))
    entity_type = "supplier" if supplier_target or (supplier_words and not item_words and not sku_count) else "sku"
    if supplier_count >= 2 and not item_words:
        entity_type = "supplier"

    comparison = bool(re.search(r"\b(compare|versus|vs)\b|\bwhich\s+has\s+more\b", text))
    explanation = bool(re.match(r"^(why|explain)\b", text))
    forecast = bool(re.search(r"\b(forecast|predict|projection)\b|\bnext\s+\d+\s+(week|period)", text))
    ranking_word = any(re.search(rf"\b{re.escape(word)}\b", text) for word in RANKING_LANGUAGE)
    singular_target = bool(re.match(r"^(which|what)\s+(sku|item|product|supplier|vendor)\b", text))
    ranking = ranking_word or (singular_target and metric is not None and sku_count + supplier_count == 0)

    if comparison:
        query_type = "comparison"
    elif forecast:
        query_type = "forecast"
    elif explanation:
        query_type = "explanation"
    elif ranking:
        query_type = "ranking"
    elif sku_count or supplier_count:
        query_type = "metric_lookup" if metric else "profile"
    elif metric and re.search(r"\b(which|show|find|list|what)\b", text):
        query_type = "filter"
    else:
        query_type = "clarification"

    clarification = None
    confidence = "high"
    if query_type == "clarification":
        confidence = "low"
        if re.search(r"\bbetter\b", text) and metric is None:
            clarification = "Which metric should I use to decide what is better?"
        else:
            clarification = "Which inventory metric or SKU/supplier would you like me to analyze?"

    limit = _requested_limit(text, ranking and singular_target, maximum_rows)
    return LanguageInterpretation(
        query_type=query_type,
        entity_type=entity_type,
        metric=metric,
        metrics=tuple(metrics),
        direction=_direction(text, metric) if ranking or query_type == "filter" else None,
        limit=limit,
        conditions=tuple(_conditions(text, metrics)),
        confidence=confidence,
        clarification=clarification,
    )


def extract_unresolved_sku_candidate(question: str) -> str | None:
    """Return only identifier-shaped explicit SKU candidates.

    A word after ``SKU`` is not sufficient evidence. The token must also look
    like an identifier (digit/separator or an all-uppercase multi-letter code).
    """
    match = re.search(
        r"\bsku(?:\s+(?:code|id|number))?\s*[:#]?\s+([A-Za-z0-9][A-Za-z0-9._/-]*)",
        question,
        re.I,
    )
    if not match:
        # Common code shape even when the user omits the word SKU.
        match = re.search(r"(?<![A-Za-z0-9])([A-Za-z]{2,}[-_/]\d+[A-Za-z0-9._/-]*)", question)
    if not match:
        return None
    candidate = match.group(1).rstrip("?.!,")
    strong_introducer = bool(re.search(r"\bsku\s+(?:code|id|number)\b|\bsku\s*[:#]", question, re.I))
    identifier_shaped = (
        any(char.isdigit() for char in candidate)
        or any(char in "-_/" for char in candidate)
        or (strong_introducer and len(candidate) >= 2 and candidate == candidate.upper())
    )
    return candidate if identifier_shaped else None
