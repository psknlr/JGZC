"""Structural contracts for the six sub-agents.

Shallow on purpose — required keys and broad types. The goal is to catch an
agent (or a future LLM-backed replacement for one) silently emitting a
differently-shaped payload, not to re-implement a type system. Content is
checked where it is clinically load-bearing, in :func:`validate`.
"""

from __future__ import annotations

from typing import Any

SCHEMAS: dict[str, dict[str, tuple[bool, tuple[type, ...]]]] = {
    "StructuredCase": {
        "facts": (True, (dict,)),
        "coverage": (True, (dict,)),
        "missing_critical": (True, (list,)),
        "derived": (True, (list,)),
    },
    "OsteoSarcDiagnosis": {
        "osteoporosis": (True, (dict,)),
        "sarcopenia": (True, (dict,)),
        "osteosarcopenia": (True, (dict,)),
        "standards": (True, (list,)),
    },
    "RiskProfile": {
        "axes": (True, (list,)),
        "radar": (True, (dict,)),
        "headline": (True, (str,)),
        "model_note": (True, (str,)),
    },
    "EvidenceSelection": {
        "applicable": (True, (list,)),
        "excluded": (True, (list,)),
        "insufficient": (True, (list,)),
        "by_topic": (True, (dict,)),
        "corpus": (True, (dict,)),
    },
    "ConflictResolution": {
        "questions": (True, (list,)),
        "counts": (True, (dict,)),
    },
    "SafetyAndFollowup": {
        "checks_run": (True, (list,)),
        "issues": (True, (list,)),
        "blocking": (True, (list,)),
        "followup": (True, (dict,)),
    },
}

#: Every safety check the audit agent must be able to say it ran. If one is
#: absent from ``checks_run`` the payload is rejected: a safety audit that
#: silently skipped the renal check is worse than no audit, because the report
#: it produces looks complete.
REQUIRED_SAFETY_CHECKS = (
    "contraindication",
    "interaction",
    "renal",
    "hypocalcemia",
    "fall_risk_drugs",
    "adherence",
    "followup",
)


def validate(schema_name: str, payload: Any) -> tuple[bool, list[str]]:
    if not schema_name:
        return True, []
    schema = SCHEMAS.get(schema_name)
    if schema is None:
        return False, [f"unknown schema {schema_name!r}"]
    if not isinstance(payload, dict):
        return False, [f"{schema_name}: payload is not a mapping"]

    problems: list[str] = []
    for field_name, (required, types) in schema.items():
        if field_name not in payload:
            if required:
                problems.append(f"{schema_name}: missing required field {field_name!r}")
            continue
        value = payload[field_name]
        if value is None:
            if required:
                problems.append(f"{schema_name}: field {field_name!r} is null")
            continue
        if not isinstance(value, types):
            expected = "/".join(t.__name__ for t in types)
            problems.append(
                f"{schema_name}: field {field_name!r} should be {expected}, got {type(value).__name__}"
            )

    if schema_name == "SafetyAndFollowup" and isinstance(payload.get("checks_run"), list):
        ran = {str(item) for item in payload["checks_run"]}
        skipped = [name for name in REQUIRED_SAFETY_CHECKS if name not in ran]
        if skipped:
            problems.append(f"SafetyAndFollowup: 安全审查漏跑了 {skipped}——缺项的审查报告比没有报告更危险")
        followup = payload.get("followup")
        if isinstance(followup, dict):
            for milestone in ("m3", "m6", "m12"):
                if not followup.get(milestone):
                    problems.append(f"SafetyAndFollowup: 随访计划缺少 {milestone} 节点")
    return not problems, problems
