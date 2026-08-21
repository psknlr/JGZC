"""Computable guideline records: a recommendation that knows when it applies.

A guideline library that stores prose can only ever be searched. This module
stores each recommendation together with a **machine-evaluable applicability
predicate**, so the question stops being "which paragraphs mention this
patient's problem" and becomes "which recommendations are *in force* for this
patient, and which are excluded, and why".

Three-valued logic is the load-bearing decision here. Clinical records are
incomplete far more often than they are wrong, so a predicate over a missing
fact must not collapse to false: ``UNKNOWN`` propagates and surfaces as
"需补充数据" instead of quietly dropping a recommendation that might well be the
most important one for this patient. ``all`` takes the minimum, ``any`` the
maximum, ``not`` swaps true/false and leaves unknown alone.

Every evaluation can also return a **trace** — the predicate tree annotated with
the facts that decided each leaf. That trace is what the console shows behind
「查看循证依据」: not only what the guideline says, but why it is being applied to
*this* patient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

# ---------------------------------------------------------------------------
# Tri-state truth
# ---------------------------------------------------------------------------

TRUE = "true"
FALSE = "false"
UNKNOWN = "unknown"

#: Ordering used by ``all`` (minimum) and ``any`` (maximum).
_ORDER = {FALSE: 0, UNKNOWN: 1, TRUE: 2}

#: Operators that answer a question *about* the record rather than its value,
#: and therefore stay determinate when the fact is absent.
_PRESENCE_OPS = {"known", "unknown"}


def tri_and(values: Iterable[str]) -> str:
    result = TRUE
    for value in values:
        if _ORDER[value] < _ORDER[result]:
            result = value
    return result


def tri_or(values: Iterable[str]) -> str:
    result = FALSE
    for value in values:
        if _ORDER[value] > _ORDER[result]:
            result = value
    return result


def tri_not(value: str) -> str:
    if value == TRUE:
        return FALSE
    if value == FALSE:
        return TRUE
    return UNKNOWN


def as_tri(value: Any) -> str:
    """Coerce a Python value into tri-state truth. ``None`` means unknown."""
    if value is None:
        return UNKNOWN
    if isinstance(value, str) and value in (TRUE, FALSE, UNKNOWN):
        return value
    return TRUE if value else FALSE


# ---------------------------------------------------------------------------
# Predicate evaluation
# ---------------------------------------------------------------------------


class ConditionError(ValueError):
    """A malformed predicate. Raised at load time, never at decision time."""


def _compare(op: str, left: Any, right: Any) -> str:
    """Apply one comparison, returning tri-state truth.

    A comparison between incompatible types is ``UNKNOWN``, not an exception:
    a corpus typo must not take down a clinical run, and the trace will show
    the operand that could not be compared.
    """
    try:
        if op == ">=":
            return as_tri(float(left) >= float(right))
        if op == ">":
            return as_tri(float(left) > float(right))
        if op == "<=":
            return as_tri(float(left) <= float(right))
        if op == "<":
            return as_tri(float(left) < float(right))
        if op == "==":
            return as_tri(left == right)
        if op == "!=":
            return as_tri(left != right)
        if op == "is_true":
            return as_tri(bool(left))
        if op == "is_false":
            return tri_not(as_tri(bool(left)))
        if op == "in":
            return as_tri(left in right)
        if op == "not_in":
            return as_tri(left not in right)
        if op == "contains":
            return as_tri(right in (left or []))
        if op == "any_in":
            haystack = set(left or [])
            return as_tri(bool(haystack & set(right or [])))
    except (TypeError, ValueError):
        return UNKNOWN
    raise ConditionError(f"unknown operator {op!r}")


#: Every operator the corpus may use. Checked at load time so a typo is a load
#: error rather than a silently-unknown predicate at the bedside.
OPERATORS = {
    ">=", ">", "<=", "<", "==", "!=", "is_true", "is_false",
    "in", "not_in", "contains", "any_in", "known", "unknown",
}


@dataclass(frozen=True)
class TraceNode:
    """One node of an applicability trace."""

    label: str
    result: str
    detail: str = ""
    children: tuple["TraceNode", ...] = ()
    #: The fact this leaf tested, kept alongside the human-readable label so
    #: callers can key on the name without parsing the display string.
    fact: str = ""

    def to_dict(self) -> dict[str, Any]:
        node: dict[str, Any] = {"label": self.label, "result": self.result}
        if self.detail:
            node["detail"] = self.detail
        if self.fact:
            node["fact"] = self.fact
        if self.children:
            node["children"] = [child.to_dict() for child in self.children]
        return node

    def missing_facts(self) -> list[str]:
        """Fact names that were unknown and mattered to the outcome."""
        found: list[str] = []
        if self.result == UNKNOWN and not self.children and self.fact and not self.detail:
            found.append(self.fact)
        for child in self.children:
            found.extend(child.missing_facts())
        # Preserve order, drop duplicates.
        seen: set[str] = set()
        return [name for name in found if not (name in seen or seen.add(name))]


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple, set)):
        return "/".join(str(item) for item in value)
    return str(value)


def evaluate(node: Mapping[str, Any] | None, facts: Mapping[str, Any]) -> tuple[str, TraceNode]:
    """Evaluate a predicate against ``facts``, returning ``(result, trace)``.

    ``None`` and ``{}`` both mean "unconditional" — a recommendation with no
    applicability predicate applies to everyone the corpus scopes it to.
    """
    if not node:
        return TRUE, TraceNode("无附加条件", TRUE)
    if not isinstance(node, Mapping):
        raise ConditionError(f"predicate must be a mapping, got {type(node).__name__}")

    if "all" in node:
        children = tuple(evaluate(child, facts)[1] for child in node["all"])
        result = tri_and(child.result for child in children)
        return result, TraceNode(node.get("label", "全部满足"), result, children=children)
    if "any" in node:
        children = tuple(evaluate(child, facts)[1] for child in node["any"])
        result = tri_or(child.result for child in children)
        return result, TraceNode(node.get("label", "任一满足"), result, children=children)
    if "not" in node:
        inner = evaluate(node["not"], facts)[1]
        result = tri_not(inner.result)
        return result, TraceNode(node.get("label", "不满足"), result, children=(inner,))
    if "always" in node:
        result = as_tri(node["always"])
        return result, TraceNode("常规适用", result)

    fact_name = node.get("fact")
    if not fact_name:
        raise ConditionError(f"predicate has no fact/all/any/not key: {dict(node)!r}")
    op = node.get("op", "is_true")
    if op not in OPERATORS:
        raise ConditionError(f"unknown operator {op!r} on fact {fact_name!r}")
    value = facts.get(fact_name)
    label = node.get("label") or _leaf_label(fact_name, op, node.get("value"))

    # Details are written with the fact's Chinese label: this string is read by
    # clinicians in the evidence drawer and in the printed report, and
    # ``egfr=32 < 35`` is worse there than ``eGFR=32 < 35``. The machine-readable
    # name travels alongside in ``TraceNode.fact``.
    from .facts import describe_fact

    name_zh = describe_fact(fact_name)
    if op in _PRESENCE_OPS:
        result = as_tri(value is not None) if op == "known" else as_tri(value is None)
        return result, TraceNode(label, result, detail=f"{name_zh}={_fmt(value)}", fact=fact_name)
    if value is None:
        # No detail: an unmeasured fact has no value to show, and the empty
        # detail is what marks this leaf as a genuine data gap.
        return UNKNOWN, TraceNode(label, UNKNOWN, fact=fact_name)

    result = _compare(op, value, node.get("value"))
    detail = f"{name_zh}={_fmt(value)}"
    if op not in ("is_true", "is_false"):
        detail += f" {_OP_TEXT.get(op, op)} {_fmt(node.get('value'))}"
    return result, TraceNode(label, result, detail=detail, fact=fact_name)


_OP_TEXT = {
    ">=": "≥", ">": ">", "<=": "≤", "<": "<", "==": "=", "!=": "≠",
    "in": "属于", "not_in": "不属于", "contains": "包含", "any_in": "命中任一",
}


def _leaf_label(fact_name: str, op: str, value: Any) -> str:
    from .facts import describe_fact

    name = describe_fact(fact_name)
    if op == "is_true":
        return name
    if op == "is_false":
        return f"非{name}"
    return f"{name} {_OP_TEXT.get(op, op)} {_fmt(value)}"


def validate_condition(node: Any, where: str) -> None:
    """Raise :class:`ConditionError` if ``node`` is not a usable predicate."""
    if node is None or node == {}:
        return
    if not isinstance(node, Mapping):
        raise ConditionError(f"{where}: predicate must be a mapping")
    for key in ("all", "any"):
        if key in node:
            if not isinstance(node[key], list) or not node[key]:
                raise ConditionError(f"{where}: {key!r} must be a non-empty list")
            for child in node[key]:
                validate_condition(child, where)
            return
    if "not" in node:
        validate_condition(node["not"], where)
        return
    if "always" in node:
        return
    if "fact" not in node:
        raise ConditionError(f"{where}: predicate has no fact/all/any/not/always key")
    op = node.get("op", "is_true")
    if op not in OPERATORS:
        raise ConditionError(f"{where}: unknown operator {op!r}")
    if op not in _PRESENCE_OPS and op not in ("is_true", "is_false") and "value" not in node:
        raise ConditionError(f"{where}: operator {op!r} needs a 'value'")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

#: Where a recommendation comes from. Used by the conflict agent to decide
#: whether two agreeing records are genuinely independent voices.
REGIONS = {"CN", "US", "EU", "INTL", "APAC"}

#: Which body of practice the source speaks for. A geriatrics guideline and an
#: endocrine guideline disagreeing is a *different* kind of conflict from two
#: endocrine guidelines disagreeing, and the console labels them differently.
TRADITIONS = {"western", "tcm", "geriatrics", "sarcopenia", "nutrition", "rehab", "pharmacy"}

STRENGTHS = {"strong", "conditional", "consensus", "expert", "good_practice"}

DIRECTIONS = {"recommend", "consider", "avoid", "require", "against"}

#: Provenance is deliberately explicit. Nothing in this repository is licensed
#: guideline text; every shipped record is an editorial paraphrase of a public
#: recommendation, and the console must say so on every screen that shows one.
PROVENANCES = {"editorial_paraphrase", "licensed_verbatim", "local_policy"}


@dataclass(frozen=True)
class ClinicalQuestion:
    """One decision the guidelines answer — the unit conflict is detected over.

    ``exclusive`` is the field that makes conflict detection meaningful. Under a
    non-exclusive question ("增加跌倒风险的药物如何处理") three sources naming three
    different actions are three complementary instructions. Under an exclusive
    one ("极高骨折风险者的初始药物选择") two sources naming different actions are a
    genuine disagreement about what to do first. Without this distinction a
    conflict detector either cries wolf on every multi-source topic or misses
    the disagreements that matter.
    """

    question_id: str
    label_zh: str
    exclusive: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "label_zh": self.label_zh,
            "exclusive": self.exclusive,
            "note": self.note,
        }


@dataclass(frozen=True)
class GuidelineSource:
    """One guideline document."""

    source_id: str
    title_zh: str
    issuer: str
    year: int
    region: str
    tradition: str
    title_en: str = ""
    url: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "title_zh": self.title_zh,
            "title_en": self.title_en,
            "issuer": self.issuer,
            "year": self.year,
            "region": self.region,
            "tradition": self.tradition,
            "url": self.url,
            "note": self.note,
        }


@dataclass(frozen=True)
class Recommendation:
    """A single recommendation, with the predicate that decides whether it fires."""

    rec_id: str
    source_id: str
    topic: str
    question: str
    action: str
    direction: str
    strength: str
    statement_zh: str
    evidence_level: str = ""
    applies_when: Mapping[str, Any] | None = None
    excluded_when: Mapping[str, Any] | None = None
    rationale: str = ""
    citation: str = ""
    provenance: str = "editorial_paraphrase"
    tags: tuple[str, ...] = ()
    verbatim: bool = False
    #: Actions this one already satisfies. "每日至少 1.0 g/kg" is satisfied by
    #: "1.2～1.5 g/kg", so two sources naming those two actions are not in
    #: conflict even under an exclusive question — one is the stricter case of
    #: the other. Without this, every nested target reads as a disagreement.
    subsumes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rec_id": self.rec_id,
            "source_id": self.source_id,
            "topic": self.topic,
            "question": self.question,
            "action": self.action,
            "direction": self.direction,
            "strength": self.strength,
            "evidence_level": self.evidence_level,
            "statement_zh": self.statement_zh,
            "rationale": self.rationale,
            "citation": self.citation,
            "provenance": self.provenance,
            "verbatim": self.verbatim,
            "tags": list(self.tags),
            "subsumes": list(self.subsumes),
        }


@dataclass(frozen=True)
class Applicability:
    """The result of testing one recommendation against one patient."""

    recommendation: Recommendation
    source: GuidelineSource
    status: str  # "applies" | "excluded" | "not_applicable" | "insufficient_data"
    applies_result: str
    excluded_result: str
    trace: TraceNode
    exclusion_trace: TraceNode | None = None
    missing_facts: tuple[str, ...] = ()

    @property
    def fires(self) -> bool:
        return self.status == "applies"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "applies_result": self.applies_result,
            "excluded_result": self.excluded_result,
            "missing_facts": list(self.missing_facts),
            "missing_facts_zh": [_fact_label(name) for name in self.missing_facts],
            "trace": self.trace.to_dict(),
            "recommendation": self.recommendation.to_dict(),
            "source": self.source.to_dict(),
        }
        if self.exclusion_trace is not None:
            payload["exclusion_trace"] = self.exclusion_trace.to_dict()
        return payload


def _fact_label(name: str) -> str:
    from .facts import describe_fact

    return describe_fact(name)


def test_recommendation(
    rec: Recommendation, source: GuidelineSource, facts: Mapping[str, Any]
) -> Applicability:
    """Decide whether ``rec`` is in force for ``facts``.

    Exclusion dominates: a recommendation whose ``excluded_when`` fires is
    ``not_applicable`` even if ``applies_when`` also fires. That asymmetry is
    intentional — the exclusions in this corpus are contraindications, and a
    contraindication that loses a tie is not a contraindication.
    """
    applies_result, trace = evaluate(rec.applies_when, facts)
    excluded_result, exclusion_trace = (FALSE, None)
    if rec.excluded_when:
        excluded_result, exclusion_trace = evaluate(rec.excluded_when, facts)

    if excluded_result == TRUE:
        status = "excluded"
    elif applies_result == FALSE:
        status = "not_applicable"
    elif applies_result == UNKNOWN or excluded_result == UNKNOWN:
        status = "insufficient_data"
    else:
        status = "applies"

    missing = list(trace.missing_facts())
    if exclusion_trace is not None:
        missing.extend(name for name in exclusion_trace.missing_facts() if name not in missing)
    return Applicability(
        recommendation=rec,
        source=source,
        status=status,
        applies_result=applies_result,
        excluded_result=excluded_result,
        trace=trace,
        exclusion_trace=exclusion_trace,
        missing_facts=tuple(missing),
    )
