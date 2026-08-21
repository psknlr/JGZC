"""Agent 4 —— 循证决策 Agent.

Not a search over guideline text. Every record in the corpus carries a
predicate, so this agent *evaluates the whole corpus against this patient* and
sorts it into four buckets:

* **applicable** — the recommendation is in force here;
* **excluded** — a contraindication or exclusion fired (``eGFR 32 < 35`` takes
  the bisphosphonate records off the table). These are clinically louder than
  the ones that apply, and are kept and cited rather than filtered away;
* **not_applicable** — the predicate is simply false for this patient;
* **insufficient_data** — the predicate could not be decided because a fact is
  missing.

That last bucket is turned into a prioritised worklist: for each missing
measurement, how many recommendations it would unlock. "补测 ASMI 可解锁 4 条推荐"
is a more useful instruction than a generic "建议完善检查".
"""

from __future__ import annotations

from typing import Any

from ..guidelines.corpus import Corpus, default_corpus
from ..guidelines.facts import describe_fact
from ..state import RunState
from .base import SubAgent

TOPIC_LABELS = {
    "diagnosis": "诊断与鉴别",
    "screening": "筛查",
    "bone_protection": "骨保护",
    "exercise": "运动与康复",
    "nutrition": "营养",
    "falls": "跌倒防治",
    "medication_safety": "用药安全",
    "monitoring": "随访监测",
    "tcm": "中医与中西医结合",
}

#: Topics that become the three plan columns in the console.
PLAN_TOPICS = ("bone_protection", "exercise", "nutrition")


class EvidenceAgent(SubAgent):
    agent_id = "evidence"
    name_zh = "循证决策 Agent"
    schema = "EvidenceSelection"

    def __init__(self, corpus: Corpus | None = None) -> None:
        self.corpus = corpus or default_corpus()

    def run(self, state: RunState) -> dict[str, Any]:
        results = self.corpus.apply(state.facts)
        ledger = state.ledger

        applicable: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        insufficient: list[dict[str, Any]] = []
        not_applicable = 0

        for item in results:
            if item.status == "applies":
                evidence_id = ledger.cite(item, self.agent_id)
                applicable.append(self._entry(item, evidence_id))
            elif item.status == "excluded":
                evidence_id = ledger.cite(item, self.agent_id)
                entry = self._entry(item, evidence_id)
                entry["excluded_because"] = self._exclusion_reason(item)
                excluded.append(entry)
            elif item.status == "insufficient_data":
                insufficient.append({
                    "rec_id": item.recommendation.rec_id,
                    "question": item.recommendation.question,
                    "topic": item.recommendation.topic,
                    "statement_zh": item.recommendation.statement_zh,
                    "missing_facts": [
                        {"fact": name, "label": describe_fact(name)}
                        for name in item.missing_facts
                    ],
                })
            else:
                not_applicable += 1

        by_topic: dict[str, Any] = {}
        for entry in applicable:
            bucket = by_topic.setdefault(entry["topic"], {
                "label": TOPIC_LABELS.get(entry["topic"], entry["topic"]),
                "items": [],
            })
            bucket["items"].append(entry)
        for bucket in by_topic.values():
            bucket["items"].sort(key=lambda e: (_strength_rank(e["strength"]), e["rec_id"]))

        return {
            "applicable": applicable,
            "excluded": excluded,
            "insufficient": insufficient,
            "not_applicable_count": not_applicable,
            "by_topic": by_topic,
            "data_gaps": self._data_gaps(insufficient),
            "corpus": self.corpus.stats(),
            "evidence": [entry["evidence_id"] for entry in applicable],
        }

    def _entry(self, item: Any, evidence_id: str) -> dict[str, Any]:
        rec = item.recommendation
        source = item.source
        return {
            "evidence_id": evidence_id,
            "rec_id": rec.rec_id,
            "topic": rec.topic,
            "topic_label": TOPIC_LABELS.get(rec.topic, rec.topic),
            "question": rec.question,
            "action": rec.action,
            "direction": rec.direction,
            "strength": rec.strength,
            "evidence_level": rec.evidence_level,
            "statement_zh": rec.statement_zh,
            "rationale": rec.rationale,
            "citation": rec.citation,
            "provenance": rec.provenance,
            "tags": list(rec.tags),
            "source": source.to_dict(),
            "trace": item.trace.to_dict(),
        }

    def _exclusion_reason(self, item: Any) -> str:
        """Name the fact that took this recommendation off the table."""
        trace = item.exclusion_trace
        if trace is None:
            return "排除条件成立"
        leaves = _true_leaves(trace)
        if leaves:
            return "；".join(f"{leaf.label}（{leaf.detail}）" if leaf.detail else leaf.label
                             for leaf in leaves)
        return trace.label

    def _data_gaps(self, insufficient: list[dict[str, Any]]) -> list[dict[str, Any]]:
        counts: dict[str, dict[str, Any]] = {}
        for item in insufficient:
            for missing in item["missing_facts"]:
                bucket = counts.setdefault(missing["fact"], {
                    "fact": missing["fact"],
                    "label": missing["label"],
                    "unlocks": 0,
                    "recommendations": [],
                })
                bucket["unlocks"] += 1
                if len(bucket["recommendations"]) < 5:
                    bucket["recommendations"].append(item["rec_id"])
        gaps = sorted(counts.values(), key=lambda gap: -gap["unlocks"])
        for gap in gaps:
            gap["hint"] = f"补测「{gap['label']}」可解锁 {gap['unlocks']} 条推荐的适用性判定"
        return gaps

    def notes(self, payload: dict[str, Any]) -> list[str]:
        notes = [
            f"语料 {payload['corpus']['recommendations']} 条 / {payload['corpus']['sources']} 部指南"
            f"（{payload['corpus']['year_range'][0]}–{payload['corpus']['year_range'][1]}），"
            f"本例命中 {len(payload['applicable'])} 条，排除 {len(payload['excluded'])} 条，"
            f"{len(payload['insufficient'])} 条因数据缺失无法判定。"
        ]
        for entry in payload["excluded"][:4]:
            notes.append(f"已排除 {entry['rec_id']}：{entry['excluded_because']}")
        if payload["corpus"]["verbatim_records"] == 0:
            notes.append("当前语料全部为编辑性转述，未含授权指南原文——结论须由医师核对原文后使用。")
        return notes


def _true_leaves(node: Any) -> list[Any]:
    if node.result != "true":
        return []
    if not node.children:
        return [node]
    leaves: list[Any] = []
    for child in node.children:
        leaves.extend(_true_leaves(child))
    return leaves or [node]


_STRENGTH_ORDER = {"strong": 0, "consensus": 1, "conditional": 2, "expert": 3, "good_practice": 4}


def _strength_rank(strength: str) -> int:
    return _STRENGTH_ORDER.get(strength, 9)
