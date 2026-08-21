"""Agent 5 —— 指南冲突消解 Agent.

The output of five guidelines on one patient is not five paragraphs. It is one
answer per clinical question, with a verdict that says how much the guidelines
actually agree:

===================  ==========================================================
一致推荐              applicable records all point the same way
条件推荐              they agree in direction but the recommendation is
                      conditional, or rests on a single source
存在争议              they genuinely disagree — opposite directions on the same
                      action, or different answers to a question whose answers
                      are mutually exclusive
不适用于本患者         every record for this question is excluded or its
                      predicate is false for this patient
===================  ==========================================================

Two further states are reported because the data forces them, and folding them
into the four above would be a lie:

* **争议已被本例事实消解** — the guidelines *do* disagree in general, but this
  patient's own facts take one side off the table. ``eGFR 32`` removes the
  bisphosphonate arm, so the international dispute about first-line therapy has
  a determinate answer *here*. Reporting this as plain 一致推荐 would hide that
  the answer is contingent on one lab value; reporting it as 存在争议 would push
  a decision back to the clinician that the patient's chart has already made.
* **数据不足** — the question cannot be decided because a fact is missing.

Where a dispute survives, the platform does **not** pick a side. It states the
positions, names the patient-specific factors that bear on the choice, and
leaves the decision to the physician.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..guidelines.corpus import Corpus, default_corpus
from ..guidelines.facts import format_value
from ..guidelines.model import Applicability
from ..state import RunState
from .base import SubAgent

CONSENSUS = "consensus"
CONDITIONAL = "conditional"
DISPUTED = "disputed"
NOT_APPLICABLE = "not_applicable"
RESOLVED_BY_PATIENT = "resolved_by_patient"
INSUFFICIENT = "insufficient_data"

VERDICT_ZH = {
    CONSENSUS: "一致推荐",
    CONDITIONAL: "条件推荐",
    DISPUTED: "存在争议",
    NOT_APPLICABLE: "不适用于本患者",
    RESOLVED_BY_PATIENT: "争议已被本例事实消解",
    INSUFFICIENT: "数据不足，无法判定",
}

#: The four verdicts the product surface leads with; the other two are
#: qualifiers on top of them.
PRIMARY_VERDICTS = (CONSENSUS, CONDITIONAL, DISPUTED, NOT_APPLICABLE)

POSITIVE = {"recommend", "require", "consider"}
NEGATIVE = {"avoid", "against"}

REGION_ZH = {"CN": "中国", "US": "美国", "EU": "欧洲", "INTL": "国际", "APAC": "亚太"}
TRADITION_ZH = {
    "western": "现代医学", "tcm": "中医", "geriatrics": "老年医学",
    "sarcopenia": "肌少症专业", "nutrition": "营养", "rehab": "康复", "pharmacy": "临床药学",
}
STRENGTH_ZH = {
    "strong": "强推荐", "conditional": "有条件推荐", "consensus": "专家共识",
    "expert": "专家意见", "good_practice": "良好实践",
}
DIRECTION_ZH = {
    "recommend": "推荐", "require": "必须", "consider": "可考虑",
    "avoid": "不推荐/避免", "against": "反对",
}

#: Where the platform has a stated policy for handling a disputed question, it
#: says so. A conflict detector that only reports "存在争议" and stops has moved
#: the work back onto the reader; where a defensible way of *holding* the dispute
#: exists (run all three standards; require two specialties to co-set the target),
#: naming it is the difference between an analysis and a shrug. None of these
#: policies picks a winner — each one keeps both sides visible.
PLATFORM_POLICY: dict[str, str] = {
    "q.sarco.diagnosis_criteria":
        "本平台的处理：三套标准全部判定并并列呈现，不替医师选定标准；"
        "中国人群部署下默认以 AWGS 亚洲切点作为主判定，并在结论中标注该选择。",
    "q.sarco.mass_required":
        "本平台的处理：肌量未测时，按 AWGS/EWGSOP2 的「可能肌少症」立即启动运动与营养干预，"
        "同时按 GLIS 标注为「不能确诊」，并把补测 ASMI 列入即刻随访——两边都不牺牲。",
    "q.nutrition.protein_target":
        "本平台的处理：不输出单一蛋白目标，转为要求肾脏科与营养科共同设定并动态监测"
        "（体重、握力、白蛋白、eGFR、代谢性酸中毒）。",
    "q.pharm.initial_agent_very_high_risk":
        "本平台的处理：不替医师选择初始药物；列出双方立场、各自的适用条件与本例的决定性因素，"
        "由主管医师决定并把理由写入病历。",
    "q.screen.dxa_age":
        "本平台的处理：按本机构所在地区的指南执行，并在报告中注明所依据的筛查年龄标准。",
}

#: Facts that most often decide a dispute, shown as decision support when one
#: survives. Kept short on purpose — a list of forty facts is not support.
_DECISION_FACTS = (
    ("egfr", "肾功能"), ("age", "年龄"), ("tscore_min", "最低 T 值"),
    ("prior_fragility_fracture", "既往脆性骨折"), ("fracture_within_12m", "近期骨折"),
    ("recent_cv_event_12m", "近期心血管事件"), ("high_fall_risk", "跌倒风险"),
    ("dx_sarcopenia_any", "肌少症"), ("dx_possible_sarcopenia", "可能肌少症"),
    ("hypocalcemia", "低钙血症"), ("vitd_deficient", "维生素 D 缺乏"),
    ("polypharmacy", "多重用药"), ("adherence_concern", "依从性"),
)


def _polarity(direction: str) -> str:
    if direction in NEGATIVE:
        return "-"
    return "+"


class ConflictAgent(SubAgent):
    agent_id = "conflict"
    name_zh = "指南冲突消解 Agent"
    schema = "ConflictResolution"

    def __init__(self, corpus: Corpus | None = None) -> None:
        self.corpus = corpus or default_corpus()

    def run(self, state: RunState) -> dict[str, Any]:
        grouped: dict[str, list[Applicability]] = {}
        for item in self.corpus.apply(state.facts):
            grouped.setdefault(item.recommendation.question, []).append(item)

        questions: list[dict[str, Any]] = []
        for question_id, items in grouped.items():
            questions.append(self._resolve(question_id, items, state))

        order = {verdict: index for index, verdict in enumerate(
            (DISPUTED, RESOLVED_BY_PATIENT, CONSENSUS, CONDITIONAL, INSUFFICIENT, NOT_APPLICABLE)
        )}
        questions.sort(key=lambda q: (order.get(q["verdict"], 9), q["question_id"]))

        counts: dict[str, int] = {}
        for question in questions:
            counts[question["verdict"]] = counts.get(question["verdict"], 0) + 1

        return {
            "questions": questions,
            "counts": counts,
            "counts_zh": {VERDICT_ZH[key]: value for key, value in counts.items()},
            "disputed": [q for q in questions if q["verdict"] == DISPUTED],
            "resolved": [q for q in questions if q["verdict"] == RESOLVED_BY_PATIENT],
            "evidence": sorted({
                eid for question in questions for eid in question["evidence"]
            }),
        }

    # -- per-question resolution -----------------------------------------
    def _resolve(self, question_id: str, items: list[Applicability], state: RunState) -> dict[str, Any]:
        question = self.corpus.question(question_id)
        live = [item for item in items if item.status == "applies"]
        excluded = [item for item in items if item.status == "excluded"]
        insufficient = [item for item in items if item.status == "insufficient_data"]
        inapplicable = [item for item in items if item.status == "not_applicable"]

        live_positions = self._positions(live, state)
        latent_positions = self._positions(items, state)

        live_conflict = self._conflict(live_positions, question.exclusive)
        latent_conflict = self._conflict(latent_positions, question.exclusive)

        evidence = state.ledger.cite_all(live + excluded, self.agent_id)

        payload: dict[str, Any] = {
            "question_id": question_id,
            "label_zh": question.label_zh,
            "exclusive": question.exclusive,
            "question_note": question.note,
            "positions": live_positions,
            "all_positions": latent_positions,
            "counts": {
                "applicable": len(live), "excluded": len(excluded),
                "insufficient": len(insufficient), "not_applicable": len(inapplicable),
            },
            "excluded_records": [
                {
                    "rec_id": item.recommendation.rec_id,
                    "source": item.source.title_zh,
                    "statement_zh": item.recommendation.statement_zh,
                    "reason": self._exclusion_reason(item),
                }
                for item in excluded
            ],
            "evidence": evidence,
            "notes": [],
        }

        if live and live_conflict["conflict"]:
            payload["verdict"] = DISPUTED
            payload["basis"] = live_conflict["why"]
            payload["divergence"] = self._divergence(live_positions)
            payload["decision_support"] = self._decision_support(state, question_id)
            payload["pending_data"] = bool(insufficient)
            if insufficient:
                payload["notes"].append(
                    f"另有 {len(insufficient)} 条推荐因数据缺失无法判定适用性，本争议可能在补数据后改变"
                )
        elif live and latent_conflict["conflict"]:
            payload["verdict"] = RESOLVED_BY_PATIENT
            payload["basis"] = "指南之间本有分歧，但本患者的事实已使其中一方不适用"
            payload["divergence"] = self._divergence(latent_positions)
            removed = [
                {
                    "action": item.recommendation.action,
                    "source": item.source.title_zh,
                    "statement_zh": item.recommendation.statement_zh,
                    "status": item.status,
                    "reason": self._exclusion_reason(item) if item.status == "excluded"
                    else self._inapplicability_reason(item),
                }
                for item in excluded + inapplicable
            ]
            payload["resolution"] = {
                "kept": [pos["action"] for pos in live_positions],
                "removed": removed,
                "deciding_facts": self._deciding_facts(excluded, inapplicable),
            }
        elif live:
            sources = {item.source.source_id for item in live}
            regions = {item.source.region for item in live}
            weak = [
                item for item in live
                if item.recommendation.strength in ("conditional", "expert", "good_practice")
                or item.recommendation.direction == "consider"
            ]
            if weak or len(live) == 1 and live[0].recommendation.strength not in ("strong", "consensus"):
                payload["verdict"] = CONDITIONAL
                payload["basis"] = (
                    f"{len(live)} 条适用推荐方向一致，但其中 {len(weak)} 条为有条件推荐或专家意见"
                    if weak else "适用推荐方向一致，但证据强度为有条件"
                )
            else:
                payload["verdict"] = CONSENSUS
                if len(sources) >= 2:
                    payload["basis"] = (
                        f"{len(sources)} 部指南（{'、'.join(sorted(REGION_ZH.get(r, r) for r in regions))}）"
                        "对本患者给出方向一致的推荐"
                    )
                else:
                    payload["basis"] = "单一来源的强推荐/共识，其他指南未就该问题作出相反推荐"
            payload["cross_region"] = len(regions) >= 2
            payload["source_count"] = len(sources)
        elif insufficient:
            payload["verdict"] = INSUFFICIENT
            payload["basis"] = "该问题下的推荐均因关键数据缺失而无法判定适用性"
            payload["missing_facts"] = sorted({
                name for item in insufficient for name in item.missing_facts
            })
        else:
            payload["verdict"] = NOT_APPLICABLE
            if excluded:
                payload["basis"] = "该问题下的推荐均因禁忌或排除条件被排除"
            else:
                payload["basis"] = "该问题下的推荐的适用条件在本患者均不成立"

        payload["verdict_zh"] = VERDICT_ZH[payload["verdict"]]
        policy = PLATFORM_POLICY.get(question_id)
        if policy and payload["verdict"] in (DISPUTED, RESOLVED_BY_PATIENT):
            payload["platform_policy"] = policy
        payload["notes"].extend(self._tradition_notes(live, question_id))
        return payload

    # -- helpers ---------------------------------------------------------
    def _positions(self, items: Iterable[Applicability], state: RunState) -> list[dict[str, Any]]:
        """Group records by (action, polarity) — one entry per distinct stance."""
        buckets: dict[tuple[str, str], dict[str, Any]] = {}
        for item in items:
            rec = item.recommendation
            key = (rec.action, _polarity(rec.direction))
            bucket = buckets.setdefault(key, {
                "action": rec.action,
                "polarity": key[1],
                "direction_zh": DIRECTION_ZH.get(rec.direction, rec.direction),
                "records": [],
                "sources": [],
                "regions": [],
                "traditions": [],
                "strengths": [],
                "subsumes": [],
            })
            bucket["records"].append({
                "rec_id": rec.rec_id,
                "evidence_id": state.ledger.for_recommendation(rec.rec_id),
                "statement_zh": rec.statement_zh,
                "strength": rec.strength,
                "strength_zh": STRENGTH_ZH.get(rec.strength, rec.strength),
                "direction": rec.direction,
                "evidence_level": rec.evidence_level,
                "status": item.status,
                "citation": rec.citation,
                "source": item.source.to_dict(),
            })
            if item.source.source_id not in bucket["sources"]:
                bucket["sources"].append(item.source.source_id)
            region = REGION_ZH.get(item.source.region, item.source.region)
            if region not in bucket["regions"]:
                bucket["regions"].append(region)
            tradition = TRADITION_ZH.get(item.source.tradition, item.source.tradition)
            if tradition not in bucket["traditions"]:
                bucket["traditions"].append(tradition)
            bucket["strengths"].append(rec.strength)
            for subsumed in rec.subsumes:
                if subsumed not in bucket["subsumes"]:
                    bucket["subsumes"].append(subsumed)
        return list(buckets.values())

    def _conflict(self, positions: list[dict[str, Any]], exclusive: bool) -> dict[str, Any]:
        by_action: dict[str, set[str]] = {}
        for position in positions:
            by_action.setdefault(position["action"], set()).add(position["polarity"])

        opposed = [action for action, polarities in by_action.items() if len(polarities) > 1]
        if opposed:
            return {"conflict": True, "kind": "polarity",
                    "why": f"同一措施上出现相反方向的推荐（{'、'.join(opposed)}）"}

        positive_actions = {
            position["action"] for position in positions if position["polarity"] == "+"
        }
        # An action that another present action already satisfies is not a rival
        # answer — drop it before counting. ("至少 1.0 g/kg" vs "1.2～1.5 g/kg")
        subsumed = {
            action
            for position in positions if position["polarity"] == "+"
            for action in position["subsumes"]
        }
        positive_actions = sorted(positive_actions - subsumed)
        if exclusive and len(positive_actions) >= 2:
            return {"conflict": True, "kind": "alternatives",
                    "why": f"这是一个互斥问题，但指南给出了 {len(positive_actions)} 种不同答案"}
        return {"conflict": False, "kind": "", "why": ""}

    def _divergence(self, positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for position in positions:
            titles = [record["source"]["title_zh"] for record in position["records"]]
            years = [record["source"]["year"] for record in position["records"]]
            out.append({
                "action": position["action"],
                "stance": position["direction_zh"],
                "polarity": position["polarity"],
                "regions": position["regions"],
                "traditions": position["traditions"],
                "sources": titles,
                "year_range": [min(years), max(years)] if years else [],
                "strongest": min(position["strengths"], key=_strength_rank) if position["strengths"] else "",
                "records": position["records"],
            })
        out.sort(key=lambda item: (-len(item["sources"]), item["action"]))
        return out

    def _deciding_facts(
        self, excluded: list[Applicability], inapplicable: list[Applicability] | None = None
    ) -> list[dict[str, Any]]:
        """The patient's own facts that took one side of the dispute away.

        Two ways a side can fall: an exclusion fired (a contraindication — the
        ``true`` leaves of ``excluded_when``), or an applicability condition
        simply did not hold (the ``false`` leaves of ``applies_when``). Both are
        answers to "why does this guideline not apply to me", so both are shown.
        """
        facts: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(leaf: Any, kind: str) -> None:
            key = f"{kind}:{leaf.detail or leaf.label}"
            if key in seen:
                return
            seen.add(key)
            facts.append({"label": leaf.label, "detail": leaf.detail,
                          "fact": leaf.fact, "kind": kind})

        for item in excluded:
            if item.exclusion_trace is None:
                continue
            for leaf in _true_leaves(item.exclusion_trace):
                add(leaf, "禁忌/排除")
        for item in inapplicable or []:
            for leaf in _false_leaves(item.trace):
                add(leaf, "适用条件不成立")
        return facts

    def _inapplicability_reason(self, item: Applicability) -> str:
        leaves = _false_leaves(item.trace)
        if leaves:
            return "适用条件不成立：" + "；".join(
                f"{leaf.label}（{leaf.detail}）" if leaf.detail else leaf.label for leaf in leaves
            )
        return "适用条件不成立"

    def _decision_support(self, state: RunState, question_id: str) -> dict[str, Any]:
        present = [
            {"fact": name, "label": label,
             "value": format_value(name, state.facts.get(name)),
             "raw": state.facts.get(name)}
            for name, label in _DECISION_FACTS
            if state.facts.get(name) is not None
        ]
        return {
            "statement": "本平台不对存在争议的问题替医师做选择；以下为可用于本例权衡的患者特异性因素。",
            "patient_factors": present,
            "next_step": "由主管医师结合可及性、患者偏好与共病负担决定，并把决策理由写入病历。",
        }

    def _tradition_notes(self, live: list[Applicability], question_id: str) -> list[str]:
        if not live:
            return []
        traditions = {item.source.tradition for item in live}
        if traditions == {"tcm"}:
            return [
                "该问题下仅中医指南作出推荐，现代医学指南未涉及——这是证据体系与关注问题不同，"
                "不等于现代医学指南持反对意见。"
            ]
        if "tcm" in traditions and len(traditions) > 1:
            return ["该问题同时有中医与现代医学来源的推荐，二者在本例中方向一致。"]
        return []

    def _exclusion_reason(self, item: Applicability) -> str:
        if item.exclusion_trace is None:
            return "排除条件成立"
        leaves = _true_leaves(item.exclusion_trace)
        if leaves:
            return "；".join(
                f"{leaf.label}（{leaf.detail}）" if leaf.detail else leaf.label for leaf in leaves
            )
        return item.exclusion_trace.label

    def notes(self, payload: dict[str, Any]) -> list[str]:
        notes: list[str] = []
        counts = payload["counts"]
        notes.append(
            "冲突消解覆盖 "
            + str(sum(counts.values()))
            + " 个临床问题："
            + "、".join(f"{VERDICT_ZH[key]} {value}" for key, value in counts.items())
        )
        for question in payload["disputed"]:
            notes.append(f"存在争议：{question['label_zh']}——{question['basis']}")
        for question in payload["resolved"]:
            facts = "、".join(
                fact["detail"] or fact["label"] for fact in question["resolution"]["deciding_facts"]
            )
            notes.append(f"争议已消解：{question['label_zh']}（决定性事实：{facts or '本患者的排除条件'}）")
        return notes


_STRENGTH_ORDER = {"strong": 0, "consensus": 1, "conditional": 2, "expert": 3, "good_practice": 4}


def _strength_rank(strength: str) -> int:
    return _STRENGTH_ORDER.get(strength, 9)


def _false_leaves(node: Any) -> list[Any]:
    """Leaves that evaluated ``false`` — why an applicability condition failed."""
    if node.result != "false":
        return []
    if not node.children:
        return [node]
    leaves: list[Any] = []
    for child in node.children:
        leaves.extend(_false_leaves(child))
    return leaves or [node]


def _true_leaves(node: Any) -> list[Any]:
    if node.result != "true":
        return []
    if not node.children:
        return [node]
    leaves: list[Any] = []
    for child in node.children:
        leaves.extend(_true_leaves(child))
    return leaves or [node]
