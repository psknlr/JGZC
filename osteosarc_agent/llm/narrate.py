"""The narrator: a model may restate conclusions, never reach them.

Everything clinical in this platform is decided before a model is consulted —
the diagnosis, the risk tiers, which recommendations are in force, which
conflicts survived, which safety gates are closed. The narrator's whole job is
to turn that structure into two paragraphs a clinician or a patient can read.

Which makes the guard the interesting part. The model is given only the
already-decided conclusions, and its output is checked before it is kept:

* **Dose scan.** Any number followed by a dose unit that does not appear
  verbatim in the deterministic output is a fabricated dose, and the whole
  narrative is discarded. Not edited — discarded. A narrative that had to be
  repaired is not one to trust with the rest of its sentences.
* **Drug-name allowlist.** Drug names the pipeline never mentioned cannot appear.
  This is what stops "可以考虑阿仑膦酸钠" from being written for a patient whose
  eGFR took that drug off the table.
* **Negation guard.** When the safety agent recorded blocking issues, a
  narrative that reads as an all-clear is rejected.

A rejected narrative degrades to the deterministic summary and says so. The
structured decision is the product; the narrative is a convenience on top of it.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from ..safety.drugs import CLASS_LABELS, DRUG_CLASSES
from .base import LLMClient, LLMError, NullLLMClient

SYSTEM_PROMPT = (
    "你是骨质疏松与肌少症临床决策平台的表述层。"
    "你的唯一任务是把下面已经确定的结论改写成通顺的中文说明。"
    "硬性约束："
    "（1）不得引入结论中没有的诊断、药物、检查或数值；"
    "（2）不得给出任何剂量、克数、毫克数、IU 数；"
    "（3）不得弱化或省略任何标注为「阻断」「高」的安全问题；"
    "（4）不得替医师在存在争议的问题上选边；"
    "（5）不使用「建议您服用」这类直接医嘱口吻，改用「本次评估提示」。"
    "输出两段：第一段面向医师（结论与依据），第二段面向患者与家属（通俗、可执行）。"
)

#: Units that make a number a dose. Deliberately broad.
_DOSE_UNITS = r"(?:mg|毫克|g\b|克|μg|ug|微克|IU|国际单位|mL|毫升|片|粒|支)"
_DOSE_PATTERN = re.compile(rf"(\d+(?:\.\d+)?\s*(?:～|~|-|—)?\s*\d*(?:\.\d+)?\s*{_DOSE_UNITS})", re.I)

#: Every drug name fragment the corpus and drug table know about.
_KNOWN_DRUG_TERMS = tuple(
    sorted({fragment for fragments in DRUG_CLASSES.values() for fragment in fragments}, key=len, reverse=True)
)

_ALL_CLEAR_PATTERNS = (
    "无安全问题", "未见明显异常", "没有需要注意", "一切正常", "无需特殊处理", "无禁忌",
    "无特殊注意事项", "安全性良好",
)

#: Words that turn a drug mention into a prohibition rather than a suggestion.
#: A heuristic, and knowingly a crude one — its failure mode is rejecting a
#: correct narrative, which costs a paragraph, not a patient.
_NEGATION_MARKERS = ("不", "避免", "禁用", "禁忌", "排除", "停用", "慎用", "无法", "不宜", "非")
_NEGATION_WINDOW = 20


def denied_classes(decision: Mapping[str, Any]) -> set[str]:
    """Drug classes this patient must not receive, per the safety audit."""
    denied: set[str] = set()
    for finding in (decision.get("safety", {}).get("issues") or []):
        if finding.get("check") == "renal" and "禁" in finding.get("action", ""):
            denied.add(str(finding.get("ref", "")))
    for finding in (decision.get("safety", {}).get("issues") or []):
        if finding.get("check") != "renal":
            continue
        ref = str(finding.get("ref", ""))
        if ref in DRUG_CLASSES and "不推荐" in finding.get("finding", ""):
            denied.add(ref)
    for entry in (decision.get("evidence") or {}).values():
        if entry.get("status") != "excluded":
            continue
        for tag in entry.get("recommendation", {}).get("tags", []):
            for drug_class, label in CLASS_LABELS.items():
                if tag == label:
                    denied.add(drug_class)
    return {cls for cls in denied if cls in DRUG_CLASSES}


class Narrator:
    """Wraps a client and enforces the guards."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or NullLLMClient()

    def narrate(self, decision: Mapping[str, Any]) -> dict[str, Any]:
        deterministic = summarise(decision)
        if isinstance(self.client, NullLLMClient):
            return {
                "mode": "deterministic",
                "text": deterministic,
                "note": "未配置模型，本段为确定性摘要。",
                "problems": [],
            }
        try:
            raw = self.client.complete(SYSTEM_PROMPT, _build_prompt(decision))
        except LLMError as exc:
            return {
                "mode": "deterministic",
                "text": deterministic,
                "note": f"模型调用失败，已回退确定性摘要：{exc}",
                "problems": [str(exc)],
            }

        ok, problems = check_narrative(raw, decision)
        if not ok:
            return {
                "mode": "rejected",
                "text": deterministic,
                "note": "模型输出未通过安全校验，已整体丢弃并回退确定性摘要。",
                "problems": problems,
                "rejected_text": raw,
            }
        return {
            "mode": "model",
            "text": raw.strip(),
            "note": "本段由模型改写自上方确定性结论，不含新的临床判断。",
            "problems": [],
        }


def _allowed_text(decision: Mapping[str, Any]) -> str:
    """Everything the pipeline said *affirmatively*, as one blob.

    Excluded records are deliberately left out. Their text mentions the very
    drugs this patient must not receive, so counting it as licensing vocabulary
    would let the narrator recommend the drug the pipeline just ruled out.
    """
    parts: list[str] = []
    for entry in (decision.get("evidence") or {}).values():
        if entry.get("status") != "applies":
            continue
        rec = entry.get("recommendation", {})
        parts.append(rec.get("statement_zh", ""))
        parts.append(rec.get("rationale", ""))
    for column in (decision.get("plans", {}).get("columns") or []):
        for item in column.get("items", []):
            parts.extend([item.get("title", ""), item.get("detail", ""), item.get("why", "")])
            parts.extend(item.get("cautions", []))
    for issue in (decision.get("safety", {}).get("issues") or []):
        # Only the *action* line, not the finding: a finding quotes the excluded
        # recommendation and would smuggle its drug names into the allowlist.
        parts.append(issue.get("action", ""))
    for bucket in ("immediate", "m3", "m6", "m12"):
        for item in (decision.get("safety", {}).get("followup", {}).get(bucket) or []):
            parts.extend([item.get("what", ""), item.get("why", "")])
    for statement in (decision.get("judgement", {}).get("statements") or []):
        parts.extend([statement.get("text", ""), statement.get("detail", "")])
    return "\n".join(part for part in parts if part)


def _affirmative_mentions(text: str, term: str) -> bool:
    """True when ``term`` appears without a negation in the preceding window.

    "不宜使用阿仑膦酸钠" is a prohibition and is fine to write; "可以考虑阿仑膦酸钠"
    is a recommendation the pipeline never made. Distinguishing them by a
    lookbehind window is crude, and errs toward rejection, which costs a
    paragraph rather than a patient.
    """
    for match in re.finditer(re.escape(term), text, re.I):
        window = text[max(0, match.start() - _NEGATION_WINDOW):match.start()]
        if not any(marker in window for marker in _NEGATION_MARKERS):
            return True
    return False


def check_narrative(text: str, decision: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Return ``(ok, problems)``. Any problem means the narrative is discarded."""
    problems: list[str] = []
    if not text or not text.strip():
        return False, ["模型返回空内容"]

    allowed = _allowed_text(decision)
    allowed_normalised = re.sub(r"\s+", "", allowed)

    for match in _DOSE_PATTERN.findall(text):
        if re.sub(r"\s+", "", match) not in allowed_normalised:
            problems.append(f"出现了确定性结论中没有的剂量表述: {match!r}")

    allowed_lower = allowed.lower()
    for term in _KNOWN_DRUG_TERMS:
        if term.lower() in allowed_lower:
            continue
        if _affirmative_mentions(text, term):
            problems.append(f"在肯定语境中提到了本例结论中未出现的药物: {term!r}")

    denied = denied_classes(decision)
    for drug_class in sorted(denied):
        for term in DRUG_CLASSES[drug_class]:
            if _affirmative_mentions(text, term):
                problems.append(
                    f"在肯定语境中提到了本例已排除的药物类别: "
                    f"{CLASS_LABELS.get(drug_class, drug_class)}（{term}）"
                )

    serious = (decision.get("safety", {}).get("blocking") or []) or [
        issue for issue in (decision.get("safety", {}).get("issues") or [])
        if issue.get("severity") == "high"
    ]
    if serious:
        for phrase in _ALL_CLEAR_PATTERNS:
            if phrase in text:
                problems.append(f"存在阻断性或高风险安全问题时出现了放行式表述: {phrase!r}")

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique = [p for p in problems if not (p in seen or seen.add(p))]
    return not unique, unique


def _build_prompt(decision: Mapping[str, Any]) -> str:
    return "以下是已经确定的结论，请据此改写：\n\n" + summarise(decision)


def summarise(decision: Mapping[str, Any]) -> str:
    """The deterministic summary — also the fallback when a narrative is rejected."""
    lines: list[str] = []
    meta = decision.get("meta", {})
    lines.append(f"【病例】{meta.get('case_label') or meta.get('case_id') or '未命名'}")

    judgement = decision.get("judgement", {})
    if judgement.get("headline"):
        lines.append(f"【总体】{judgement['headline']}")

    diagnosis = decision.get("diagnosis", {})
    if diagnosis:
        osteo = diagnosis.get("osteoporosis", {})
        lines.append(f"【骨】{osteo.get('diagnosis_zh', '')}——{osteo.get('basis', '')}")
        verdicts = "；".join(
            f"{standard['name_zh']} {standard['verdict_zh']}"
            for standard in diagnosis.get("standards", [])
        )
        lines.append(f"【肌】{verdicts}")
        lines.append(f"【共病】{diagnosis.get('osteosarcopenia', {}).get('text', '')}")

    radar = decision.get("risk", {}).get("radar", {}).get("axes", [])
    if radar:
        lines.append("【风险】" + "；".join(
            f"{axis['label']} {axis['score']}（{axis['tier_zh']}）" for axis in radar
        ))

    conflicts = decision.get("conflicts", {})
    for question in conflicts.get("disputed", []) or []:
        lines.append(f"【争议】{question['label_zh']}：{question.get('basis', '')}")
    for question in conflicts.get("resolved", []) or []:
        lines.append(f"【争议已消解】{question['label_zh']}")

    for column in decision.get("plans", {}).get("columns", []):
        titles = "；".join(item["title"] for item in column.get("items", [])[:4])
        if titles:
            lines.append(f"【{column['label']}】{titles}")

    safety = decision.get("safety", {})
    for issue in (safety.get("blocking") or []):
        lines.append(f"【阻断】{issue['title']}：{issue['action']}")
    high = [issue for issue in (safety.get("issues") or []) if issue.get("severity") == "high"]
    if high:
        lines.append("【高风险提示】" + "；".join(issue["title"] for issue in high[:5]))

    followup = safety.get("followup", {})
    for key, label in (("immediate", "即刻"), ("m3", "3 个月"), ("m6", "6 个月"), ("m12", "12 个月")):
        items = followup.get(key) or []
        if items:
            lines.append(f"【随访·{label}】" + "；".join(item["what"] for item in items[:4]))

    for note in (judgement.get("uncertainties") or [])[:4]:
        lines.append(f"【不确定性】{note}")
    if meta.get("disclaimer"):
        lines.append(f"【声明】{meta['disclaimer']}")
    return "\n".join(lines)
