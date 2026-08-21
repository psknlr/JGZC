"""Terminal rendering of a decision.

Same content as the console, in the order a clinician reads it: who the patient
is, what the platform concluded, what it is unsure about, what to do, what must
be resolved first, and when to look again. Evidence ids are printed inline —
``[E07]`` — and resolve against the ledger printed at the end, so a paper copy
is as auditable as the web page.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

_RULE = "─" * 74


def _head(title: str) -> str:
    return f"\n{title}\n{_RULE}"


def _cite(ids: Sequence[str]) -> str:
    return f" [{'/'.join(ids)}]" if ids else ""


def render(decision: Mapping[str, Any], *, evidence: bool = True, width: int = 74) -> str:
    lines: list[str] = []
    meta = decision.get("meta", {})
    status_zh = {"ok": "已完成", "needs_action": "存在阻断性问题", "degraded": "部分环节失败"}
    lines.append(f"{meta.get('platform', '')} v{meta.get('version', '')}")
    lines.append(f"病例：{meta.get('case_label') or meta.get('case_id') or '未命名'}")
    lines.append(f"状态：{status_zh.get(decision.get('status'), decision.get('status'))}")
    corpus = meta.get("corpus", {})
    lines.append(
        f"语料：{corpus.get('recommendations')} 条推荐 / {corpus.get('sources')} 部指南 / "
        f"{corpus.get('questions')} 个临床问题（{corpus.get('year_range', ['', ''])[0]}–"
        f"{corpus.get('year_range', ['', ''])[1]}）"
    )

    # --- 患者数字画像 ---------------------------------------------------
    lines.append(_head("一、患者数字画像"))
    for group in decision.get("profile", {}).get("groups", []):
        items = "；".join(f"{item['label']} {item['value']}" for item in group["items"])
        lines.append(f"  [{group['label']}] {items}")
    missing = decision.get("profile", {}).get("missing_critical", [])
    if missing:
        lines.append("  ⚠ 关键缺失：")
        for item in missing:
            lines.append(f"      · {item['label']} —— 影响{item['blocks']}")

    # --- 诊断 ------------------------------------------------------------
    diagnosis = decision.get("diagnosis", {})
    if diagnosis:
        lines.append(_head("二、骨肌诊断（多标准并列）"))
        osteo = diagnosis["osteoporosis"]
        lines.append(f"  骨：{osteo['diagnosis_zh']}——{osteo['basis']}")
        if osteo["very_high_risk"]:
            drivers = "、".join(d["label"] for d in osteo["risk_drivers"])
            lines.append(f"      骨折风险分层：极高（{drivers}）")
        for caveat in osteo.get("caveats", []):
            lines.append(f"      ⚠ {caveat}")
        lines.append("  肌：")
        for standard in diagnosis["standards"]:
            lines.append(f"      {standard['name_zh']:<24s} {standard['verdict_zh']}")
            lines.append(f"          依据：{standard['reason']}")
            for axis, axis_zh in (("strength", "肌力"), ("performance", "躯体功能"), ("mass", "肌量")):
                component = standard["components"][axis]
                mark = {"true": "低", "false": "正常", "unknown": "未测"}[component["result"]]
                lines.append(f"          · {axis_zh:<5s}{mark:<4s}{component['detail']}")
        for note in diagnosis["sarcopenia"]["cross_standard_notes"]:
            lines.append(f"      ※ {note}")
        lines.append(f"  共病：{diagnosis['osteosarcopenia']['text']}")
        for interaction in diagnosis["osteosarcopenia"]["interactions"]:
            lines.append(f"      · {interaction}")

    # --- 风险 ------------------------------------------------------------
    risk = decision.get("risk", {})
    if risk:
        lines.append(_head("三、骨肌风险雷达"))
        for axis in risk["axes"]:
            bar = "█" * round(axis["score"] / 5)
            lines.append(
                f"  {axis['name_zh']:<8s} {axis['score']:>3d} {axis['tier_zh']:<4s} "
                f"置信 {axis['confidence']:.0%}  {bar}"
            )
            drivers = "、".join(f"{d['label']}(+{d['points']})" for d in axis["drivers"][:4])
            if drivers:
                lines.append(f"      主要因素：{drivers}")
            if axis.get("note"):
                lines.append(f"      ⚠ {axis['note']}")
        lines.append(f"  说明：{risk['model_note']}")

    # --- AI 判断 ----------------------------------------------------------
    judgement = decision.get("judgement", {})
    if judgement:
        lines.append(_head("四、AI 判断"))
        lines.append(f"  {judgement.get('headline', '')}")
        for statement in judgement.get("statements", []):
            lines.append(f"  · [{statement['kind']}] {statement['text']}{_cite(statement['evidence'])}")
            if statement.get("detail"):
                lines.append(f"        {statement['detail']}")
        if judgement.get("uncertainties"):
            lines.append("  不确定性与数据缺口：")
            for note in judgement["uncertainties"]:
                lines.append(f"      · {note}")

    # --- 冲突 -------------------------------------------------------------
    conflicts = decision.get("conflicts", {})
    if conflicts:
        lines.append(_head("五、指南冲突消解"))
        counts = "；".join(f"{k} {v}" for k, v in conflicts.get("counts_zh", {}).items())
        lines.append(f"  {counts}")
        for question in conflicts.get("disputed", []) + conflicts.get("resolved", []):
            lines.append(f"\n  【{question['verdict_zh']}】{question['label_zh']}")
            lines.append(f"      {question['basis']}")
            for position in question.get("divergence", []):
                sources = "；".join(position["sources"])
                lines.append(
                    f"      ▸ {position['stance']}「{position['action']}」"
                    f"（{'/'.join(position['regions'])} · {'/'.join(position['traditions'])}）"
                )
                lines.append(f"          {sources}")
            for removed in question.get("resolution", {}).get("removed", []):
                lines.append(f"      ✖ 去除「{removed['action']}」（{removed['source']}）：{removed['reason']}")
            if question.get("platform_policy"):
                lines.append(f"      ⚙ {question['platform_policy']}")
            if question.get("decision_support"):
                support = question["decision_support"]
                factors = "、".join(
                    f"{f['label']}={f['value']}" for f in support["patient_factors"][:6]
                )
                lines.append(f"      ⚖ {support['statement']}")
                lines.append(f"          本例可用于权衡的因素：{factors}")

    # --- 方案 -------------------------------------------------------------
    plans = decision.get("plans", {})
    if plans:
        lines.append(_head("六、个体化治疗路径"))
        for column in plans["columns"]:
            lines.append(f"\n  【{column['label']}】")
            for item in column["items"]:
                gate = "  ⛔已被安全审查阻断" if item.get("gated") else ""
                lines.append(f"   {item['priority']}. {item['title']}{_cite(item['evidence'])}{gate}")
                lines.append(f"      {item['detail']}")
                lines.append(f"      理由：{item['why']}")
                if item.get("owner"):
                    lines.append(f"      执行：{item['owner']}")
                for caution in item["cautions"]:
                    lines.append(f"      ⚠ {caution}")
                for conflict in item["conflicts"]:
                    lines.append(f"      ⚖ 本条涉及「{conflict['label_zh']}」：{conflict['verdict_zh']}")
        if plans.get("suppressed"):
            lines.append("\n  未生成的条目（其支持性推荐在本例中不成立）：")
            for entry in plans["suppressed"]:
                lines.append(f"      · {entry['title']} —— {entry['reason']}")

    # --- 安全 -------------------------------------------------------------
    safety = decision.get("safety", {})
    if safety:
        lines.append(_head("七、安全审查"))
        lines.append(f"  已跑检查：{'、'.join(safety['checks_zh'])}")
        lines.append(f"  结论：{ {'clear': '未见需处理问题', 'caution': '有需处理问题', 'blocked': '存在阻断性问题'}[safety['clearance']] }")
        for issue in safety["issues"]:
            lines.append(f"  [{issue['severity_zh']}] {issue['check_zh']} · {issue['title']}")
            lines.append(f"        {issue['finding']}")
            lines.append(f"        → {issue['action']}")
        for action in safety.get("fall_safety", []):
            lines.append(f"  · 跌倒与居家：{action['title']} —— {action['detail']}")

        lines.append(_head("八、随访计划"))
        for key, label in (("immediate", "即刻（本次就诊完成）"), ("m3", "3 个月"),
                           ("m6", "6 个月"), ("m12", "12 个月")):
            items = safety["followup"].get(key, [])
            if not items:
                continue
            lines.append(f"  【{label}】")
            for item in items:
                owner = f"（{item['owner']}）" if item.get("owner") else ""
                lines.append(f"      · {item['what']}{owner}")
                lines.append(f"          为什么：{item['why']}")

    # --- 证据台账 ---------------------------------------------------------
    if evidence and decision.get("evidence"):
        lines.append(_head("九、证据台账"))
        for evidence_id, entry in decision["evidence"].items():
            rec = entry["recommendation"]
            source = entry["source"]
            status = {"applies": "适用", "excluded": "已排除",
                      "not_applicable": "不适用", "insufficient_data": "数据不足"}[entry["status"]]
            lines.append(f"  {evidence_id} [{status}] {source['title_zh']}（{source['issuer']}，{source['year']}）")
            lines.append(f"      {rec['statement_zh']}")
            lines.append(f"      强度：{rec['strength']}  出处：{rec['citation']}  来源类型：{rec['provenance']}")

    narrative = decision.get("narrative")
    if narrative:
        lines.append(_head("十、叙述性小结"))
        lines.append(f"  （{narrative['note']}）")
        for line in narrative["text"].splitlines():
            lines.append(f"  {line}")

    lines.append("")
    lines.append(_RULE)
    lines.append(meta.get("disclaimer", ""))
    return "\n".join(lines)
