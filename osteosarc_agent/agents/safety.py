"""Agent 6 —— 安全审查与随访 Agent.

Runs last, over the assembled plan, in a fixed order:

    禁忌证 → 药物相互作用 → 肾功能 → 低钙风险 → 跌倒风险 → 依从性 → 随访节点

The order is the point. Checking interactions before knowing renal function
produces a tidy report about a drug the patient should not be given at all; and
the calcium gate has to close before any anti-resorptive is issued, not after.

Two rules this agent enforces on the platform itself:

* **A check that did not run is reported as not-run.** The schema requires every
  check name in ``checks_run``; a partially-run audit is rejected rather than
  published, because a safety report with a silent gap reads as an all-clear.
* **Blocking issues block.** Anything in ``blocking`` means the drug portion of
  the plan is not ready to issue. The orchestrator marks the whole run
  ``needs_action`` and the console shows it before anything else.

Follow-up is generated from what this specific patient is about to receive —
a denosumab pathway in CKD earns early calcium checks that a bisphosphonate
pathway does not — and always lands on the 3 / 6 / 12-month milestones.
"""

from __future__ import annotations

from typing import Any

from ..safety import drugs
from ..state import RunState
from .base import SubAgent

CHECK_ORDER = (
    ("contraindication", "禁忌证"),
    ("interaction", "药物相互作用"),
    ("renal", "肾功能"),
    ("hypocalcemia", "低钙风险"),
    ("fall_risk_drugs", "跌倒风险"),
    ("adherence", "依从性"),
    ("followup", "随访节点"),
)

BLOCKING = "blocking"
HIGH = "high"
MODERATE = "moderate"
ADVISORY = "advisory"

SEVERITY_ZH = {BLOCKING: "阻断", HIGH: "高", MODERATE: "中", ADVISORY: "提示"}


class SafetyAgent(SubAgent):
    agent_id = "safety"
    name_zh = "安全审查与随访 Agent"
    schema = "SafetyAndFollowup"

    def run(self, state: RunState) -> dict[str, Any]:
        facts = state.facts
        classes = set(facts.get("medication_classes") or [])
        egfr = _num(facts.get("egfr"))
        issues: list[dict[str, Any]] = []
        checks_run: list[str] = []

        # 1) 禁忌证 ------------------------------------------------------
        checks_run.append("contraindication")
        issues.extend(self._contraindications(state, classes, egfr))

        # 2) 药物相互作用 -------------------------------------------------
        checks_run.append("interaction")
        for found in drugs.find_interactions(classes):
            issues.append({
                "check": "interaction",
                "severity": HIGH if found["severity"] == "high" else MODERATE,
                "title": "、".join(found["classes"]) + " 相互作用",
                "finding": found["finding"],
                "action": found["action"],
                "ref": found["rule_id"],
            })
        unrecognised = [
            med["name"] for med in state.payload("intake").get("medications", [])
            if not med["recognised"]
        ]
        if unrecognised:
            issues.append({
                "check": "interaction",
                "severity": HIGH,
                "title": "存在未能识别的药物",
                "finding": f"未识别：{'、'.join(unrecognised)}——相互作用与跌倒风险统计未包含它们",
                "action": "由临床药师人工核对这些药物后重跑安全审查",
                "ref": "",
            })

        # 3) 肾功能 ------------------------------------------------------
        checks_run.append("renal")
        candidate_classes = classes | ({"bisphosphonate", "denosumab"} if facts.get("dx_osteoporosis") else set())
        for finding in drugs.renal_findings(candidate_classes, egfr):
            severity = BLOCKING if finding["verdict"] == "unknown" else (
                HIGH if finding["verdict"] == "contraindicated" else MODERATE
            )
            issues.append({
                "check": "renal",
                "severity": severity,
                "title": f"肾功能门槛：{finding['label']}",
                "finding": finding["note"],
                "action": (
                    "用药前必须先查 eGFR" if finding["verdict"] == "unknown"
                    else "按肾功能选择药物并记录选择理由"
                ),
                "ref": finding.get("drug_class", ""),
            })
        if egfr is not None and egfr < 45:
            issues.append({
                "check": "renal",
                "severity": MODERATE,
                "title": "需鉴别 CKD-MBD",
                "finding": f"eGFR {egfr:g}（{facts.get('ckd_stage') or '分期未定'}），"
                           "此时的骨骼异常需先鉴别慢性肾病矿物质与骨代谢异常",
                "action": "查 PTH、血磷、碱性磷酸酶，必要时肾内科共管",
                "ref": "LABEL.EXCLUDE_SECONDARY",
            })

        # 4) 低钙风险 ----------------------------------------------------
        checks_run.append("hypocalcemia")
        issues.extend(self._calcium_gate(state, classes, egfr))

        # 5) 跌倒风险 ----------------------------------------------------
        checks_run.append("fall_risk_drugs")
        issues.extend(self._fall_safety(state, classes))

        # 6) 依从性 ------------------------------------------------------
        checks_run.append("adherence")
        issues.extend(self._adherence(state, classes))

        # 7) 随访 --------------------------------------------------------
        checks_run.append("followup")
        followup = self._followup(state, classes, egfr)

        order = {BLOCKING: 0, HIGH: 1, MODERATE: 2, ADVISORY: 3}
        issues.sort(key=lambda issue: order.get(issue["severity"], 9))
        for issue in issues:
            issue["severity_zh"] = SEVERITY_ZH.get(issue["severity"], issue["severity"])
            issue["check_zh"] = dict(CHECK_ORDER).get(issue["check"], issue["check"])

        blocking = [issue for issue in issues if issue["severity"] == BLOCKING]
        return {
            "checks_run": checks_run,
            "checks_zh": [label for _, label in CHECK_ORDER],
            "issues": issues,
            "blocking": blocking,
            "followup": followup,
            "fall_safety": self._fall_actions(state),
            "clearance": "blocked" if blocking else ("caution" if issues else "clear"),
        }

    # -- individual checks ----------------------------------------------
    def _contraindications(self, state: RunState, classes: set[str], egfr: float | None) -> list[dict[str, Any]]:
        facts = state.facts
        found: list[dict[str, Any]] = []

        # Several guidelines usually exclude the same action for the same
        # reason. Report the action once with all its sources, rather than once
        # per source — a list that repeats itself reads as more findings than
        # there are, and the reader stops counting.
        by_action: dict[str, dict[str, Any]] = {}
        for entry in state.payload("evidence").get("excluded", []):
            if entry["direction"] in ("avoid", "against"):
                continue
            if entry["topic"] not in ("bone_protection", "nutrition", "exercise", "medication_safety"):
                continue
            bucket = by_action.setdefault(entry["action"], {
                "check": "contraindication",
                "severity": HIGH,
                "title": f"已排除的措施：{entry['action']}",
                "finding": f"{entry['statement_zh'][:60]}…",
                "action": f"排除依据：{entry['excluded_because']}——不得按该条实施",
                "ref": entry["rec_id"],
                "evidence": [],
                "sources": [],
            })
            bucket["evidence"].append(entry["evidence_id"])
            bucket["sources"].append(entry["source"]["title_zh"])
        for bucket in by_action.values():
            if len(bucket["sources"]) > 1:
                bucket["finding"] += f"（{len(bucket['sources'])} 部指南一致排除）"
            found.append(bucket)

        if facts.get("recent_cv_event_12m"):
            found.append({
                "check": "contraindication",
                "severity": BLOCKING,
                "title": "罗莫索单抗禁用",
                "finding": "近 12 个月内心肌梗死或脑卒中——说明书列为禁忌",
                "action": "在药物选择中排除罗莫索单抗",
                "ref": "US.ACP.2023.ROMOSOZUMAB_CAUTION",
            })
        if facts.get("esophageal_disease") or facts.get("cannot_stay_upright"):
            found.append({
                "check": "contraindication",
                "severity": HIGH,
                "title": "口服双膦酸盐剂型禁忌",
                "finding": "活动性食管疾病或不能服后保持直立 30 分钟",
                "action": "若确需双膦酸盐，改用静脉剂型；并复核肾功能门槛",
                "ref": "US.ES.2020.ORAL_BP_ADMIN",
            })
        if facts.get("skeletal_radiotherapy") or facts.get("malignancy"):
            found.append({
                "check": "contraindication",
                "severity": HIGH,
                "title": "促骨形成药物需先排除",
                "finding": "骨骼放疗史或恶性肿瘤史——特立帕肽使用前须排除骨恶性肿瘤与骨转移",
                "action": "完成排查并记录后再考虑促骨形成药物",
                "ref": "US.ES.2020.TERIPARATIDE_LIMITS",
            })
        if facts.get("dental_extraction_pending"):
            found.append({
                "check": "contraindication",
                "severity": HIGH,
                "title": "口腔病灶未处理",
                "finding": "有待处理的拔牙或牙病，此时启动抗骨吸收药物增加颌骨坏死风险",
                "action": "先完成口腔处理，再启动抗骨吸收治疗",
                "ref": "US.ES.2020.DENTAL_BEFORE_ANTIRESORPTIVE",
            })
        return found

    def _calcium_gate(self, state: RunState, classes: set[str], egfr: float | None) -> list[dict[str, Any]]:
        facts = state.facts
        found: list[dict[str, Any]] = []
        antiresorptive_planned = bool(facts.get("dx_osteoporosis")) or bool(
            classes & {"denosumab", "bisphosphonate"}
        )
        calcium = _num(facts.get("corrected_calcium"))
        vitd = _num(facts.get("vitamin_d_25oh"))

        if antiresorptive_planned and calcium is None:
            found.append({
                "check": "hypocalcemia",
                "severity": BLOCKING,
                "title": "未查校正血钙",
                "finding": "抗骨吸收治疗前必须知道血钙水平",
                "action": "查校正血钙（或血钙 + 白蛋白）后再决定用药",
                "ref": "CN.OP.2022.CORRECT_BEFORE_ANTIRESORPTIVE",
            })
        elif calcium is not None and facts.get("hypocalcemia"):
            found.append({
                "check": "hypocalcemia",
                "severity": BLOCKING,
                "title": "低钙血症未纠正",
                "finding": f"校正血钙 {calcium:g} mmol/L 低于正常下限",
                "action": "先纠正低钙与维生素 D，达标后再启动抗骨吸收治疗",
                "ref": "CN.OP.2022.CORRECT_BEFORE_ANTIRESORPTIVE",
            })

        if antiresorptive_planned and vitd is None:
            found.append({
                "check": "hypocalcemia",
                "severity": BLOCKING,
                "title": "未查 25-OH 维生素 D",
                "finding": "维生素 D 缺乏是抗骨吸收药物相关严重低钙血症的主要促发因素",
                "action": "查 25-OH-D 并补足后再启动治疗",
                "ref": "CN.OP.2022.CORRECT_BEFORE_ANTIRESORPTIVE",
            })
        elif facts.get("vitd_deficient"):
            found.append({
                "check": "hypocalcemia",
                "severity": HIGH,
                "title": "维生素 D 缺乏",
                "finding": f"25-OH-D {vitd:g} ng/mL < 20",
                "action": "按常规每日剂量补足并复查；不使用大剂量冲击方案",
                "ref": "US.BHOF.2022.VITD_TARGET",
            })

        if egfr is not None and egfr < 35 and (antiresorptive_planned or "denosumab" in classes):
            found.append({
                "check": "hypocalcemia",
                "severity": HIGH,
                "title": "晚期 CKD 使用地舒单抗的低钙血症风险",
                "finding": f"eGFR {egfr:g}——说明书对该人群有严重症状性低钙血症警示",
                "action": "用药前纠正钙与维 D；用药后 1～2 周、1 个月复查血钙；"
                          "告知手足抽搐、口周麻木需立即就诊",
                "ref": "LABEL.DENOSUMAB_HYPOCALCEMIA_CKD",
            })
        return found

    def _fall_safety(self, state: RunState, classes: set[str]) -> list[dict[str, Any]]:
        facts = state.facts
        found: list[dict[str, Any]] = []
        frids = sorted(classes & drugs.FALL_RISK_CLASSES)
        if frids:
            labels = "、".join(drugs.CLASS_LABELS.get(cls, cls) for cls in frids)
            severity = HIGH if len(frids) >= 2 or facts.get("high_fall_risk") else MODERATE
            found.append({
                "check": "fall_risk_drugs",
                "severity": severity,
                "title": f"增加跌倒风险的药物（{len(frids)} 类）",
                "finding": f"{labels}——镇静催眠与抗胆碱负荷是可逆的跌倒危险因素",
                "action": "制定渐进减停计划并随访；不可骤停苯二氮䓬类",
                "ref": "INTL.WFG.2022.MED_REVIEW",
            })
        harmful = sorted(classes & drugs.BONE_HARMFUL_CLASSES)
        if harmful:
            found.append({
                "check": "fall_risk_drugs",
                "severity": MODERATE,
                "title": "对骨骼有害的药物",
                "finding": "、".join(drugs.CLASS_LABELS.get(cls, cls) for cls in harmful),
                "action": "评估能否减量或替换；若必须继续，按更低阈值启动骨保护",
                "ref": "CN.GIOP.2021.LOWER_THRESHOLD",
            })
        if facts.get("high_fall_risk") and not facts.get("home_hazards"):
            found.append({
                "check": "fall_risk_drugs",
                "severity": ADVISORY,
                "title": "居家环境未评估",
                "finding": "已判定为高跌倒风险，但记录中没有居家环境信息",
                "action": "完成居家危险因素清单（扶手、防滑、照明、门槛、鞋具）",
                "ref": "CN.FALLS.2022.HOME_ENV",
            })
        return found

    def _adherence(self, state: RunState, classes: set[str]) -> list[dict[str, Any]]:
        facts = state.facts
        found: list[dict[str, Any]] = []
        if facts.get("polypharmacy"):
            found.append({
                "check": "adherence",
                "severity": MODERATE,
                "title": f"多重用药（{facts.get('medication_count')} 种）",
                "finding": "药物越多，漏服与相互作用越多，抗骨质疏松治疗的实际暴露越不确定",
                "action": "结构化用药重整；简化给药频次；考虑长间隔剂型",
                "ref": "EU.STOPP.2023.POLYPHARMACY",
            })
        if facts.get("cognitive_impairment") or facts.get("living_alone"):
            reasons = []
            if facts.get("cognitive_impairment"):
                reasons.append("认知障碍")
            if facts.get("living_alone"):
                reasons.append("独居")
            found.append({
                "check": "adherence",
                "severity": MODERATE,
                "title": "服药管理需要外部支持",
                "finding": "、".join(reasons) + "——口服每周或每日方案的漏服风险高",
                "action": "家属或社区协助；优先考虑医疗机构给药的长间隔方案；建立提醒机制",
                "ref": "",
            })
        if facts.get("dx_osteoporosis") and not facts.get("on_antiosteoporosis_therapy"):
            found.append({
                "check": "adherence",
                "severity": HIGH,
                "title": "处方遗漏",
                "finding": "已诊断骨质疏松但用药清单中没有抗骨质疏松药物（钙与维 D 属基础用药，不计入）",
                "action": "按 START 标准纠正处方遗漏",
                "ref": "EU.STOPP.2023.START_BONE",
            })
        if "denosumab" in classes:
            found.append({
                "check": "adherence",
                "severity": HIGH,
                "title": "地舒单抗的给药时间是安全问题，不只是依从性问题",
                "finding": "延迟或停用后骨转换反跳，存在多发椎体骨折风险",
                "action": "把每 6 个月的给药日期写入随访计划，并预设序贯衔接方案",
                "ref": "CN.OP.2022.DENOSUMAB_NO_STOP",
            })
        return found

    def _fall_actions(self, state: RunState) -> list[dict[str, Any]]:
        facts = state.facts
        actions: list[dict[str, Any]] = []
        if facts.get("high_fall_risk") or (facts.get("falls_12m") or 0) >= 1:
            actions.append({
                "title": "多因素跌倒评估与干预",
                "detail": "步态平衡训练、用药重整、卧立位血压、视力与足部鞋具、居家改造、骨健康评估同时推进",
                "ref": "INTL.WFG.2022.MULTIFACTORIAL",
            })
            actions.append({
                "title": "居家环境改造",
                "detail": "卫生间扶手与防滑垫、去门槛与松动地毯、夜间照明、常用物品放在易取高度、合脚防滑鞋",
                "ref": "CN.FALLS.2022.HOME_ENV",
            })
        if facts.get("frid_count"):
            actions.append({
                "title": "跌倒风险药物渐进减停",
                "detail": "按类别逐一评估指征与替代方案，渐进减量并在每次调整后随访跌倒与睡眠",
                "ref": "INTL.WFG.2022.MED_REVIEW",
            })
        if facts.get("fear_of_falling"):
            actions.append({
                "title": "打破「害怕跌倒—回避活动—更易跌倒」循环",
                "detail": "在监督下恢复活动，从低难度平衡任务起步并记录成功体验",
                "ref": "INTL.WFG.2022.EXERCISE_BALANCE",
            })
        return actions

    def _followup(self, state: RunState, classes: set[str], egfr: float | None) -> dict[str, Any]:
        facts = state.facts
        denosumab_path = bool(egfr is not None and egfr < 35 and facts.get("dx_osteoporosis")) or "denosumab" in classes
        antiresorptive = denosumab_path or bool(classes & {"bisphosphonate"}) or bool(facts.get("dx_osteoporosis"))

        immediate = [
            {"what": "校正血钙、25-OH 维生素 D、eGFR、血磷、碱性磷酸酶、PTH",
             "why": "抗骨吸收治疗的用药前置条件，同时用于排除继发因素", "owner": "门诊"},
        ]
        if not facts.get("vertebral_imaging_done"):
            immediate.append({
                "what": "胸腰椎侧位 X 线片或 VFA",
                "why": "查找无症状椎体骨折——一处椎体骨折即改变风险分层与用药选择",
                "owner": "影像科",
            })
        if not facts.get("asmi"):
            immediate.append({
                "what": "DXA 体成分或 BIA 测四肢骨骼肌量（ASMI）",
                "why": "三套肌少症标准的结论目前不一致，肌量是让它们收敛的关键测量",
                "owner": "康复科/体成分室",
            })

        m3: list[dict[str, Any]] = [
            {"what": "握力与 5 次起坐复测", "why": "运动干预的最早可见反应出现在肌力与功能，而不在骨密度", "owner": "康复治疗师"},
            {"what": "运动方案执行情况与进阶", "why": "抗阻训练需渐进增加负荷才有效", "owner": "康复治疗师"},
            {"what": "蛋白质摄入是否达到既定目标、体重变化", "why": "营养不足时运动无法转化为肌肉", "owner": "营养科"},
            {"what": "跌倒日记回顾与用药重整落实情况", "why": "跌倒危险因素中最可逆的一项是药物", "owner": "临床药师"},
        ]
        if antiresorptive:
            m3.insert(0, {"what": "复查校正血钙与 25-OH-D", "why": "抗骨吸收治疗期间的低钙监测", "owner": "门诊"})
        if denosumab_path:
            m3.insert(0, {
                "what": "地舒单抗给药后 1～2 周及 1 个月加查血钙",
                "why": "晚期 CKD 使用地舒单抗有严重症状性低钙血症风险（说明书警示）",
                "owner": "门诊 + 患者自我监测症状",
            })

        m6 = [
            {"what": "握力、步速、SPPB 与肌少症标准重新判定", "why": "评估干预是否改变了诊断层级", "owner": "老年科/康复科"},
            {"what": "肾功能与电解质复查", "why": "药物选择与蛋白目标都建立在 eGFR 之上", "owner": "门诊"},
            {"what": "营养状态复评（MNA-SF、体重、白蛋白）", "why": "营养目标需按疗效动态调整", "owner": "营养科"},
            {"what": "用药清单复核与跌倒风险药物减停进度", "why": "减停是渐进过程，需要复核而不是一次性医嘱", "owner": "临床药师"},
        ]
        if denosumab_path:
            m6.insert(0, {
                "what": "地舒单抗按期给药（每 6 个月），并确认下一次日期已预约",
                "why": "延迟给药即进入反跳风险窗口",
                "owner": "门诊",
            })

        m12 = [
            {"what": "DXA 复查（同一设备、同一部位）", "why": "疗效监测与治疗目标评估", "owner": "骨密度室"},
            {"what": "ASMI 复测", "why": "肌量是肌少症诊断与疗效的核心指标", "owner": "体成分室"},
            {"what": "五轴风险重评（骨折/跌倒/肌少症/营养/功能）", "why": "风险构成会随干预改变，方案应随之调整", "owner": "老年科"},
            {"what": "抗骨质疏松治疗疗程评估与后续策略", "why": "疗程、药物假期与序贯方案都需要按期重新决策", "owner": "骨质疏松专科"},
        ]
        if facts.get("tcm_pattern") or "tcm_patent" in classes:
            m3.append({"what": "肝功能复查", "why": "部分补肾壮骨类中成药有肝损伤报告", "owner": "中医科"})
            m12.append({"what": "中医证型复诊与方案调整", "why": "证型随病程变化，方药需随证调整", "owner": "中医骨伤科"})

        return {"immediate": immediate, "m3": m3, "m6": m6, "m12": m12}

    def notes(self, payload: dict[str, Any]) -> list[str]:
        notes = [f"安全审查已跑完 {len(payload['checks_run'])} 项：{'、'.join(payload['checks_zh'])}"]
        if payload["blocking"]:
            notes.append(
                f"存在 {len(payload['blocking'])} 项阻断性问题，药物部分在解决前不得下达：" +
                "；".join(issue["title"] for issue in payload["blocking"])
            )
        return notes


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
