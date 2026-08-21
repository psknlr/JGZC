"""Individualised care paths: 骨保护 / 运动训练 / 营养.

One invariant governs this layer: **no plan item without live evidence.** A
plan rule fires only when its clinical trigger holds *and* at least one of the
guideline records it names came back as applicable for this patient. Rules whose
supporting evidence was excluded (``eGFR 32`` removing the bisphosphonate
records) do not appear, and are reported in ``suppressed`` with the reason —
that list is often the more instructive half.

Consequences of the invariant:

* every card in the console has something behind 「查看循证依据」, by construction
  rather than by hoping;
* replacing the corpus with a licensed one changes the plans, which is the whole
  point of keeping guidelines computable;
* a rule whose evidence disappears fails loudly (it shows up as suppressed)
  rather than turning into an unsourced assertion.

Doses are never generated. Administration rules, targets expressed as guideline
ranges, and monitoring intervals are safety information and are stated;
"how many milligrams for this patient" is a physician's decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

BONE = "bone_protection"
EXERCISE = "exercise"
NUTRITION = "nutrition"

COLUMN_LABELS = {
    BONE: "骨保护方案",
    EXERCISE: "运动与训练方案",
    NUTRITION: "营养方案",
}


@dataclass(frozen=True)
class PlanRule:
    plan_id: str
    column: str
    title: str
    detail: str
    why: str
    supported_by: tuple[str, ...]
    trigger: Callable[[Mapping[str, Any]], bool] = lambda facts: True
    priority: int = 5           # 1 = 最先做
    cautions: tuple[str, ...] = ()
    owner: str = ""
    tags: tuple[str, ...] = ()


def _t(name: str) -> Callable[[Mapping[str, Any]], bool]:
    return lambda facts: bool(facts.get(name))


def _any(*names: str) -> Callable[[Mapping[str, Any]], bool]:
    return lambda facts: any(bool(facts.get(name)) for name in names)


def _lt(name: str, bound: float) -> Callable[[Mapping[str, Any]], bool]:
    def check(facts: Mapping[str, Any]) -> bool:
        value = facts.get(name)
        try:
            return value is not None and float(value) < bound
        except (TypeError, ValueError):
            return False
    return check


def _always(facts: Mapping[str, Any]) -> bool:
    return True


#: Site names a clinical record may use for a vertebral fracture.
_VERTEBRAL_SITES = {"vertebra", "椎体", "脊柱", "spine"}


def _vertebral_fracture(facts: Mapping[str, Any]) -> bool:
    """A vertebral fracture by any route — reported site or imaging.

    Triggering only on ``morphometric_vertebral_fracture`` would withhold the
    spinal-precaution card from every patient whose fracture is recorded as a
    site rather than an imaging finding, which is most of them.
    """
    if facts.get("morphometric_vertebral_fracture"):
        return True
    sites = {str(site).strip() for site in (facts.get("fracture_sites") or [])}
    return bool(sites & _VERTEBRAL_SITES)


PLAN_RULES: tuple[PlanRule, ...] = (
    # --- 骨保护 ---------------------------------------------------------
    PlanRule(
        "bone.correct_first", BONE,
        "第一步：先纠正钙与维生素 D，再启动抗骨吸收治疗",
        "复查校正血钙与 25-OH-D；在血钙与维生素 D 达标前不启动地舒单抗或唑来膦酸。达标后方可进入下一步。",
        "顺序颠倒是抗骨吸收药物相关严重低钙血症最主要的可预防原因，肾功能不全者风险更高。",
        ("CN.OP.2022.CORRECT_BEFORE_ANTIRESORPTIVE", "CN.OP.2022.CALCIUM_VITD_BASE"),
        trigger=_always, priority=1,
        cautions=("这是用药前置条件，不是可选项",),
        owner="骨质疏松专科/内分泌科", tags=("用药前置", "安全"),
    ),
    PlanRule(
        "bone.secondary_prevention", BONE,
        "脆性骨折后必须启动抗骨质疏松药物治疗",
        "纳入骨折联络服务（FLS）管理；把「处理了骨折、没处理骨质疏松」列为不可接受的结局。",
        "骨折后一年内再骨折风险最高，而现实中二级预防的启动率极低。",
        ("CN.FRACTURE.2022.SECONDARY_PREVENTION", "EU.STOPP.2023.START_BONE"),
        trigger=_t("prior_fragility_fracture"), priority=1,
        owner="骨科/骨质疏松专科", tags=("二级预防",),
    ),
    PlanRule(
        "bone.exclude_secondary", BONE,
        "启动治疗前排除继发性骨质疏松",
        "查甲状旁腺激素、TSH、血磷、碱性磷酸酶；骨痛合并贫血与肾损时加查血清蛋白电泳排除多发性骨髓瘤；eGFR 下降者需鉴别 CKD-MBD。",
        "把肾性骨病或骨髓瘤当作原发骨质疏松治疗，既无效又延误。",
        ("LABEL.EXCLUDE_SECONDARY",),
        trigger=_always, priority=2,
        owner="内分泌科/肾内科", tags=("鉴别诊断",),
    ),
    PlanRule(
        "bone.renal_pathway", BONE,
        "肾功能不全下的药物路径：避开双膦酸盐，按地舒单抗路径评估",
        "eGFR 低于 35 时不选双膦酸盐；地舒单抗不经肾清除可作为候选，但必须先纠正钙与维生素 D 并安排早期血钙复查。具体药物与剂量由主管医师决定。",
        "肾功能不全解锁了地舒单抗，同时把风险从肾脏转移到血钙——用药可及不等于用药安全。",
        ("CN.OP.2022.DENOSUMAB_CKD", "CN.OP.2022.BISPHOSPHONATE_RENAL", "EU.ESCEO.2019.BISPHOSPHONATE_RENAL"),
        trigger=_lt("egfr", 35), priority=2,
        cautions=("地舒单抗在晚期 CKD 中有严重低钙血症风险，须按说明书警示监测",),
        owner="骨质疏松专科 + 肾内科", tags=("肾功能", "药物选择"),
    ),
    PlanRule(
        "bone.denosumab_continuity", BONE,
        "若使用地舒单抗：预先写好停药衔接方案",
        "把每 6 个月一次的给药日期写入随访计划；任何停用或延迟都必须预先安排双膦酸盐等序贯衔接。",
        "停药后骨转换反跳可致多发椎体骨折——这是该药最重要的单条安全信息。",
        ("CN.OP.2022.DENOSUMAB_NO_STOP",),
        trigger=_any("dx_osteoporosis"), priority=3,
        cautions=("不可自行停药或推迟给药",),
        owner="骨质疏松专科", tags=("地舒单抗", "停药反跳"),
    ),
    PlanRule(
        "bone.dental", BONE,
        "用药前完成口腔评估",
        "拟长期使用抗骨吸收药物者，先处理需拔牙的病灶，用药期间保持口腔卫生并定期口腔检查。",
        "降低颌骨坏死风险；侵入性口腔操作最好安排在启动治疗之前。",
        ("US.ES.2020.DENTAL_BEFORE_ANTIRESORPTIVE",),
        trigger=_always, priority=3,
        owner="口腔科", tags=("颌骨坏死",),
    ),
    PlanRule(
        "bone.vertebral_imaging", BONE,
        "加做胸腰椎侧位片或 VFA 查找无症状椎体骨折",
        "身高下降、驼背或不明原因背痛者应完成一次椎体骨折评估，结果直接改变风险分层与用药选择。",
        "约三分之二的椎体骨折无症状，靠「痛不痛」筛不出来。",
        ("CN.OP.2022.VERTEBRAL_IMAGING",),
        trigger=_always, priority=2,
        owner="影像科", tags=("影像", "补数据"),
    ),
    PlanRule(
        "bone.gc_threshold", BONE,
        "糖皮质激素相关骨质疏松按更低阈值处理",
        "长期使用激素者不沿用一般人群的 T ≤ -2.5 阈值，中高风险者在开始激素治疗的同时即启动抗骨质疏松药物与钙、维生素 D。",
        "激素性骨质疏松的骨折发生在骨密度下降之前。",
        ("CN.GIOP.2021.LOWER_THRESHOLD",),
        trigger=_any("glucocorticoid_current"), priority=2,
        owner="风湿免疫科/骨质疏松专科", tags=("激素",),
    ),
    PlanRule(
        "bone.tcm_adjunct", BONE,
        "中医药作为辅助：补肾壮骨、缓解骨痛，不替代抗骨质疏松药物",
        "由中医师四诊后按证型拟定方药或中成药；用药期间监测肝功能。本平台不生成任何剂量。",
        "中成药可改善腰背疼痛与生活质量、减少老年人 NSAIDs 暴露，但不具备替代抗骨折治疗的证据。",
        ("CN.TCM.PATENT.2021.PAIN_RELIEF", "CN.TCM.OP.2020.KIDNEY_PATTERN", "CN.TCM.PATENT.2021.ADJUNCT_ONLY"),
        trigger=_any("tcm_kidney_deficiency", "tcm_pattern", "tcm_spleen_deficiency"), priority=6,
        cautions=("含淫羊藿、补骨脂类制剂有肝损伤报告，用药期间监测肝功能",
                  "不得替代具有抗骨折证据的抗骨质疏松药物"),
        owner="中医骨伤科", tags=("中医",),
    ),
    PlanRule(
        "bone.monitor_dxa", BONE,
        "疗效监测：复查骨密度与骨转换标志物",
        "治疗期间在同一台设备、同一部位复查骨密度；结合骨转换标志物评估依从性与治疗反应。",
        "未达治疗目标或治疗中再骨折，应重新评估方案而不是延长观察。",
        ("CN.OP.2022.DXA_FOLLOWUP", "EU.ESCEO.2019.TREAT_TO_TARGET"),
        trigger=_t("dx_osteoporosis"), priority=7,
        owner="骨质疏松专科", tags=("随访",),
    ),

    # --- 运动与训练 ------------------------------------------------------
    PlanRule(
        "ex.resistance", EXERCISE,
        "渐进抗阻训练：每周 2～3 次，8～10 个大肌群动作",
        "每组 8～12 次重复、1～3 组，自觉用力中等偏上并逐步递增负荷；起始阶段可用弹力带或自重完成。",
        "抗阻训练是肌少症唯一具有一致获益证据的核心干预，同时改善骨骼负荷刺激。",
        ("US.ACSM.2019.PROGRESSIVE_RESISTANCE", "INTL.ICFSR.2021.RESISTANCE_FIRST"),
        trigger=_any("dx_sarcopenia_any", "dx_possible_sarcopenia", "dx_osteoporosis"), priority=1,
        owner="康复治疗师", tags=("抗阻", "核心干预"),
    ),
    PlanRule(
        "ex.balance", EXERCISE,
        "平衡与功能性训练：每周至少 3 次，可用太极拳/八段锦",
        "以有挑战性的平衡训练为主，长期坚持；传统功法在社区场景依从性好且证据支持。",
        "以平衡与功能训练为主的方案可直接降低跌倒发生率。",
        ("INTL.WFG.2022.EXERCISE_BALANCE", "CN.REHAB.2023.TAICHI", "CN.INTEGRATED.2023.EXERCISE_QIGONG"),
        trigger=_always, priority=1,
        owner="康复治疗师/社区", tags=("平衡", "跌倒预防", "中医导引"),
    ),
    PlanRule(
        "ex.spinal_precautions", EXERCISE,
        "椎体骨折者的动作改良：不停练，但改练法",
        "避免负重下的重复性脊柱前屈、躯干旋转下的提举与仰卧起坐类动作；改以中立位核心稳定与渐进抗阻为主。",
        "「有椎体骨折就别练了」造成的失用与再骨折风险，大于经过改良的训练本身。",
        ("EU.ROS.2018.EXERCISE_AFTER_VF",),
        trigger=_vertebral_fracture, priority=1,
        cautions=("动作模式须由康复治疗师先行示教并确认",),
        owner="康复治疗师", tags=("椎体骨折", "禁忌动作"),
    ),
    PlanRule(
        "ex.back_extensor", EXERCISE,
        "背伸肌力量与姿势训练",
        "纳入常规方案，配合日常姿势提示（坐立位胸椎伸展、避免长时间前屈）。",
        "可减轻胸椎后凸进展、改善平衡并降低椎体再骨折风险。",
        ("EU.ROS.2018.BACK_EXTENSOR",),
        trigger=_always, priority=2,
        owner="康复治疗师", tags=("姿势", "背伸肌"),
    ),
    PlanRule(
        "ex.impact", EXERCISE,
        "分级冲击性负荷（快走、上下台阶、原地踏步）",
        "在能力与安全允许的前提下加入低至中等冲击活动，为骨骼提供负荷刺激。",
        "骨骼需要冲击性负荷刺激；但高冲击训练不适用于近期椎体骨折或平衡障碍者。",
        ("EU.ROS.2018.IMPACT_GRADED",),
        trigger=_t("dx_osteoporosis"), priority=4,
        owner="康复治疗师", tags=("冲击负荷",),
    ),
    PlanRule(
        "ex.supervised", EXERCISE,
        "起始阶段在康复治疗师指导下进行",
        "学会动作模式与安全边界后再转为居家训练，并安排定期复评与进阶。",
        "高跌倒风险或有骨折史者，无监督起步的受伤风险与放弃率都更高。",
        ("CN.REHAB.2023.SUPERVISED_START",),
        trigger=_any("high_fall_risk", "prior_fragility_fracture"), priority=2,
        owner="康复医学科", tags=("安全", "监督"),
    ),
    PlanRule(
        "ex.some_is_better", EXERCISE,
        "达不到推荐量也要动：先建立频率，再加量",
        "从每次 5～10 分钟、每天 2～3 次开始累积；减少久坐，以任何强度的活动替代长时间静坐。",
        "对功能受限的老年人，「做一些总比不做好」本身就是指南推荐。",
        ("INTL.WHO.PA.2020.SOME_IS_BETTER", "INTL.WHO.PA.2020.MULTICOMPONENT"),
        trigger=_always, priority=3,
        owner="患者与家属", tags=("起步", "久坐"),
    ),

    # --- 营养 -----------------------------------------------------------
    PlanRule(
        "nut.protein_target", NUTRITION,
        "每日蛋白质目标（本例需多学科共同确定）",
        "肌少症方向建议 1.2～1.5 g/kg/d，非透析 CKD 方向建议 0.6～0.8 g/kg/d；两者方向相反，须由肾脏科与营养科共同设定目标并动态监测体重、握力、白蛋白、eGFR 与代谢性酸中毒。",
        "把蛋白目标交给任一单学科单向设定，都会在另一端造成伤害。",
        ("INTL.KDIGO.2024.INDIVIDUALIZE_SARCOPENIA", "EU.ESPEN.2022.PROTEIN_MIN",
         "CN.SARCO.2021.PROTEIN_EXERCISE", "INTL.KDIGO.2024.PROTEIN_RESTRICTION",
         "INTL.PROTAGE.2013.PROTEIN_ILLNESS"),
        trigger=_always, priority=1,
        cautions=("本条存在跨指南分歧，详见「指南冲突消解」",),
        owner="营养科 + 肾内科", tags=("蛋白质", "争议"),
    ),
    PlanRule(
        "nut.protein_distribution", NUTRITION,
        "蛋白质均匀分配到三餐",
        "每餐 25～30 g 优质蛋白，优先动物性蛋白与大豆蛋白；集中在晚餐一次摄入不利于肌肉合成。",
        "同样的总量，分配方式不同，肌肉合成效果不同。",
        ("CN.DIET.2022.PROTEIN_DISTRIBUTION",),
        trigger=_always, priority=2,
        owner="营养科", tags=("蛋白质", "分配"),
    ),
    PlanRule(
        "nut.calcium", NUTRITION,
        "钙以膳食为主，补充剂只补差额",
        "每日 300～400 mL 奶或等量奶制品，配合豆制品与深绿色蔬菜；补充剂分次服用，单次元素钙不超过 500 mg，并与左甲状腺素、喹诺酮类、口服双膦酸盐、铁剂分开服用。",
        "超量补钙未见额外抗骨折获益，且与肾结石和胃肠道不适相关。",
        ("CN.DIET.2022.DAIRY_CALCIUM", "US.BHOF.2022.DIET_CALCIUM_FIRST", "CN.OP.2022.CALCIUM_VITD_BASE"),
        trigger=_always, priority=2,
        owner="营养科", tags=("钙",),
    ),
    PlanRule(
        "nut.vitamin_d", NUTRITION,
        "维生素 D 按缺乏程度补足，不用大剂量冲击",
        "以常规每日剂量补足并复查 25-OH-D；不使用月度或年度大剂量冲击方案。",
        "大剂量间断给药在随机试验中反而增加跌倒与骨折；纠正缺乏对肌力与平衡也有获益。",
        ("US.BHOF.2022.VITD_TARGET", "INTL.WFG.2022.NO_HIGH_DOSE_VITD", "CN.SARCO.2021.VITD_MUSCLE"),
        trigger=_always, priority=1,
        owner="营养科/骨质疏松专科", tags=("维生素D",),
    ),
    PlanRule(
        "nut.screening", NUTRITION,
        "营养不良筛查与随访",
        "用 MNA-SF 等经验证工具筛查，阳性者进入营养评定；膳食强化仍不达标时给予口服营养补充并明确复评时间。",
        "营养不足时，蛋白与运动干预都无法转化为肌肉。",
        ("EU.ESPEN.2022.SCREEN_MALNUTRITION", "EU.ESPEN.2022.ORAL_SUPPLEMENT", "EU.ESPEN.2022.ENERGY"),
        trigger=_always, priority=3,
        owner="营养科", tags=("筛查", "ONS"),
    ),
    PlanRule(
        "nut.spleen_tcm", NUTRITION,
        "中医健脾益气，与营养干预协同",
        "食少纳呆、倦怠乏力、肌肉瘦削者，由中医师辨证后以健脾益气法佐助，配合膳食调整。",
        "「脾主肌肉」与现代营养—肌肉轴在干预目标上高度重合。",
        ("CN.TCM.OP.2020.SPLEEN_PATTERN",),
        trigger=_any("tcm_spleen_deficiency", "appetite_loss"), priority=6,
        owner="中医科", tags=("中医", "健脾"),
    ),
)


@dataclass
class PlanItem:
    plan_id: str
    column: str
    title: str
    detail: str
    why: str
    priority: int
    evidence: list[str] = field(default_factory=list)
    rec_ids: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    owner: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "column": self.column,
            "title": self.title,
            "detail": self.detail,
            "why": self.why,
            "priority": self.priority,
            "evidence": list(self.evidence),
            "rec_ids": list(self.rec_ids),
            "cautions": list(self.cautions),
            "conflicts": list(self.conflicts),
            "owner": self.owner,
            "tags": list(self.tags),
        }


def build(
    facts: Mapping[str, Any],
    applicable: Sequence[Mapping[str, Any]],
    excluded: Sequence[Mapping[str, Any]],
    conflicts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble the three care paths from live evidence."""
    live = {entry["rec_id"]: entry for entry in applicable}
    blocked = {entry["rec_id"]: entry for entry in excluded}
    conflict_by_question = {entry["question_id"]: entry for entry in conflicts}

    columns: dict[str, list[PlanItem]] = {BONE: [], EXERCISE: [], NUTRITION: []}
    suppressed: list[dict[str, Any]] = []

    for rule in PLAN_RULES:
        if not rule.trigger(facts):
            continue
        supporting = [rec_id for rec_id in rule.supported_by if rec_id in live]
        if not supporting:
            reasons = [
                f"{rec_id}：{blocked[rec_id]['excluded_because']}"
                for rec_id in rule.supported_by if rec_id in blocked
            ]
            suppressed.append({
                "plan_id": rule.plan_id,
                "column": rule.column,
                "title": rule.title,
                "reason": "；".join(reasons) if reasons else "本例无适用的支持性推荐",
                "rec_ids": list(rule.supported_by),
            })
            continue

        item = PlanItem(
            plan_id=rule.plan_id,
            column=rule.column,
            title=rule.title,
            detail=rule.detail,
            why=rule.why,
            priority=rule.priority,
            evidence=[live[rec_id]["evidence_id"] for rec_id in supporting],
            rec_ids=supporting,
            cautions=list(rule.cautions),
            owner=rule.owner,
            tags=list(rule.tags),
        )
        questions = {live[rec_id]["question"] for rec_id in supporting}
        for question_id in sorted(questions):
            entry = conflict_by_question.get(question_id)
            if entry and entry["verdict"] in ("disputed", "resolved_by_patient"):
                item.conflicts.append({
                    "question_id": question_id,
                    "label_zh": entry["label_zh"],
                    "verdict": entry["verdict"],
                    "verdict_zh": entry["verdict_zh"],
                    "basis": entry.get("basis", ""),
                })
        # A rule whose supporting evidence was partially excluded keeps the
        # exclusion visible on the card rather than dropping it silently.
        for rec_id in rule.supported_by:
            if rec_id in blocked:
                item.cautions.append(
                    f"已排除：{blocked[rec_id]['statement_zh'][:40]}…（{blocked[rec_id]['excluded_because']}）"
                )
        columns[rule.column].append(item)

    for items in columns.values():
        items.sort(key=lambda item: (item.priority, item.plan_id))

    return {
        "columns": [
            {
                "column": column,
                "label": COLUMN_LABELS[column],
                "items": [item.to_dict() for item in items],
            }
            for column, items in columns.items()
        ],
        "suppressed": suppressed,
        "item_count": sum(len(items) for items in columns.values()),
    }
