"""Drug classification, fall-risk flags and interaction rules.

Two jobs, kept separate because they fail differently:

* **Classification** turns free-text medication names (Chinese trade and generic
  names, plus English) into the classes the guideline predicates reason about.
  An unrecognised name is reported as unrecognised — never silently dropped,
  because "no interaction found" over a list the platform could not read is a
  false reassurance.
* **Interaction and contraindication rules** are stated as data with the
  clinical consequence attached, so the safety agent can quote the reason
  rather than emit a bare flag.

No dose is produced anywhere in this module or anywhere else in the platform.
Administration *rules* (fasting, upright, separation intervals) are safety
constraints and are stated; the amount to give is a physician's decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

#: class -> (Chinese and English name fragments matched case-insensitively)
DRUG_CLASSES: dict[str, tuple[str, ...]] = {
    "bisphosphonate": ("阿仑膦酸", "福善美", "唑来膦酸", "密固达", "利塞膦酸", "伊班膦酸",
                       "alendronate", "zoledronic", "risedronate", "ibandronate", "膦酸盐"),
    "denosumab": ("地舒单抗", "普罗力", "安加维", "denosumab", "prolia"),
    "romosozumab": ("罗莫索单抗", "romosozumab", "evenity"),
    "teriparatide": ("特立帕肽", "teriparatide", "forteo", "复泰奥"),
    "calcitonin": ("降钙素", "鲑降钙素", "密盖息", "calcitonin"),
    "serm": ("雷洛昔芬", "raloxifene"),
    "calcium": ("碳酸钙", "钙尔奇", "枸橼酸钙", "乳酸钙", "calcium", "钙片", "补钙"),
    "vitamin_d": ("维生素d", "维生素 d", "vitamin d", "胆钙化醇", "cholecalciferol", "骨化二醇"),
    "active_vitamin_d": ("骨化三醇", "罗盖全", "阿法骨化醇", "阿法迪三", "calcitriol", "alfacalcidol"),
    "ppi": ("奥美拉唑", "泮托拉唑", "雷贝拉唑", "艾司奥美拉唑", "兰索拉唑", "omeprazole",
            "pantoprazole", "esomeprazole", "拉唑"),
    "benzodiazepine": ("地西泮", "艾司唑仑", "阿普唑仑", "劳拉西泮", "氯硝西泮", "咪达唑仑",
                       "diazepam", "estazolam", "alprazolam", "lorazepam", "西泮", "唑仑"),
    "z_drug": ("唑吡坦", "佐匹克隆", "右佐匹克隆", "扎来普隆", "zolpidem", "zopiclone"),
    "anticholinergic": ("苯海拉明", "氯苯那敏", "奥昔布宁", "托特罗定", "索利那新", "阿米替林",
                        "东莨菪碱", "山莨菪碱", "diphenhydramine", "oxybutynin", "tolterodine",
                        "amitriptyline"),
    "antipsychotic": ("喹硫平", "奥氮平", "利培酮", "氟哌啶醇", "quetiapine", "olanzapine",
                      "risperidone", "haloperidol"),
    "antidepressant": ("舍曲林", "帕罗西汀", "氟西汀", "西酞普兰", "米氮平", "文拉法辛",
                       "sertraline", "paroxetine", "fluoxetine", "mirtazapine"),
    "opioid": ("曲马多", "羟考酮", "吗啡", "芬太尼", "可待因", "tramadol", "oxycodone", "morphine"),
    "nsaid": ("布洛芬", "双氯芬酸", "塞来昔布", "美洛昔康", "洛索洛芬", "萘普生",
              "ibuprofen", "diclofenac", "celecoxib", "meloxicam"),
    "antihypertensive": ("氨氯地平", "硝苯地平", "缬沙坦", "厄贝沙坦", "培哚普利", "美托洛尔",
                         "比索洛尔", "特拉唑嗪", "amlodipine", "valsartan", "metoprolol",
                         "地平", "沙坦", "普利"),
    "diuretic": ("呋塞米", "托拉塞米", "螺内酯", "氢氯噻嗪", "furosemide", "hydrochlorothiazide",
                 "spironolactone"),
    "anticoagulant": ("华法林", "利伐沙班", "达比加群", "阿哌沙班", "warfarin", "rivaroxaban",
                      "dabigatran", "apixaban"),
    "antiplatelet": ("阿司匹林", "氯吡格雷", "aspirin", "clopidogrel"),
    "levothyroxine": ("左甲状腺素", "优甲乐", "levothyroxine"),
    "glucocorticoid": ("泼尼松", "强的松", "甲泼尼龙", "地塞米松", "prednisone",
                       "methylprednisolone", "dexamethasone"),
    "thiazolidinedione": ("吡格列酮", "罗格列酮", "pioglitazone", "rosiglitazone"),
    "sglt2": ("达格列净", "恩格列净", "卡格列净", "dapagliflozin", "empagliflozin", "列净"),
    "quinolone": ("左氧氟沙星", "莫西沙星", "环丙沙星", "levofloxacin", "moxifloxacin", "沙星"),
    "iron": ("硫酸亚铁", "琥珀酸亚铁", "多糖铁", "ferrous", "铁剂"),
    "tcm_patent": ("仙灵骨葆", "骨疏康", "强骨胶囊", "金天格", "淫羊藿", "补骨脂", "壮骨"),
}

#: Classes that increase fall risk in older adults (FRIDs).
FALL_RISK_CLASSES = frozenset({
    "benzodiazepine", "z_drug", "anticholinergic", "antipsychotic",
    "antidepressant", "opioid", "antihypertensive", "diuretic", "sglt2",
})

#: Classes that count as anti-osteoporosis therapy for the "processing omission"
#: check. Calcium and vitamin D are *base* therapy and deliberately excluded —
#: counting them would let "她在补钙" pass as treated.
ANTIOSTEOPOROSIS_CLASSES = frozenset({
    "bisphosphonate", "denosumab", "romosozumab", "teriparatide", "serm", "calcitonin",
})

#: Classes that harm bone and should be surfaced whenever present.
BONE_HARMFUL_CLASSES = frozenset({"glucocorticoid", "thiazolidinedione"})

CLASS_LABELS = {
    "bisphosphonate": "双膦酸盐", "denosumab": "地舒单抗", "romosozumab": "罗莫索单抗",
    "teriparatide": "特立帕肽", "calcitonin": "降钙素", "serm": "雷洛昔芬",
    "calcium": "钙剂", "vitamin_d": "维生素 D", "active_vitamin_d": "活性维生素 D",
    "ppi": "质子泵抑制剂", "benzodiazepine": "苯二氮䓬类", "z_drug": "非苯二氮䓬类催眠药",
    "anticholinergic": "抗胆碱能药", "antipsychotic": "抗精神病药", "antidepressant": "抗抑郁药",
    "opioid": "阿片类", "nsaid": "非甾体抗炎药", "antihypertensive": "降压药",
    "diuretic": "利尿剂", "anticoagulant": "抗凝药", "antiplatelet": "抗血小板药",
    "levothyroxine": "左甲状腺素", "glucocorticoid": "糖皮质激素",
    "thiazolidinedione": "噻唑烷二酮类", "sglt2": "SGLT2 抑制剂", "quinolone": "喹诺酮类",
    "iron": "铁剂", "tcm_patent": "补肾壮骨类中成药",
}


def classify(name: str) -> str | None:
    """Map one medication name to a class, or ``None`` if unrecognised."""
    text = str(name).strip().lower()
    if not text:
        return None
    for drug_class, fragments in DRUG_CLASSES.items():
        for fragment in fragments:
            if fragment.lower() in text:
                return drug_class
    return None


@dataclass(frozen=True)
class MedicationEntry:
    name: str
    drug_class: str | None
    label: str
    raw: Any = None

    @property
    def recognised(self) -> bool:
        return self.drug_class is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "class": self.drug_class,
            "label": self.label,
            "recognised": self.recognised,
        }


def parse_medications(entries: Iterable[Any]) -> list[MedicationEntry]:
    """Accept strings or ``{"name": ..., "class": ...}`` mappings."""
    parsed: list[MedicationEntry] = []
    for entry in entries or []:
        if isinstance(entry, Mapping):
            name = str(entry.get("name", "")).strip()
            declared = entry.get("class")
            drug_class = str(declared) if declared else classify(name)
        else:
            name = str(entry).strip()
            drug_class = classify(name)
        if not name:
            continue
        label = CLASS_LABELS.get(drug_class or "", "未识别")
        parsed.append(MedicationEntry(name, drug_class, label, entry))
    return parsed


@dataclass(frozen=True)
class InteractionRule:
    rule_id: str
    when_classes: tuple[str, ...]
    severity: str  # high | moderate | advisory
    finding: str
    action: str
    requires_all: bool = True
    when_facts: tuple[tuple[str, str, float], ...] = ()  # (fact, op, value)


INTERACTION_RULES: tuple[InteractionRule, ...] = (
    InteractionRule(
        "IX.CALCIUM_LEVOTHYROXINE", ("calcium", "levothyroxine"), "high",
        "钙剂与左甲状腺素同服会螯合并显著降低甲状腺素吸收",
        "两者间隔 4 小时以上服用；调整后复查 TSH",
    ),
    InteractionRule(
        "IX.CALCIUM_QUINOLONE", ("calcium", "quinolone"), "high",
        "钙剂与喹诺酮类同服显著降低抗生素吸收，可致治疗失败",
        "抗生素疗程期间与钙剂间隔至少 2～4 小时",
    ),
    InteractionRule(
        "IX.CALCIUM_ORAL_BP", ("calcium", "bisphosphonate"), "high",
        "钙剂与口服双膦酸盐螯合，使后者几乎无法吸收",
        "双膦酸盐晨起空腹单独服用，服后 30 分钟内不得服钙剂或进食",
    ),
    InteractionRule(
        "IX.CALCIUM_IRON", ("calcium", "iron"), "moderate",
        "钙与铁互相竞争吸收",
        "分开服用，间隔 2 小时以上",
    ),
    InteractionRule(
        "IX.PPI_BISPHOSPHONATE", ("ppi", "bisphosphonate"), "moderate",
        "长期质子泵抑制剂可能降低口服双膦酸盐疗效，且自身与骨折风险升高相关",
        "评估 PPI 是否仍有指征；若需长期使用，优先考虑静脉双膦酸盐或其他药物",
    ),
    InteractionRule(
        "IX.ACTIVE_VITD_CALCIUM", ("active_vitamin_d", "calcium"), "moderate",
        "活性维生素 D（骨化三醇/阿法骨化醇）与钙剂联用有高钙血症与高尿钙风险",
        "定期监测血钙、尿钙与肾功能，不与普通维生素 D 叠加超量",
    ),
    InteractionRule(
        "IX.TCM_ANTICOAGULANT", ("tcm_patent", "anticoagulant"), "moderate",
        "活血类中药与抗凝药联用可能增加出血风险",
        "监测出血征象与凝血指标，由中西医共同评估是否联用",
    ),
    InteractionRule(
        "IX.THIAZIDE_HYPERCALCEMIA", ("diuretic", "calcium"), "advisory",
        "噻嗪类利尿剂减少尿钙排泄，与钙剂及维生素 D 联用时高钙血症风险升高",
        "监测血钙",
    ),
)


def find_interactions(classes: Iterable[str]) -> list[dict[str, Any]]:
    present = set(classes)
    found: list[dict[str, Any]] = []
    for rule in INTERACTION_RULES:
        if all(cls in present for cls in rule.when_classes):
            found.append({
                "rule_id": rule.rule_id,
                "severity": rule.severity,
                "classes": [CLASS_LABELS.get(c, c) for c in rule.when_classes],
                "finding": rule.finding,
                "action": rule.action,
            })
    return found


@dataclass(frozen=True)
class RenalRule:
    drug_class: str
    threshold: float
    verdict: str  # contraindicated | caution | preferred
    note: str


RENAL_RULES: tuple[RenalRule, ...] = (
    RenalRule("bisphosphonate", 35.0, "contraindicated",
              "双膦酸盐经肾排泄，eGFR <35 时不推荐使用；且需先排除 CKD-MBD"),
    RenalRule("denosumab", 35.0, "caution",
              "地舒单抗不经肾清除，肾功能不全时可用，但严重低钙血症风险显著升高，须先纠正钙与维 D 并密切监测血钙"),
    RenalRule("nsaid", 45.0, "contraindicated",
              "eGFR <45 的老年人应避免使用非甾体抗炎药，可致急性肾损伤与血压升高"),
    RenalRule("active_vitamin_d", 45.0, "caution",
              "活性维生素 D 在 CKD 中需按血钙、血磷与 PTH 调整，由肾科共同管理"),
)


def renal_findings(classes: Iterable[str], egfr: float | None) -> list[dict[str, Any]]:
    """Renal gates for the classes in play. Unknown eGFR is itself a finding."""
    present = set(classes)
    if egfr is None:
        relevant = sorted(present & {rule.drug_class for rule in RENAL_RULES})
        if relevant:
            return [{
                "drug_class": "—",
                "label": "、".join(CLASS_LABELS.get(c, c) for c in relevant),
                "verdict": "unknown",
                "note": "未提供 eGFR，无法判定上述药物的肾功能门槛——用药前必须先查肾功能",
            }]
        return []
    findings: list[dict[str, Any]] = []
    for rule in RENAL_RULES:
        if rule.drug_class in present and egfr < rule.threshold:
            findings.append({
                "drug_class": rule.drug_class,
                "label": CLASS_LABELS.get(rule.drug_class, rule.drug_class),
                "verdict": rule.verdict,
                "threshold": rule.threshold,
                "egfr": egfr,
                "note": rule.note,
            })
    return findings


def ckd_stage(egfr: float | None) -> str | None:
    if egfr is None:
        return None
    if egfr >= 90:
        return "G1"
    if egfr >= 60:
        return "G2"
    if egfr >= 45:
        return "G3a"
    if egfr >= 30:
        return "G3b"
    if egfr >= 15:
        return "G4"
    return "G5"
