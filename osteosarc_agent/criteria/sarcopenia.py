"""Multi-standard sarcopenia determination: AWGS 2019, EWGSOP2, GLIS 2024.

The three standards do not merely use different numbers — they disagree about
*what sarcopenia is*. EWGSOP2 leads with strength and treats mass as
confirmation. AWGS keeps the classic mass-plus-one structure but adds a
"possible sarcopenia" state so a community clinic without a DXA can still act.
GLIS 2024 goes the other way and makes low muscle mass the mandatory core,
demoting physical performance to an outcome.

So "which standard" is not a formatting preference: for a real patient the
three can return three different answers, and the clinically useful output is
all three plus **why they differ**. That is what this module produces — one
verdict per standard, each with the component that decided it, and a set of
cross-standard notes naming the measurement that crossed one cutoff but not
another.

Every component is tri-state. A missing ASMI is not "mass normal"; it is the
reason GLIS cannot conclude and AWGS can only say "可能肌少症".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..guidelines.model import FALSE, TRUE, UNKNOWN

# --- verdict vocabulary ----------------------------------------------------

SARCOPENIA = "sarcopenia"
SEVERE = "severe_sarcopenia"
PROBABLE = "probable_sarcopenia"
POSSIBLE = "possible_sarcopenia"
NOT_SARCOPENIA = "not_sarcopenia"
INDETERMINATE = "indeterminate"

VERDICT_ZH = {
    SARCOPENIA: "肌少症",
    SEVERE: "严重肌少症",
    PROBABLE: "可能肌少症（肌力已下降，待肌量确诊）",
    POSSIBLE: "可能肌少症（可先行干预）",
    NOT_SARCOPENIA: "未达肌少症标准",
    INDETERMINATE: "无法判定（关键测量缺失）",
}

#: Verdicts that should trigger intervention even though they are not a
#: confirmed diagnosis. AWGS and EWGSOP2 both say so explicitly; the platform
#: must not treat "not yet confirmed" as "nothing to do".
ACTIONABLE = {SARCOPENIA, SEVERE, PROBABLE, POSSIBLE}

#: Verdicts that count as a confirmed diagnosis for downstream predicates.
CONFIRMED = {SARCOPENIA, SEVERE}


@dataclass(frozen=True)
class Component:
    """One of the three axes (strength / performance / mass) under one standard."""

    axis: str
    result: str  # TRUE = low/abnormal, FALSE = normal, UNKNOWN = not measured
    detail: str
    measures: tuple[str, ...] = ()

    @property
    def is_low(self) -> bool:
        return self.result == TRUE

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "result": self.result,
            "detail": self.detail,
            "measures": list(self.measures),
        }


@dataclass(frozen=True)
class StandardVerdict:
    standard_id: str
    name_zh: str
    verdict: str
    strength: Component
    performance: Component
    mass: Component
    reason: str
    missing: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def verdict_zh(self) -> str:
        return VERDICT_ZH[self.verdict]

    @property
    def actionable(self) -> bool:
        return self.verdict in ACTIONABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "standard_id": self.standard_id,
            "name_zh": self.name_zh,
            "verdict": self.verdict,
            "verdict_zh": self.verdict_zh,
            "actionable": self.actionable,
            "confirmed": self.verdict in CONFIRMED,
            "reason": self.reason,
            "components": {
                "strength": self.strength.to_dict(),
                "performance": self.performance.to_dict(),
                "mass": self.mass.to_dict(),
            },
            "missing": list(self.missing),
            "notes": list(self.notes),
        }


# --- cutoffs ---------------------------------------------------------------


@dataclass(frozen=True)
class Cutoffs:
    """One standard's numeric thresholds, kept as data so they can be cited."""

    standard_id: str
    name_zh: str
    grip: Mapping[str, float]
    asmi_dxa: Mapping[str, float]
    asmi_bia: Mapping[str, float] = field(default_factory=dict)
    asm_absolute: Mapping[str, float] = field(default_factory=dict)
    gait_low: float | None = None
    sts5_low: float | None = None
    sppb_low: int | None = None
    tug_low: float | None = None
    sts5_strength: float | None = None  # chair-stand used as a *strength* proxy
    source_note: str = ""


AWGS_2019 = Cutoffs(
    standard_id="AWGS2019",
    name_zh="AWGS 2019（亚洲）",
    grip={"M": 28.0, "F": 18.0},
    asmi_dxa={"M": 7.0, "F": 5.4},
    asmi_bia={"M": 7.0, "F": 5.7},
    gait_low=1.0,
    sts5_low=12.0,
    sppb_low=9,
    source_note="亚洲肌少症工作组 2019 共识",
)

EWGSOP2_2019 = Cutoffs(
    standard_id="EWGSOP2",
    name_zh="EWGSOP2（欧洲）",
    grip={"M": 27.0, "F": 16.0},
    asmi_dxa={"M": 7.0, "F": 5.5},
    asmi_bia={"M": 7.0, "F": 5.5},
    asm_absolute={"M": 20.0, "F": 15.0},
    gait_low=0.8,
    sppb_low=8,
    tug_low=20.0,
    sts5_strength=15.0,
    source_note="EWGSOP2 2019 修订共识",
)

#: GLIS ships a concept, not cutoffs. The platform must therefore *borrow* a
#: numeric threshold to operationalise it — and say so on every screen where a
#: GLIS verdict appears. Defaulting to the Asian cutoffs is a deployment
#: choice for a Chinese population, not a claim GLIS endorses them.
GLIS_2024 = Cutoffs(
    standard_id="GLIS2024",
    name_zh="GLIS 2024（全球概念框架）",
    grip={"M": 28.0, "F": 18.0},
    asmi_dxa={"M": 7.0, "F": 5.4},
    asmi_bia={"M": 7.0, "F": 5.7},
    gait_low=1.0,
    sts5_low=12.0,
    source_note="GLIS 2024 概念框架；切点借用 AWGS 2019，属操作化选择",
)

STANDARDS: dict[str, Cutoffs] = {
    AWGS_2019.standard_id: AWGS_2019,
    EWGSOP2_2019.standard_id: EWGSOP2_2019,
    GLIS_2024.standard_id: GLIS_2024,
}

DEFAULT_STANDARDS = ("AWGS2019", "EWGSOP2", "GLIS2024")


# --- helpers ---------------------------------------------------------------


def _sex(facts: Mapping[str, Any]) -> str | None:
    value = facts.get("sex")
    if isinstance(value, str) and value.upper() in ("M", "F"):
        return value.upper()
    return None


def _num(facts: Mapping[str, Any], name: str) -> float | None:
    value = facts.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _grip_component(cut: Cutoffs, facts: Mapping[str, Any], *, allow_chair_stand: bool) -> Component:
    """Low muscle strength, by grip and (EWGSOP2 only) chair stand."""
    sex = _sex(facts)
    grip = _num(facts, "grip_kg")
    parts: list[str] = []
    results: list[str] = []
    measures: list[str] = []

    if grip is not None and sex is not None:
        threshold = cut.grip[sex]
        low = grip < threshold
        results.append(TRUE if low else FALSE)
        parts.append(f"握力 {grip:g} kg {'<' if low else '≥'} {threshold:g} kg")
        measures.append("grip_kg")
    elif grip is not None and sex is None:
        parts.append(f"握力 {grip:g} kg（性别未知，无法取切点）")
        results.append(UNKNOWN)
    else:
        parts.append("握力未测")
        results.append(UNKNOWN)

    if allow_chair_stand and cut.sts5_strength is not None:
        sts5 = _num(facts, "sts5_s")
        if sts5 is not None:
            low = sts5 > cut.sts5_strength
            results.append(TRUE if low else FALSE)
            parts.append(f"5 次起坐 {sts5:g} s {'>' if low else '≤'} {cut.sts5_strength:g} s")
            measures.append("sts5_s")

    # Either criterion being low makes strength low; both normal makes it
    # normal; otherwise unknown.
    if TRUE in results:
        result = TRUE
    elif FALSE in results:
        result = FALSE
    else:
        result = UNKNOWN
    return Component("strength", result, "；".join(parts), tuple(measures))


def _performance_component(cut: Cutoffs, facts: Mapping[str, Any]) -> Component:
    parts: list[str] = []
    results: list[str] = []
    measures: list[str] = []

    checks: list[tuple[str, float | None, str, str]] = [
        ("gait_speed_ms", cut.gait_low, "6 米步速", "m/s"),
        ("sts5_s", cut.sts5_low, "5 次起坐", "s"),
        ("sppb", cut.sppb_low, "SPPB", "分"),
        ("tug_s", cut.tug_low, "TUG", "s"),
    ]
    for fact_name, threshold, label, unit in checks:
        if threshold is None:
            continue
        value = _num(facts, fact_name)
        if value is None:
            continue
        measures.append(fact_name)
        # Gait speed and SPPB are "lower is worse"; chair stand and TUG are
        # "higher is worse".
        if fact_name in ("gait_speed_ms", "sppb"):
            low = value < threshold if fact_name == "gait_speed_ms" else value <= threshold
            symbol = "<" if fact_name == "gait_speed_ms" else "≤"
        else:
            low = value >= threshold if fact_name == "sts5_s" else value >= threshold
            symbol = "≥"
        results.append(TRUE if low else FALSE)
        parts.append(f"{label} {value:g} {unit} {symbol if low else '未达' + symbol} {threshold:g}")

    if not results:
        return Component("performance", UNKNOWN, "步速/起坐/SPPB 均未测", ())
    result = TRUE if TRUE in results else FALSE
    return Component("performance", result, "；".join(parts), tuple(measures))


def _mass_component(cut: Cutoffs, facts: Mapping[str, Any]) -> Component:
    sex = _sex(facts)
    asmi = _num(facts, "asmi")
    asm = _num(facts, "asm_kg")
    method = str(facts.get("muscle_mass_method") or "DXA").upper()

    if sex is None:
        return Component("mass", UNKNOWN, "性别未知，无法取肌量切点", ())

    if asmi is not None:
        table = cut.asmi_bia if method == "BIA" and cut.asmi_bia else cut.asmi_dxa
        threshold = table[sex]
        low = asmi < threshold
        return Component(
            "mass",
            TRUE if low else FALSE,
            f"ASMI {asmi:g} kg/m² {'<' if low else '≥'} {threshold:g} kg/m²（{method}）",
            ("asmi",),
        )
    if asm is not None and cut.asm_absolute:
        threshold = cut.asm_absolute[sex]
        low = asm < threshold
        return Component(
            "mass",
            TRUE if low else FALSE,
            f"ASM {asm:g} kg {'<' if low else '≥'} {threshold:g} kg",
            ("asm_kg",),
        )
    return Component("mass", UNKNOWN, "未测四肢骨骼肌量（ASMI/ASM）", ())


# --- per-standard algorithms ----------------------------------------------


def _assess_awgs(facts: Mapping[str, Any]) -> StandardVerdict:
    cut = AWGS_2019
    strength = _grip_component(cut, facts, allow_chair_stand=False)
    performance = _performance_component(cut, facts)
    mass = _mass_component(cut, facts)
    missing = tuple(c.axis for c in (strength, performance, mass) if c.result == UNKNOWN)
    notes: list[str] = []

    if mass.result == TRUE and strength.is_low and performance.is_low:
        verdict, reason = SEVERE, "肌量减少 + 肌力下降 + 躯体功能下降（AWGS 严重肌少症）"
    elif mass.result == TRUE and (strength.is_low or performance.is_low):
        verdict, reason = SARCOPENIA, "肌量减少并伴肌力或功能下降（AWGS 诊断成立）"
    elif mass.result == FALSE:
        verdict, reason = NOT_SARCOPENIA, "肌量未达 AWGS 减少标准，不满足诊断"
        if strength.is_low or performance.is_low:
            notes.append("肌量正常但肌力/功能已下降，应查找疼痛、关节病、神经病变等其他原因")
    elif strength.is_low or performance.is_low:
        verdict = POSSIBLE
        reason = "肌力或躯体功能下降、肌量未测：AWGS 判为「可能肌少症」，可直接启动运动与营养干预"
        notes.append("AWGS 2019 明确允许在肌量未测时按可能肌少症干预，不必等待 DXA/BIA")
    else:
        verdict, reason = INDETERMINATE, "肌力、功能、肌量均无可用测量"
    return StandardVerdict(cut.standard_id, cut.name_zh, verdict, strength, performance, mass,
                           reason, missing, tuple(notes))


def _assess_ewgsop2(facts: Mapping[str, Any]) -> StandardVerdict:
    cut = EWGSOP2_2019
    strength = _grip_component(cut, facts, allow_chair_stand=True)
    performance = _performance_component(cut, facts)
    mass = _mass_component(cut, facts)
    missing = tuple(c.axis for c in (strength, performance, mass) if c.result == UNKNOWN)
    notes: list[str] = []

    if not strength.is_low:
        if strength.result == FALSE:
            verdict, reason = NOT_SARCOPENIA, "肌力未达 EWGSOP2 下降标准（该标准以肌力为入口）"
        else:
            verdict, reason = INDETERMINATE, "肌力无可用测量，EWGSOP2 流程无法启动"
        return StandardVerdict(cut.standard_id, cut.name_zh, verdict, strength, performance, mass,
                               reason, missing, tuple(notes))

    if mass.result == TRUE and performance.is_low:
        verdict, reason = SEVERE, "肌力下降 + 肌量减少 + 功能下降（EWGSOP2 严重肌少症）"
    elif mass.result == TRUE:
        verdict, reason = SARCOPENIA, "肌力下降并经肌量证实（EWGSOP2 确诊）"
    else:
        verdict = PROBABLE
        reason = "肌力下降、肌量未证实：EWGSOP2 判为「可能肌少症（probable）」，应立即干预并查因"
        notes.append("EWGSOP2 明确：probable sarcopenia 即可开始治疗，肌量测定用于确诊而非启动条件")
    return StandardVerdict(cut.standard_id, cut.name_zh, verdict, strength, performance, mass,
                           reason, missing, tuple(notes))


def _assess_glis(facts: Mapping[str, Any]) -> StandardVerdict:
    cut = GLIS_2024
    strength = _grip_component(cut, facts, allow_chair_stand=False)
    performance = _performance_component(cut, facts)
    mass = _mass_component(cut, facts)
    missing = tuple(c.axis for c in (strength, mass) if c.result == UNKNOWN)
    notes = ["GLIS 未发布统一切点，本判定借用 AWGS 2019 肌量与握力切点（操作化选择，需在报告中声明）"]

    if mass.result == UNKNOWN:
        verdict = INDETERMINATE
        reason = "GLIS 以肌量减少为必备核心要素，未测肌量即不能作出确定性诊断"
        notes.append("这正是 GLIS 与 AWGS/EWGSOP2 的实质分歧：能否在没有肌量的情况下下诊断")
    elif mass.result == FALSE:
        verdict, reason = NOT_SARCOPENIA, "肌量未减少，不满足 GLIS 必备核心要素"
    elif strength.is_low:
        verdict, reason = SARCOPENIA, "肌量减少（必备）+ 肌力下降（支持），GLIS 诊断成立"
        if performance.is_low:
            notes.append("躯体功能下降在 GLIS 中作为结局与严重程度指标，不参与诊断成立与否")
    elif strength.result == FALSE:
        verdict, reason = INDETERMINATE, "肌量减少但肌力未下降，GLIS 框架下需结合肌肉特异性力量进一步判断"
    else:
        verdict, reason = INDETERMINATE, "肌量减少但肌力未测，无法完成 GLIS 判定"
    return StandardVerdict(cut.standard_id, cut.name_zh, verdict, strength, performance, mass,
                           reason, tuple(missing), tuple(notes))


_ASSESSORS = {
    "AWGS2019": _assess_awgs,
    "EWGSOP2": _assess_ewgsop2,
    "GLIS2024": _assess_glis,
}


def assess(standard_id: str, facts: Mapping[str, Any]) -> StandardVerdict:
    try:
        assessor = _ASSESSORS[standard_id]
    except KeyError:
        raise ValueError(f"未知肌少症标准: {standard_id!r}") from None
    return assessor(facts)


def assess_all(facts: Mapping[str, Any], standards: tuple[str, ...] = DEFAULT_STANDARDS) -> list[StandardVerdict]:
    return [assess(standard_id, facts) for standard_id in standards]


# --- cross-standard reading ------------------------------------------------


def cross_standard_notes(verdicts: list[StandardVerdict], facts: Mapping[str, Any]) -> list[str]:
    """Explain *why* the standards disagree, naming the deciding measurement.

    A user who sees three different verdicts and no explanation learns nothing.
    A user who reads "握力 17 kg 低于 AWGS 的 18 kg 但高于 EWGSOP2 的 16 kg" can
    decide for themselves which standard to trust for this patient.
    """
    notes: list[str] = []
    labels = {v.standard_id: v.name_zh for v in verdicts}
    outcomes = {v.standard_id: v.verdict for v in verdicts}
    if len(set(outcomes.values())) <= 1:
        if outcomes:
            notes.append("三套标准结论一致。")
        return notes

    sex = _sex(facts)
    grip = _num(facts, "grip_kg")
    if grip is not None and sex is not None:
        straddled = [
            (sid, STANDARDS[sid].grip[sex])
            for sid in outcomes
            if sid in STANDARDS
        ]
        below = [(sid, cut) for sid, cut in straddled if grip < cut]
        above = [(sid, cut) for sid, cut in straddled if grip >= cut]
        if below and above:
            below_txt = "、".join(f"{labels[s]} <{c:g}" for s, c in below)
            above_txt = "、".join(f"{labels[s]} <{c:g}" for s, c in above)
            notes.append(
                f"握力 {grip:g} kg 落在切点之间：低于 {below_txt}，但未低于 {above_txt}"
                "——同一个数值跨过了一条线而没跨过另一条。"
            )

    sts5 = _num(facts, "sts5_s")
    if sts5 is not None:
        ew = next((v for v in verdicts if v.standard_id == "EWGSOP2"), None)
        if ew is not None and ew.strength.is_low and "sts5_s" in ew.strength.measures:
            notes.append(
                f"EWGSOP2 把 5 次起坐 {sts5:g} s 用作肌力替代判据（>15 s），"
                "AWGS 则把它算作躯体功能——同一项测量在两套标准里归属不同的轴。"
            )

    mass_missing = [v for v in verdicts if v.mass.result == UNKNOWN]
    if mass_missing:
        notes.append(
            "肌量未测是本次分歧的主因：GLIS 以肌量为必备核心，未测即无法确诊；"
            "AWGS 与 EWGSOP2 都允许在肌量缺失时先按「可能肌少症」干预。"
            "补测 ASMI（DXA 或 BIA）可让三套标准收敛。"
        )
    return notes


def summarise(verdicts: list[StandardVerdict]) -> dict[str, Any]:
    """Collapse the per-standard verdicts into the flags downstream code needs."""
    confirmed = [v for v in verdicts if v.verdict in CONFIRMED]
    actionable = [v for v in verdicts if v.actionable]
    return {
        "any_confirmed": bool(confirmed),
        "any_actionable": bool(actionable),
        "confirmed_by": [v.standard_id for v in confirmed],
        "actionable_by": [v.standard_id for v in actionable],
        "severe": any(v.verdict == SEVERE for v in verdicts),
        "agreement": len({v.verdict for v in verdicts}) == 1,
    }
