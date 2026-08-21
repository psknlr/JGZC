"""Loading and querying the computable guideline corpus.

The corpus is **data, not code**: JSON (or YAML, if PyYAML happens to be
installed) files under ``library/``, so an operator who licenses real guideline
text can replace the shipped paraphrases without touching a Python file. The
loader is fail-closed in the ways that matter:

* every ``fact:`` referenced by a predicate must exist in :mod:`.facts` — an
  unknown fact name is a load error, not a predicate that quietly evaluates to
  unknown forever;
* every recommendation must name a source that was actually loaded;
* enumerated fields (direction, strength, region, provenance) are checked
  against the vocabularies in :mod:`.model`;
* duplicate ``rec_id`` values are rejected, because the evidence ledger
  addresses recommendations by id and a duplicate would make a citation
  ambiguous.

Nothing here is licensed guideline text. See ``docs/GUIDELINES.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import facts as fact_table
from .model import (
    DIRECTIONS, PROVENANCES, REGIONS, STRENGTHS, TRADITIONS,
    Applicability, ClinicalQuestion, ConditionError, GuidelineSource, Recommendation,
    test_recommendation, validate_condition,
)

LIBRARY_DIR = Path(__file__).parent / "library"


class CorpusError(ValueError):
    """A corpus file that cannot be trusted to drive clinical decisions."""


def _read_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
            raise CorpusError(
                f"{path.name}: 需要 PyYAML 才能读取 YAML 语料；改用 .json 或安装 pyyaml"
            ) from exc
        return yaml.safe_load(text)
    return json.loads(text)


def _collect_fact_names(node: Any, out: set[str]) -> None:
    if isinstance(node, Mapping):
        if "fact" in node:
            out.add(str(node["fact"]))
        for key in ("all", "any"):
            for child in node.get(key, []) or []:
                _collect_fact_names(child, out)
        if "not" in node:
            _collect_fact_names(node["not"], out)


@dataclass
class Corpus:
    """A loaded corpus: sources plus recommendations, indexed for querying."""

    sources: dict[str, GuidelineSource] = field(default_factory=dict)
    recommendations: list[Recommendation] = field(default_factory=list)
    questions_index: dict[str, ClinicalQuestion] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)

    # -- construction ----------------------------------------------------
    @classmethod
    def load(cls, paths: Sequence[str | Path] | None = None) -> "Corpus":
        """Load every corpus file in ``paths`` (default: the shipped library)."""
        corpus = cls()
        targets: list[Path] = []
        for entry in paths or [LIBRARY_DIR]:
            path = Path(entry)
            if path.is_dir():
                targets.extend(sorted(
                    child for child in path.iterdir()
                    if child.suffix in (".json", ".yaml", ".yml")
                ))
            elif path.exists():
                targets.append(path)
            else:
                raise CorpusError(f"语料路径不存在: {path}")
        if not targets:
            raise CorpusError("未找到任何语料文件")
        for path in targets:
            corpus._ingest(path)
        corpus._check_references()
        return corpus

    def _ingest(self, path: Path) -> None:
        try:
            payload = _read_file(path)
        except json.JSONDecodeError as exc:
            raise CorpusError(f"{path.name}: JSON 解析失败 — {exc}") from exc
        if not isinstance(payload, Mapping):
            raise CorpusError(f"{path.name}: 顶层必须是对象")
        self.files.append(path.name)

        for raw in payload.get("questions", []) or []:
            question = self._build_question(raw, path.name)
            if question.question_id in self.questions_index:
                raise CorpusError(f"{path.name}: 重复的 question_id {question.question_id!r}")
            self.questions_index[question.question_id] = question

        for raw in payload.get("sources", []) or []:
            source = self._build_source(raw, path.name)
            if source.source_id in self.sources:
                raise CorpusError(f"{path.name}: 重复的 source_id {source.source_id!r}")
            self.sources[source.source_id] = source

        known_ids = {rec.rec_id for rec in self.recommendations}
        for raw in payload.get("recommendations", []) or []:
            rec = self._build_recommendation(raw, path.name)
            if rec.rec_id in known_ids:
                raise CorpusError(f"{path.name}: 重复的 rec_id {rec.rec_id!r}")
            known_ids.add(rec.rec_id)
            self.recommendations.append(rec)

    @staticmethod
    def _build_question(raw: Mapping[str, Any], filename: str) -> ClinicalQuestion:
        missing = [k for k in ("question_id", "label_zh") if k not in raw]
        if missing:
            raise CorpusError(f"{filename}: question 缺少字段 {missing}")
        return ClinicalQuestion(
            question_id=str(raw["question_id"]),
            label_zh=str(raw["label_zh"]),
            exclusive=bool(raw.get("exclusive", False)),
            note=str(raw.get("note", "")),
        )

    @staticmethod
    def _build_source(raw: Mapping[str, Any], filename: str) -> GuidelineSource:
        missing = [k for k in ("source_id", "title_zh", "issuer", "year", "region", "tradition") if k not in raw]
        if missing:
            raise CorpusError(f"{filename}: source 缺少字段 {missing}")
        if raw["region"] not in REGIONS:
            raise CorpusError(f"{filename}: 未知 region {raw['region']!r}")
        if raw["tradition"] not in TRADITIONS:
            raise CorpusError(f"{filename}: 未知 tradition {raw['tradition']!r}")
        return GuidelineSource(
            source_id=str(raw["source_id"]),
            title_zh=str(raw["title_zh"]),
            title_en=str(raw.get("title_en", "")),
            issuer=str(raw["issuer"]),
            year=int(raw["year"]),
            region=str(raw["region"]),
            tradition=str(raw["tradition"]),
            url=str(raw.get("url", "")),
            note=str(raw.get("note", "")),
        )

    @staticmethod
    def _build_recommendation(raw: Mapping[str, Any], filename: str) -> Recommendation:
        required = ("rec_id", "source_id", "topic", "question", "action", "direction", "strength", "statement_zh")
        missing = [k for k in required if k not in raw]
        if missing:
            raise CorpusError(f"{filename}: recommendation {raw.get('rec_id', '?')} 缺少字段 {missing}")
        rec_id = str(raw["rec_id"])
        if raw["direction"] not in DIRECTIONS:
            raise CorpusError(f"{filename}/{rec_id}: 未知 direction {raw['direction']!r}")
        if raw["strength"] not in STRENGTHS:
            raise CorpusError(f"{filename}/{rec_id}: 未知 strength {raw['strength']!r}")
        provenance = str(raw.get("provenance", "editorial_paraphrase"))
        if provenance not in PROVENANCES:
            raise CorpusError(f"{filename}/{rec_id}: 未知 provenance {provenance!r}")

        for key in ("applies_when", "excluded_when"):
            node = raw.get(key)
            try:
                validate_condition(node, f"{filename}/{rec_id}.{key}")
            except ConditionError as exc:
                raise CorpusError(str(exc)) from exc
            names: set[str] = set()
            _collect_fact_names(node, names)
            unknown = sorted(name for name in names if not fact_table.is_known_fact(name))
            if unknown:
                raise CorpusError(
                    f"{filename}/{rec_id}.{key}: 引用了未声明的事实 {unknown}"
                    " —— 请先在 guidelines/facts.py 中声明，否则该条件永远无法判定"
                )

        return Recommendation(
            rec_id=rec_id,
            source_id=str(raw["source_id"]),
            topic=str(raw["topic"]),
            question=str(raw["question"]),
            action=str(raw["action"]),
            direction=str(raw["direction"]),
            strength=str(raw["strength"]),
            statement_zh=str(raw["statement_zh"]),
            evidence_level=str(raw.get("evidence_level", "")),
            applies_when=raw.get("applies_when"),
            excluded_when=raw.get("excluded_when"),
            rationale=str(raw.get("rationale", "")),
            citation=str(raw.get("citation", "")),
            provenance=provenance,
            tags=tuple(str(tag) for tag in raw.get("tags", []) or ()),
            verbatim=bool(raw.get("verbatim", False)),
            subsumes=tuple(str(item) for item in raw.get("subsumes", []) or ()),
        )

    def _check_references(self) -> None:
        dangling = sorted({rec.source_id for rec in self.recommendations} - set(self.sources))
        if dangling:
            raise CorpusError(f"recommendation 引用了未定义的 source_id: {dangling}")
        # A recommendation filed under an undeclared question would be invisible
        # to conflict detection — it would form a one-record group and always
        # read as consensus. That is the quietest possible failure, so it is a
        # load error.
        undeclared = sorted({rec.question for rec in self.recommendations} - set(self.questions_index))
        if undeclared:
            raise CorpusError(
                f"recommendation 使用了未在问题目录中声明的 question: {undeclared}"
                " —— 未声明的问题无法参与冲突消解，必须先在 00_questions.json 中登记"
            )

    # -- querying --------------------------------------------------------
    def source_for(self, rec: Recommendation) -> GuidelineSource:
        return self.sources[rec.source_id]

    def question(self, question_id: str) -> ClinicalQuestion:
        return self.questions_index.get(
            question_id, ClinicalQuestion(question_id, question_id, False)
        )

    def by_id(self, rec_id: str) -> Recommendation | None:
        for rec in self.recommendations:
            if rec.rec_id == rec_id:
                return rec
        return None

    def topics(self) -> list[str]:
        seen: list[str] = []
        for rec in self.recommendations:
            if rec.topic not in seen:
                seen.append(rec.topic)
        return seen

    def questions(self) -> list[str]:
        seen: list[str] = []
        for rec in self.recommendations:
            if rec.question not in seen:
                seen.append(rec.question)
        return seen

    def select(
        self,
        *,
        topics: Iterable[str] | None = None,
        questions: Iterable[str] | None = None,
        regions: Iterable[str] | None = None,
        traditions: Iterable[str] | None = None,
    ) -> list[Recommendation]:
        topic_set = set(topics) if topics else None
        question_set = set(questions) if questions else None
        region_set = set(regions) if regions else None
        tradition_set = set(traditions) if traditions else None
        out: list[Recommendation] = []
        for rec in self.recommendations:
            source = self.sources[rec.source_id]
            if topic_set and rec.topic not in topic_set:
                continue
            if question_set and rec.question not in question_set:
                continue
            if region_set and source.region not in region_set:
                continue
            if tradition_set and source.tradition not in tradition_set:
                continue
            out.append(rec)
        return out

    def apply(self, facts: Mapping[str, Any], **filters: Any) -> list[Applicability]:
        """Test the (filtered) corpus against ``facts``.

        Returns every record with its status — including the ones that did *not*
        fire. The excluded ones are the clinically interesting half: "本患者
        eGFR 32，双膦酸盐条目被排除" is a finding, not an absence.
        """
        return [
            test_recommendation(rec, self.sources[rec.source_id], facts)
            for rec in self.select(**filters)
        ]

    def stats(self) -> dict[str, Any]:
        by_region: dict[str, int] = {}
        by_tradition: dict[str, int] = {}
        by_topic: dict[str, int] = {}
        for rec in self.recommendations:
            source = self.sources[rec.source_id]
            by_region[source.region] = by_region.get(source.region, 0) + 1
            by_tradition[source.tradition] = by_tradition.get(source.tradition, 0) + 1
            by_topic[rec.topic] = by_topic.get(rec.topic, 0) + 1
        return {
            "files": list(self.files),
            "sources": len(self.sources),
            "questions": len(self.questions_index),
            "exclusive_questions": sum(1 for q in self.questions_index.values() if q.exclusive),
            "recommendations": len(self.recommendations),
            "by_region": by_region,
            "by_tradition": by_tradition,
            "by_topic": by_topic,
            "year_range": [
                min((s.year for s in self.sources.values()), default=0),
                max((s.year for s in self.sources.values()), default=0),
            ],
            "verbatim_records": sum(1 for rec in self.recommendations if rec.verbatim),
        }


_DEFAULT: Corpus | None = None


def default_corpus() -> Corpus:
    """The shipped library, loaded once per process."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Corpus.load()
    return _DEFAULT
