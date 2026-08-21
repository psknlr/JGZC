"""Command line entry point.

    python -m osteosarc_agent assess --case demo
    python -m osteosarc_agent assess --file ./case.json --json
    python -m osteosarc_agent guidelines --stats
    python -m osteosarc_agent conflicts --case demo
    python -m osteosarc_agent ui --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import cases as demo_cases
from .criteria import sarcopenia as sarco
from .guidelines.corpus import Corpus, CorpusError, default_corpus
from .orchestrator import Orchestrator
from .render import render


def _load_case(args: argparse.Namespace) -> dict[str, Any]:
    if args.file:
        path = Path(args.file)
        if not path.exists():
            raise SystemExit(f"病例文件不存在: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SystemExit("病例文件的顶层必须是 JSON 对象")
        return payload
    try:
        return demo_cases.get(args.case)
    except KeyError as exc:
        raise SystemExit(str(exc)) from None


def _corpus(args: argparse.Namespace) -> Corpus:
    if getattr(args, "corpus", None):
        return Corpus.load([args.corpus])
    return default_corpus()


def _orchestrator(args: argparse.Namespace) -> Orchestrator:
    standards = tuple(args.standards.split(",")) if getattr(args, "standards", None) else sarco.DEFAULT_STANDARDS
    unknown = [s for s in standards if s not in sarco.STANDARDS]
    if unknown:
        raise SystemExit(f"未知肌少症标准 {unknown}，可用: {', '.join(sarco.STANDARDS)}")
    narrator = None
    if getattr(args, "llm", None):
        from .llm import Narrator, build_client, LLMError
        try:
            narrator = Narrator(build_client(args.llm, getattr(args, "model", None)))
        except LLMError as exc:
            print(f"[warn] 模型不可用，按确定性路径运行：{exc}", file=sys.stderr)
            narrator = Narrator()
    return Orchestrator(corpus=_corpus(args), standards=standards, narrator=narrator)


def cmd_assess(args: argparse.Namespace) -> int:
    decision = _orchestrator(args).run(_load_case(args))
    if args.json:
        print(json.dumps(decision, ensure_ascii=False, indent=2))
    else:
        print(render(decision, evidence=not args.no_evidence))
    # Exit code carries the run status so a script can gate on it.
    return {"ok": 0, "needs_action": 2, "degraded": 3}.get(decision["status"], 1)


def cmd_conflicts(args: argparse.Namespace) -> int:
    decision = _orchestrator(args).run(_load_case(args))
    conflicts = decision["conflicts"]
    print("；".join(f"{k} {v}" for k, v in conflicts["counts_zh"].items()))
    for question in conflicts["questions"]:
        if args.all or question["verdict"] in ("disputed", "resolved_by_patient"):
            print(f"\n【{question['verdict_zh']}】{question['label_zh']}")
            print(f"  {question['basis']}")
            for position in question.get("divergence", []) or question.get("positions", []):
                stance = position.get("stance") or position.get("direction_zh")
                regions = "/".join(position.get("regions", []))
                print(f"   ▸ {stance}「{position['action']}」（{regions}）")
            if question.get("platform_policy"):
                print(f"   ⚙ {question['platform_policy']}")
    return 0


def cmd_guidelines(args: argparse.Namespace) -> int:
    try:
        corpus = _corpus(args)
    except CorpusError as exc:
        print(f"语料加载失败：{exc}", file=sys.stderr)
        return 1
    if args.show:
        rec = corpus.by_id(args.show)
        if rec is None:
            print(f"未找到 {args.show}", file=sys.stderr)
            return 1
        source = corpus.source_for(rec)
        print(json.dumps({"recommendation": rec.to_dict(), "source": source.to_dict(),
                          "applies_when": rec.applies_when, "excluded_when": rec.excluded_when},
                         ensure_ascii=False, indent=2))
        return 0
    if args.questions:
        for question_id in corpus.questions():
            question = corpus.question(question_id)
            count = len(corpus.select(questions=[question_id]))
            flag = "互斥" if question.exclusive else "互补"
            print(f"{question_id:<46s} [{flag}] {count:>2d} 条  {question.label_zh}")
        return 0
    if args.list:
        for rec in corpus.recommendations:
            source = corpus.source_for(rec)
            print(f"{rec.rec_id:<42s} {source.region:<5s} {rec.topic:<18s} {rec.statement_zh[:40]}")
        return 0
    print(json.dumps(corpus.stats(), ensure_ascii=False, indent=2))
    return 0


def cmd_agents(args: argparse.Namespace) -> int:
    orchestrator = _orchestrator(args)
    print("主智能体：OsteoSarc Orchestrator")
    for index, agent in enumerate(orchestrator.agent_catalog(), start=1):
        print(f"  {index}. {agent['name_zh']:<20s} id={agent['agent_id']:<10s} schema={agent['schema']}")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from .ui.server import serve
    serve(host=args.host, port=args.port, corpus_path=getattr(args, "corpus", None),
          llm=getattr(args, "llm", None), model=getattr(args, "model", None))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    orchestrator = _orchestrator(args)
    for name in demo_cases.CASES:
        decision = orchestrator.run(demo_cases.get(name))
        meta = decision["meta"]
        radar = "  ".join(
            f"{axis['label']}={axis['score']}({axis['tier_zh']})"
            for axis in decision["risk"]["radar"]["axes"]
        )
        counts = "；".join(f"{k}{v}" for k, v in decision["conflicts"]["counts_zh"].items())
        print(f"\n[{name}] {meta['case_label']}  → {decision['status']}")
        print(f"  诊断：{decision['diagnosis']['osteoporosis']['diagnosis_zh']} / "
              + " · ".join(f"{s['standard_id']}:{s['verdict_zh']}" for s in decision['diagnosis']['standards']))
        print(f"  风险：{radar}")
        print(f"  冲突：{counts}")
        print(f"  方案：{decision['plans']['item_count']} 条；安全：{len(decision['safety']['issues'])} 项"
              f"（阻断 {len(decision['safety']['blocking'])}）")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="osteosarc-agent",
        description="筋骨智策 OsteoSarc-Agent —— 骨质疏松与肌少症全周期智能决策平台",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser, *, case: bool = True) -> None:
        if case:
            sub.add_argument("--case", default="demo", help=f"示例病例：{', '.join(demo_cases.CASES)}")
            sub.add_argument("--file", help="病例 JSON 文件路径")
            sub.add_argument("--standards", help="肌少症标准，逗号分隔（默认三套全跑）")
        sub.add_argument("--corpus", help="自定义语料目录或文件（替换内置语料）")
        sub.add_argument("--llm", help="可选表述层：anthropic / openai")
        sub.add_argument("--model", help="模型名")

    assess = subparsers.add_parser("assess", help="对一个病例跑完整评估")
    add_common(assess)
    assess.add_argument("--json", action="store_true", help="输出完整 JSON")
    assess.add_argument("--no-evidence", action="store_true", help="不打印证据台账")
    assess.set_defaults(func=cmd_assess)

    conflicts = subparsers.add_parser("conflicts", help="只看指南冲突消解结果")
    add_common(conflicts)
    conflicts.add_argument("--all", action="store_true", help="列出全部问题而不仅是有分歧的")
    conflicts.set_defaults(func=cmd_conflicts)

    guidelines = subparsers.add_parser("guidelines", help="查看/校验语料")
    add_common(guidelines, case=False)
    guidelines.add_argument("--stats", action="store_true", help="语料统计（默认行为）")
    guidelines.add_argument("--list", action="store_true", help="列出全部推荐")
    guidelines.add_argument("--questions", action="store_true", help="列出全部临床问题")
    guidelines.add_argument("--show", help="显示某条推荐的完整定义（含适用条件）")
    guidelines.set_defaults(func=cmd_guidelines)

    agents = subparsers.add_parser("agents", help="列出主智能体与子智能体")
    add_common(agents, case=False)
    agents.set_defaults(func=cmd_agents)

    demo = subparsers.add_parser("demo", help="跑完全部示例病例并对比")
    add_common(demo, case=False)
    demo.set_defaults(func=cmd_demo)

    ui = subparsers.add_parser("ui", help="启动可视化控制台")
    add_common(ui, case=False)
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8000)
    ui.set_defaults(func=cmd_ui)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CorpusError as exc:
        print(f"语料错误：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
