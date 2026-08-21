"""可计算循证指南层：把"指南说了什么"变成"这条对本患者是否成立"。"""

from .corpus import Corpus, CorpusError, default_corpus
from .model import (
    Applicability, GuidelineSource, Recommendation, TraceNode,
    FALSE, TRUE, UNKNOWN, evaluate, test_recommendation,
)

__all__ = [
    "Corpus", "CorpusError", "default_corpus",
    "Applicability", "GuidelineSource", "Recommendation", "TraceNode",
    "TRUE", "FALSE", "UNKNOWN", "evaluate", "test_recommendation",
]
