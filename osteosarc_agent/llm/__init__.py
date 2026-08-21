"""可选认知层：模型只被允许改写已确定的结论，不能作出临床判断。"""

from .base import LLMClient, LLMError, NullLLMClient
from .narrate import Narrator, check_narrative, summarise
from .providers import build_client, describe

__all__ = [
    "LLMClient", "LLMError", "NullLLMClient",
    "Narrator", "check_narrative", "summarise",
    "build_client", "describe",
]
