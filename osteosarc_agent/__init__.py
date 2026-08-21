"""筋骨智策 OsteoSarc-Agent —— 骨质疏松与肌少症全周期智能决策平台。

可计算循证指南 + 一个主智能体 + 六个子智能体。确定性内核，语料可替换，
每一条结论都绑定到在本例中成立的指南条目。
"""

from .orchestrator import Orchestrator, assess

__all__ = ["Orchestrator", "assess", "__version__"]
__version__ = "0.1.0"
