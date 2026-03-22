# core/BaseAgent.py
from abc import ABC, abstractmethod
from typing import Dict, List, Callable
from core.LLMClient import BaseLLMClient

class BaseAgent(ABC):

    def __init__(self, llm: BaseLLMClient):
        self.llm = llm
        self.history: List[Dict[str, str]] = []
        self.history.append({
            "role": "system",
            "content": self.system_prompt()
        })

    @abstractmethod
    def system_prompt(self) -> str:
        pass

    @abstractmethod
    def run(self, user_input: str, tool_executor: Callable = None) -> str:
        # ↑ 加上 tool_executor 参数，子类覆写时签名就对齐了
        pass

    def clear_history(self):
        self.history = [self.history[0]]