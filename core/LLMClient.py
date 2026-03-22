import os
from abc import ABC, abstractmethod
from typing import Dict, List, Iterable
from openai import OpenAI


class BaseLLMClient(ABC):
    """
    LLM 抽象基类，定义所有 LLM 客户端必须实现的接口。
    """

    @abstractmethod
    def think(self, messages: List[Dict[str, str]]) -> Iterable[str]:
        """子类必须实现：调用 LLM，流式返回响应"""
        pass


class OpenAILLMClient(BaseLLMClient):
    """
    兼容 OpenAI 接口的 LLM 客户端实现。
    支持所有兼容 OpenAI 接口的服务（DeepSeek、Qwen、Kimi 等）。
    """

    def __init__(
        self,
        model: str = None,
        apiKey: str = None,
        baseUrl: str = None,
        timeout: int = None,
        temperature: float = 1.0,
    ):
        """
        初始化客户端。优先使用传入参数，如果未提供则从环境变量加载。
        """
        self.model       = model   or os.getenv("LLM_MODEL_ID")
        self.apiKey      = apiKey  or os.getenv("LLM_API_KEY")
        self.baseUrl     = baseUrl or os.getenv("LLM_BASE_URL")
        self.timeout     = timeout or int(os.getenv("LLM_TIMEOUT", 60))
        self.temperature = temperature

        if not all([self.model, self.apiKey, self.baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在 .env 文件中定义。")

        self.client = OpenAI(
            api_key=self.apiKey,
            base_url=self.baseUrl,
            timeout=self.timeout,
        )

    def think(self, messages: List[Dict[str, str]]) -> Iterable[str]:
        """
        调用 LLM，流式 yield 响应内容。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stream=True,
            )
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
            print()

        except Exception as e:
            print(f"❌ 调用 LLM API 时发生错误: {e}")
            return