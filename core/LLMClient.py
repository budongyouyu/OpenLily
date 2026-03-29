import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Iterable, Any
from openai import OpenAI

# ── 新增：工具调用响应结构 ──

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str          # JSON 字符串，用 json.loads() 解析

@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class BaseLLMClient(ABC):

    @abstractmethod
    def stream(self, messages: List[Dict[str, str]]) -> Iterable[str]:
        """流式调用，yield 文本片段"""
        pass

    @abstractmethod
    def invoke(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict],
    ) -> LLMResponse:
        """非流式调用，支持 Function Calling，返回完整响应"""
        pass


class LLMClient(BaseLLMClient):

    def __init__(
        self,
        model: str = None,
        apiKey: str = None,
        baseUrl: str = None,
        timeout: int = None,
        temperature: float = 1.0,
    ):
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

    def stream(self, messages: List[Dict[str, str]]) -> Iterable[str]:
        """流式调用，yield 文本片段（原有逻辑不动）"""
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

    def invoke(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict],
        tool_choice: str = "auto",
    ) -> LLMResponse:
        """
        非流式调用，支持 Function Calling。
        tool_choice:
            "auto"     — LLM 自己决定是否调用工具
            "required" — 强制必须调用工具
            "none"     — 禁止调用工具
        """

        print(f"🧠 正在调用 {self.model} 模型（Function Calling）...")
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                tools=tools,
                tool_choice=tool_choice,
                stream=False,       # Function Calling 必须关流式
            )
            msg = resp.choices[0].message

            # 解析 tool_calls
            tool_calls = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ))

            return LLMResponse(
                content=msg.content or "",
                model=resp.model,
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
                tool_calls=tool_calls,
                raw=resp,
            )

        except Exception as e:
            print(f"❌ 调用 LLM API 时发生错误: {e}")
            return LLMResponse(content="", model=self.model)