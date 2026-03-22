from typing import Callable, Type
from LLMClient import BaseLLMClient
from BaseAgent import BaseAgent
from core.ReactAgent import ReActAgent
from tools.ToolsRegister import ToolsRegister

class AgentFactory:
    """
    Agent 工厂类，负责组装 LLM + Tools + Agent。

    用法：
        agent = AgentFactory.create_react_agent(
            llm=OpenAILLMClient(),
            tools=[("write_txt", "保存内容到文件", write_txt)]
        )
        agent.run("帮我写首诗并保存", tool_executor=registry.execute)
    """

    # 已注册的 Agent 类型: name → AgentClass
    _registry: dict[str, Type[BaseAgent]] = {
        "react": ReActAgent,
    }

    @classmethod
    def register_agent(cls, name: str, agent_class: Type[BaseAgent]):
        """
        注册自定义 Agent 类型。

        :param name:        Agent 类型名称
        :param agent_class: Agent 类，必须继承 BaseAgent
        """
        cls._registry[name] = agent_class
        print(f"✅ 已注册 Agent 类型: {name}")

    @classmethod
    def create(
        cls,
        agent_type: str,
        llm: BaseLLMClient,
        tools: list[tuple[str, str, Callable]] = None,
    ) -> tuple[BaseAgent, ToolsRegister]:
        """
        通用创建方法，根据 agent_type 创建对应 Agent。

        :param agent_type: Agent 类型，如 "react"
        :param llm:        LLM 客户端实例
        :param tools:      工具列表，每个元素为 (name, description, func)
        :return:           (agent 实例, registry 实例)
        """
        agent_class = cls._registry.get(agent_type)
        if not agent_class:
            supported = list(cls._registry.keys())
            raise ValueError(f"不支持的 Agent 类型: '{agent_type}'，可选: {supported}")

        registry = cls._build_registry(tools)
        agent    = agent_class(llm=llm)
        return agent, registry

    @classmethod
    def create_react_agent(
        cls,
        llm: BaseLLMClient,
        tools: list[tuple[str, str, Callable]] = None,
    ) -> tuple[ReActAgent, ToolsRegister]:
        """
        快捷创建 ReAct Agent。

        :param llm:   LLM 客户端实例
        :param tools: 工具列表，每个元素为 (name, description, func)
        :return:      (ReActAgent 实例, ToolsRegister 实例)
        """
        registry = cls._build_registry(tools)
        agent    = ReActAgent(llm=llm)
        return agent, registry

    @classmethod
    def _build_registry(cls, tools: list[tuple[str, str, Callable]] = None) -> ToolsRegister:
        """构建并注册工具"""
        registry = ToolsRegister()
        if tools:
            for name, description, func in tools:
                registry.register(name, description, func)
        return registry