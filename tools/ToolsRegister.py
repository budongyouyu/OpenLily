from typing import Callable, Dict

class Tool:
    """单个工具的封装"""

    def __init__(self, name: str, description: str, func: Callable[[str], str]):
        """
        :param name:        工具名称，LLM 通过这个名字调用
        :param description: 工具描述，会注入到 system_prompt 告知 LLM
        :param func:        实际执行函数，接收字符串参数，返回字符串结果
        """
        self.name        = name
        self.description = description
        self.func        = func

    def run(self, input: str) -> str:
        return self.func(input)

    def __repr__(self):
        return f"Tool(name={self.name!r})"


class ToolsRegister:
    """
    工具注册表，负责注册、管理和执行工具。

    用法：
        registry = ToolsRegister()
        registry.register("get_weather", "查询城市天气", get_weather_func)

        # 传给 ReActAgent
        agent.run("上海天气？", tool_executor=registry.execute)

        # 传给 system_prompt 注入工具描述
        tools_desc = registry.descriptions()
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, name: str, description: str, func: Callable[[str], str]) -> "ToolsRegister":
        """
        注册一个工具。支持链式调用。

        :param name:        工具名称
        :param description: 工具描述
        :param func:        执行函数
        """
        self._tools[name] = Tool(name=name, description=description, func=func)
        print(f"🔧 已注册工具: {name}")
        return self

    def execute(self, action: str, action_input: str) -> str:
        """
        执行指定工具，供 ReActAgent 的 tool_executor 使用。

        :param action:       工具名称
        :param action_input: 工具输入参数
        """
        tool = self._tools.get(action)
        if not tool:
            return f"错误：工具 '{action}' 不存在，可用工具: {self.names()}"
        try:
            return tool.run(action_input)
        except Exception as e:
            return f"工具 '{action}' 执行出错: {e}"

    def descriptions(self) -> str:
        """返回所有工具的描述，用于注入 system_prompt"""
        if not self._tools:
            return "（暂无可用工具）"
        return "\n".join(
            f"- {tool.name}: {tool.description}"
            for tool in self._tools.values()
        )

    def names(self) -> list[str]:
        """返回所有已注册的工具名称列表"""
        return list(self._tools.keys())

    def to_openai_schema(self) -> list[dict]:
        """生成 Function Calling 格式的工具 schema"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "工具的输入内容",
                            }
                        },
                        "required": ["content"],
                    },
                },
            }
            for tool in self._tools.values()
        ]

    def __len__(self):
        return len(self._tools)

    def __repr__(self):
        return f"ToolsRegister(tools={self.names()})"