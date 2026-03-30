import json
import os
from typing import Callable
from core.LLMClient import LLMClient
from dotenv import load_dotenv
from tools.ToolsRegister import ToolsRegister

load_dotenv()

class WriterAgent:

    def __init__(self, llm: LLMClient, message: str = None, tools: ToolsRegister = None):
        self.llm = llm
        self.message = message
        self.tools = tools or ToolsRegister()

    def system_prompt(self) -> str:
        return """
            你是一个专业的写作专家，擅长悬疑的故事风格和连续反转的故事设计。
            用户会给你一个主题，你需要构思并写出完整故事。
            故事风格：你擅长在一开头就以极高的悬念来吸引读者，擅长连续反转
            写完故事后，必须调用 save_to_file 工具将故事保存到文件，并且文件路径设置在outputs目录中。
            文件名格式：story_主题关键词.txt
        """

    def act(self, user_input: str) -> str:

        messages = [
            {"role": "system", "content": self.system_prompt() },
            {"role": "user"  , "content": user_input            },
        ]

        while True:

            response = self.llm.invoke(messages, tools=self.tools.to_openai_schema())

            print(response.tool_calls)

            if response.has_tool_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": response.content or "",  # 确保不是 None
                }

                # 如果存在 reasoning_content，添加到 assistant 消息级别
                if hasattr(response, 'reasoning_content') and response.reasoning_content:
                    assistant_msg["reasoning_content"] = response.reasoning_content

                # tool_calls 数组
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages.append(assistant_msg)

                for tc in response.tool_calls:
                    args = json.loads(tc.arguments)
                    print(f"\n[工具调用] {tc.name}({args})")

                    # 之前：TOOL_EXECUTOR.get(tc.name)
                    # 现在：直接用注册表执行
                    action_input = next(iter(args.values())) if args else ""
                    result = self.tools.execute(tc.name, action_input)
                    print(f"[工具结果] {result}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
            else:
                print(response.content)
                return response.content
