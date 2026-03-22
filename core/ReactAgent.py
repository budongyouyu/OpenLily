import re
from typing import Dict, List, Callable
from core.BaseAgent import BaseAgent
from core.LLMClient import BaseLLMClient


class ReActAgent(BaseAgent):
    """
    基于 ReAct 框架的 Agent。
    工具通过外部注入，由 ToolsRegistry 管理。

    ReAct 循环：
        Thought    → LLM 思考下一步做什么
        Action     → LLM 决定调用哪个工具
        Action Input → 工具的输入参数
        Observation → 执行工具后的结果，喂回 LLM
        ... 循环直到输出 Final Answer
    """

    MAX_STEPS = 10

    def __init__(
        self,
        llm: BaseLLMClient,
        tools: Dict[str, Callable[[str], str]],  # { tool_name: func }
        tools_desc: str,                          # 工具描述，由 ToolsRegistry 生成
    ):
        self.tools      = tools
        self.tools_desc = tools_desc
        super().__init__(llm)

    def system_prompt(self) -> str:
        return f"""你是一个智能助手，通过"思考→行动→观察"的循环来解决问题。
                    你可以使用以下工具：
                         {self.tools_desc}
            
                    每次回复必须严格遵守以下格式之一：
            
                    格式一（需要调用工具时）：
                        Thought: 你的思考过程
                        Action: 工具名称
                        Action Input: 工具的输入参数
            
                    格式二（可以直接回答时）：
                        Thought: 我已经知道答案了
                        Final Answer: 最终回答内容
            
                     规则：
                        - 每次只能调用一个工具
                         - Action 必须是工具列表中的名称，不能编造
                        - 看到 Observation 后继续思考，直到给出 Final Answer
                """

    def run(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        print(f"\n👤 用户: {user_input}\n")

        for step in range(self.MAX_STEPS):
            print(f"── Step {step + 1} ──────────────────────")

            # ① LLM 思考
            llm_output = "".join(self.llm.think(self.history))
            print(f"🤖 LLM:\n{llm_output}\n")

            # ② 是否已有最终答案
            final_answer = self._parse_final_answer(llm_output)
            if final_answer:
                self.history.append({"role": "assistant", "content": llm_output})
                print(f"✅ Final Answer: {final_answer}")
                return final_answer

            # ③ 解析 Action
            action, action_input = self._parse_action(llm_output)
            if not action:
                self.history.append({"role": "assistant", "content": llm_output})
                return llm_output

            # ④ 执行工具
            observation = self._execute_tool(action, action_input)
            print(f"🔧 Tool [{action}] → {observation}\n")

            # ⑤ 把结果追加到历史，进入下一轮
            self.history.append({"role": "assistant", "content": llm_output})
            self.history.append({"role": "user",      "content": f"Observation: {observation}"})

        return "❌ 超过最大步骤数，未能得出答案"

    # ── 解析 ──────────────────────────────────────────────────────────────────

    def _parse_final_answer(self, text: str) -> str | None:
        match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
        return match.group(1).strip() if match else None

    def _parse_action(self, text: str) -> tuple[str | None, str]:
        action_match = re.search(r"Action:\s*(.+)", text)
        input_match  = re.search(r"Action Input:\s*(.+)", text, re.DOTALL)
        action       = action_match.group(1).strip() if action_match else None
        action_input = input_match.group(1).strip()  if input_match  else ""
        return action, action_input

    def _execute_tool(self, action: str, action_input: str) -> str:
        func = self.tools.get(action)
        if not func:
            return f"错误：工具 '{action}' 不存在，可用工具: {list(self.tools.keys())}"
        try:
            return func(action_input)
        except Exception as e:
            return f"工具执行出错: {e}"