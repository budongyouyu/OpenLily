# test/test_writer_agent.py

import sys
import os

# 把项目根目录加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from core.LLMClient import LLMClient
from tools.ToolsRegister import ToolsRegister
from tools.save_file_tool import make_save_file_tool
from agent.WriterAgent import WriterAgent


def test_writer_save_to_file():

    # 1. 初始化 LLM
    llm = LLMClient()

    # 2. 注册工具
    registry = ToolsRegister()
    make_save_file_tool(registry, filename="outputs/story.txt")

    # 3. 创建 Agent
    writer = WriterAgent(llm=llm, tools=registry)

    # 4. 运行
    result = writer.act(
        "请帮我以《青春的保护色》为主题，写一篇无限流悬疑小说，要求多重反转"
    )

    # 5. 验证文件是否生成
    assert os.path.exists("outputs/story.txt"), "❌ 文件未生成"
    assert len(open("outputs/story.txt", encoding="utf-8").read()) > 0, "❌ 文件内容为空"
    print("✅ 测试通过，文件已生成")

if __name__ == "__main__":
    test_writer_save_to_file()
