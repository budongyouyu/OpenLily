from tools.ToolsRegister import ToolsRegister
from tools.write_txt import write_txt
from core.ReactAgent import ReActAgent
from core.LLMClient import BaseLLMClient
from dotenv import load_dotenv

load_dotenv()

registry = ToolsRegister()
registry.register(
    "write_txt",
    "将内容保存到 txt 文件。格式：'文件名.txt|内容' 或直接传内容自动命名",
    write_txt
)

llm   = BaseLLMClient(temperature=1)
agent = ReActAgent(llm=llm)
agent.run("帮我写一首关于春天的诗，并保存到 spring.txt", tool_executor=registry.execute)
