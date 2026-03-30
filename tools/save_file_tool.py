# tools/save_file_tool.py

def make_save_file_tool(register: "ToolsRegister", filename: str = "output.txt"):
    """
    创建并注册保存文件工具。

    :param register: ToolsRegister 实例
    :param filename: 保存的文件名，默认 output.txt
    """

    def save_to_file(content: str) -> str:
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            return f"✅ 已保存到 {filename}，共 {len(content)} 字"
        except Exception as e:
            return f"❌ 保存失败: {e}"

    register.register(
        name="save_to_file",
        description=f"将内容保存到本地文件 {filename}，输入参数为要保存的完整文本内容",
        func=save_to_file,
    )