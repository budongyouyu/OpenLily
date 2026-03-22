import os
from datetime import datetime

def write_txt(input: str) -> str:
    """
    将内容写入 txt 文件。

    input 格式支持两种：
        1. 只传内容:         "你好世界"                  → 自动生成文件名
        2. 指定文件名+内容:  "output.txt|你好世界"        → 保存到指定文件名
    """
    # 解析 input
    if "|" in input:
        filename, content = input.split("|", maxsplit=1)
        filename = filename.strip()
    else:
        content  = input
        filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # 统一保存到 outputs/ 目录
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return f"已保存到 {filepath}（{len(content)} 字符）"