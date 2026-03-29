from core.LLMClient import BaseLLMClient, LLMClient
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":

    try:
        llmClient = LLMClient()

        exampleMessages = [
            {"role": "system", "content": "你是一个言情小说高手."},
            {"role": "user", "content": "写一个言情小说"}
        ]

        print("--- 调用LLM ---")

        responseText = llmClient.think(exampleMessages)

        print("\n\n--- 完整模型响应 ---")

        for chunk in responseText:
            print(chunk, end="", flush=True)  # 实时打印每个片段

    except ValueError as e:
        print(e)
