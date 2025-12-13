import os
import requests
import json

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY 环境变量未设置")

url = "https://cloud.infini-ai.com/maas/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

data = {
    "model": "deepseek-v3.2",
    "messages": [
        {"role": "user", "content": "你是谁"}
    ]
}

response = requests.post(url, headers=headers, json=data, timeout=30)

# 基本错误检查
response.raise_for_status()

result = response.json()

# 打印完整返回（调试用）
print(json.dumps(result, ensure_ascii=False, indent=2))

# 只取模型回复（常用）
reply = result["choices"][0]["message"]["content"]
print("\n模型回复：")
print(reply)