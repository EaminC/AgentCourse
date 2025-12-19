import os
import requests
import json

#API_KEY = os.getenv("API_KEY")
API_KEY="sk-23qfb76qghixbui2"
if not API_KEY:
    raise RuntimeError("API_KEY 环境变量未设置")


def get_ai_response(prompt):
    """
    调用 AI API 获取回答
    
    参数:
        prompt (str): 用户输入的问题或提示词
        
    返回:
        str: AI 模型的回复内容
    """
    url = "https://cloud.infini-ai.com/maas/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    data = {
        "model": "deepseek-v3.2",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    response = requests.post(url, headers=headers, json=data, timeout=300)
    
    # 基本错误检查
    response.raise_for_status()
    
    result = response.json()
    
    # 只取模型回复并返回
    reply = result["choices"][0]["message"]["content"]
    return reply


# 使用示例
if __name__ == "__main__":
    prompt = "你好 请你介绍一下上海市有什么好吃的"
    response = get_ai_response(prompt)
    print("模型回复：")
    print(response)