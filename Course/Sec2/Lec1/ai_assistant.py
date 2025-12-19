"""
AI 助手类 - 用于调用 AI API

这个模块提供了 AIAssistant 类，用于与 AI API 进行交互。
"""

import requests


class AIAssistant:
    """AI 助手类，用于调用 AI API"""
    
    def __init__(self, api_key):
        """
        初始化 AI 助手
        :param api_key: API 密钥
        """
        self.api_key = api_key
        self.api_url = "https://cloud.infini-ai.com/maas/v1/chat/completions"
    
    def get_response(self, prompt):
        """
        调用 AI API 获取回答
        :param prompt: 用户输入的问题或提示词
        :return: AI 模型的回复内容
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": "deepseek-v3.2",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=300)
            response.raise_for_status()
            result = response.json()
            reply = result["choices"][0]["message"]["content"]
            return reply
        except Exception as e:
            print(f"❌ AI 调用失败: {e}")
            return ""

