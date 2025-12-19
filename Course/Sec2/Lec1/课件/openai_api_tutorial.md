# OpenAI API 使用教程

## 目录

1. [API 简介](#api-简介)
2. [安装和配置](#安装和配置)
3. [基础使用](#基础使用)
4. [常用功能](#常用功能)
5. [错误处理](#错误处理)
6. [实际应用示例](#实际应用示例)
7. [使用无问 API（Infini-AI）](#使用无问-apiinfini-ai)
8. [最佳实践](#最佳实践)

---

## API 简介

OpenAI API 提供了强大的 AI 功能，包括：

- **文本生成**：生成文章、对话、代码等
- **文本理解**：总结、翻译、问答等
- **代码生成**：根据描述生成代码
- **对话系统**：构建聊天机器人

### 主要模型

- `gpt-3.5-turbo`：快速、经济，适合大多数任务
- `gpt-4`：更强大，适合复杂任务
- `text-embedding-ada-002`：文本嵌入模型

---

## 安装和配置

### 1. 安装所需库

**安装 OpenAI 库**

```bash
pip install openai
```

**安装 requests 库（用于调用第三方 API）**

```bash
pip install requests
```

### 2. 获取 API Key

1. 访问 [OpenAI 官网](https://platform.openai.com/)
2. 注册/登录账号
3. 进入 API Keys 页面
4. 创建新的 API Key

### 3. 配置 API Key

**方法 1：环境变量（推荐）**

```bash
# Windows
set OPENAI_API_KEY=your-api-key-here

# Linux/Mac
export OPENAI_API_KEY=your-api-key-here
```

**方法 2：在代码中设置（仅用于测试，不推荐用于生产环境）**

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

---

## 基础使用

### 导入库

```python
from openai import OpenAI

# 创建客户端（会自动从环境变量读取 API Key）
client = OpenAI()
```

### 最简单的文本生成

```python
from openai import OpenAI

client = OpenAI()

# 调用 API
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "用一句话介绍 Python"}
    ]
)

# 获取回复
answer = response.choices[0].message.content
print(answer)
```

### 理解响应结构

```python
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "你好"}
    ]
)

# 响应对象的主要属性
print(response.id)                    # 请求 ID
print(response.model)                  # 使用的模型
print(response.choices[0].message)    # 消息对象
print(response.choices[0].message.content)  # 回复内容
print(response.usage.total_tokens)    # 使用的 token 数量
```

---

## 常用功能

### 1. 对话系统（多轮对话）

```python
from openai import OpenAI

client = OpenAI()

# 对话历史
messages = [
    {"role": "system", "content": "你是一个友好的助手"},
    {"role": "user", "content": "你好，我是小明"},
    {"role": "assistant", "content": "你好小明！很高兴认识你。"},
    {"role": "user", "content": "你能帮我写代码吗？"}
]

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)

print(response.choices[0].message.content)
```

### 2. 文本总结

```python
from openai import OpenAI

client = OpenAI()

long_text = """
人工智能（AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。
机器学习是 AI 的一个子集，它使计算机能够从数据中学习，而无需明确编程。
深度学习是机器学习的一个子集，使用神经网络来模拟人脑的工作方式。
"""

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": f"请用一句话总结以下内容：\n{long_text}"}
    ]
)

print(response.choices[0].message.content)
```

### 3. 代码生成

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "用 Python 写一个函数，计算斐波那契数列的第 n 项"}
    ]
)

code = response.choices[0].message.content
print(code)
```

### 4. 文本翻译

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "将以下英文翻译成中文：Hello, how are you?"}
    ]
)

print(response.choices[0].message.content)
```

### 5. 参数控制

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "写一首关于春天的诗"}
    ],
    temperature=0.7,      # 控制随机性（0-2），值越大越随机
    max_tokens=200,       # 最大生成 token 数
    top_p=0.9             # 核采样参数
)

print(response.choices[0].message.content)
```

**参数说明：**

- `temperature`：0-2，控制输出的随机性
  - 0：完全确定，适合需要准确答案的任务
  - 1：平衡（默认）
  - 2：非常随机，适合创意写作
- `max_tokens`：限制生成的最大长度
- `top_p`：核采样，控制输出的多样性

---

## 错误处理

### 常见错误类型

```python
from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError

client = OpenAI()

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "你好"}
        ]
    )
    print(response.choices[0].message.content)

except RateLimitError as e:
    print(f"请求过于频繁，请稍后再试: {e}")

except APIConnectionError as e:
    print(f"网络连接错误: {e}")

except APIError as e:
    print(f"API 错误: {e}")

except Exception as e:
    print(f"其他错误: {e}")
```

### 检查 API Key

```python
from openai import OpenAI
import os

# 检查环境变量
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("错误：未设置 OPENAI_API_KEY 环境变量")
    exit(1)

client = OpenAI()
```

---

## 实际应用示例

### 示例 1：智能问答助手

```python
from openai import OpenAI

class ChatBot:
    def __init__(self):
        self.client = OpenAI()
        self.messages = [
            {"role": "system", "content": "你是一个友好的助手，用简洁的语言回答问题。"}
        ]

    def chat(self, user_input):
        """添加用户消息并获取回复"""
        self.messages.append({"role": "user", "content": user_input})

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.messages,
            temperature=0.7
        )

        assistant_reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_reply})

        return assistant_reply

# 使用示例
bot = ChatBot()
print(bot.chat("Python 是什么？"))
print(bot.chat("它有什么优点？"))
```

### 示例 2：文本摘要工具

```python
from openai import OpenAI

def summarize_text(text, max_length=100):
    """总结文本内容"""
    client = OpenAI()

    prompt = f"请用不超过 {max_length} 字总结以下内容：\n\n{text}"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_length,
        temperature=0.3  # 较低的温度，更准确
    )

    return response.choices[0].message.content

# 使用示例
article = """
人工智能正在改变我们的生活方式。从智能手机的语音助手到自动驾驶汽车，
AI 技术已经渗透到各个领域。机器学习算法能够从大量数据中学习模式，
帮助医生诊断疾病，帮助科学家发现新药，帮助工程师优化设计。
"""
summary = summarize_text(article, max_length=50)
print(summary)
```

### 示例 3：代码审查助手

````python
from openai import OpenAI

def review_code(code):
    """审查代码并提供改进建议"""
    client = OpenAI()

    prompt = f"""请审查以下 Python 代码，指出潜在问题并提供改进建议：

```python
{code}
```"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

# 使用示例
code = """
def calculate_sum(numbers):
    total = 0
    for i in range(len(numbers)):
        total = total + numbers[i]
    return total
"""

review = review_code(code)
print(review)
````

---

## 使用无问 API（Infini-AI）

除了 OpenAI 官方 API，还有一些兼容 OpenAI 格式的第三方 API 服务，例如无问 API（Infini-AI）。这些服务提供了更多模型选择，可能在某些场景下更加经济实惠。

### 1. 配置无问 API

**获取 API Key**

1. 访问 [Infini-AI 平台](https://cloud.infini-ai.com/)
2. 注册并获取 API Key

**设置环境变量**

```bash
# Linux/Mac
export API_KEY=your-api-key-here

# Windows
set API_KEY=your-api-key-here
```

### 2. 基础使用（使用 requests 库）

无问 API 兼容 OpenAI 的接口格式，但使用不同的端点。以下是使用 requests 库的示例：

```python
import os
import requests
import json

# 获取 API Key
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY 环境变量未设置")

# API 端点
url = "https://cloud.infini-ai.com/maas/v1/chat/completions"

# 设置请求头
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# 构建请求数据
data = {
    "model": "deepseek-v3.2",  # 使用的模型
    "messages": [
        {"role": "user", "content": "你是谁"}
    ]
}

# 发送请求
response = requests.post(url, headers=headers, json=data, timeout=30)

# 检查错误
response.raise_for_status()

# 解析结果
result = response.json()

# 获取模型回复
reply = result["choices"][0]["message"]["content"]
print("模型回复：")
print(reply)
```

### 3. 完整示例：对话助手

```python
import os
import requests
import json

class InfiniChatBot:
    def __init__(self, model="deepseek-v3.2"):
        self.api_key = os.getenv("API_KEY")
        if not self.api_key:
            raise RuntimeError("API_KEY 环境变量未设置")

        self.url = "https://cloud.infini-ai.com/maas/v1/chat/completions"
        self.model = model
        self.messages = []

    def chat(self, user_input, system_prompt=None):
        """发送消息并获取回复"""
        # 如果是第一条消息且提供了系统提示
        if system_prompt and not self.messages:
            self.messages.append({"role": "system", "content": system_prompt})

        # 添加用户消息
        self.messages.append({"role": "user", "content": user_input})

        # 构建请求
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "messages": self.messages,
            "temperature": 0.7
        }

        try:
            # 发送请求
            response = requests.post(self.url, headers=headers, json=data, timeout=30)
            response.raise_for_status()

            result = response.json()

            # 获取助手回复
            assistant_reply = result["choices"][0]["message"]["content"]
            self.messages.append({"role": "assistant", "content": assistant_reply})

            return assistant_reply

        except requests.exceptions.RequestException as e:
            return f"请求错误: {e}"
        except KeyError as e:
            return f"响应格式错误: {e}"

    def reset(self):
        """重置对话历史"""
        self.messages = []

# 使用示例
bot = InfiniChatBot()
print(bot.chat("你好，请介绍一下 Python 语言", system_prompt="你是一个编程专家"))
print("\n" + "="*50 + "\n")
print(bot.chat("它有哪些优点？"))
```

### 4. 使用 OpenAI 库调用无问 API

无问 API 兼容 OpenAI 的接口格式，因此也可以使用 OpenAI 库，只需修改 base_url：

```python
from openai import OpenAI
import os

# 创建客户端，指定自定义端点
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://cloud.infini-ai.com/maas/v1"
)

# 调用方式与 OpenAI 官方 API 完全相同
response = client.chat.completions.create(
    model="deepseek-v3.2",
    messages=[
        {"role": "user", "content": "用一句话介绍 Python"}
    ]
)

print(response.choices[0].message.content)
```

### 5. 错误处理

```python
import os
import requests
import json

def call_infini_api(prompt, model="deepseek-v3.2"):
    """调用无问 API 并处理错误"""
    api_key = os.getenv("API_KEY")
    if not api_key:
        return "错误: API_KEY 环境变量未设置"

    url = "https://cloud.infini-ai.com/maas/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    except requests.exceptions.Timeout:
        return "错误: 请求超时，请稍后重试"

    except requests.exceptions.HTTPError as e:
        return f"错误: HTTP 错误 {response.status_code} - {e}"

    except requests.exceptions.ConnectionError:
        return "错误: 网络连接失败"

    except KeyError:
        return "错误: 返回数据格式不正确"

    except Exception as e:
        return f"错误: {str(e)}"

# 使用示例
result = call_infini_api("介绍一下深度学习")
print(result)
```

### 6. 可用模型

无问 API 提供多种模型选择，例如：

- `deepseek-v3.2`：强大的通用模型
- 其他模型请参考官方文档

### 7. OpenAI API vs 无问 API 对比

| 特性         | OpenAI API                  | 无问 API                               |
| ------------ | --------------------------- | -------------------------------------- |
| **接口格式** | OpenAI 标准格式             | 兼容 OpenAI 格式                       |
| **端点**     | `https://api.openai.com/v1` | `https://cloud.infini-ai.com/maas/v1`  |
| **认证方式** | Bearer Token                | Bearer Token                           |
| **主要模型** | GPT-3.5, GPT-4              | DeepSeek, 其他国内模型                 |
| **使用方式** | `from openai import OpenAI` | requests 或 OpenAI 库（指定 base_url） |

---

## 最佳实践

### 1. 管理 API 成本

```python
# 设置合理的 max_tokens，避免生成过长内容
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    max_tokens=500  # 根据需要设置
)

# 检查 token 使用量
print(f"使用了 {response.usage.total_tokens} 个 token")
```

### 2. 缓存常用响应

```python
# 对于相同的问题，可以缓存结果，避免重复调用
cache = {}

def get_cached_response(prompt):
    if prompt in cache:
        return cache[prompt]

    response = client.chat.completions.create(...)
    result = response.choices[0].message.content
    cache[prompt] = result
    return result
```

### 3. 批量处理

```python
# 如果需要处理多个请求，考虑批量处理
def process_multiple_questions(questions):
    results = []
    for question in questions:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": question}]
        )
        results.append(response.choices[0].message.content)
    return results
```

---

## 总结

OpenAI API 提供了强大的 AI 能力，主要步骤：

1. **安装库**：`pip install openai`
2. **配置 API Key**：设置环境变量
3. **创建客户端**：`client = OpenAI()`
4. **调用 API**：使用 `client.chat.completions.create()`
5. **处理响应**：获取 `response.choices[0].message.content`

### 注意事项

- 保护好 API Key，不要提交到代码仓库
- 注意 API 调用成本，合理设置 `max_tokens`
- 处理网络错误和 API 限制
- 根据任务选择合适的 `temperature` 参数

### 继续学习

- [OpenAI 官方文档](https://platform.openai.com/docs)
- [API 参考](https://platform.openai.com/docs/api-reference)
- 探索更多模型和功能
