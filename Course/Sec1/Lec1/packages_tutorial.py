"""
Python 包函数使用说明 - 新闻爬虫项目

本文件介绍 news.py 项目中实际用到的函数和用法
"""

# ============================================================================
# 1. feedparser - RSS 解析
# ============================================================================
print("1. feedparser 包")
print("-" * 60)

import feedparser

# feedparser.parse(url) - 解析 RSS 订阅源
rss_url = "http://news.baidu.com/n?cmd=4&class=civilnews&tn=rss"
feed = feedparser.parse(rss_url)

# feed.entries - 新闻条目列表
print(f"新闻数量: {len(feed.entries)}")

# entry.title - 新闻标题
# entry.link - 新闻链接
if len(feed.entries) > 0:
    entry = feed.entries[0]
    print(f"标题: {entry.title}")
    print(f"链接: {entry.link}")

print()

# ============================================================================
# 2. requests - HTTP 请求
# ============================================================================
print("2. requests 包")
print("-" * 60)

import requests

# requests.get(url, timeout=10) - 发送 GET 请求
url = "https://www.example.com"
response = requests.get(url, timeout=10)

# response.encoding - 设置编码
response.encoding = response.apparent_encoding

# response.text - 获取响应文本内容
html = response.text
print(f"网页内容长度: {len(html)} 字符")

print()

# ============================================================================
# 3. BeautifulSoup - HTML 解析
# ============================================================================
print("3. BeautifulSoup 包")
print("-" * 60)

from bs4 import BeautifulSoup

# BeautifulSoup(html, "html.parser") - 创建解析对象
html = "<html><body><p>段落1</p><p>段落2</p></body></html>"
soup = BeautifulSoup(html, "html.parser")

# soup.find_all("p") - 查找所有 <p> 标签
paragraphs = soup.find_all("p")
print(f"找到 {len(paragraphs)} 个段落")

# p.get_text(strip=True) - 获取标签内的文本，strip=True 去除首尾空白
for p in paragraphs:
    text = p.get_text(strip=True)
    print(f"  段落: {text}")

print()

# ============================================================================
# 4. os - 文件系统操作
# ============================================================================
print("4. os 包")
print("-" * 60)

import os

# os.path.exists(path) - 检查文件/文件夹是否存在
path = "data"
if os.path.exists(path):
    print(f"{path} 存在")
else:
    print(f"{path} 不存在")

# os.makedirs(path) - 创建文件夹
if not os.path.exists(path):
    os.makedirs(path)
    print(f"创建文件夹: {path}")

# os.path.join(dir, filename) - 组合文件路径
filename = "news.txt"
filepath = os.path.join(path, filename)
print(f"文件路径: {filepath}")

print()

# ============================================================================
# 5. 字符串方法
# ============================================================================
print("5. 字符串方法")
print("-" * 60)

text = "  Hello: World/Test  "

# str.replace(old, new) - 替换字符
new_text = text.replace(":", "").replace("/", "")
print(f"替换后: {new_text}")

# str.strip() - 去除首尾空白字符
clean_text = new_text.strip()
print(f"去除空白: {clean_text}")

# len(str) - 获取字符串长度
print(f"长度: {len(clean_text)}")

# str[:50] - 字符串切片，取前50个字符
short_text = clean_text[:50]
print(f"前50字符: {short_text}")

print()

# ============================================================================
# 6. 文件操作（内置函数）
# ============================================================================
print("6. 文件操作")
print("-" * 60)

# open(filepath, "w", encoding="utf-8") - 打开文件（写入模式）
filepath = "test.txt"
with open(filepath, "w", encoding="utf-8") as f:
    # f.write(text) - 写入内容
    f.write("第一行\n")
    f.write("第二行\n")

print(f"已写入文件: {filepath}")

print()
print("=" * 60)
print("函数说明完成")
print("=" * 60)
