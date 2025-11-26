"""
新闻爬虫 - Python 初学者入门案例

这个脚本演示了以下 Python 基础概念：
1. 导入模块 (import)
2. 变量和数据类型
3. 字符串操作
4. 列表 (list) 和循环 (for)
5. 条件判断 (if/else)
6. 异常处理 (try/except)
7. 文件操作
8. 函数定义
"""

# ========== 第一步：导入模块 ==========
import feedparser      # 解析 RSS 订阅源
from bs4 import BeautifulSoup  # 解析 HTML
import os              # 文件系统操作
import requests        # 发送 HTTP 请求

# ========== 第二步：设置新闻源并解析 ==========
rss_url = "http://news.baidu.com/n?cmd=4&class=civilnews&tn=rss"
feed = feedparser.parse(rss_url)

print(f"🗞️ 找到 {len(feed.entries)} 条新闻\n")

# ========== 第三步：创建保存文件夹 ==========
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# ========== 第四步：定义清理文件名的函数 ==========
def clean_filename(title):
    """清理文件名，移除特殊字符"""
    filename = title
    for char in [":", "/", "\\", "?", "*", "<", ">", "|", '"']:
        filename = filename.replace(char, "")
    filename = filename.strip()
    if len(filename) > 50:
        filename = filename[:50]
    return filename

# ========== 第五步：遍历新闻并保存 ==========
news_count = 0

for entry in feed.entries[:10]:
    title = entry.title
    link = entry.link
    
    print("📰 标题:", title)
    
    # 抓取新闻正文
    full_text = ""
    try:
        # 下载网页
        response = requests.get(link, timeout=10)
        response.encoding = response.apparent_encoding
        
        # 解析 HTML
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 提取所有段落
        paragraphs = soup.find_all("p")
        text_list = []
        for p in paragraphs:
            text = p.get_text(strip=True)
            if len(text) > 30:  # 只保留较长的段落
                text_list.append(text)
        
        full_text = "\n\n".join(text_list)
    except Exception as e:
        print(f"❌ 抓取失败: {e}\n")
        continue  # 跳过这条新闻
    
    # 如果正文为空，跳过
    if len(full_text) < 50:
        print("⚠️  正文为空，跳过\n")
        continue
    
    print(f"📄 正文: {len(full_text)} 字\n")
    
    # 保存到文件
    filename = clean_filename(title) + ".txt"
    filepath = os.path.join(data_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("标题: " + title + "\n")
        f.write("链接: " + link + "\n")
        f.write("\n" + "=" * 50 + "\n\n")
        f.write(full_text)
    
    print(f"💾 已保存: {filename}\n")
    news_count += 1

print(f"✅ 完成！共保存 {news_count} 条新闻")
