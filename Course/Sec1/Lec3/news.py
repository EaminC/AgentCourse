"""
百度新闻 RSS 爬虫 - 保存标题、链接和二级页面 HTML
尝试抓正文，如果抓不到，至少保存 HTML
"""

import feedparser
from bs4 import BeautifulSoup
import os
import requests

# RSS 链接
rss_url = "http://news.baidu.com/n?cmd=4&class=civilnews&tn=rss"
feed = feedparser.parse(rss_url)

print(f"🗞️ 找到 {len(feed.entries)} 条新闻\n")

# 创建保存文件夹
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 清理文件名函数
def clean_filename(title):
    filename = title
    for char in [":", "/", "\\", "?", "*", "<", ">", "|", '"']:
        filename = filename.replace(char, "")
    filename = filename.strip()
    if len(filename) > 50:
        filename = filename[:50]
    return filename

news_count = 0

for entry in feed.entries[:100]:
    title = entry.title
    link = entry.link
    print("📰 标题:", title)
    print("🔗 链接:", link)
    
    full_text = ""
    html_content = ""
    try:
        # 下载二级页面 HTML
        response = requests.get(link, timeout=10)
        response.encoding = response.apparent_encoding
        html_content = response.text
        
        # 尝试解析正文
        soup = BeautifulSoup(html_content, "html.parser")
        
        # 百家号正文通常在 class 包含 "article" 或 "article-content" 的 div 内
        article_div = soup.find("div", class_="article") or soup.find("div", class_="article-content")
        if article_div:
            paragraphs = article_div.find_all("p")
            text_list = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 10]
            full_text = "\n\n".join(text_list)
    except Exception as e:
        print(f"❌ 下载或解析失败: {e}")
    
    # 如果正文为空，提示用户
    if len(full_text) < 30:
        full_text = "⚠️ 正文抓取失败或为空，可打开对应 HTML 查看\n"
    
    # 保存文本信息
    filename_base = clean_filename(title)
    txt_filepath = os.path.join(data_dir, filename_base + ".txt")
    try:
        with open(txt_filepath, "w", encoding="utf-8") as f:
            f.write("标题: " + title + "\n")
            f.write("链接: " + link + "\n")
            f.write("\n" + "="*50 + "\n\n")
            f.write(full_text)
        print(f"✅ 已保存文本: {txt_filepath}")
    except Exception as e:
        print(f"❌ 保存文本失败: {e}")
    
    # 另外保存原始 HTML，方便之后手动查看或重新解析
    if html_content:
        html_filepath = os.path.join(data_dir, filename_base + ".html")
        try:
            with open(html_filepath, "w", encoding="utf-8") as f_html:
                f_html.write(html_content)
            print(f"✅ 已保存HTML: {html_filepath}")
        except Exception as e:
            print(f"❌ 保存HTML失败: {e}")
    
    news_count += 1
    print()

print(f"\n🎉 完成！共保存了 {news_count} 条新闻")
