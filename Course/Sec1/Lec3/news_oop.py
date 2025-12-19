"""
新闻爬虫 - Python 初学者入门案例（面向对象版本）

这个脚本演示了以下 Python 基础概念：
1. 导入模块 (import)
2. 变量和数据类型
3. 字符串操作
4. 列表 (list) 和循环 (for)
5. 条件判断 (if/else)
6. 异常处理 (try/except)
7. 文件操作
8. 函数定义
9. 面向对象编程（类、对象、方法）

注意：AI 相关功能已移动到 Sec2/Lec1，如需使用 AI 功能，请从 Sec2/Lec1 导入 AIAssistant 类。
"""

# ========== 第一步：导入模块 ==========
import feedparser      # 解析 RSS 订阅源
from bs4 import BeautifulSoup  # 解析 HTML
import os              # 文件系统操作
import requests        # 发送 HTTP 请求
import json            # JSON 处理

# 尝试从 Sec2/Lec1 导入 AIAssistant（如果可用）
try:
    import sys
    from pathlib import Path
    # 添加 Sec2/Lec1 目录到路径
    # news_oop.py 在 Sec1/Lec3，需要找到 Sec2/Lec1
    current_path = Path(__file__).resolve()
    # 从 Sec1/Lec3 回到 Course，然后进入 Sec2/Lec1
    sec2_lec1_path = current_path.parent.parent.parent / "Sec2" / "Lec1"
    if str(sec2_lec1_path) not in sys.path:
        sys.path.insert(0, str(sec2_lec1_path))
    from ai_assistant import AIAssistant
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    AIAssistant = None


# ========== 第二步：定义 NewsCrawler 类 ==========
class NewsCrawler:
    """新闻爬虫类"""
    
    def __init__(self, rss_url, api_key=None, data_dir="data"):
        """
        初始化爬虫
        :param rss_url: RSS 订阅源地址
        :param api_key: AI API 密钥（可选，如果提供则启用 AI 摘要功能）
        :param data_dir: 数据保存目录
        """
        self.rss_url = rss_url
        self.data_dir = data_dir
        self.news_count = 0
        
        # 如果提供了 API key 且 AI 功能可用，则初始化 AI 助手
        if api_key and AI_AVAILABLE:
            self.ai_assistant = AIAssistant(api_key)
            self.ai_enabled = True
        else:
            self.ai_assistant = None
            self.ai_enabled = False
            if api_key and not AI_AVAILABLE:
                print("⚠️  警告：AI 功能不可用，将跳过 AI 摘要功能")
        
        # 创建保存文件夹
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def clean_filename(self, title):
        """清理文件名，移除特殊字符"""
        filename = title
        for char in [":", "/", "\\", "?", "*", "<", ">", "|", '"']:
            filename = filename.replace(char, "")
        filename = filename.strip()
        if len(filename) > 50:
            filename = filename[:50]
        return filename
    
    def download_article(self, link):
        """
        下载并解析新闻正文
        :param link: 新闻链接
        :return: (HTML内容, 纯文本内容)，失败返回("", "")
        """
        try:
            # 下载网页
            response = requests.get(link, timeout=10)
            response.encoding = response.apparent_encoding
            html_content = response.text
            
            # 解析 HTML
            soup = BeautifulSoup(html_content, "html.parser")
            
            # 尝试找到文章主体
            article_div = soup.find("div", class_="article") or soup.find("div", class_="article-content")
            if article_div:
                paragraphs = article_div.find_all("p")
            else:
                # 如果找不到特定的 div，就提取所有段落
                paragraphs = soup.find_all("p")
            
            text_list = []
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 10:  # 只保留较长的段落
                    text_list.append(text)
            
            full_text = "\n\n".join(text_list)
            return html_content, full_text
        except Exception as e:
            print(f"❌ 抓取失败: {e}\n")
            return "", ""
    
    def summarize_content(self, title, content):
        """
        使用 AI 总结新闻内容（如果 AI 功能可用）
        :param title: 新闻标题
        :param content: 新闻内容
        :return: AI 生成的摘要，如果 AI 不可用则返回提示信息
        """
        if not self.ai_enabled:
            return "AI 摘要功能未启用（需要 API key 且 AI 模块可用）"
        
        if not content or len(content) < 50:
            return "内容过短，无法生成摘要"
        
        # 限制内容长度，避免 token 过多
        content_preview = content[:2000]
        
        prompt = f"""请用中文总结以下新闻内容，要求：
1. 简洁明了，3-5句话
2. 突出关键信息
3. 保持客观中立

新闻标题：{title}

新闻内容：
{content_preview}
"""
        
        print("🤖 正在生成 AI 摘要...")
        summary = self.ai_assistant.get_response(prompt)
        return summary if summary else "AI 摘要生成失败"
    
    def save_article(self, title, link, html_content, text_content):
        """
        保存新闻到文件（包括 HTML 和文本）
        :param title: 新闻标题
        :param link: 新闻链接
        :param html_content: HTML 原始内容
        :param text_content: 纯文本内容
        """
        filename_base = self.clean_filename(title)
        
        # 1. 保存 HTML 文件
        if html_content:
            html_filepath = os.path.join(self.data_dir, filename_base + ".html")
            try:
                with open(html_filepath, "w", encoding="utf-8") as f:
                    f.write(html_content)
                print(f"💾 已保存 HTML: {filename_base}.html")
            except Exception as e:
                print(f"❌ 保存 HTML 失败: {e}")
        
        # 2. 生成 AI 摘要（如果启用）
        summary = self.summarize_content(title, text_content) if self.ai_enabled else None
        
        # 3. 保存文本文件（包含原文和 AI 摘要，如果有）
        txt_filepath = os.path.join(self.data_dir, filename_base + ".txt")
        try:
            with open(txt_filepath, "w", encoding="utf-8") as f:
                f.write("标题: " + title + "\n")
                f.write("链接: " + link + "\n")
                if summary:
                    f.write("\n" + "=" * 50 + "\n")
                    f.write("AI 摘要：\n")
                    f.write(summary + "\n")
                    f.write("\n" + "=" * 50 + "\n\n")
                f.write("原文内容：\n\n")
                f.write(text_content if text_content else "⚠️ 正文抓取失败或为空")
            print(f"💾 已保存文本: {filename_base}.txt\n")
        except Exception as e:
            print(f"❌ 保存文本失败: {e}\n")
        
        self.news_count += 1
    
    def run(self, max_news=10):
        """
        运行爬虫主流程
        :param max_news: 最多爬取多少条新闻
        """
        # 解析 RSS 订阅源
        feed = feedparser.parse(self.rss_url)
        print(f"🗞️ 找到 {len(feed.entries)} 条新闻\n")
        
        # 遍历新闻并保存
        for entry in feed.entries[:max_news]:
            title = entry.title
            link = entry.link
            
            print("📰 标题:", title)
            print("🔗 链接:", link)
            
            # 下载新闻 HTML 和正文
            html_content, text_content = self.download_article(link)
            
            # 如果正文太短，标记但仍然保存
            if len(text_content) < 30:
                print("⚠️  正文较短或抓取不完整")
                text_content = "⚠️ 正文抓取失败或为空，可打开对应 HTML 查看"
            else:
                print(f"📄 正文: {len(text_content)} 字")
            
            # 保存到文件（包括 HTML 和文本，如果启用 AI 则包含摘要）
            self.save_article(title, link, html_content, text_content)
        
        print(f"✅ 完成！共保存 {self.news_count} 条新闻")


# ========== 第四步：使用类创建对象并运行 ==========
if __name__ == "__main__":
    # API 密钥（建议使用环境变量）
    API_KEY = os.getenv("API_KEY") or "sk-23qfb76qghixbui2"
    
    if not API_KEY:
        raise RuntimeError("API_KEY 未设置，请设置环境变量或在代码中指定")
    
    # 创建爬虫对象
    rss_url = "http://news.baidu.com/n?cmd=4&class=civilnews&tn=rss"
    crawler = NewsCrawler(rss_url, api_key=API_KEY, data_dir="data")
    
    # 运行爬虫（默认爬取 10 条）
    crawler.run(max_news=10)

