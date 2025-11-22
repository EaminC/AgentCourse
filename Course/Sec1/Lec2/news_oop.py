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
"""

# ========== 第一步：导入模块 ==========
import feedparser      # 解析 RSS 订阅源
from bs4 import BeautifulSoup  # 解析 HTML
import os              # 文件系统操作
import requests        # 发送 HTTP 请求


# ========== 第二步：定义 NewsCrawler 类 ==========
class NewsCrawler:
    """新闻爬虫类"""
    
    def __init__(self, rss_url, data_dir="data"):
        """
        初始化爬虫
        :param rss_url: RSS 订阅源地址
        :param data_dir: 数据保存目录
        """
        self.rss_url = rss_url
        self.data_dir = data_dir
        self.news_count = 0
        
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
        :return: 新闻正文内容，失败返回空字符串
        """
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
            
            return "\n\n".join(text_list)
        except Exception as e:
            print(f"❌ 抓取失败: {e}\n")
            return ""
    
    def save_article(self, title, link, content):
        """
        保存新闻到文件
        :param title: 新闻标题
        :param link: 新闻链接
        :param content: 新闻内容
        """
        filename = self.clean_filename(title) + ".txt"
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("标题: " + title + "\n")
            f.write("链接: " + link + "\n")
            f.write("\n" + "=" * 50 + "\n\n")
            f.write(content)
        
        print(f"💾 已保存: {filename}\n")
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
            
            # 下载新闻正文
            full_text = self.download_article(link)
            
            # 如果正文为空，跳过
            if len(full_text) < 50:
                print("⚠️  正文为空，跳过\n")
                continue
            
            print(f"📄 正文: {len(full_text)} 字\n")
            
            # 保存到文件
            self.save_article(title, link, full_text)
        
        print(f"✅ 完成！共保存 {self.news_count} 条新闻")


# ========== 第三步：使用类创建对象并运行 ==========
if __name__ == "__main__":
    # 创建爬虫对象
    rss_url = "http://news.baidu.com/n?cmd=4&class=civilnews&tn=rss"
    crawler = NewsCrawler(rss_url, data_dir="data")
    
    # 运行爬虫
    crawler.run(max_news=10)

