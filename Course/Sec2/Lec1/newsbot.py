from news_oop import NewsCrawler

if __name__ == "__main__":
    crawler = NewsCrawler(rss_url="http://news.baidu.com/n?cmd=4&class=civilnews&tn=rss", api_key="sk-23qfb76qghixbui2", data_dir="data")
    crawler.run(max_news=10)