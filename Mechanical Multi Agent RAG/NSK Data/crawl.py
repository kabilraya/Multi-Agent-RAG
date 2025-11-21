import asyncio
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai import CacheMode, BrowserConfig, CrawlerRunConfig,AsyncWebCrawler
"""
data css_selectors = td.list-attr-product
"""
async def product_specification_extraction():
    browser_conf = BrowserConfig(headless = True, verbose = True, viewport_width=1400, viewport_height=940)
    run_conf = CrawlerRunConfig(
        word_count_threshold=10,
        scan_full_page=True,
        cache_mode=CacheMode.DISABLED,
        scroll_delay=1.5,
        mean_delay=1.0,
        css
    ) 