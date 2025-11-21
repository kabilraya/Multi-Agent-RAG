import asyncio
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai import CacheMode, BrowserConfig, CrawlerRunConfig,AsyncWebCrawler
"""
data css_selectors = td.list-attr-product
"""
async def product_specification_extraction():
    browser_conf = BrowserConfig(headless = False, verbose = True, viewport_width=1400, viewport_height=940)
    run_conf = CrawlerRunConfig(
        
        scan_full_page=True,
        cache_mode=CacheMode.DISABLED,
        scroll_delay=1.5,
        page_timeout=60000,
        mean_delay=1.0,
        css_selector=".list-attr-product",
        exclude_domains=[],
        delay_before_return_html=10.0,
        exclude_external_links=True,
        exclude_internal_links=True,
        exclude_social_media_domains=[],
        exclude_social_media_links=True
    )
    async with AsyncWebCrawler(config = browser_conf) as crawler:
        result = await crawler.arun(url = "https://www.oss.nsk.com/products/bearing-accessories/nut/an02.html",config = run_conf)
        if result.success:
            print(result.markdown)

asyncio.run(product_specification_extraction())