import asyncio
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai import AsyncWebCrawler, CacheMode, BrowserConfig, CrawlerRunConfig
from bs4 import BeautifulSoup
import uuid
"""
css selector for internal_links: tr.nsk-product-info>td:nth-child(3)>a
css_selector for a.action.next
"""
async def internal_links_extraction():
    browser_conf = BrowserConfig(headless=False,verbose = True, viewport_width=1400,viewport_height=940)
    internal_links = []
    session_id = str(uuid.uuid4())
    page_number = 1
    has_next_page_button = True
    run_conf = CrawlerRunConfig(
        word_count_threshold=10,
        cache_mode=CacheMode.BYPASS,
        exclude_all_images=True,
        exclude_domains=[],
        exclude_external_images=True,
        exclude_social_media_domains=[],
        exclude_social_media_links=True,
        excluded_tags=["nav","footer","img"],
        scan_full_page=True,
        scroll_delay=1.5,
        mean_delay=1.0,
        css_selector="tr.nsk-product-info>td:nth-child(3)>a,.next",
        
)
    async with AsyncWebCrawler(config = browser_conf) as crawler:
        print(has_next_page_button)
        while has_next_page_button:
            
            if page_number == 1:
                run_conf.delay_before_return_html = 20.0
                run_conf.session_id = session_id
            else:
                run_conf.delay_before_return_html = 20.0
                run_conf.js_code = """
        const nextButtonSelector = '.next';
        const nextButton = document.querySelector(nextButtonSelector);

    if (nextButton && nextButton.getAttribute('aria-disabled') !== 'true' && nextButton.offsetWidth > 0) {
        nextButton.click();
        return true;
    }
    return false;

        """
                run_conf.js_only = True
                run_conf.session_id = session_id
            result = await crawler.arun(url = "https://www.oss.nsk.com/products/bearing-accessories/nut.html", config=run_conf)
            soup = BeautifulSoup(result.html,"html.parser")
            next_button = soup.select_one('.next')
            print(next_button)
            for link in result.links["internal"]:
                url = link.get("href")
                if "?p=" in url:
                    continue
                if url not in internal_links:
                    internal_links.append(url)
            page_number +=1

            if next_button is None:
                has_next_page_button = False
            else:
                continue
        with open("internal_links_nsk.txt","a",encoding="utf-8") as f:
            for internal_link in internal_links:
                f.write(f"{internal_link}\n")
    return internal_links




        