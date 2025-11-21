from crawl4ai import AsyncWebCrawler,CacheMode,BrowserConfig,CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
import asyncio
import uuid

"""
div.SeriesListControl_seriesListControl__qpxrG>div:nth-child(2)>div>li:nth-child(2)>a[href="?Page=2"]
"""

async def product_links_extraction():
    session_id = str(uuid.uuid4())
    internal_links = []
    browser_config = BrowserConfig(headless =False, verbose = True)
    async with AsyncWebCrawler(config = browser_config) as crawler:

        for i in range(1,3):
            if i==1:
                run_conf = CrawlerRunConfig(
            cache_mode = CacheMode.DISABLED,
            css_selector = "a.PhotoItem_seriesNameLink__pq1vO",
            excluded_tags=["nav","header","footer","aside","img"],
            exclude_all_images=True,
            exclude_domains=[],
            exclude_external_links=True,
            exclude_social_media_domains=[],
            exclude_social_media_links=True,
            markdown_generator=DefaultMarkdownGenerator(),
            simulate_user=True,
            delay_before_return_html=10.0,
            mean_delay=2.0,
            remove_overlay_elements=True,
            word_count_threshold=10,
            check_robots_txt=False,
            session_id=session_id,
            # scan_full_page=True
                )
                result = await crawler.arun(url = "https://vn.misumi-ec.com/vona2/mech/M0800000000/M0810000000/M0810010000/", config = run_conf)
            else:
                run_conf = CrawlerRunConfig(
                    cache_mode = CacheMode.DISABLED,
                    css_selector = "a.PhotoItem_seriesNameLink__pq1vO",
                    excluded_tags=["nav","header","footer","aside","img"],
                    exclude_all_images=True,
                    exclude_domains=[],
                    exclude_external_links=True,
                    exclude_social_media_domains=[],
                    exclude_social_media_links=True,
                    markdown_generator=DefaultMarkdownGenerator(),
                    simulate_user=True,
                    delay_before_return_html=15.0,
                    mean_delay=2.0,
                    remove_overlay_elements=True,
                    word_count_threshold=10,
                    check_robots_txt=False,
                    # scroll_delay=1.5,
                    js_code = f"""
                    setTimeout(()=>{{
                        document.querySelector('div.SeriesListControl_seriesListControl__qpxrG>div:nth-child(2)>div>li:nth-child(2)>a[href="?Page={i}"]').click();
                    }},5000)

                    """,
                    js_only=True,
                    session_id=session_id,
                    # scan_full_page=True
                    )
                result =await crawler.arun(url = "https://vn.misumi-ec.com/vona2/mech/M0800000000/M0810000000/M0810010000/",config = run_conf)
            if result.success:
                new_links_found = 0
                for link in result.links["internal"]:
                    url = link.get("href")
                    if url not in internal_links:
                        internal_links.append(url)
                        new_links_found+=1
                    print(new_links_found)
        with open("internal_links.txt","a",encoding="utf-8") as f:
            for link in internal_links:
                f.write(link + "\n")
    return internal_links

