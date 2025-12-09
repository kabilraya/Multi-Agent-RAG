import asyncio
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai import CacheMode, BrowserConfig, CrawlerRunConfig,AsyncWebCrawler
from bs4 import BeautifulSoup
import pandas as pd
from internal_links_crawl import internal_links_extraction
"""
data css_selectors = td.list-attr-product
"""
async def product_specification_extraction():
    all_links = await internal_links_extraction()
    all_products = []
    
    browser_conf = BrowserConfig(headless = False, verbose = True, viewport_width=1400, viewport_height=940)
    run_conf = CrawlerRunConfig(
        
        scan_full_page=True,
        cache_mode=CacheMode.DISABLED,
        scroll_delay=1.5,
        page_timeout=60000,
        mean_delay=1.0,
        
        exclude_domains=[],
        delay_before_return_html=10.0,
        exclude_external_links=True,
        exclude_internal_links=True,
        exclude_social_media_domains=[],
        exclude_social_media_links=True
    )
    async with AsyncWebCrawler(config = browser_conf) as crawler:
        for url in all_links:
            row_data = {}
            result = await crawler.arun(url = url,config = run_conf)
            if result.success:
                soup = BeautifulSoup(result.html,"html.parser")

                designation_name = soup.select_one("tr.attr-row.tr-designations>td:nth-child(2)")
                product_name = designation_name.text.strip() if designation_name else "Unknown"

                row_data["product_name"] = product_name
                tables = soup.select("table.nsk-table-product-attr")
                spec_tables = tables[1:]
                for table in spec_tables:
                    for tr in table.find_all("tr"):
                        cols = [td.get_text(strip = True) for td in tr.find_all("td")]

                        value = cols[1]
                        unit = cols[2]
                        attribute_name = cols[-1]
                        symbol = cols[0]

                        final_value = f"{value} {unit}".strip()
                        final_attribute = f"{attribute_name} ({symbol})"

                        row_data[final_attribute] = final_value
            row_data["URL"] = url
            all_products.append(row_data)
    df = pd.DataFrame(all_products)
    df.to_excel("nsk.xlsx", index=False)
    print(df)

asyncio.run(product_specification_extraction())