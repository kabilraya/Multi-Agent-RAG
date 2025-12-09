import asyncio

from crawl4ai import AsyncWebCrawler, CacheMode, BrowserConfig, CrawlerRunConfig,LLMConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
import re
from bs4 import BeautifulSoup
import json
import pandas as pd
import os
from dotenv import load_dotenv
import uuid
import time

async def crawl_with_pagination():
    all_data = []
    total_crawl_time = 0
    total_extract_time = 0
    
    page_count = 0
    base_url = "https://vn.misumi-ec.com/vona2/detail/221005675716?list=PageCategory"
    browser_config = BrowserConfig(headless = False, verbose = True, viewport_width=1400,viewport_height=940)
    run_config = CrawlerRunConfig(
        word_count_threshold=10,
        cache_mode=CacheMode.DISABLED,
        mean_delay=2.0,
        page_timeout=30.0,
        exclude_all_images=True,
        exclude_domains=[],
        exclude_external_links=True,
        remove_forms=True,
        remove_overlay_elements=True,
        check_robots_txt=True,
        css_selector = "div.PartNumberList_mainOuter__d74Qg,div.PartNumberList_head__DR_PF,dl.SpecTable_specTable__XNeik"
        )
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    session_id = str(uuid.uuid4())
    query = "Generate a schema to extract product details.The column names from the tables MUST exact. Only use the table column names as the key_fields in the json schema(Split the Part Number header into two keys - Part Number Name and Part Number URL)  You MUST use CSS selectors, not XPath. The base selector MUST ONLY  be 'div.mergedRow'."

    schema_conf = CrawlerRunConfig(
        word_count_threshold=10,
        mean_delay=2.0,
        delay_before_return_html=40.0,
        js_code="""
                return new Promise((resolve)=>{
                setTimeout(()=>{
                const langButton = document.querySelector('a[lang="en"]');
                if (langButton) langButton.click();
                },3000)
                
                setTimeout(() => {
                const partTab = document.querySelector('a#codeList');
                if (partTab) partTab.click();
                
            setTimeout(()=>{
            const partRows = document.querySelectorAll('tr.PartNumberColumn_dataRow__43D6Y');
            const priceRows = document.querySelectorAll('tr.PartNumberAsideColumns_dataRow__OUw8N');
            const specRows = document.querySelectorAll('tr.PartNumberSpecColumns_dataRow__M4B4a');

            const mainContainer = document.querySelector('div.PartNumberList_mainOuter__d74Qg');
            const mergedContainer = document.createElement('div');
            mergedContainer.classList.add('mergedRowContainer');
            const numRows = Math.min(partRows.length, priceRows.length, specRows.length);
            for (let i = 0; i < numRows; i++) {
            const row = document.createElement('div');
            row.classList.add('mergedRow');
            row.appendChild(partRows[i]);
            row.appendChild(priceRows[i]);
            row.appendChild(specRows[i]);
            mainContainer.appendChild(row)
            }
            resolve(true);
            },10000);
            }, 20000);
            });
            """,
        session_id=session_id
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        
            crawl_start = time.perf_counter()
            initial_result = await crawler.arun(url = base_url, config=schema_conf)
            crawl_end = time.perf_counter()
            crawl_duration = crawl_end - crawl_start
            total_crawl_time += crawl_duration

            extract_start = time.perf_counter()
            soup = BeautifulSoup(initial_result.html,"html.parser")
            table_container = soup.select_one("div.PartNumberList_mainOuter__d74Qg")
            title_element = soup.select_one("h1.PageHeading_wrap__K1c1n")
            sub_category = title_element.get_text(strip=True) if title_element else "unknown_product" 

            cleaned_html = str(table_container)
            css_schema = JsonCssExtractionStrategy.generate_schema(
            html=cleaned_html,
            schema_type="css", 
            query=query,
            llm_config = LLMConfig(provider="gemini/gemini-2.5-flash",api_token=api_key)
            )
            extract_end = time.perf_counter()
            extract_duration = extract_end - extract_start
            total_extract_time += extract_duration
            print(css_schema)

            #extraction starts from here

            current_page = 1
            has_next_page = True
            while has_next_page:
                page_count+=1
                if current_page == 1:
                    run_config.js_only = True
                    run_config.session_id = session_id
                    run_config.delay_before_return_html = 30.0
                    run_config.extraction_strategy = JsonCssExtractionStrategy(schema=css_schema)



                else:
                    run_config.delay_before_return_html = 60.0
                    run_config.js_code = """
                    return new Promise((resolve)=>{
                                setTimeout(()=>{
                        document.querySelector('a#codeList').click();

                        setTimeout(() => {


                        const nextButton = document.querySelector('a.Pagination_link__5eELX.Pagination_next__5fRp8');
                        if(nextButton){
                            nextButton.click();

                        setTimeout(()=>{
                                const mainContainer = document.querySelector('div.PartNumberList_mainOuter__d74Qg');


                                const existingRows = mainContainer.querySelectorAll('div.mergedRow');
                                existingRows.forEach(r => r.remove());


                                    // Step 4: Run the row-merging logic on the NEW data for page 2.
                                    const partRows = document.querySelectorAll('tr.PartNumberColumn_dataRow__43D6Y');
                                    const priceRows = document.querySelectorAll('tr.PartNumberAsideColumns_dataRow__OUw8N');
                                    const specRows = document.querySelectorAll('tr.PartNumberSpecColumns_dataRow__M4B4a');
                                    const numRows = Math.min(partRows.length, priceRows.length, specRows.length);


                                    for (let j = 0; j < numRows; j++) {
                                        const row = document.createElement('div');
                                        row.classList.add('mergedRow');
                                        row.appendChild(partRows[j]);
                                        row.appendChild(priceRows[j]);
                                        row.appendChild(specRows[j]);
                                        mainContainer.appendChild(row);
                                    }
                                },20000);
                                }
                                resolve(true);
                                },3000);
                                },5000);
                                });
                            """,
                    run_config.extraction_strategy = JsonCssExtractionStrategy(schema=css_schema)
                    run_config.js_only = True
                    run_config.session_id = session_id
                crawl_start = time.perf_counter()
                result = await crawler.arun(url = base_url, config = run_config)
                crawl_end = time.perf_counter()
                crawl_duration = crawl_end - crawl_start
                total_crawl_time += crawl_duration

                extract_start = time.perf_counter()
                soup = BeautifulSoup(result.html,"html.parser")
                next_button = soup.select_one("a.Pagination_link__5eELX.Pagination_next__5fRp8")
                data = json.loads(result.extracted_content)
                extract_end = time.perf_counter()
                extract_duration = extract_end - extract_start
                total_extract_time += extract_duration
                all_data.extend(data)
                current_page += 1
                if next_button == None:
                    has_next_page = False
                else:
                    continue
            print(f"Average Crawl Time: {total_crawl_time/page_count}")
            print(f"Average Extraction Time: {total_extract_time/page_count}")
            total_time = total_crawl_time + total_extract_time
            print(f"Average Scrape Time: {total_time/page_count}")
            print(len(all_data))
            df = pd.DataFrame(all_data)
            sub_category = re.sub(r'[/ - , : ( ) -]+', '_', sub_category)
            file_name = f"{sub_category}.xlsx".lower()
            df.to_excel(file_name, index=False)
            with open("crawl_time.txt","a",encoding="utf-8") as f:
                f.write(f"{sub_category}\n\nAverage time to crawl:{total_time/page_count}\nPage Count: {page_count}\n\n")

asyncio.run(crawl_with_pagination())