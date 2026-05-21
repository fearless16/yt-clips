import asyncio
import aiohttp
from urllib.parse import urljoin, urlparse
from collections import Counter
import time


CONCURRENCY_LIMIT = 10
REQUEST_DELAY = 0.5
OUTPUT_FILE = "unique_urls.txt"
TIMEOUT = aiohttp.ClientTimeout(total=30)


async def fetch_page(session, url, semaphore):
    async with semaphore:
        await asyncio.sleep(REQUEST_DELAY)
        try:
            async with session.get(url, timeout=TIMEOUT) as response:
                if response.status == 200:
                    return await response.text()
                print(f"[{response.status}] {url}")
                return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"[ERROR] {url}: {e}")
            return None


def parse_hrefs(html, base_url):
    hrefs = []
    start = 0
    while True:
        idx = html.find("<a", start)
        if idx == -1:
            break
        end = html.find(">", idx)
        tag = html[idx:end + 1]
        for attr in ['href="', "href='"]:
            astart = tag.find(attr)
            if astart != -1:
                astart += len(attr)
                aend = tag.find(attr[0], astart)
                href = tag[astart:aend]
                full_url = urljoin(base_url, href)
                hrefs.append(full_url)
                break
        start = end + 1
    return hrefs


async def scrape(urls):
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    all_hrefs = []

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_page(session, url, semaphore) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for url, html in zip(urls, results):
        if isinstance(html, str):
            hrefs = parse_hrefs(html, url)
            all_hrefs.extend(hrefs)

    unique_urls = sorted(set(all_hrefs))
    with open(OUTPUT_FILE, "w") as f:
        for url in unique_urls:
            f.write(url + "\n")

    print(f"Scraped {len(urls)} pages, found {len(all_hrefs)} links, {len(unique_urls)} unique saved to {OUTPUT_FILE}")
    return unique_urls


if __name__ == "__main__":
    import sys

    default_urls = [
        "https://example.com",
        "https://httpbin.org",
        "https://python.org",
        "https://github.com",
        "https://news.ycombinator.com",
    ]
    urls = sys.argv[1:] if len(sys.argv) > 1 else default_urls

    start = time.monotonic()
    asyncio.run(scrape(urls))
    elapsed = time.monotonic() - start
    print(f"Completed in {elapsed:.2f}s")
