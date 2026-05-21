import asyncio
import aiohttp
import aiofiles
from urllib.parse import urljoin, urlparse
from collections import deque


async def fetch(session, url, semaphore):
    async with semaphore:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    print(f'Error {response.status} for {url}')
        except Exception as e:
            print(f'Exception {e} for {url}')


async def parse_links(base_url, html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    links = set()
    for link in soup.find_all('a', href=True):
        full_url = urljoin(base_url, link['href'])
        if is_valid_url(full_url):
            links.add(full_url)
    return links


def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


async def main(start_urls, rate_limit=10, max_pages=10):
    conn = aiohttp.TCPConnector(limit=rate_limit)
    sem = asyncio.Semaphore(rate_limit)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [fetch(session, url, sem) for url in start_urls]
        html_contents = await asyncio.gather(*tasks)
        all_links = deque(start_urls)
        unique_links = set(start_urls)
        while len(all_links) < max_pages and html_contents:
            new_links = set()
            for html in html_contents:
                if html:
                    links = await parse_links(all_links[0], html)
                    new_links.update(links - unique_links)
            all_links.extend(new_links)
            unique_links.update(new_links)
            tasks = [fetch(session, url, sem) for url in list(new_links)[:max_pages - len(all_links)]]
            html_contents = await asyncio.gather(*tasks)
        async with aiofiles.open('unique_urls.txt', 'w') as f:
            for url in unique_links:
                await f.write(f'{url}\n')


if __name__ == '__main__':
    start_urls = ['https://example.com'] * 10  # Replace with actual URLs
    asyncio.run(main(start_urls))