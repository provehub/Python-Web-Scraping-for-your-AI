import csv
import time
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser
import requests
from bs4 import BeautifulSoup
from typing import Dict

BASE_URL = "https://quotes.toscrape.com/"

HEADERS: Dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (compatible; PyScraper/1.0)"
}

CSV_PATH = "quotes.csv"
REQUEST_DELAY = 1.0

def can_fetch(url: str) -> bool:
    rp = RobotFileParser()
    rp.set_url(urljoin(BASE_URL, "robots.txt"))
    rp.read()
    return rp.can_fetch(HEADERS["User-Agent"], url)

def fetch_html(url: str) -> str:
    if not can_fetch(url):
        raise PermissionError(f"Blocked by robots.txt: {url}")
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    return response.text

def parse_page(html: str) -> tuple[list[dict], str | None]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict] = []

    for q in soup.select(".quote"):
        text = q.select_one(".text").get_text(strip=True)
        author = q.select_one(".author").get_text(strip=True)
        tags = [a.get_text(strip=True) for a in q.select(".tags a.tag")]
        results.append({
            "text": text,
            "author": author,
            "tags": ",".join(tags)
        })

    next_link = soup.select_one("li.next > a")
    next_url = urljoin(BASE_URL, next_link["href"]) if next_link else None
    return results, next_url

def write_csv(rows: list[dict], path: str) -> None:
    fieldnames = ["text", "author", "tags"]
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def scrape_all(start_url: str) -> list[dict]:
    all_rows: list[dict] = []
    url = start_url
    while url:
        print(f"Scraping {url}")
        html = fetch_html(url)
        rows, next_url = parse_page(html)
        all_rows.extend(rows)
        time.sleep(REQUEST_DELAY)
        url = next_url
    return all_rows

if __name__ == "__main__":
    data = scrape_all(BASE_URL)
    write_csv(data, CSV_PATH)
    print(f"Finished scraping {len(data)} records → {CSV_PATH}")