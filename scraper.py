"""
Step 2 - scrape extra Hindi text to enlarge the vocabulary.

Two modes (both write one paragraph per line to data/raw/scraped_corpus.txt):

  wiki   Hindi Wikipedia through the official MediaWiki API (fast, allowed, clean).
         python scraper.py wiki --requests 100          # ~20 article intros per request

  web    Polite crawler for any Hindi site you have permission to scrape.
         python scraper.py web --seed https://example.com/hindi-news --max-pages 100

The web crawler obeys robots.txt, identifies itself, rate-limits requests and only
stays on the seed's domain. Check each site's terms of use before scraping it.
"""
import argparse
import time
import urllib.robotparser
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

import config
from preprocessing import devanagari_ratio, normalize_text

UA = "HindiSpellCheckerResearchBot/1.0 (student NLP project; contact: your_email@example.com)"
HEADERS = {"User-Agent": UA}


def _keep(paragraph: str) -> bool:
    return len(paragraph) >= 30 and devanagari_ratio(paragraph) >= 0.7


def _append(paragraphs, seen):
    config.SCRAPED_FILE.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    with open(config.SCRAPED_FILE, "a", encoding="utf8") as f:
        for p in paragraphs:
            p = normalize_text(p)
            if _keep(p) and p not in seen:
                seen.add(p)
                f.write(p + "\n")
                added += 1
    return added


def _load_seen():
    if config.SCRAPED_FILE.exists():
        return set(config.SCRAPED_FILE.read_text(encoding="utf8").splitlines())
    return set()


# ------------------------------------------------------------------ Wikipedia
def scrape_wiki(n_requests: int, delay: float):
    api = "https://hi.wikipedia.org/w/api.php"
    params = {"action": "query", "format": "json", "generator": "random", "grnnamespace": 0,
              "grnlimit": 20, "prop": "extracts", "exintro": 1, "explaintext": 1,
              "exlimit": "max", "maxlag": 5}
    session = requests.Session()
    session.headers.update(HEADERS)
    seen, total, done, fails = _load_seen(), 0, 0, 0
    try:
        while done < n_requests:
            try:
                r = session.get(api, params=params, timeout=20)
            except requests.RequestException as e:
                fails += 1
                print(f"network error: {e}")
                if fails > 6:
                    print("Too many network errors in a row - stopping.")
                    break
                time.sleep(min(120, 10 * 2 ** fails))
                continue

            if r.status_code in (429, 503):
                fails += 1
                if fails > 6:
                    print("Still rate-limited after 6 retries - stopping. Try again in a few minutes.")
                    break
                retry_after = r.headers.get("Retry-After", "")
                wait = int(retry_after) if retry_after.isdigit() else min(120, 10 * 2 ** fails)
                print(f"Rate limited ({r.status_code}). Waiting {wait}s before retry {fails}/6...")
                time.sleep(wait)
                continue

            r.raise_for_status()
            data = r.json()
            if "error" in data:  # e.g. 'maxlag': Wikipedia servers are busy
                print("Server busy:", data["error"].get("code"), "- waiting 10s")
                time.sleep(10)
                continue

            fails = 0
            done += 1
            pages = data.get("query", {}).get("pages", {}).values()
            paragraphs = [p for pg in pages for p in pg.get("extract", "").split("\n")]
            total += _append(paragraphs, seen)
            print(f"[{done}/{n_requests}] paragraphs saved this run: {total}")
            time.sleep(delay)
    except KeyboardInterrupt:
        print("\nStopped by user. Everything scraped so far is already saved.")


# ------------------------------------------------------------------ generic web
def scrape_web(seed: str, max_pages: int, delay: float):
    domain = urlparse(seed).netloc
    rp = urllib.robotparser.RobotFileParser(urljoin(seed, "/robots.txt"))
    try:
        rp.read()
    except Exception:
        print("Could not read robots.txt - refusing to crawl.")
        return
    queue, visited, seen, total = deque([seed]), set(), _load_seen(), 0
    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited or not rp.can_fetch(UA, url):
            continue
        visited.add(url)
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if "text/html" not in r.headers.get("content-type", ""):
                continue
            soup = BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            print("skip", url, e)
            continue
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        total += _append([p.get_text(" ", strip=True) for p in soup.find_all("p")], seen)
        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"]).split("#")[0]
            if urlparse(link).netloc == domain and link not in visited:
                queue.append(link)
        print(f"[{len(visited)}/{max_pages}] {url}  (paragraphs saved: {total})")
        time.sleep(delay)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    w = sub.add_parser("wiki")
    w.add_argument("--requests", type=int, default=50)
    w.add_argument("--delay", type=float, default=1.0)
    s = sub.add_parser("web")
    s.add_argument("--seed", required=True)
    s.add_argument("--max-pages", type=int, default=50)
    s.add_argument("--delay", type=float, default=1.5)
    a = ap.parse_args()
    if a.mode == "wiki":
        scrape_wiki(a.requests, a.delay)
    else:
        scrape_web(a.seed, a.max_pages, a.delay)
