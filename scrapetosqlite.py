import logging
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser


from sqlalchemy import String, Integer, DateTime, func, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy import create_engine


BASE_URL = "https://quotes.toscrape.com/"
DB_URL = "postgresql+psycopg://quotes_user:quotes_pass@localhost:5432/quotes"
REQUEST_DELAY = 1.0
TIMEOUT_SECS = 20

USER_AGENTS: List[str] = [
    "Mozilla/5.0 (compatible; PyScraper/1.0; +https://example.com/docs)",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/123.0 Safari/537.36",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s level=%(levelname)s msg=%(message)s",
)
log = logging.getLogger("pipeline")


class Base(DeclarativeBase):
    """Base class for SQLAlchemy ORM models."""
    pass

class Quote(Base):
    """Stores one quote row."""
    __tablename__ = "quotes"
    __table_args__ = (UniqueConstraint("text", "author", name="uq_text_author"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(String(1000))
    author: Mapped[str] = mapped_column(String(255))
    tags_csv: Mapped[str] = mapped_column(String(500), default="")
    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


engine = create_engine(DB_URL, pool_pre_ping=True, pool_size=5, max_overflow=10, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def init_db() -> None:
    """Create tables if they don't exist."""
    Base.metadata.create_all(bind=engine)

def make_session() -> requests.Session:
    """Create a requests Session with robust Retry/backoff."""
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

@dataclass
class RobotsGate:
    """Caches robots.txt and checks if a UA can fetch a URL."""
    base_url: str
    rp: RobotFileParser

    @classmethod
    def from_base(cls, base_url: str) -> "RobotsGate":
        rp = RobotFileParser()
        rp.set_url(urljoin(base_url, "robots.txt"))
        rp.read()
        return cls(base_url=base_url, rp=rp)

    def can_fetch(self, ua: str, url: str) -> bool:
        """Return True if UA may fetch URL according to robots.txt."""
        return self.rp.can_fetch(ua, url)

def pick_user_agent() -> str:
    """Pick a UA string (very simple rotation)."""
    return random.choice(USER_AGENTS)

def fetch_html(session: requests.Session, robots: RobotsGate, url: str) -> Tuple[str, str]:

    ua = pick_user_agent()
    if not robots.can_fetch(ua, url):
        raise PermissionError(f"Blocked by robots.txt for UA '{ua}': {url}")
    headers: Dict[str, str] = {"User-Agent": ua}
    log.info("fetch start url=%s ua=%s", url, ua)
    resp = session.get(url, headers=headers, timeout=TIMEOUT_SECS)
    resp.raise_for_status()
    log.info("fetch ok url=%s status=%s bytes=%s", url, resp.status_code, len(resp.content))
    return resp.text, ua

def parse_page(html: str) -> Tuple[List[Dict], Optional[str]]:
    """Parse a quotes page; return (rows, next_url_or_none)."""
    soup = BeautifulSoup(html, "html.parser")
    rows: List[Dict] = []

    for q in soup.select(".quote"):
        text = q.select_one(".text").get_text(strip=True)
        author = q.select_one(".author").get_text(strip=True)
        tags = [a.get_text(strip=True) for a in q.select(".tags a.tag")]
        rows.append({"text": text, "author": author, "tags_csv": ",".join(tags)})

    next_link = soup.select_one("li.next > a")
    next_url = urljoin(BASE_URL, next_link["href"]) if next_link else None
    return rows, next_url


def save_rows(rows: List[Dict]) -> int:
    """Insert rows into DB; skip duplicates via uniqueness constraint.
       Returns number of rows inserted."""
    inserted = 0
    with SessionLocal() as db:
        for r in rows:
            try:

                db.add(Quote(text=r["text"], author=r["author"], tags_csv=r["tags_csv"]))
                db.commit()
                inserted += 1
            except Exception as e:
                db.rollback()

                log.warning("insert skipped reason=%s author=%s", type(e).__name__, r.get("author", ""))
    return inserted


def crawl_all() -> None:
    """Crawl all pages from BASE_URL and persist results."""
    init_db()
    session = make_session()
    robots = RobotsGate.from_base(BASE_URL)
    url = BASE_URL

    total = 0
    while url:
        try:
            html, ua = fetch_html(session, robots, url)
            rows, next_url = parse_page(html)
            inserted = save_rows(rows)
            total += inserted
            log.info("page done url=%s found=%d inserted=%d next=%s",
                     url, len(rows), inserted, bool(next_url))
            time.sleep(REQUEST_DELAY)
            url = next_url
        except PermissionError as pe:
            log.error("blocked url=%s err=%s", url, pe)
            break
        except requests.RequestException as re:
            log.error("request error url=%s err=%s", url, re)
            break
        except Exception as e:
            log.exception("unexpected error url=%s err=%s", url, e)
            break

    log.info("crawl finished total_inserted=%d db=%s", total, DB_URL)

if __name__ == "__main__":
    crawl_all()
